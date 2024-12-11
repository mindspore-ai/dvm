/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unordered_map>
#include <cmath>
#include <vector>
#include "dvm.h"
#include "kernel.h"
#include "xkernel.h"
#include "msprof.h"
#include "tuning.h"
#include "comm.h"

namespace dvm {
namespace {
static const BinarySOpType binary_map[kBinaryOpEnd] = {
  kEquals, kNotEquals,    kGreaters,     kGreaterEquals, kLesss,    kLessEquals,   kAdds,         kBinarySOpEnd,
  kMuls,   kBinarySOpEnd, kBinarySOpEnd, kMaximums,      kMinimums, kBinarySOpEnd, kBinarySOpEnd,
};

static const BinarySOpType lhs_val_binary_map[kBinaryOpEnd] = {
  kEquals, kNotEquals, kLesss,        kLessEquals, kGreaters, kGreaterEquals, kAdds,         kBinarySOpEnd,
  kMuls,   kDivs,      kBinarySOpEnd, kMaximums,   kMinimums, kBinarySOpEnd,  kBinarySOpEnd,
};

template <typename T>
bool isInteger(const T &value) {
  if constexpr (std::is_integral<T>::value) {
    return true;
  } else if constexpr (std::is_floating_point<T>::value) {
    return std::floor(value) == value;
  }
  return false;
}

template <typename T>
NDObject *PowS(Kernel *kernel, NDObject *obj, const T &value) {
  int64_t iter_num = std::abs(static_cast<int64_t>(value));
  if (iter_num == 0) {
    return kernel->Broadcast(static_cast<T>(1), obj->shape_ref_, obj->type_id_, false);
  }
  NDObject *res = nullptr;
  if (iter_num == 1) {
    res = kernel->Copy(obj);
  } else {
    while (iter_num) {
      if (iter_num & 1) {
        res = res == nullptr ? obj : kernel->Binary(BinaryOpType::kMul, res, obj);
      }
      if (iter_num != 1) {
        obj = kernel->Binary(BinaryOpType::kMul, obj, obj);
      }
      iter_num >>= 1;
    }
  }
  if (value < 0) {
    res = kernel->Unary(UnaryOpType::kReciprocal, res);
  }
  return res;
}

template <typename T, bool rhs_val>
NDObject *GetBinaryS(Kernel *kernel, int op_type, T val, NDObject *input) {
  auto vkernel = kernel->GetImpl();
  switch (op_type) {
    case BinaryOpType::kAdd:
    case BinaryOpType::kMul:
    case BinaryOpType::kMaximum:
    case BinaryOpType::kMinimum: {
      auto obj = new BinaryScalarOp<T>(binary_map[op_type], input, val);
      vkernel->Append(obj);
      return obj;
    }
    case BinaryOpType::kEqual:
    case BinaryOpType::kNotEqual:
    case BinaryOpType::kGreater:
    case BinaryOpType::kLessEqual:
    case BinaryOpType::kGreaterEqual:
    case BinaryOpType::kLess: {
      if constexpr (std::is_same<T, float>::value) {
        auto obj = new CompareScalarOp(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input, val);
        vkernel->Append(obj);
        return obj;
      }
      return nullptr;
    }
    case BinaryOpType::kPow: {
      if (rhs_val) {
        if (isInteger(val)) return PowS(kernel, input, val);
      } else if (val > 0) {
        if (input->type_id_ != kFloat32) {
          auto result = GetBinaryS<float, false>(kernel, BinaryOpType::kPow, (float)val, kernel->Cast(input, kFloat32));
          return kernel->Cast(result, input->type_id_);
        }
        auto tmp = GetBinaryS<float, true>(kernel, BinaryOpType::kMul, std::log((float)val), input);
        return kernel->Unary(UnaryOpType::kExp, tmp);
      }
      return nullptr;
    }
    case BinaryOpType::kSub: {
      if (rhs_val) {
        return GetBinaryS<T, true>(kernel, BinaryOpType::kAdd, -val, input);
      } else {
        auto neg_input = GetBinaryS<T, true>(kernel, BinaryOpType::kMul, static_cast<T>(-1.0), input);
        return GetBinaryS<T, true>(kernel, BinaryOpType::kAdd, val, neg_input);
      }
    }
    case BinaryOpType::kDiv: {
      if (rhs_val) {
        return GetBinaryS<T, true>(kernel, BinaryOpType::kMul, static_cast<T>(1.0 / val), input);
      } else {
        auto obj = new BinaryScalarOp<T>(lhs_val_binary_map[op_type], input, val);
        vkernel->Append(obj);
        return obj;
      }
    }
    default:
      return nullptr;
  }
}
}  // namespace

Comm::~Comm() {
  if (comm_) delete comm_;
}

bool Comm::Init(int rank_id, int rank_size) {
  if (!comm_) {
    comm_ = new Communicator(rank_id, rank_size);
  }
  return comm_->Init();
}

Kernel::Kernel() : kernel_{nullptr}, msprof_helper_{nullptr} {}

Kernel::~Kernel() {
  delete kernel_;
  delete msprof_helper_;
}

void Kernel::Reset(KernelType type) {
  if (kernel_) {
    delete kernel_;
  }
  if (type == kStaticShape) {
    kernel_ = new VKernelS();
  } else if (type == kDynShape) {
    kernel_ = new VKernelD();
  } else if (type == kStaticParallel) {
    kernel_ = new VKernelP();
  } else if (type == kStaticMix) {
    kernel_ = new MixKernel();
  } else if (type == kStaticStages) {
    kernel_ = new StagesKernel();
  } else {
    ASSERT(0);
  }
}

NDObject *Kernel::Load(void *addr, ShapeRef *shape, DType type) {
  auto ktype = kernel_->KType();
  if (ktype == kStaticStages) {
    ktype = static_cast<StagesKernel *>(kernel_)->Current()->KType();
  }
  NDObject *obj;
  if (ktype == kStaticMix) {
    obj = new NDSLoad(static_cast<uint8_t *>(addr), shape, type);
  } else {
    obj = new NDLoad(static_cast<uint8_t *>(addr), shape, type);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DType type) {
  auto obj = new NDSliceLoad(static_cast<uint8_t *>(addr), shape, start, size, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step,
                                   DType type) {
  auto obj = new NDStridedSliceLoad(static_cast<uint8_t *>(addr), shape, start, end, step, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Unary(int op_type, NDObject *input) {
  if (GetDType(input) == kInt32) {
    if (op_type == UnaryOpType::kAbs) {
      return Binary(BinaryOpType::kMaximum, input, Binary(BinaryOpType::kMul, input, -1));
    }
  }
  if (op_type == UnaryOpType::kLogicalNot) {
    if (input->type_id_ == kBool) {
      return Cast(Binary(BinaryOpType::kSub, 1.0f, Cast(input, kFloat16)), kBool);
    } else if (input->type_id_ == kInt32) {
      return Binary(BinaryOpType::kSub, 1, input);
    } else {
      return Binary(BinaryOpType::kSub, 1.0f, input);
    }
  }
  if (op_type == UnaryOpType::kReciprocal) {
    if (input->type_id_ == kInt32 || input->type_id_ == kBool) {
      return Cast(Binary(BinaryOpType::kDiv, 1.0f, Cast(input, kFloat32)), input->type_id_);
    } else {
      return Binary(BinaryOpType::kDiv, 1.0f, input);
    }
  }
  NDObject *obj;
  obj = new UnaryOp(op_type, input);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Binary(int op_type, NDObject *lhs, NDObject *rhs) {
  if (lhs->type_id_ == DType::kBool) {
    // Binary may introduce broadcast which is not supported in Bool. So Cast to f16.
    auto cast1 = Cast(lhs, DType::kFloat16);
    auto cast2 = Cast(rhs, DType::kFloat16);
    auto bin = Binary(op_type, cast1, cast2);
    auto cast3 = Cast(bin, DType::kBool);
    return cast3;
  }
  if (op_type < V_CMP_ALL && lhs->type_id_ == kInt32) {
    if (op_type == kEqual || op_type == kNotEqual) {
      auto sub = Binary(BinaryOpType::kSub, rhs, lhs);
      auto abs = Binary(BinaryOpType::kMaximum, sub, Binary(BinaryOpType::kMul, sub, -1));
      auto min = Binary(BinaryOpType::kMinimum, abs, 1);
      if (op_type == kEqual) {
        return Binary(BinaryOpType::kSub, 1, min);
      }
      return min;
    } else {
      NDObject *sub;
      if (op_type == kGreaterEqual || op_type == kGreater) {
        sub = Binary(BinaryOpType::kSub, lhs, rhs);
      } else {
        sub = Binary(BinaryOpType::kSub, rhs, lhs);
      }
      if (op_type == kLessEqual || op_type == kGreaterEqual) {
        sub = Binary(BinaryOpType::kAdd, sub, 1);
      }
      auto ret = Binary(BinaryOpType::kMinimum, sub, 1);
      ret = Binary(BinaryOpType::kMaximum, ret, 0);
      return ret;
    }
  }
  NDObject *obj;
  if (op_type == BinaryOpType::kPow) {
    if (lhs->type_id_ == kFloat16) {
      obj = new PowerOp(Cast(lhs, kFloat32), Cast(rhs, kFloat32));
      kernel_->Append(obj);
      return Cast(obj, kFloat16);
    }
    obj = new PowerOp(lhs, rhs);
  } else if (op_type < V_CMP_ALL) {
    obj = new CompareOp(op_type, lhs, rhs);
  } else {
    obj = new BinaryOp(op_type, lhs, rhs);
  }
  kernel_->Append(obj);
  return obj;
}

template <typename T>
NDObject *Kernel::Binary(int op_type, T val, NDObject *rhs) {
  NDObject *obj = GetBinaryS<T, false>(this, op_type, val, rhs);
  if (obj == nullptr) {
    NDObject *broadcast = new BroadcastScalarOp<T>(val, rhs->shape_ref_, rhs->type_id_, nullptr);
    kernel_->Append(broadcast);
    return Binary(op_type, broadcast, rhs);
  }
  return obj;
}

template <typename T>
NDObject *Kernel::Binary(int op_type, NDObject *lhs, T val) {
  NDObject *obj = GetBinaryS<T, true>(this, op_type, val, lhs);
  if (obj == nullptr) {
    NDObject *broadcast = new BroadcastScalarOp<T>(val, lhs->shape_ref_, lhs->type_id_, nullptr);
    kernel_->Append(broadcast);
    return Binary(op_type, lhs, broadcast);
  }
  return obj;
}

template NDObject *Kernel::Binary<float>(int op_type, NDObject *lhs, float val);
template NDObject *Kernel::Binary<int32_t>(int op_type, NDObject *lhs, int32_t val);
template NDObject *Kernel::Binary<float>(int op_type, float val, NDObject *rhs);
template NDObject *Kernel::Binary<int32_t>(int op_type, int32_t val, NDObject *rhs);

NDObject *Kernel::Select(NDObject *cond, NDObject *lhs, NDObject *rhs) {
  if (cond->type_id_ != lhs->type_id_) {
    cond = this->Cast(cond, lhs->type_id_);
  }
  auto obj = new SelectOp(cond, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Cast(NDObject *input, DType type) {
  static const int g_cast_staff_type[kTypeEnd][kTypeEnd] = {
    {-1, -1, kFloat16, kFloat16, kFloat16},  // V_BOOL
    {-1, -1, kFloat32, -1, -1},              // V_FLOAT16
    {kFloat32, kFloat32, -1, -1, -1},        // V_BFLOAT16
    {kFloat16, -1, -1, -1, -1},              // V_FLOAT32
    {kFloat16, -1, kFloat32, -1, -1},        // V_INT32
  };
  if (input->type_id_ == type) {
    return input;
  }
  auto input_obj_type = input->GetObjectType();
  if (type == kBool && input_obj_type != kCompare && input_obj_type != kCompareS) {
    if (input->type_id_ == kBFloat16) {
      input = Cast(input, kFloat32);
    }
    if (input->type_id_ == kInt32) {
      input = Binary(BinaryOpType::kNotEqual, input, 0);
    } else {
      input = Binary(BinaryOpType::kNotEqual, input, 0.0f);
    }
  }
  auto stuff_type = g_cast_staff_type[input->type_id_][type];
  while (stuff_type != -1) {
    input = new CastOp(input, static_cast<DType>(stuff_type));
    kernel_->Append(input);
    stuff_type = g_cast_staff_type[input->type_id_][type];
  }
  auto obj = new CastOp(input, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Copy(NDObject *input) {
  auto obj = new CopyOp(input);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::ElemAny(NDObject *input) {
  auto obj = new ElementAnyOp(input);
  kernel_->Append(obj);
  return obj;
}

template <typename T>
NDObject *Kernel::Broadcast(T val, ShapeRef *shape, DType type, bool dummy_load) {
  NDObject *load = nullptr;
  if (dummy_load) {
    load = new NDLoadDummy(type);
    kernel_->Append(load);
  }
  auto obj = new BroadcastScalarOp<T>(val, shape, type, load);
  kernel_->Append(obj);
  return obj;
}

template NDObject *Kernel::Broadcast<float>(float val, ShapeRef *shape, DType type, bool dummy_load);
template NDObject *Kernel::Broadcast<int32_t>(int32_t val, ShapeRef *shape, DType type, bool dummy_load);

NDObject *Kernel::Broadcast(NDObject *input, ShapeRef *shape) {
  if (input->type_id_ == DType::kBool) {
    auto cast1 = Cast(input, DType::kFloat16);
    auto obj = Broadcast(cast1, shape);
    auto cast2 = Cast(obj, DType::kBool);
    return cast2;
  }
  auto obj = new BroadcastOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Reshape(NDObject *input, ShapeRef *shape) {
  auto obj = new ReshapeOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Reduce(int op_type, NDObject *input, ShapeRef *dims, bool keepdims) {
  if (input->type_id_ != DType::kFloat32) {
    return nullptr;
  }
  auto obj = new ReduceOp(input, ReduceOp::SUM, dims, keepdims);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Store(void *addr, NDObject *input) {
  if (input->IsLoad() || input->IsComm()) {
    input = Copy(input);
  }
  auto ktype = kernel_->KType();
  if (ktype == kEager) {
    if (auto store = VKernelE::GetStore(input)) {
      store->gm_ = static_cast<uint8_t *>(addr);
      store->flags_ |= OBJ_FLAG_EAGER;
      return store;
    }
  } else if (ktype == kStaticStages) {
    ktype = static_cast<StagesKernel *>(kernel_)->Current()->KType();
  }
  NDObject *obj;
  if (ktype == kStaticMix) {
    obj = new NDSStore(static_cast<uint8_t *>(addr), input);
  } else {
    obj = new NDStore(static_cast<uint8_t *>(addr), input);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::PadStore(void *addr, NDObject *input, ShapeRef *pad_shape) {
  auto ktype = kernel_->KType();
  if (ktype == kStaticStages) {
    ktype = static_cast<StagesKernel *>(kernel_)->Current()->KType();
  }
  NDObject *obj;
  obj = new NDPadStore(static_cast<uint8_t *>(addr), input, pad_shape);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::AllReduce(NDObject *input, const Comm *comm) {
  if (input->IsLoad()) {
    input = Copy(input);
  }
  NDObject *obj = new AllReduceOp(input, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias) {
  CubeOp *obj = new CubeOp(lhs, rhs, trans_a, trans_b, bias);
  if (kernel_->KType() == KernelType::kEager) {
    return static_cast<VKernelE*>(kernel_)->AppendCube(obj);
  }
  kernel_->Append(obj);
  return obj;
}

int Kernel::ParallelNext() {
  if (kernel_->KType() == KernelType::kStaticParallel) {
    static_cast<VKernelP *>(kernel_)->AppendNext();
  } else if (kernel_->KType() == KernelType::kStaticStages) {
    static_cast<StagesKernel *>(kernel_)->ParallelSwitch();
  } else {
    ASSERT(0);
  }
  return 0;
}

void Kernel::StageSwitch(KernelType type) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  static_cast<StagesKernel *>(kernel_)->StageSwitch(type);
}

NDObject *Kernel::StageLoad(NDObject *stage_store) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  auto op = static_cast<StagesKernel *>(kernel_)->Current()->KType() == kStaticMix
              ? new NDSLoad(nullptr, stage_store->shape_ref_, stage_store->type_id_)
              : new NDLoad(nullptr, stage_store->shape_ref_, stage_store->type_id_);
  static_cast<StagesKernel *>(kernel_)->StageLoad(op, static_cast<NDStore *>(stage_store));
  return op;
}

NDObject *Kernel::StageStore(NDObject *input) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  auto op = static_cast<StagesKernel *>(kernel_)->Current()->KType() == kStaticMix ? new NDSStore(nullptr, input)
                                                                                   : new NDStore(nullptr, input);
  static_cast<StagesKernel *>(kernel_)->StageStore(op);
  return op;
}

NDObject *Kernel::StagePadStore(NDObject *input, ShapeRef *pad_shape) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  auto op = new NDPadStore(nullptr, input, pad_shape);
  static_cast<StagesKernel *>(kernel_)->StageStore(op);
  return op;
}

ShapeRef *Kernel::GetShape(NDObject *op) const { return op->shape_ref_; }

DType Kernel::GetDType(NDObject *op) const { return op->type_id_; }

uint64_t Kernel::CodeGen() {
  uint64_t ws_size = kernel_->CodeGen();
  return kernel_->code_.ReserveWorkspace(ws_size);
}

int Kernel::Launch(void *workspace, void *stream) {
  auto &code = kernel_->code_;
  return code.RelocLaunch(workspace, stream);
}

int Kernel::MsProfLaunch(const char *op_name, const char *op_fullname, const RelocTable &reloc_table, void **inputs,
                         void **outputs, void *workspace, void *stream) {
  if (msprof_helper_ == nullptr) {
    msprof_helper_ = new MsProfHelper();
    auto &info = msprof_helper_->info_;
    info.op_name = op_name;
    info.op_fullname = op_fullname;
    info.input_size = reloc_table.inputs_size;
    info.output_size = reloc_table.outputs_size;
    info.kernel_type = kernel_->KType();
    info.block_dim = kernel_->code_.block_dim_;
    auto loads = reinterpret_cast<NDAccess **>(reloc_table.inputs);
    for (size_t i = 0; i < reloc_table.inputs_size; ++i) {
      info.shapes.emplace_back(GetShape(*loads));
      info.data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[GetDType(*loads)]);
      loads++;
    }
    auto stores = reinterpret_cast<NDAccess **>(reloc_table.outputs);
    for (size_t i = 0; i < reloc_table.outputs_size; ++i) {
      info.shapes.emplace_back(GetShape(*stores));
      info.data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[GetDType(*stores)]);
      stores++;
    }
    msprof_helper_->InitReportNode();
  } else if (kernel_->KType() == kDynShape) {
    msprof_helper_->UpdateReportNode(kernel_->code_.block_dim_);
  }
  msprof_helper_->UpdateBeginTime();
  auto ret = Launch(reloc_table, inputs, outputs, workspace, stream);
  msprof_helper_->ReportTask();
  return ret;
}

int Kernel::EagerMsProfLaunch(void *stream) {
  auto kernel = static_cast<VKernelE *>(kernel_);
  auto extern_code = kernel->ExternCode();
  int kernel_used;
  const auto &kernels = kernel->GetKernels(kernel_used);
  for (int i = 0; i < kernel_used; ++i) {
    MsProfHelper msprof_helper;
    auto &info = msprof_helper.info_;
    auto vector_kernel = reinterpret_cast<VectorKernel *>(kernels[i]);
    info.kernel_type = vector_kernel->KType();
    info.block_dim = vector_kernel->code_.block_dim_;
    std::ostringstream oss;
    oss << "Dvm";
    if (vector_kernel->code_.target_ > Code::kTargetVec) {
      oss << "MatMul";
    }
    for (auto op : vector_kernel->objects_) {
      if (op->flags_ & OBJ_FLAG_EAGER) {
        if (op->IsLoad()) {
          info.shapes.emplace_back(GetShape(op));
          info.data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[GetDType(op)]);
          info.input_size++;
        } else if (!op->IsStore()) {
          op->Dump(false, oss);
        }
      }
    }
    for (auto op : vector_kernel->objects_) {
      if (op->flags_ & OBJ_FLAG_EAGER) {
        if (op->IsStore()) {
          info.shapes.emplace_back(GetShape(op));
          info.data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[GetDType(op)]);
          info.output_size++;
        }
      }
    }
    auto prof_name = oss.str();
    info.op_name = prof_name.c_str();
    info.op_fullname = info.op_name;
    msprof_helper.InitReportNode();
    msprof_helper.UpdateBeginTime();
    vector_kernel->code_.Launch(extern_code, stream);
    msprof_helper.ReportTask();
  }
  return 0;
}

int Kernel::Launch(const RelocTable &reloc_table, void **inputs, void **outputs, void *workspace, void *stream) {
  auto loads = reinterpret_cast<NDAccess **>(reloc_table.inputs);
  for (size_t i = 0; i < reloc_table.inputs_size; ++i) {
    (*loads++)->Reloc(*inputs++);
  }
  auto stores = reinterpret_cast<NDAccess **>(reloc_table.outputs);
  for (size_t i = 0; i < reloc_table.outputs_size; ++i) {
    (*stores++)->Reloc(*outputs++);
  }
  auto &code = kernel_->code_;
  return code.RelocLaunch(workspace, stream);
}

void Kernel::EagerReset(WsAllocFunc ws_alloc, void *user_data) {
  if (kernel_) {
    delete kernel_;
  }
  kernel_ = new VKernelE(ws_alloc, user_data);
}

void Kernel::EagerCodeGen(const RelocEntry *reloc_table, size_t reloc_size) {
  ASSERT(kernel_->KType() == KernelType::kEager);
  for (auto reloc = reloc_table; reloc < reloc_table + reloc_size; ++reloc) {
    static_cast<NDAccess *>(reloc->io)->gm_ = static_cast<uint8_t *>(reloc->addr);
  }
  auto kernel = static_cast<VKernelE *>(kernel_);
  kernel->VKernelE::CodeGen();
}

int Kernel::EagerLaunch(void *stream) {
  auto kernel = static_cast<VKernelE *>(kernel_);
  kernel->Launch(stream);
  return 0;
}

void Kernel::EagerClear() {
  auto kernel = static_cast<VKernelE *>(kernel_);
  kernel->Clear();
}

const char *Kernel::Dump() const {
  std::string &graph = kernel_->DumpGraph();
  return graph.c_str();
}

const char *Kernel::Das() const {
  std::string &das = kernel_->DisAssemble();
  return das.c_str();
}

void SetDeterministic(bool enable) { System::Instance().deterministic_ = enable; }

void SetOnlineTuning(bool enable) {
  auto &sys = System::Instance();
  if (enable) {
    sys.online_tuner_ = new OnlineCubeTuner();
    sys.lazy_tuner_ = new LazyCubeTuner();
  } else {
    delete sys.online_tuner_;
    delete sys.lazy_tuner_;
    sys.online_tuner_ = nullptr;
    sys.lazy_tuner_ = nullptr;
  }
}
}  // namespace dvm
