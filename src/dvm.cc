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

#include <dlfcn.h>
#include "dvm.h"
#include "kernel.h"

// rts_runtime
#if defined(__cplusplus)
extern "C" {
#endif
#define RT_DEV_BINARY_MAGIC_ELF        0x43554245U
#define RT_DEV_BINARY_MAGIC_ELF_AICPU  0x41415243U
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC  0x41415246U
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

typedef int32_t rtError_t;
typedef char char_t;
const int32_t RT_ERROR_NONE = 0; // success
typedef void *rtStream_t;
struct tagRtSmCtrl;
typedef struct tagRtSmCtrl rtSmDesc_t;

typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);
rtError_t rtDevBinaryUnRegister(void *hdl);
rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                         const void *kernelInfoExt, uint32_t funcMode);
rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                         rtSmDesc_t *smDesc, rtStream_t stm);
#if defined(__cplusplus)
}
#endif

extern const unsigned char g_vkernel_bin[];
extern unsigned int  g_vkernel_bin_len;
extern const unsigned char g_vkernel_910b_bin[];
extern unsigned int  g_vkernel_910b_bin_len;

namespace dvm {
namespace {
const char FUNC_NAME[] = "vmain";
using namespace dvm;

class VKernelHolder {
 public:
  VKernelHolder();
  ~VKernelHolder() = default;

  static VKernelHolder &Instance() {
    static VKernelHolder instance;
    return instance;
  }
  void *StubFunc() { return reinterpret_cast<void*>(this); }

  rtError_t (*Launch)(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                              rtSmDesc_t *smDesc, rtStream_t stm);
};

VKernelHolder::VKernelHolder() {
#ifdef VK_SIM_MODEL
  auto rt_binary_register = rtDevBinaryRegister;
  auto rt_function_register = rtFunctionRegister;
  Launch = rtKernelLaunch;
#else
  void *handle = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(handle == nullptr, "Load libruntime.so failed");
  auto rt_binary_register =
    reinterpret_cast<rtError_t (*)(const rtDevBinary_t *, void **)>(dlsym(handle, "rtDevBinaryRegister"));
  EXCEPTION_IF(rt_binary_register == nullptr, "load rt_binary_register symbol failed");
  auto rt_function_register =
    reinterpret_cast<rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t)>(
      dlsym(handle, "rtFunctionRegister"));
  EXCEPTION_IF(rt_function_register == nullptr, "load rt_function_register symbol failed");
  Launch = reinterpret_cast<rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t)>(
    dlsym(handle, "rtKernelLaunch"));
  EXCEPTION_IF(Launch == nullptr, "load rt_kernel_launch symbol failed");
#endif

  void *module = nullptr;
  rtDevBinary_t dev_bin;
  if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
    dev_bin.data = g_vkernel_910b_bin;
    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    dev_bin.length = g_vkernel_910b_bin_len;
  } else {
    dev_bin.data = g_vkernel_bin;
    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    dev_bin.length = g_vkernel_bin_len;
  }
  dev_bin.version = 0;
  auto stub_func = reinterpret_cast<void*>(this);
  rtError_t err = rt_binary_register(&dev_bin, &module);
  if (err != RT_ERROR_NONE) {
    std::cerr << "reg binary failed: " << static_cast<int>(err) << std::endl;
    exit(0);
  }
  err = rt_function_register(module, stub_func, FUNC_NAME, FUNC_NAME, 0);
  if (err != RT_ERROR_NONE) {
    std::cerr << "reg function failed: " << static_cast<int>(err) << std::endl;
    exit(0);
  }
}

template <typename T>
NDObject *GetBinaryS(int op_type, T val, NDObject *rhs) {
  if (rhs->type_id_ == kInt32 && DeviceInfo::Instance().Arch() != kAiCore_C220) {
    return nullptr;
  }
  switch (op_type) {
    case BinaryOpType::kAdd:
      return new BinaryScalarOp<T>(BinarySOpType::kAdds, rhs, val);
    case BinaryOpType::kMul:
      return new BinaryScalarOp<T>(BinarySOpType::kMuls, rhs, val);
    case BinaryOpType::kMaximum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        return new BinaryScalarOp<T>(BinarySOpType::kMaximums, rhs, val);
      }
    case BinaryOpType::kMinimum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        return new BinaryScalarOp<T>(BinarySOpType::kMinimums, rhs, val);
      }
    default:
      return nullptr;
  }
}
}  // namespace

Kernel::Kernel() : kernel_{nullptr} {
  static bool init = false;
  if (!init) {
    (void)VKernelHolder::Instance();
    init = true;
  }
}

Kernel::~Kernel() {
  delete kernel_;
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
  } else {
    ASSERT(0);
  }
}

NDObject* Kernel::Load(void *addr, ShapeRef *shape, DType type) {
  auto obj = new NDLoad(static_cast<uint8_t*>(addr), shape, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DType type) {
  auto obj = new NDSliceLoad(static_cast<uint8_t *>(addr), shape, start, size, type);
  kernel_->Append(obj);
  return obj;
}


NDObject *Kernel::StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step, DType type) {
  auto obj = new NDStridedSliceLoad(static_cast<uint8_t *>(addr), shape, start, end, step, type);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Unary(int op_type, NDObject* input) {
  if (GetDType(input) == kInt32) {
    if (op_type == UnaryOpType::kAbs) {
      return Binary(BinaryOpType::kMaximum, input, Binary(BinaryOpType::kMul, input, -1));
    }
  }
  if (op_type == UnaryOpType::kLogicalNot && input->type_id_ != kInt8) {
    if (input->type_id_ == kInt32) {
      return Binary(BinaryOpType::kSub, 1, input);
    } else {
      return Binary(BinaryOpType::kSub, 1.0f, input);
    }
  }
  auto obj = new UnaryOp(op_type, input);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Binary(int op_type, NDObject* lhs, NDObject* rhs) {
  if (op_type < V_CMP_ALL && lhs->type_id_ == kInt32) {
    if (op_type == kEqual || op_type == kNotEqual) {
      auto sub = Binary(BinaryOpType::kSub, rhs, lhs);
      auto abs = Binary(BinaryOpType::kMaximum, sub, Binary(BinaryOpType::kMul, sub, -1));;
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
  auto obj = new BinaryOp(op_type, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

template<typename T>
NDObject *Kernel::Binary(int op_type, T val, NDObject *rhs) {
  NDObject *obj = GetBinaryS(op_type, val, rhs);
  if (obj == nullptr) {
    NDObject *broadcast = new BroadcastScalarOp<T>(val, rhs->shape_ref_, rhs->type_id_, nullptr);
    kernel_->Append(broadcast);
    return Binary(op_type, broadcast, rhs);
  }
  kernel_->Append(obj);
  return obj;
}

template<typename T>
NDObject *Kernel::Binary(int op_type, NDObject *lhs, T val) {
  NDObject *obj = GetBinaryS(op_type, val, lhs);
  if (obj == nullptr) {
    NDObject *broadcast = new BroadcastScalarOp<T>(val, lhs->shape_ref_, lhs->type_id_, nullptr);
    kernel_->Append(broadcast);
    return Binary(op_type, lhs, broadcast);
  }
  kernel_->Append(obj);
  return obj;
}

template NDObject *Kernel::Binary<float>(int op_type, NDObject *lhs, float val);
template NDObject *Kernel::Binary<int32_t>(int op_type, NDObject *lhs, int32_t val);
template NDObject *Kernel::Binary<float>(int op_type, float val, NDObject *rhs);
template NDObject *Kernel::Binary<int32_t>(int op_type, int32_t val, NDObject *rhs);

NDObject* Kernel::Select(NDObject* cond, NDObject* lhs, NDObject* rhs) {
  if (cond->type_id_ != lhs->type_id_) {
    cond = this->Cast(cond, lhs->type_id_);
  }
  auto obj = new SelectOp(cond, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Cast(NDObject* input, DType type) {
  static const int g_cast_staff_type[kTypeEnd][kTypeEnd] = {
    {-1, -1, kFloat16, kFloat16, kFloat16},  // V_INT8
    {-1, -1, kFloat32, -1, -1},              // V_FLOAT16
    {kFloat32, kFloat32, -1, -1, -1},        // V_BFLOAT16
    {kFloat16, -1, -1, -1, -1},              // V_FLOAT32
    {kFloat16, -1, kFloat32, -1, -1},        // V_INT32
  };
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

NDObject* Kernel::Copy(NDObject* input) {
  auto obj = new CopyOp(input);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::ElemAny(NDObject* input) {
  auto obj = new ElementAnyOp(input);
  kernel_->Append(obj);
  return obj;
}

template<typename T>
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

NDObject* Kernel::Broadcast(NDObject* input, ShapeRef *shape) {
  auto obj = new BroadcastOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Reshape(NDObject* input, ShapeRef *shape) {
  auto obj = new ReshapeOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Reduce(int op_type, NDObject* input, ShapeRef *dims, bool keepdims) {
  if (input->type_id_ != DType::kFloat32) {
    return nullptr;
  }
  auto obj = new ReduceOp(input, ReduceOp::SUM, dims, keepdims);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Store(void *addr, NDObject* input) {
  auto obj = new NDStore(static_cast<uint8_t*>(addr), input);
  kernel_->Append(obj);
  return obj;
}

void Kernel::Reserve(size_t size) {
  auto ktype = kernel_->KType();
  if (ktype == KernelType::kStaticParallel) {
    static_cast<VKernelP*>(kernel_)->Reserve(size);
  } else {
    static_cast<VKernelBase*>(kernel_)->Reserve(size);
  }
}

int Kernel::ParallelNext() {
  if (kernel_->KType() != KernelType::kStaticParallel) {
    ASSERT(0);
    return -1;
  }
  auto p_kernel = static_cast<VKernelP*>(kernel_);
  p_kernel->AppendNext();
  return 0;
}

ShapeRef* Kernel::GetShape(NDObject* op) const {
  return op->shape_ref_;
}

DType Kernel::GetDType(NDObject* op) const {
  return op->type_id_;
}

uint64_t Kernel::CodeGen() {
  kernel_->CodeGen();
  return kernel_->GetCode()->data_size_;
}

int Kernel::Launch(void* stream) {
  CodeBase* code = kernel_->GetCode();
  if (code->data_ == nullptr) {
    return -1;
  }
  auto stub_func = VKernelHolder::Instance().StubFunc();
  if (!code->atomic_clean_.empty()) {
    for (auto atomic: code->atomic_clean_) {
      auto ret = VKernelHolder::Instance().Launch(stub_func, atomic->block_dim_, atomic->data_, atomic->data_size_, nullptr, stream);
      if (ret != RT_ERROR_NONE) {
        return ret;
      }
    }
  }
  auto ret = VKernelHolder::Instance().Launch(stub_func, code->block_dim_, code->data_, code->data_size_, nullptr, stream);
  return ret;
}

int Kernel::Launch(const RelocTable &reloc_table, void** inputs, void** outputs, void* stream) {
  auto loads = reinterpret_cast<NDLoad**>(reloc_table.inputs);
  for (size_t i = 0; i < reloc_table.inputs_size; ++i) {
    (*loads++)->Reloc(*inputs++);
  }
  auto stores = reinterpret_cast<NDStore**>(reloc_table.outputs);
  for (size_t i = 0; i < reloc_table.outputs_size; ++i) {
    (*stores++)->Reloc(*outputs++);
  }
  return Launch(stream);
}

int Kernel::Launch(NDObject **op, int size, void* stream) {
  return 0;
}

const char* Kernel::Dump() const {
  std::string &graph = kernel_->DumpGraph();
  return graph.c_str();
}

const char* Kernel::Das() const {
  std::string &das = kernel_->DisAssemble();
  return das.c_str();
}
} // namespace dvm
