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

#include "dvm.h"
#include "kernel.h"
#include "acl_ext.h"

extern unsigned char g_vkernel_bin[];
extern unsigned int  g_vkernel_bin_len;
extern unsigned char g_vkernel_910b_bin[];
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
};

VKernelHolder::VKernelHolder() {
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
  rtError_t err = rtDevBinaryRegister(&dev_bin, &module);
  if (err != RT_ERROR_NONE) {
    std::cerr << "reg binary failed: " << static_cast<int>(err) << std::endl;
    exit(0);
  }
  err = rtFunctionRegister(module, stub_func, FUNC_NAME, FUNC_NAME, 0);
  if (err != RT_ERROR_NONE) {
    std::cerr << "reg function failed: " << static_cast<int>(err) << std::endl;
    exit(0);
  }
}
}

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
  } else {
    ASSERT(0);
  }
}

NDObject* Kernel::Load(void *addr, ShapeRef *shape, DType type) {
  auto obj = new NDLoad(static_cast<uint8_t*>(addr), shape, type);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Unary(int op_type, NDObject* input) {
  auto obj = new UnaryOp(op_type, input);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Binary(int op_type, NDObject* lhs, NDObject* rhs) {
  auto obj = new BinaryOp(op_type, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Binary(int op_type, float val, NDObject *rhs) {
  NDObject *obj = nullptr;
  switch (op_type) {
    case BinaryOpType::kAdd:
      obj = new BinaryScalarOp(BinarySOpType::kAdds, rhs, val);
      break;
    case BinaryOpType::kMul:
      obj = new BinaryScalarOp(BinarySOpType::kMuls, rhs, val);
      break;
    case BinaryOpType::kMaximum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        obj = new BinaryScalarOp(BinarySOpType::kMaximums, rhs, val);
        break;
      }
    case BinaryOpType::kMinimum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        obj = new BinaryScalarOp(BinarySOpType::kMinimums, rhs, val);
        break;
      }
    default:
      auto broadcast = new BroadcastScalarOp(val, rhs->shape_ref_, rhs->type_id_, nullptr);
      kernel_->Append(broadcast);
      obj = new BinaryOp(op_type, broadcast, rhs);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Binary(int op_type, NDObject* lhs, float val) {
  NDObject *obj = nullptr;
  switch (op_type) {
    case BinaryOpType::kAdd:
      obj = new BinaryScalarOp(BinarySOpType::kAdds, lhs, val);
      break;
    case BinaryOpType::kMul:
      obj = new BinaryScalarOp(BinarySOpType::kMuls, lhs, val);
      break;
    case BinaryOpType::kMaximum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        obj = new BinaryScalarOp(BinarySOpType::kMaximums, lhs, val);
        break;
      }
    case BinaryOpType::kMinimum:
      if (DeviceInfo::Instance().Arch() == kAiCore_C220) {
        obj = new BinaryScalarOp(BinarySOpType::kMinimums, lhs, val);
        break;
      }
    default:
      auto broadcast = new BroadcastScalarOp(val, lhs->shape_ref_, lhs->type_id_, nullptr);
      kernel_->Append(broadcast);
      obj = new BinaryOp(op_type, lhs, broadcast);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Select(NDObject* cond, NDObject* lhs, NDObject* rhs) {
  auto obj = new SelectOp(cond, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

NDObject* Kernel::Cast(NDObject* input, DType type) {
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

NDObject *Kernel::Broadcast(float val, ShapeRef *shape, DType type, bool dummy_load) {
  NDObject *load = nullptr;
  if (dummy_load) {
    load = new NDLoadDummy(type);
    kernel_->Append(load);
  }
  auto obj = new BroadcastScalarOp(val, shape, type, load);
  kernel_->Append(obj);
  return obj;
}

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

}

ShapeRef* Kernel::GetShape(NDObject* op) const {
  return op->shape_ref_;
}

DType Kernel::GetDType(NDObject* op) const {
  return op->type_id_;
}

int Kernel::CodeGen() {
  kernel_->CodeGen();
  return 0;
}

int Kernel::Launch(void* stream) {
  Code* code = kernel_->GetCode();
  if (code->data == nullptr) {
    return -1;
  }
  auto stub_func = VKernelHolder::Instance().StubFunc();
  auto block_dim = code->BlockDim();
  ASSERT(code->size <= 4096);
  auto ret = rtKernelLaunch(stub_func, block_dim, code->data, code->size, nullptr, stream);
  return ret;
}

int Kernel::Launch(const RelocTable &reloc_table, void** inputs, void** outputs, void* stream) {
  for (size_t i = 0; i < reloc_table.inputs_size; ++i) {
    auto load = static_cast<NDLoad*>(reloc_table.inputs[i]);
    load->Reloc(*inputs++, true);
  }
  for (size_t i = 0; i < reloc_table.outputs_size; ++i) {
    auto store = static_cast<NDStore*>(reloc_table.outputs[i]);
    store->Reloc(*outputs++, true);
  }
  Code* code = kernel_->GetCode();
  if (code->data == nullptr) {
    return -1;
  }
  auto stub_func = VKernelHolder::Instance().StubFunc();
  auto block_dim = code->BlockDim();
  ASSERT(code->size <= 4096);
  auto ret = rtKernelLaunch(stub_func, block_dim, code->data, code->size, nullptr, stream);
  return ret;
}

int Kernel::Launch(NDObject **op, int size, void* stream) {
  return 0;
}

const char* Kernel::DisAssemble() {
  std::string &das = kernel_->DisAssemble();
  return das.c_str();
}
} // namespace dvm
