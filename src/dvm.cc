/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <cstring>
#include "dvm.h"
#include "kernel.h"
#include "xkernel.h"
#include "msprof.h"
#include "tuning.h"
#include "comm.h"

namespace dvm {
std::mutex g_rt_kernel_launch_mutex;
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
struct TypeTrait {
  static constexpr DType ID = kTypeEnd;
};
template <>
struct TypeTrait<int32_t> {
  static constexpr DType ID = kInt32;
};
template <>
struct TypeTrait<float> {
  static constexpr DType ID = kFloat32;
};
template <>
struct TypeTrait<Float16> {
  static constexpr DType ID = kFloat16;
};
template <>
struct TypeTrait<BFloat16> {
  static constexpr DType ID = kBFloat16;
};

union Union32 {
  uint32_t u;
  float f;
};

float ToFloat32(const Float16 &f16) {
  static constexpr uint16_t value_mask = 0x7fff;
  constexpr uint32_t mu_value = 113 << 23;
  Union32 magic;
  magic.u = mu_value;
  constexpr uint32_t exponent_adjust = ((127 - 15) << 23);
  constexpr uint32_t inf_extra_exp_adjust = ((128 - 16) << 23);
  constexpr uint32_t zero_extra_exp_adjust = (1 << 23);
  constexpr uint32_t sign_mask = 0x8000;
  constexpr unsigned int shifted_exp = (0x7c00 << 13);  // Exponent mask after shift.
  constexpr unsigned int exponent_bits = 13;
  constexpr unsigned int sign_bit_shift = 16;
  // Exponent/mantissa bits.
  Union32 f32;
  f32.u = (static_cast<uint32_t>(f16.int_value() & value_mask) << exponent_bits);
  // Just the exponent.
  unsigned int exp = (shifted_exp & f32.u);
  f32.u += exponent_adjust;
  // Handle exponent special cases.
  if (exp == shifted_exp) {
    // Inf/NaN, extra exp adjust.
    f32.u += inf_extra_exp_adjust;
  } else if (exp == 0) {
    // Zero/Denormal, extra exp adjust and renormalize.
    f32.u += zero_extra_exp_adjust;
    f32.f -= magic.f;
  }
  // Set sign bit.
  f32.u |= ((f16.int_value() & sign_mask) << sign_bit_shift);
  return f32.f;
}

uint16_t EncoderFP16(float f32) {
  static constexpr uint16_t nan_value = 0x7e00;
  static constexpr uint16_t inf_value = 0x7c00;
  constexpr uint32_t magic = {113 << 23};
  constexpr uint32_t f32infty_value = 255 << 23;
  Union32 f32infty;
  f32infty.u = f32infty_value;
  constexpr uint32_t f16max_value = (127 + 16) << 23;
  Union32 f16max;
  f16max.u = f16max_value;
  constexpr uint32_t denorm_magic_value = ((127 - 15) + (23 - 10) + 1) << 23;
  Union32 denorm_magic;
  denorm_magic.u = denorm_magic_value;
  constexpr unsigned int exponent_bits = 13;
  constexpr unsigned int sign_bit_shift = 16;
  constexpr unsigned int sign_mask = 0x80000000u;
  constexpr uint32_t rouding_bias_part1 = (static_cast<unsigned int>(15 - 127) << 23) + 0xfff;

  Union32 f;
  f.f = f32;
  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;
  uint16_t result = 0;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).
  if (f.u >= f16max.u) {
    // Result is Inf or NaN (all exponent bits set).
    result = (f.u > f32infty.u) ? nan_value : inf_value;
  } else if (f.u < magic) {
    // (De)normalized number or zero; resulting FP16 is subnormal or zero.
    // Use a magic value to align our 10 mantissa bits at the bottom of
    // the float. as long as FP addition is round-to-nearest-even this
    // just works.
    f.f += denorm_magic.f;
    // And one integer subtract of the bias later, we have our final float!
    result = static_cast<uint16_t>(f.u - denorm_magic.u);
  } else {
    // Resulting mantissa is odd.
    unsigned int mant_odd = (f.u >> exponent_bits) & 1;
    // Update exponent, rounding bias part 1;
    f.u += rouding_bias_part1;
    // Rounding bias part 2;
    f.u += mant_odd;
    // Take the bits!
    result = static_cast<uint16_t>(f.u >> exponent_bits);
  }
  // Set sign bit.
  result |= static_cast<uint16_t>(sign >> sign_bit_shift);
  return result;
}

float ToFloat32(const BFloat16 &bf16) {
  float f32 = 0;
  uint32_t f32_tmp = bf16.int_value();
  f32_tmp <<= 16;
  memcpy(&f32, &f32_tmp, sizeof(f32_tmp));
  return f32;
}

uint16_t EncoderBF16(float f32) {
  static constexpr uint16_t nan_value = 0x7fc0;
  if (std::isnan(f32)) {
    return nan_value;
  } else {
    union {
      uint32_t U32;
      float F32;
    };
    F32 = f32;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

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
  int32_t iter_num = std::abs(static_cast<int32_t>(value));
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
  if (value < T(0)) {
    res = kernel->Unary(UnaryOpType::kReciprocal, res);
  }
  return res;
}

inline Float16 operator/(const Float16 &a, const Float16 &b) {
  return Float16(static_cast<float>(a) / static_cast<float>(b));
}
inline Float16 operator-(const Float16 &a) {
  constexpr uint16_t sign_mask = 0x8000;
  return Float16(a.int_value() ^ sign_mask);
}
inline bool operator>(const Float16 &a, const Float16 &b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator<(const Float16 &a, const Float16 &b) { return static_cast<float>(a) < static_cast<float>(b); }

inline BFloat16 operator/(const BFloat16 &a, const BFloat16 &b) {
  return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}
inline BFloat16 operator-(const BFloat16 &a) {
  constexpr uint16_t sign_mask = 0x8000;
  return BFloat16(a.int_value() ^ sign_mask);
}
inline bool operator>(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator<(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) < static_cast<float>(b); }

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
      if (input->type_id_ != kInt32) {
        auto obj = new CompareScalarOp<T>(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input, val);
        vkernel->Append(obj);
        return obj;
      }
      return nullptr;
    }
    case BinaryOpType::kPow: {
      if (rhs_val) {
        if (isInteger(val)) return PowS(kernel, input, val);
      } else if (val > T(0)) {
        if (input->type_id_ != kFloat32) {
          auto result = GetBinaryS<float, false>(kernel, BinaryOpType::kPow, static_cast<float>(val),
                                                 kernel->Cast(input, kFloat32));
          return kernel->Cast(result, input->type_id_);
        }
        auto tmp = GetBinaryS<float, true>(kernel, BinaryOpType::kMul, std::log(static_cast<float>(val)), input);
        return kernel->Unary(UnaryOpType::kExp, tmp);
      }
      return nullptr;
    }
    case BinaryOpType::kSub: {
      if (rhs_val) {
        return GetBinaryS<T, true>(kernel, BinaryOpType::kAdd, -val, input);
      } else {
        auto neg_input = GetBinaryS<T, true>(kernel, BinaryOpType::kMul, static_cast<T>(-1), input);
        return GetBinaryS<T, true>(kernel, BinaryOpType::kAdd, val, neg_input);
      }
    }
    case BinaryOpType::kDiv: {
      if (rhs_val) {
        return GetBinaryS<T, true>(kernel, BinaryOpType::kMul, T(1) / val, input);
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

Float16::Float16(const float &v) : Float16(EncoderFP16(v)) {}
Float16::Float16(const int32_t &v) : Float16(static_cast<float>(v)) {}
Float16::operator float() const { return ToFloat32(*this); }
Float16::operator int32_t() const { return static_cast<int32_t>(ToFloat32(*this)); }

BFloat16::BFloat16(const float &v) : BFloat16(EncoderBF16(v)) {}
BFloat16::BFloat16(const int32_t &v) : BFloat16(static_cast<float>(v)) {}
BFloat16::operator float() const { return ToFloat32(*this); }
BFloat16::operator int32_t() const { return static_cast<int32_t>(ToFloat32(*this)); }

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
  } else if (type == kDynMix) {
    kernel_ = new DynMixKernel();
  } else {
    ASSERT(0);
  }
}

NDObject *Kernel::Load(void *addr, ShapeRef *shape, DType type) {
  NDObject *obj = new NDLoad(addr, shape, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DType type) {
  auto obj = new NDSliceLoad(addr, shape, start, size, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step,
                                   DType type) {
  auto obj = new NDStridedSliceLoad(addr, shape, start, end, step, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::MultiLoad(void *addr, ShapeRef *shape, DType type, const Comm *comm) {
  NDObject *obj = new NDMultiLoad(static_cast<uint8_t *>(addr), shape, type, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Unary(int op_type, NDObject *input) {
  if (op_type >= UnaryOpType::kRound && op_type <= UnaryOpType::kTrunc && GetDType(input) == kFloat16) {
    return Cast(Unary(op_type, Cast(input, kFloat32)), kFloat16);
  }
  if (GetDType(input) == kInt32) {
    if (op_type == UnaryOpType::kAbs) {
      return Binary(BinaryOpType::kMaximum, input, Binary(BinaryOpType::kMul, input, -1));
    }
  }
  if (op_type == UnaryOpType::kLogicalNot) {
    if (input->type_id_ == kBool) {
      return Cast(Binary(BinaryOpType::kSub, 1.0f, Cast(input, kFloat16)), kBool);
    } else {
      return Binary(BinaryOpType::kSub, 1, input);
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
    switch (op_type) {
      case kGreaterEqual:
        return Binary(BinaryOpType::kEqual, Binary(BinaryOpType::kMaximum, lhs, rhs), lhs);
      case kLess:
        return Binary(BinaryOpType::kNotEqual, Binary(BinaryOpType::kMaximum, lhs, rhs), lhs);
      case kLessEqual:
        return Binary(BinaryOpType::kEqual, Binary(BinaryOpType::kMinimum, lhs, rhs), lhs);
      case kGreater:
        return Binary(BinaryOpType::kNotEqual, Binary(BinaryOpType::kMinimum, lhs, rhs), lhs);
      default:
        break;
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
template NDObject *Kernel::Binary<Float16>(int op_type, NDObject *lhs, Float16 val);
template NDObject *Kernel::Binary<BFloat16>(int op_type, NDObject *lhs, BFloat16 val);
template NDObject *Kernel::Binary<float>(int op_type, float val, NDObject *rhs);
template NDObject *Kernel::Binary<int32_t>(int op_type, int32_t val, NDObject *rhs);
template NDObject *Kernel::Binary<Float16>(int op_type, Float16 val, NDObject *rhs);
template NDObject *Kernel::Binary<BFloat16>(int op_type, BFloat16 val, NDObject *rhs);

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
    {-1, -1, -1, -1, -1},                    // V_INT64
  };
  if (input->type_id_ == type) {
    return input;
  }
  auto input_obj_type = input->GetObjectType();
  if (type == kBool && input_obj_type != kCompare && input_obj_type != kCompareS) {
    if (input->type_id_ == kBFloat16) {
      input = Cast(input, kFloat32);
    }
    input = Binary(BinaryOpType::kNotEqual, input, 0);
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
template NDObject *Kernel::Broadcast<Float16>(Float16 val, ShapeRef *shape, DType type, bool dummy_load);
template NDObject *Kernel::Broadcast<BFloat16>(BFloat16 val, ShapeRef *shape, DType type, bool dummy_load);

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

template <typename T>
NDObject *Kernel::OneHot(NDObject *indices, ShapeRef *depth, int axis, T on_value, T off_value) {
  auto obj = new OneHotOp<T>(indices, depth, axis, on_value, off_value, TypeTrait<T>::ID);
  kernel_->Append(obj);
  return obj;
}

template NDObject *Kernel::OneHot<float>(NDObject *, ShapeRef *, int, float, float);
template NDObject *Kernel::OneHot<int32_t>(NDObject *, ShapeRef *, int, int32_t, int32_t);
template NDObject *Kernel::OneHot<Float16>(NDObject *, ShapeRef *, int, Float16, Float16);
template NDObject *Kernel::OneHot<BFloat16>(NDObject *, ShapeRef *, int, BFloat16, BFloat16);

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
  if (input->IsLoad()) {
    input = Copy(input);
  }
  auto ktype = kernel_->KType();
  if (ktype == kEager) {
    if (auto store = VKernelE::GetStore(input)) {
      store->addr_.gm = addr;
      store->flags_ |= OBJ_FLAG_EAGER;
      return store;
    }
  }
  NDObject *obj = new NDStore(addr, input);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::PadStore(void *addr, NDObject *input, int64_t pad_size) {
  auto ktype = kernel_->KType();
  if (ktype == kStaticStages) {
    ktype = static_cast<StagesKernel *>(kernel_)->Current()->KType();
  }
  NDObject *obj = new NDPadStore(addr, input, pad_size);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::AllReduce(NDObject *input, const Comm *comm) {
  NDObject *obj;
  if (input->type_id_ == DType::kBFloat16) {
    obj = new AllReduceOp<true>(input, comm->GetImpl());
  } else {
    obj = new AllReduceOp<false>(input, comm->GetImpl());
  }
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::AllGather(NDObject *input, const Comm *comm) {
  NDObject *obj = new AllGatherOp(input, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::AllGatherV2(NDObject *input, const Comm *comm) {
  NDObject *obj = new AllGatherV2Op(input, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::ReduceScatter(NDObject *input, const Comm *comm) {
  NDObject *obj = new ReduceScatterOp(input, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias) {
  CubeOp *obj = new CubeOp(lhs, rhs, trans_a, trans_b, bias);
  if (kernel_->KType() == KernelType::kEager) {
    return static_cast<VKernelE *>(kernel_)->AppendCube(obj);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::GroupedMatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias,
                                NDObject *group_list, GroupType group_type) {
  GmmOp *obj = new GmmOp(lhs, rhs, trans_a, trans_b, bias, group_list, group_type);
  if (kernel_->KType() == KernelType::kEager) {
    return static_cast<VKernelE *>(kernel_)->AppendCube(obj);
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
  auto op = new NDLoad(nullptr, stage_store->shape_ref_, stage_store->type_id_);
  static_cast<StagesKernel *>(kernel_)->StageLoad(op, static_cast<NDStore *>(stage_store));
  return op;
}

NDObject *Kernel::StageStore(NDObject *input) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  auto op = new NDStore(nullptr, input);
  static_cast<StagesKernel *>(kernel_)->StageStore(op);
  return op;
}

NDObject *Kernel::StagePadStore(NDObject *input, int64_t pad_size) {
  ASSERT(kernel_->KType() == KernelType::kStaticStages);
  auto op = new NDPadStore(nullptr, input, pad_size);
  static_cast<StagesKernel *>(kernel_)->StageStore(op);
  return op;
}

ShapeRef *Kernel::GetShape(NDObject *op) const { return op->shape_ref_; }

DType Kernel::GetDType(NDObject *op) const { return op->type_id_; }

uint64_t Kernel::CodeGen() {
  uint64_t ws_size = kernel_->CodeGen();
  return kernel_->code_.ReserveWorkspace(ws_size);
}

void Kernel::Infer() {
  if (kernel_->KType() != KernelType::kDynShape) {
    return;
  }
  auto dyn_kernel = static_cast<VKernelD *>(kernel_);
  dyn_kernel->Normalize();
}

int Kernel::Launch(void *workspace, void *stream) {
  auto &code = kernel_->code_;
  code.RelocBinds(workspace);
  return code.Launch(workspace, stream);
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
  std::lock_guard<std::mutex> lock(g_rt_kernel_launch_mutex);
  ScopedValueGuard<LaunchFunc> guard(
    System::Instance().rt_kernel_launch_,
    [real_rt_launch = System::Instance().rt_kernel_launch_, this](const void *stub, auto &&...rest_args) {
      uint32_t target = reinterpret_cast<const uint8_t *>(stub) - reinterpret_cast<uint8_t *>(&System::Instance());
      msprof_helper_->Update(target);
      auto ret = real_rt_launch(stub, std::forward<decltype(rest_args)>(rest_args)...);
      msprof_helper_->ReportTask();
      return ret;
    });
  return Launch(reloc_table, inputs, outputs, workspace, stream);
}

int Kernel::EagerMsProfLaunch(void *stream) {
  auto kernel = static_cast<VKernelE *>(kernel_);
  int kernel_begin, kernel_end;
  const auto &kernels = kernel->GetKernels(kernel_begin, kernel_end);
  for (int i = kernel_begin; i < kernel_end; ++i) {
    MsProfHelper msprof_helper;
    auto &info = msprof_helper.info_;
    auto vector_kernel = reinterpret_cast<VectorKernel *>(kernels[i]);
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
    msprof_helper.Update(kernel->code_.target_);
    kernel->Launch(i, stream);
    msprof_helper.ReportTask();
  }
  return 0;
}

int Kernel::Launch(const RelocTable &reloc_table, void **inputs, void **outputs, void *workspace, void *stream) {
  auto loads = reinterpret_cast<NDAccess **>(reloc_table.inputs);
  for (size_t i = 0; i < reloc_table.inputs_size; ++i) {
    (*loads++)->addr_.Reloc(*inputs++);
  }
  auto stores = reinterpret_cast<NDAccess **>(reloc_table.outputs);
  for (size_t i = 0; i < reloc_table.outputs_size; ++i) {
    (*stores++)->addr_.Reloc(*outputs++);
  }
  auto &code = kernel_->code_;
  code.RelocBinds(workspace);
  return code.Launch(workspace, stream);
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
    static_cast<NDAccess *>(reloc->io)->addr_.gm = reloc->addr;
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
