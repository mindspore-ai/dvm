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
#include "dvm.h"
#include "kernel.h"
#include "xkernel.h"
#include "comm.h"

namespace dvm {
namespace {
static const BinarySOpType binary_map[kBinaryTypeEnd] = {
  kEquals, kNotEquals, kGreaters,     kGreaterEquals, kLesss,    kLessEquals,   kAdds,         kBinarySOpEnd,
  kMuls,   kDivs,      kBinarySOpEnd, kMaximums,      kMinimums, kBinarySOpEnd, kBinarySOpEnd,
};

static const BinarySOpType lhs_val_binary_map[kBinaryTypeEnd] = {
  kEquals, kNotEquals, kLesss,        kLessEquals, kGreaters, kGreaterEquals, kAdds,         kBinarySOpEnd,
  kMuls,   ksDiv,      kBinarySOpEnd, kMaximums,   kMinimums, kBinarySOpEnd,  kBinarySOpEnd,
};
template <typename T>
struct TypeTrait {
  static constexpr DataType ID = kDataTypeEnd;
  using code_t = uint8_t;
};
template <>
struct TypeTrait<int32_t> {
  static constexpr DataType ID = kInt32;
  using code_t = uint32_t;
};
template <>
struct TypeTrait<float> {
  static constexpr DataType ID = kFloat32;
  using code_t = uint32_t;
};
template <>
struct TypeTrait<Float16> {
  static constexpr DataType ID = kFloat16;
  using code_t = uint16_t;
};
template <>
struct TypeTrait<BFloat16> {
  static constexpr DataType ID = kBFloat16;
  using code_t = uint16_t;
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
  union {
    float f32;
    uint32_t u32;
  };
  u32 = static_cast<uint32_t>(bf16.int_value()) << 16;
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
scode_t EncodeScalar(T scalar) {
  constexpr auto id = TypeTrait<T>::ID;
  if constexpr (id == kFloat16 || id == kBFloat16) {
    return scalar.int_value();
  } else {
    union Code {
      T val;
      typename TypeTrait<T>::code_t code;
    } data{scalar};
    return data.code;
  }
}

template <typename T>
scode_t EncodeScalar(T scalar, int type) {
  constexpr auto id = TypeTrait<T>::ID;
  if constexpr (id == kFloat16 || id == kBFloat16) {
    return scalar.int_value();
  } else {
    switch (type) {
      case kFloat16:
        return static_cast<Float16>(static_cast<float>(scalar)).int_value();
      case kBFloat16:
        return static_cast<BFloat16>(static_cast<float>(scalar)).int_value();
      case kFloat32:
        return EncodeScalar(static_cast<float>(scalar));
      case kInt32:
        return EncodeScalar(static_cast<int32_t>(scalar));
      default:
        return 0;
    }
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

template <typename T>
NDObject *PowS(Kernel *kernel, NDObject *obj, const T &value) {
  int32_t iter_num = std::abs(static_cast<int32_t>(value));
  if (iter_num == 0) {
    return kernel->Broadcast(static_cast<T>(1), obj->shape_ref_, obj->type_id_);
  }
  NDObject *res = nullptr;
  if (iter_num == 1) {
    res = kernel->Copy(obj);
  } else {
    while (iter_num) {
      if (iter_num & 1) {
        res = res == nullptr ? obj : kernel->Binary<BinaryType::kMul>(res, obj);
      }
      if (iter_num != 1) {
        obj = kernel->Binary<BinaryType::kMul>(obj, obj);
      }
      iter_num >>= 1;
    }
  }
  if (value < T(0)) {
    res = kernel->Unary<UnaryType::kReciprocal>(res);
  }
  return res;
}

scode_t EncodeScalarRef(const ScalarRef *ref, DataType type_id) {
  scode_t code;
  switch (ref->type) {
    case kFloat16:
    case kBFloat16: {
      ASSERT(type_id == ref->type);
      code = ref->f16;
      break;
    }
    case kFloat32: {
      code = EncodeScalar(ref->f32, type_id);
      break;
    }
    case kInt32: {
      code = EncodeScalar(ref->i32, type_id);
      break;
    }
    case kInt64: {
      code = EncodeScalar(ref->i64, type_id);
      break;
    }
    default: {
      DvmException("unsupport scalar ref type");
      code = 0;
      break;
    }
  };
  return code;
}

class BroadcastScalarRefOp : public BroadcastScalarOp {
 public:
  BroadcastScalarRefOp(ScalarRef *scalar, IntArrayRef *shape_ref, DataType type_id)
      : BroadcastScalarOp(0, shape_ref, type_id), scalar_ref_(scalar) {}
  uint64_t Emit(VectorKernel &k) {
    scalar_ = EncodeScalarRef(scalar_ref_, type_id_);
    return BroadcastScalarOp::Emit(k);
  }
  NDObject *Clone(CloneHelper &h) override {
    auto shape_ref = h.GetClone(shape_ref_);
    return new BroadcastScalarRefOp(h.GetClone(scalar_ref_), shape_ref, type_id_);
  }

 private:
  ScalarRef *scalar_ref_;
};

class CompareScalarRefOp : public CompareScalarOp {
 public:
  CompareScalarRefOp(int op_type, NDObject *input, ScalarRef *scalar)
      : CompareScalarOp(op_type, input, 0), scalar_ref_(scalar) {}

  uint64_t Emit(VectorKernel &k) override {
    scalar_ = EncodeScalarRef(scalar_ref_, type_id_);
    return CompareScalarOp::Emit(k);
  }
  NDObject *Clone(CloneHelper &h) override { return new CompareScalarRefOp(cmp_op_, h.GetClone(lhs_), h.GetClone(scalar_ref_)); }

 private:
  ScalarRef *scalar_ref_;
};

class BinaryScalarRefOp : public BinaryScalarOp {
 public:
  BinaryScalarRefOp(int op_type, NDObject *input, ScalarRef *scalar)
      : BinaryScalarOp(op_type, input, 0), scalar_ref_(scalar) {}

  uint64_t Emit(VectorKernel &k) override {
    scalar_ = EncodeScalarRef(scalar_ref_, type_id_);
    return BinaryScalarOp::Emit(k);
  }
  NDObject *Clone(CloneHelper &h) override { return new BinaryScalarRefOp(op_type_, h.GetClone(lhs_), h.GetClone(scalar_ref_)); }

 private:
  ScalarRef *scalar_ref_;
};

template <BinaryType op_type>
NDObject *BinaryPromotion(Kernel *kernel, dvm::DataType promote_dtype, NDObject *lhs, NDObject *rhs) {
  auto orig_dtype = lhs->type_id_;
  lhs = kernel->Cast(lhs, promote_dtype);
  rhs = kernel->Cast(rhs, promote_dtype);
  auto result = kernel->Binary<op_type>(lhs, rhs);
  return kernel->Cast(result, orig_dtype);
}

template <UnaryType op_type>
NDObject *UnaryPromotion(Kernel *kernel, dvm::DataType promote_dtype, NDObject *lhs) {
  auto orig_dtype = lhs->type_id_;
  lhs = kernel->Cast(lhs, promote_dtype);
  auto result = kernel->Unary<op_type>(lhs);
  return kernel->Cast(result, orig_dtype);
}

template <BinaryType op_type, typename T, bool rhs_val>
NDObject *GetBinaryS(Kernel *kernel, T val, NDObject *input) {
  auto vkernel = kernel->GetImpl();
  if constexpr (op_type == BinaryType::kAdd || op_type == BinaryType::kMul || op_type == BinaryType::kMinimum ||
                op_type == BinaryType::kMaximum) {
    NDObject *obj;
    if constexpr (std::is_same<T, ScalarRef *>::value) {
      obj = new BinaryScalarRefOp(binary_map[op_type], input, val);
    } else {
      obj = new BinaryScalarOp(binary_map[op_type], input, EncodeScalar(val, input->type_id_));
    }
    vkernel->Append(obj);
    return obj;
  } else if constexpr (op_type == BinaryType::kDiv) {
    if (input->type_id_ == kBFloat16) {
      auto result = GetBinaryS<op_type, T, rhs_val>(kernel, val, kernel->Cast(input, kFloat32));
      return kernel->Cast(result, input->type_id_);
    }
    NDObject *obj;
    if constexpr (std::is_same<T, ScalarRef *>::value) {
      obj = new BinaryScalarRefOp(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input, val);
    } else {
      obj = new BinaryScalarOp(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input,
                               EncodeScalar(val, input->type_id_));
    }
    vkernel->Append(obj);
    return obj;
  } else if constexpr (op_type == BinaryType::kEqual || op_type == BinaryType::kNotEqual ||
                       op_type == BinaryType::kGreater || op_type == BinaryType::kGreaterEqual ||
                       op_type == BinaryType::kLess || op_type == BinaryType::kLessEqual) {
    auto type_id = input->type_id_;
    if (type_id != kInt32 || g_system.Arch() == kAiCore_C310) {
      NDObject *obj;
      if constexpr (std::is_same<T, ScalarRef *>::value) {
        obj = new CompareScalarRefOp(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input, val);
      } else {
        obj = new CompareScalarOp(rhs_val ? binary_map[op_type] : lhs_val_binary_map[op_type], input,
                                  EncodeScalar(val, type_id));
      }
      vkernel->Append(obj);
      return obj;
    }
    return nullptr;
  } else if constexpr (op_type == BinaryType::kPow) {
    if constexpr (std::is_pointer<T>::value) {
      return nullptr;
    } else {
      if constexpr (rhs_val) {
        if (isInteger(val)) {
          return PowS(kernel, input, val);
        }
      } else if (val > T(0)) {
        if (input->type_id_ != kFloat32) {
          auto result = GetBinaryS<BinaryType::kPow, float, false>(kernel, static_cast<float>(val),
                                                                     kernel->Cast(input, kFloat32));
          return kernel->Cast(result, input->type_id_);
        }
        auto tmp = GetBinaryS<BinaryType::kMul, float, true>(kernel, std::log(static_cast<float>(val)), input);
        return kernel->Unary<UnaryType::kExp>(tmp);
      }
      return nullptr;
    }
  } else if constexpr (op_type == BinaryType::kSub) {
    if constexpr (std::is_pointer<T>::value) {
      return nullptr;
    } else {
      if constexpr (rhs_val) {
        return GetBinaryS<BinaryType::kAdd, T, true>(kernel, -val, input);
      } else {
        auto neg_input = GetBinaryS<BinaryType::kMul, T, true>(kernel, static_cast<T>(-1), input);
        return GetBinaryS<BinaryType::kAdd, T, true>(kernel, val, neg_input);
      }
    }
  }
  return nullptr;
}

class NDSliceLoad : public NDViewLoad {
 public:
  NDSliceLoad(void *src, IntArrayRef *src_ref, IntArrayRef *start_ref, IntArrayRef *size_ref, DataType type_id)
      : NDViewLoad(src, size_ref, &stride_data_, type_id), start_ref_(start_ref), src_ref_(src_ref) {
    MESS(offset_data_, 10);
  }
  void Normalize(std::vector<NDObject *> &run_ops) {
    stride_data_[src_ref_->size - 1] = 1;
    for (size_t i = src_ref_->size - 1; i > 0; --i) {
      stride_data_[i - 1] = stride_data_[i] * src_ref_->data[i];
    }
    offset_data_ = 0;
    if (start_ref_) {
      ASSERT(start_ref_->size == src_ref_->size);
      auto get_start = [this](size_t idx) -> int64_t {
        auto start = start_ref_->data[idx];
        return start >= 0 ? start : start + src_ref_->data[idx];
      };
      offset_data_ = get_start(start_ref_->size - 1);
      for (size_t i = src_ref_->size - 1; i > 0; --i) {
        offset_data_ += stride_data_[i - 1] * get_start(i - 1);
      }
    }
    NDViewLoad::Normalize(run_ops);
    offset_bytes_ = static_cast<uint64_t>(offset_data_) * ITEM_SIZE[type_id_];
  }
  NDObject *Clone(CloneHelper &h) override {
    auto src_ref = h.GetClone(src_ref_);
    auto start_ref = h.GetClone(start_ref_);
    auto shape_ref = h.GetClone(shape_ref_);
    return new NDSliceLoad(addr_.gm, src_ref, start_ref, shape_ref, type_id_);
  }
 protected:
  IntArrayRef *start_ref_;
  IntArrayRef *src_ref_;
  ShapeWithRef stride_data_;
  int64_t offset_data_;
};

class NDStridedSliceLoad : public NDSliceLoad {
 public:
  NDStridedSliceLoad(void *src, IntArrayRef *src_ref, IntArrayRef *start_ref, IntArrayRef *end_ref, IntArrayRef *step_ref,
                     DataType type_id = kFloat32)
      : NDSliceLoad(src, src_ref, start_ref, &shape_, type_id), end_ref_(end_ref), step_ref_(step_ref) {}
  void Normalize(std::vector<NDObject *> &run_ops) {
    ASSERT(std::all_of(step_ref_->data, step_ref_->data + step_ref_->size, [](int64_t i) { return i == 1; }));
    shape_.Resize(src_ref_->size);
    for (size_t i = 0; i < src_ref_->size; i++) {
      int64_t end = end_ref_->data[i] < 0 ? end_ref_->data[i] + src_ref_->data[i] : end_ref_->data[i];
      int64_t start = start_ref_ == nullptr
                        ? 0
                        : (start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i]);
      shape_[i] = std::min(end, src_ref_->data[i]) - start;
    }
    NDSliceLoad::Normalize(run_ops);
  }
  NDObject *Clone(CloneHelper &h) override {
    auto shape_ref = h.GetClone(shape_ref_);
    auto src_stride_ref = h.GetClone(src_stride_ref_);
    auto end_ref = h.GetClone(end_ref_);
    auto step_ref = h.GetClone(step_ref_);
    return new NDStridedSliceLoad(addr_.gm, shape_ref, src_stride_ref, end_ref, step_ref, type_id_);
  }

 private:
  ShapeWithRef shape_;
  IntArrayRef *end_ref_;
  IntArrayRef *step_ref_;
};

class ReshapeRankOp : public ReshapeOp {
 public:
  ReshapeRankOp(NDObject *input, const Communicator *comm) : ReshapeOp(input, nullptr), comm_(comm) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) {
    size_t last_idx = lhs_->nd_.size() - 1;
    for (size_t i = 0; i < last_idx; ++i) {
      ndd_.dims[i] = lhs_->nd_[i];
    }
    auto last_dim = lhs_->nd_[last_idx];
    auto rank_size = comm_->GetRankSize();
    if (last_dim != rank_size) {
      ndd_.dims[last_idx] = last_dim / rank_size;
      last_idx++;
    }
    ndd_.dims[last_idx] = rank_size;
    ndd_.dims.resize(last_idx + 1);
  }
  NDObject *Clone(CloneHelper &h) override { return new ReshapeRankOp(h.GetClone(lhs_), comm_); }
  const Communicator *comm_;
};

VKernel *NewKernel(KernelType type, uint32_t flags) {
  VKernel *kernel;
  switch (type) {
    case KernelType::kVector: {
      bool dynamic = flags & KernelFlag::kDynamic;
      if (flags & KernelFlag::kSpeculate) {
        if (dynamic) {
          kernel = new SpecVector<true>();
        } else {
          kernel = new SpecVector<false>();
        }
      } else {
        kernel = dynamic ? new VKernelD() : new VKernelS();
      }
      break;
    }
    case KernelType::kCube:
    case KernelType::kMix: {
      kernel = flags & KernelFlag::kDynamic ? new DynMixKernel() : new MixKernel();
      break;
    }
    case KernelType::kParallel: {
      kernel = new ParallelKernel(flags);
      break;
    }
    case KernelType::kSequence: {
      kernel = new SequenceKernel(flags);
      break;
    }
    case KernelType::kSplit: {
      bool unify_ws = flags & KernelFlag::kUnifyWS;
      if (flags & KernelFlag::kDynamic) {
        kernel = unify_ws ? new SplitGraphDW() : new SplitGraphD();
      } else {
        kernel = new SplitGraphS(unify_ws);
      }
      break;
    }
    case KernelType::kEager: {
      kernel = flags & KernelFlag::kUnifyWS ? new SplitEagerW() : new VKernelE();
      break;
    }
    default: {
      kernel = nullptr;
      break;
    }
  }
  return kernel;
}
}  // namespace

Float16::Float16(float v) : Float16(EncoderFP16(v)) {}
Float16::Float16(int32_t v) : Float16(static_cast<float>(v)) {}
Float16::operator float() const { return ToFloat32(*this); }
Float16::operator int32_t() const { return static_cast<int32_t>(ToFloat32(*this)); }

BFloat16::BFloat16(float v) : BFloat16(EncoderBF16(v)) {}
BFloat16::BFloat16(int32_t v) : BFloat16(static_cast<float>(v)) {}
BFloat16::operator float() const { return ToFloat32(*this); }
BFloat16::operator int32_t() const { return static_cast<int32_t>(ToFloat32(*this)); }

Comm::~Comm() {
  if (comm_) delete comm_;
}

bool Comm::Init(int rank_id, int rank_size, int comm_type, const uint32_t *group_ranks) {
  if (comm_) {
    delete comm_;
  }
  if (comm_type == Comm::kMemory) {
    comm_ = new MemoryComm(rank_id, rank_size, group_ranks);
    if (!static_cast<MemoryComm *>(comm_)->Init()) return false;
  } else if (comm_type == Comm::kHccl) {
    // TODO: create hccl_comm
    comm_ = new HcclComm(nullptr);
  } else {
    ASSERT(comm_type == Comm::kDummy);
    comm_ = new DummyComm(rank_id, rank_size);
  }
  return true;
}

void Comm::Init(void *hccl_comm) {
  if (comm_) {
    delete comm_;
  }
  comm_ = new HcclComm(hccl_comm);
}

Kernel::Kernel() : kernel_{nullptr} {}

Kernel::~Kernel() { delete kernel_; }

void Kernel::SetNameHint(const char *name, const char *fullname) { kernel_->SetNameHint(name, fullname); }

void Kernel::Reset(KernelType type, uint32_t flags) {
  g_system.Init();
  if (kernel_) {
    delete kernel_;
  }
  kernel_ = NewKernel(type, flags);
}

void Kernel::Clone(const Kernel &base, CloneHelper &helper) {
  if (kernel_) {
    delete kernel_;
  }
  auto k = base.GetImpl();
  kernel_ = NewKernel(k->KType(), k->Flags());
  ASSERT(kernel_->KType() == k->KType() && kernel_->Flags() == k->Flags());
  kernel_->Clone(k, helper);
}

NDObject *Kernel::Load(void *addr, IntArrayRef *shape, DataType type) {
  NDObject *obj = new NDLoad(addr, shape, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Load(void *addr, IntArrayRef *shape, IntArrayRef *stride, DataType type) {
  NDObject *obj;
  if (stride) {
    obj = new NDViewLoad(addr, shape, stride, type);
  } else {
    obj = new NDLoad(addr, shape, type);
  }
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::SliceLoad(void *addr, IntArrayRef *shape, IntArrayRef *start, IntArrayRef *size, DataType type) {
  auto obj = new NDSliceLoad(addr, shape, start, size, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::StridedSliceLoad(void *addr, IntArrayRef *shape, IntArrayRef *start, IntArrayRef *end, IntArrayRef *step,
                                   DataType type) {
  auto obj = new NDStridedSliceLoad(addr, shape, start, end, step, type);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::MultiLoad(void *addr, IntArrayRef *shape, DataType type, const Comm *comm) {
  NDObject *obj = new NDMultiLoad(static_cast<uint8_t *>(addr), shape, type, comm->GetImpl());
  kernel_->Append(obj);
  return obj;
}

template <UnaryType op_type>
NDObject *Kernel::Unary(NDObject *input) {
  switch (input->type_id_) {
    case kBool:
      return UnaryPromotion<op_type>(this, DataType::kFloat16, input);
    case kBFloat16:
      return UnaryPromotion<op_type>(this, DataType::kFloat32, input);
    case kFloat16: {
      if constexpr (op_type >= UnaryType::kRound && op_type <= UnaryType::kTrunc) {
        return UnaryPromotion<op_type>(this, DataType::kFloat32, input);
      }
      break;
    }
    case kInt32: {
      if constexpr (op_type == UnaryType::kAbs) {
        if (g_system.Arch() == kAiCore_C220) {
          return Binary<BinaryType::kMaximum>(input, Binary<BinaryType::kMul>(input, -1));
        }
      }
      break;
    }
    default:
      break;
  }
  if constexpr (op_type == UnaryType::kLogicalNot) {
    return Binary<BinaryType::kSub>(1, input);
  }
  if constexpr (op_type == UnaryType::kReciprocal) {
    return Binary<BinaryType::kDiv>(1, input);
  }
  NDObject *obj = new UnaryOp(op_type, input);
  kernel_->Append(obj);
  return obj;
}

#define DEF_UNARY(op) template NDObject *Kernel::Unary<op>(NDObject *)
DEF_UNARY(UnaryType::kSqrt);
DEF_UNARY(UnaryType::kAbs);
DEF_UNARY(UnaryType::kLog);
DEF_UNARY(UnaryType::kExp);
DEF_UNARY(UnaryType::kReciprocal);
DEF_UNARY(UnaryType::kIsFinite);
DEF_UNARY(UnaryType::kLogicalNot);
DEF_UNARY(UnaryType::kRound);
DEF_UNARY(UnaryType::kFloor);
DEF_UNARY(UnaryType::kCeil);
DEF_UNARY(UnaryType::kTrunc);

template <BinaryType op_type, typename L, typename R>
NDObject *Kernel::Binary(L lhs, R rhs) {
  if constexpr (!std::is_same<L, NDObject *>::value) {
    if (rhs->type_id_ == kBool) {
      return Cast(Binary<op_type>(lhs, Cast(rhs, kFloat16)), kBool);
    }
    NDObject *obj = GetBinaryS<op_type, L, false>(this, lhs, rhs);
    if (obj == nullptr) {
      return Binary<op_type>(Broadcast(lhs, rhs->shape_ref_, rhs->type_id_), rhs);
    }
    return obj;
  } else if constexpr (!std::is_same<R, NDObject *>::value) {
    if (lhs->type_id_ == kBool) {
      return Cast(Binary<op_type>(Cast(lhs, kFloat16), rhs), kBool);
    }
    NDObject *obj = GetBinaryS<op_type, R, true>(this, rhs, lhs);
    if (obj == nullptr) {
      return Binary<op_type>(lhs, Broadcast(rhs, lhs->shape_ref_, lhs->type_id_));
    }
    return obj;
  } else {
    switch (lhs->type_id_) {
      case kBool:
        return BinaryPromotion<op_type>(this, DataType::kFloat16, lhs, rhs);
      case kBFloat16: {
        if constexpr (op_type == BinaryType::kPow || op_type == BinaryType::kDiv) {
          return BinaryPromotion<op_type>(this, DataType::kFloat32, lhs, rhs);
        }
        break;
      }
      case kFloat16: {
        if constexpr (op_type == BinaryType::kPow) {
          return BinaryPromotion<op_type>(this, DataType::kFloat32, lhs, rhs);
        }
        break;
      }
      case kInt32: {
        if (g_system.Arch() == kAiCore_C220) {
          if constexpr (op_type == kGreaterEqual) {
            return Binary<BinaryType::kEqual>(Binary<BinaryType::kMaximum>(lhs, rhs), lhs);
          } else if constexpr (op_type == kLess) {
            return Binary<BinaryType::kNotEqual>(Binary<BinaryType::kMaximum>(lhs, rhs), lhs);
          } else if constexpr (op_type == kLessEqual) {
            return Binary<BinaryType::kEqual>(Binary<BinaryType::kMinimum>(lhs, rhs), lhs);
          } else if constexpr (op_type == kGreater) {
            return Binary<BinaryType::kNotEqual>(Binary<BinaryType::kMinimum>(lhs, rhs), lhs);
          }
        }
        break;
      }
      default:
        break;
    }
    NDObject *obj;
    if constexpr (op_type == BinaryType::kPow) {
      obj = new PowerOp(lhs, rhs);
    } else if constexpr (static_cast<int>(op_type) < V_CMP_ALL) {
      obj = new CompareOp(op_type, lhs, rhs);
    } else {
      obj = new BinaryOp(op_type, lhs, rhs);
    }
    kernel_->Append(obj);
    return obj;
  }
}

#define DEF_BINARY(op)  \
  template NDObject *Kernel::Binary<op>(NDObject *, NDObject *); \
  template NDObject *Kernel::Binary<op>(NDObject *, float);      \
  template NDObject *Kernel::Binary<op>(NDObject *, int32_t);    \
  template NDObject *Kernel::Binary<op>(NDObject *, Float16);    \
  template NDObject *Kernel::Binary<op>(NDObject *, BFloat16);   \
  template NDObject *Kernel::Binary<op>(float, NDObject *);      \
  template NDObject *Kernel::Binary<op>(int32_t, NDObject *);    \
  template NDObject *Kernel::Binary<op>(Float16, NDObject *);    \
  template NDObject *Kernel::Binary<op>(BFloat16, NDObject *);   \
  template NDObject *Kernel::Binary<op>(NDObject *, ScalarRef *);\
  template NDObject *Kernel::Binary<op>(ScalarRef *, NDObject *)

DEF_BINARY(BinaryType::kEqual);
DEF_BINARY(BinaryType::kNotEqual);
DEF_BINARY(BinaryType::kGreater);
DEF_BINARY(BinaryType::kGreaterEqual);
DEF_BINARY(BinaryType::kLess);
DEF_BINARY(BinaryType::kLessEqual);
DEF_BINARY(BinaryType::kAdd);
DEF_BINARY(BinaryType::kSub);
DEF_BINARY(BinaryType::kMul);
DEF_BINARY(BinaryType::kDiv);
DEF_BINARY(BinaryType::kPow);
DEF_BINARY(BinaryType::kMaximum);
DEF_BINARY(BinaryType::kMinimum);
DEF_BINARY(BinaryType::kLogicalAnd);
DEF_BINARY(BinaryType::kLogicalOr);

NDObject *Kernel::Select(NDObject *cond, NDObject *lhs, NDObject *rhs) {
  if (cond->type_id_ != lhs->type_id_) {
    cond = this->Cast(cond, lhs->type_id_);
  }
  auto obj = new SelectOp(cond, lhs, rhs);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Cast(NDObject *input, DataType type) {
  static const int g_cast_staff_type[kDataTypeEnd][kDataTypeEnd] = {
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
    if (input->type_id_ == kBFloat16 && g_system.Arch() == kAiCore_C220) {
      input = Cast(input, kFloat32);
    }
    input = Binary<BinaryType::kNotEqual>(input, 0);
  }
  auto stuff_type = g_cast_staff_type[input->type_id_][type];
  while (stuff_type != -1) {
    input = new CastOp(input, static_cast<DataType>(stuff_type));
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
NDObject *Kernel::Broadcast(T val, IntArrayRef *shape, DataType type) {
  if (type == DataType::kBool) {
    return Cast(Broadcast(val, shape, DataType::kFloat16), DataType::kBool);
  }
  NDObject *obj;
  if constexpr (std::is_same<T, ScalarRef *>::value) {
    obj = new BroadcastScalarRefOp(val, shape, type);
  } else {
    obj = new BroadcastScalarOp(EncodeScalar(val, type), shape, type);
  }
  kernel_->Append(obj);
  return obj;
}

template NDObject *Kernel::Broadcast<float>(float val, IntArrayRef *shape, DataType type);
template NDObject *Kernel::Broadcast<int32_t>(int32_t val, IntArrayRef *shape, DataType type);
template NDObject *Kernel::Broadcast<Float16>(Float16 val, IntArrayRef *shape, DataType type);
template NDObject *Kernel::Broadcast<BFloat16>(BFloat16 val, IntArrayRef *shape, DataType type);
template NDObject *Kernel::Broadcast<ScalarRef *>(ScalarRef *val, IntArrayRef *shape, DataType type);

NDObject *Kernel::Broadcast(NDObject *input, IntArrayRef *shape) {
  if (input->type_id_ == DataType::kBool) {
    auto cast1 = Cast(input, DataType::kFloat16);
    auto obj = Broadcast(cast1, shape);
    auto cast2 = Cast(obj, DataType::kBool);
    return cast2;
  }
  auto obj = new BroadcastOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

template <typename T>
NDObject *Kernel::OneHot(NDObject *indices, IntArrayRef *depth, int axis, T on_value, T off_value) {
  auto on_code = EncodeScalar(on_value);
  auto off_code = EncodeScalar(off_value);
  auto obj = new OneHotOp(indices, depth, axis, on_code, off_code, TypeTrait<T>::ID);
  kernel_->Append(obj);
  return obj;
}

template NDObject *Kernel::OneHot<float>(NDObject *, IntArrayRef *, int, float, float);
template NDObject *Kernel::OneHot<int32_t>(NDObject *, IntArrayRef *, int, int32_t, int32_t);
template NDObject *Kernel::OneHot<Float16>(NDObject *, IntArrayRef *, int, Float16, Float16);
template NDObject *Kernel::OneHot<BFloat16>(NDObject *, IntArrayRef *, int, BFloat16, BFloat16);

NDObject *Kernel::Reshape(NDObject *input, IntArrayRef *shape) {
  auto obj = new ReshapeOp(input, shape);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::_Reduce(int op_type, NDObject *input, IntArrayRef *dims, bool keepdims) {
  if (input->type_id_ != DataType::kFloat32 && op_type == kSum) {
    return nullptr;
  }
  auto obj = new ReduceOp(input, op_type, dims, keepdims);
  kernel_->Append(obj);
  return obj;
}

NDObject *Kernel::Store(void *addr, NDObject *input) {
  if (input->IsLoad()) {
    input = Copy(input);
  }
  auto ktype = kernel_->KType();
  if (ktype == KernelType::kEager) {
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
  NDObject *obj = new NDPadStore(addr, input, pad_size);
  kernel_->Append(obj);
  return obj;
}

void Kernel::SetStoreInplace(NDObject *store) {
  ASSERT(kernel_->IsSplit());
  _SplitKernel::SetStoreInplace(store, 1);
}

NDObject *Kernel::_AllReduce(int op_type, NDObject *input, const Comm *comm) {
  NDObject *obj;
  if (input->type_id_ == DataType::kBFloat16) {
    obj = new AllReduceOp<true>(op_type, input, comm->GetImpl());
  } else {
    obj = new AllReduceOp<false>(op_type, input, comm->GetImpl());
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
  if (input->obj_id_ != ObjectType::kMultiLoad) {
    input = new ReshapeRankOp(input, comm->GetImpl());
    kernel_->Append(input);
  }
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
                                NDObject *group_list, GmmSplitType group_type, GmmListType group_list_type) {
  GmmOp *obj = new GmmOp(lhs, rhs, trans_a, trans_b, bias, group_list, group_type, group_list_type);
  if (kernel_->KType() == KernelType::kEager) {
    return static_cast<VKernelE *>(kernel_)->AppendCube(obj);
  }
  kernel_->Append(obj);
  return obj;
}

void Kernel::ParallelAdd(KernelType type, uint32_t flags, size_t thread_limit) {
  ASSERT(kernel_->KType() == KernelType::kParallel);
  static_cast<ParallelKernel*>(kernel_)->AddKernel(type, flags, thread_limit);
}

void Kernel::SequenceAdd(KernelType type, uint32_t flags) {
  ASSERT(kernel_->KType() == KernelType::kSequence);
  if (unlikely(type >= KernelType::kSequence)) {
    DvmException("invalid sequence add kernel type");
  }
  if (kernel_->IsDynamic()) {
    flags |= KernelFlag::kDynamic;
  }
  auto kernel = NewKernel(type, flags);
  static_cast<SequenceKernel *>(kernel_)->AddStage(kernel);
}

void Kernel::SpecNext() { static_cast<_SpecVector *>(kernel_)->Next(); }

IntArrayRef *Kernel::GetShape(NDObject *op) { return op->shape_ref_; }

DataType Kernel::GetDType(NDObject *op) { return op->type_id_; }

size_t Kernel::PreCodeGen() {
  uint64_t ws_size = kernel_->CodeGen();
  return kernel_->code_.ReserveWorkspace(ws_size);
}

void Kernel::Normalize() { kernel_->Normalize(); }

int Kernel::Launch(const RelocEntry *relocs, size_t reloc_size, void *workspace, void *stream) {
  kernel_->UpdatePreWS(workspace);
  kernel_->CodeGenR(relocs, reloc_size, nullptr);
  return kernel_->Launch(stream);
}

int Kernel::Launch(void *stream) { return kernel_->Launch(stream); }

void Kernel::CodeGen(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  kernel_->CodeGenR(relocs, reloc_size, ws_alloc);
}

void Kernel::Clear() {
  if (kernel_->KType() == KernelType::kEager) {
    static_cast<VKernelE *>(kernel_)->Clear();
  }
}

const char *Kernel::Dump() const {
  std::string &graph = kernel_->DumpGraph();
  return graph.c_str();
}

const char *Kernel::Das() const {
  std::string &das = kernel_->DisAssemble();
  return das.c_str();
}

Config &Config::Instance() { return g_system; }
}  // namespace dvm
