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

#ifndef _DVM_H_
#define _DVM_H_

#include <cstdint>
#include <vector>

namespace dvm {
enum DataType {
  kBool = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kInt32,
  kInt64,
  kDataTypeEnd,
};

enum UnaryType {
  kSqrt = 0,
  kAbs,
  kLog,
  kExp,
  kReciprocal,
  kIsFinite,
  kLogicalNot,
  kRound,
  kFloor,
  kCeil,
  kTrunc,
  kUnaryTypeEnd,
};

enum BinaryType {
  kEqual = 0,
  kNotEqual,
  kGreater,
  kGreaterEqual,
  kLess,
  kLessEqual,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kPow,
  kMaximum,
  kMinimum,
  kLogicalAnd,
  kLogicalOr,
  kBinaryTypeEnd,
};

enum ReduceType {
  kSum = 0,
  kMax,
  kMin,
  kReduceTypeEnd,
};

enum GmmSplitType {
  kSplit_M = 0,
  kSplit_N,
  kSplit_K,
  kGroupTypeEnd,
};

enum GmmListType {
  kListBound = 0,
  kListSize,
  kEnd,
};

enum KernelType {
  kVector = 0,
  kCube,
  kMix,
  kParallel,
  kSequence,
  kSplit,
  kEager,
  kKernelTypeEnd,
};

enum KernelFlag {
  kDynamic = 0x1,
  kUnifyWS = 0x2,
  kSpeculate = 0x4,
};

class NDObject;
class VKernel;
class MsprofHelper;
class Communicator;

struct ShapeRef {
  ShapeRef() {}
  explicit ShapeRef(const std::vector<int64_t> &other) : data(other.data()), size(other.size()) {}
  ShapeRef &operator=(const std::vector<int64_t> &other) {
    data = other.data();
    size = other.size();
    return *this;
  }
  const int64_t *data;
  size_t size;
};

class Float16 {
 public:
  explicit Float16(uint16_t v) : value_(v) {}
  explicit Float16(float v);
  explicit Float16(int32_t v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

class BFloat16 {
 public:
  explicit BFloat16(uint16_t v) : value_(v) {}
  explicit BFloat16(float v);
  explicit BFloat16(int32_t v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

struct RelocEntry {
  RelocEntry() {}
  RelocEntry(NDObject *p, void *a) : io(p), addr(a) {}
  NDObject *io;
  void *addr;
};

struct WsAllocator {
  virtual void *Alloc(size_t size) = 0;
};

class Comm {
 public:
  enum { kMemory, kHccl, kDummy };
  Comm() = default;
  ~Comm();
  bool Init(int rank_id, int rank_size, int comm_type, const uint32_t *group_ranks = nullptr);
  bool Init(int rank_id, int rank_size) { return Init(rank_id, rank_size, kMemory); }
  void Init(void *hccl_comm);
  inline const Communicator *GetImpl() const { return comm_; }

 protected:
  Communicator *comm_{nullptr};
};

class Kernel {
 public:
  Kernel();
  ~Kernel();

  void Reset(KernelType type, uint32_t flags);
  void SetNameHint(const char *name, const char *fullname) {
    op_name_ = name;
    op_fullname_ = fullname;
  }

  NDObject *Load(void *addr, ShapeRef *shape, DataType type);
  NDObject *Load(void *addr, ShapeRef *shape, ShapeRef *stride, const int64_t *offset, DataType type);
  NDObject *SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DataType type);
  NDObject *StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step, DataType type);
  NDObject *MultiLoad(void *addr, ShapeRef *shape, DataType type, const Comm *comm);
  NDObject *Store(void *addr, NDObject *input);
  NDObject *PadStore(void *addr, NDObject *input, int64_t pad_size);
  void SetStoreInplace(NDObject *store);

  template <UnaryType op_type>
  NDObject *Unary(NDObject *input);
  template <BinaryType op_type, typename L, typename R>
  NDObject *Binary(L lhs, R rhs);
  template <ReduceType op_type>
  NDObject *Reduce(NDObject *input, ShapeRef *dims, bool keepdims) { return _Reduce(op_type, input, dims, keepdims); }
  NDObject *Select(NDObject *cond, NDObject *lhs, NDObject *rhs);
  NDObject *Cast(NDObject *input, DataType type);
  NDObject *Broadcast(NDObject *input, ShapeRef *shape);
  template <typename T>
  NDObject *Broadcast(T val, ShapeRef *shape, DataType type);
  NDObject *Reshape(NDObject *input, ShapeRef *shape);
  NDObject *Copy(NDObject *input);
  template <typename T>
  NDObject *OneHot(NDObject *indices, ShapeRef *depth, int axis, T on_value, T off_value);
  NDObject *ElemAny(NDObject *input);

  NDObject *MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias);
  NDObject *GroupedMatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias,
                          NDObject *group_list, GmmSplitType group_type, GmmListType group_list_type);

  // collective communication
  template <ReduceType op_type>
  NDObject *AllReduce(NDObject *input, const Comm *comm) { return _AllReduce(op_type, input, comm); }
  NDObject *AllGather(NDObject *input, const Comm *comm);
  NDObject *AllGatherV2(NDObject *input, const Comm *comm);
  NDObject *ReduceScatter(NDObject *input, const Comm *comm);

  void ParallelNext();
  void SpecNext();
  void SequenceAdd(KernelType type, uint32_t flags);

  size_t CodeGen();
  int Launch(const RelocEntry *relocs, size_t reloc_size, void *workspace, void *stream);

  void Infer();
  void CodeGen(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc);
  int Launch(void *stream);
  void Clear();

  ShapeRef *GetShape(NDObject *op) const;
  DataType GetDType(NDObject *op) const;

  const char *Dump() const;
  const char *Das() const;

  VKernel *GetImpl() const { return kernel_; }

 protected:
  NDObject *_Reduce(int op_type, NDObject *input, ShapeRef *dims, bool keepdims);
  NDObject *_AllReduce(int op_type, NDObject *input, const Comm *comm);

  VKernel *kernel_;
  MsprofHelper *msprof_helper_;
  const char *op_name_;
  const char *op_fullname_;

  /* DEPRECATED */
 public:
  NDObject *Unary(int op_type, NDObject *input) {
    switch (op_type) {
      case UnaryType::kSqrt:
        return Unary<UnaryType::kSqrt>(input);
      case UnaryType::kAbs:
        return Unary<UnaryType::kAbs>(input);
      case UnaryType::kLog:
        return Unary<UnaryType::kLog>(input);
      case UnaryType::kExp:
        return Unary<UnaryType::kExp>(input);
      case UnaryType::kReciprocal:
        return Unary<UnaryType::kReciprocal>(input);
      case UnaryType::kIsFinite:
        return Unary<UnaryType::kIsFinite>(input);
      case UnaryType::kLogicalNot:
        return Unary<UnaryType::kLogicalNot>(input);
      case UnaryType::kRound:
        return Unary<UnaryType::kRound>(input);
      case UnaryType::kFloor:
        return Unary<UnaryType::kFloor>(input);
      case UnaryType::kCeil:
        return Unary<UnaryType::kCeil>(input);
      case UnaryType::kTrunc:
        return Unary<UnaryType::kTrunc>(input);
      default:
        return nullptr;
    }
  }
  template <typename L, typename R>
  NDObject *Binary(int op_type, L lhs, R rhs) {
    switch (op_type) {
      case BinaryType::kEqual:
        return Binary<BinaryType::kEqual>(lhs, rhs);
      case BinaryType::kNotEqual:
        return Binary<BinaryType::kNotEqual>(lhs, rhs);
      case BinaryType::kGreater:
        return Binary<BinaryType::kGreater>(lhs, rhs);
      case BinaryType::kGreaterEqual:
        return Binary<BinaryType::kGreaterEqual>(lhs, rhs);
      case BinaryType::kLess:
        return Binary<BinaryType::kLess>(lhs, rhs);
      case BinaryType::kLessEqual:
        return Binary<BinaryType::kLessEqual>(lhs, rhs);
      case BinaryType::kAdd:
        return Binary<BinaryType::kAdd>(lhs, rhs);
      case BinaryType::kSub:
        return Binary<BinaryType::kSub>(lhs, rhs);
      case BinaryType::kMul:
        return Binary<BinaryType::kMul>(lhs, rhs);
      case BinaryType::kDiv:
        return Binary<BinaryType::kDiv>(lhs, rhs);
      case BinaryType::kPow:
        return Binary<BinaryType::kPow>(lhs, rhs);
      case BinaryType::kMaximum:
        return Binary<BinaryType::kMaximum>(lhs, rhs);
      case BinaryType::kMinimum:
        return Binary<BinaryType::kMinimum>(lhs, rhs);
      case BinaryType::kLogicalAnd:
        return Binary<BinaryType::kLogicalAnd>(lhs, rhs);
      case BinaryType::kLogicalOr:
        return Binary<BinaryType::kLogicalOr>(lhs, rhs);
      default:
        return nullptr;
    }
  }
  NDObject *Reduce(int op_type, NDObject *input, ShapeRef *dims, bool keepdims) { return _Reduce(op_type, input, dims, keepdims); }
  NDObject *AllReduce(int op_type, NDObject *input, const Comm *comm) { return _AllReduce(op_type, input, comm); }
};

class Config {
 public:
  Config() = default;
  ~Config() = default;
  static Config &Instance();
  virtual Config &SetDeterm() = 0;
  virtual Config &UnsetDeterm() = 0;
  virtual Config &SetOnlineTuner() = 0;
  virtual Config &UnsetOnlineTuner() = 0;
  virtual Config &SetLazyTuner() = 0;
  virtual Config &UnsetLazyTuner() = 0;
};

/* DEPRECATED */
using DType = DataType;
using UnaryOpType = UnaryType;
using BinaryOpType = BinaryType;
using ReduceOpType = ReduceType;
using GroupType = GmmSplitType;
using GroupListType = GmmListType;
}  // namespace dvm
#endif  // _DVM_H_
