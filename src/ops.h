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

#ifndef _DVM_OPS_H_
#define _DVM_OPS_H_

#include <cstdint>
#include <vector>
#include <mutex>
#include "isa.h"
#include "system.h"
#include "code.h"

namespace dvm {
enum ObjectType {
  // Load
  kLoadDummy = 0,
  kMultiLoad,
  kSLoad,
  kLoad,

  // Store
  kPadStore,
  kSStore,
  kStore,

  // Comm
  kReduceScatter,
  kAllGather,
  kAllGatherV2,
  kAllReduce,

  // Simd
  kReshape,
  kCopy,
  kUnary,
  kBinary,
  kCast,
  kBinaryS,
  kBroadcastTo,
  kBroadcastS,
  kReduce,
  kSelect,
  kElementAny,
  kRemovePad,
  kPower,
  kCompare,
  kCompareS,
  kCubeOp,
  kObjectBulk
};

// Tile axis range: [x, y, z] -> [tile/tail, 1, 1].  x*y*z=tile*num + (tail ? tail - tile : 0)
struct TileParam {
  int start;
  int end;
  int64_t num;
  int64_t tile;
  int64_t tail;
  bool group_tile;
};

struct PropRange {
  enum {
    ELEMWISE = 0,
    BROADCAST,
    REDUCE,
  };
  int base;
  int depth;
  int affine{ELEMWISE};
  int64_t space;
};

std::ostream &operator<<(std::ostream &oss, const ShapeRef &shape);
std::ostream &operator<<(std::ostream &oss, const Float16 &scalar);
std::ostream &operator<<(std::ostream &oss, const BFloat16 &scalar);

static inline void _DimCopy(int64_t *dst, const int64_t *src, size_t size) {
  switch (size) {
    case 10:
      dst[9] = src[9];
    case 9:
      dst[8] = src[8];
    case 8:
      dst[7] = src[7];
    case 7:
      dst[6] = src[6];
    case 6:
      dst[5] = src[5];
    case 5:
      dst[4] = src[4];
    case 4:
      dst[3] = src[3];
    case 3:
      dst[2] = src[2];
    case 2:
      dst[1] = src[1];
    case 1:
      dst[0] = src[0];
      break;
    default:
      break;
  }
}

class DimArray {
 public:
  enum { kMaxDimSize = 10 };
  DimArray() : size_(0) {}
  DimArray &operator=(const DimArray &other) {
    size_ = other.size();
    if (size_ > 0) _DimCopy(data_, other.data(), size_);
    return *this;
  }
  DimArray &operator=(const std::vector<int64_t> &other) {
    size_ = other.size();
    if (size_ > 0) _DimCopy(data_, other.data(), size_);
    return *this;
  }
  bool operator==(const DimArray &other) {
    if (size_ != other.size()) return false;
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i] != other[i]) return false;
    }
    return true;
  }
  void resize(size_t size) {
    ASSERT(size <= kMaxDimSize);
    size_ = size;
  }
  void resize(size_t size, int64_t val) {
    if (size_ < size) {
      for (size_t i = size_; i < size; ++i) data_[i] = val;
    }
    size_ = size;
  }
  size_t size() const { return size_; }
  size_t prod() const {
    size_t res = 1;
    for (size_t i = 0; i < size_; ++i) {
      res *= data_[i];
    }
    return res;
  }
  int64_t &operator[](size_t i) { return data_[i]; }
  const int64_t &operator[](size_t i) const { return data_[i]; }
  int64_t &back() { return *(data_ + size_ - 1); }
  const int64_t &back() const { return *(data_ + size_ - 1); }
  const int64_t *data() const { return data_; }
  bool empty() const { return size_ == 0; }
  void push_back(int64_t val) {
    ASSERT(size_ < kMaxDimSize);
    data_[size_++] = val;
  }

 private:
  int64_t data_[DimArray::kMaxDimSize];
  size_t size_;
};

inline std::ostream &operator<<(std::ostream &oss, const DimArray &nd) {
  oss << "[";
  if (nd.size() > 0) {
    for (size_t i = 0; i < nd.size() - 1; ++i) {
      oss << nd[i] << ",";
    }
    oss << nd.back();
  }
  oss << "]";
  return oss;
}

template <size_t N>
struct ShapeRefData : public ShapeRef {
  ShapeRefData() {
    data = shape;
    size = 0;
  }
  ShapeRefData &operator=(const ShapeRef &other) {
    size = other.size;
    _DimCopy(shape, other.data, size);
    return *this;
  }
  int64_t &operator[](int i) { return shape[i]; }
  void Resize(size_t s) { size = s; }

  int64_t shape[N];
};
using ShapeWithRef = ShapeRefData<DimArray::kMaxDimSize>;

template <size_t BLOCK_SIZE, size_t POOL_SIZE>
class MemPool {
 public:
  MemPool() = default;
  ~MemPool() {
    for (size_t i = 0; i < top_; ++i) {
      std::free(pool_[i]);
    }
  }
  void *Get(size_t size) {
    if (top_ > 0 && size <= BLOCK_SIZE) {
      std::lock_guard<std::mutex> guard(mutex_);
      auto top = top_;
      if (top > 0) {
        top_ = top - 1;
        return pool_[top - 1];
      }
    }
    return std::malloc(size <= BLOCK_SIZE ? BLOCK_SIZE : size);
  }
  void Put(void *mem) {
    if (top_ < POOL_SIZE) {
      std::lock_guard<std::mutex> guard(mutex_);
      auto top = top_;
      if (top < POOL_SIZE) {
        top_ = top + 1;
        pool_[top] = mem;
        return;
      }
    }
    std::free(mem);
  }

 private:
  void *pool_[POOL_SIZE];
  volatile size_t top_{0};
  std::mutex mutex_;
};

class VectorKernel;

#define OBJ_FLAG_FREE_LHS 1
#define OBJ_FLAG_FREE_RHS 2
#define OBJ_FLAG_REUSE_LHS 4
#define OBJ_FLAG_REUSE_RHS 8
#define OBJ_FLAG_DEAD 16
#define OBJ_FLAG_FLEX_RREE_XHS (1u << 5)
#define OBJ_FLAG_FLEX_REUSE_WS (1u << 6)

#define OBJ_FLAG_WORKSPACE (1u << 16)
#define OBJ_FLAG_XHS (2u << 16)
#define OBJ_FLAG_EAGER (8u << 16)
#define OBJ_FLAG_STAGE_IO (16u << 16)
#define OBJ_FLAG_FLEX_INPL_WS (1u << 20)

class NDObject {
 public:
  NDObject(NDObject *lhs, NDObject *rhs, DType type_id, ObjectType obj_id) : lhs_(lhs), rhs_(rhs), obj_id_(obj_id) {
    type_id_ = type_id;
  }
  NDObject(const NDObject &) = delete;
  NDObject &operator=(const NDObject &) = delete;
  virtual ~NDObject() = default;

  // re-infer shape(nd_) from its inputs nd_
  virtual void Normalize(std::vector<NDObject *> &run_ops) {}
  // fold axis right alignment: [base-depth+1, base]
  virtual void FoldProp(PropRange &range) {}
  // fold axis left alignment:  [0, depth-1]
  virtual void AlignProp(PropRange &range) {}
  // tile nd range
  virtual void Tile(const TileParam &tp);
  virtual int Emit(VectorKernel &k) = 0;
  virtual void Dump(bool verbose, std::ostringstream &oss);

  void *operator new(size_t size) { return mem_pool_.Get(size); }

  void operator delete(void *ptr) { std::free(ptr); }

  void UpdateStride(uint64_t simd_width);

  int64_t Size();
  int64_t LeadAlign() const { return strides_[lead_dim_]; }
  uint64_t GetBlocks(int64_t size) const { return (size * ITEM_SIZE[type_id_] + 31) >> 5; }
  ObjectType GetObjectType() const { return obj_id_; }
  // Now we have comm op, which will cross different pipe. This method should be deprecated
  int Pipe() const { return obj_id_ <= kLoad ? V_PIPE_LOAD : (obj_id_ <= kStore ? V_PIPE_STORE : V_PIPE_SIMD); }
  bool IsLoad() const { return obj_id_ <= kLoad; }
  bool IsStore() const { return obj_id_ <= kStore && obj_id_ > kLoad; }
  bool IsComm() const { return obj_id_ > kStore && obj_id_ <= kAllReduce; }
  // Comm op is considered a simd op, remember use !IsComm() to exclude comm op
  bool IsSimd() const { return obj_id_ > kStore; }
  bool NeedTailCopy() const { return obj_id_ > kReduceScatter && obj_id_ <= kAllReduce; }
  void SetFlag(uint32_t mask) { flags_ |= mask; }
  bool CheckFlag(uint32_t mask) const { return flags_ & mask; }

  void Clear(int index) {
    index_ = index;
    xbuf_ = 0;
    lead_dim_ = 0;
    reuse_dep_ = 0;
    flags_ &= 0xffff0000u;
  }

  DimArray nd_;
  DimArray strides_;
  NDObject *lhs_;
  NDObject *rhs_;
  uint64_t xbuf_;
  ShapeRef *shape_ref_;
  NDObject *pd_next_;  // PropDomain next
  ObjectType obj_id_;
  DType type_id_;
  int lead_dim_;
  int index_;
  int reuse_dep_;
  uint32_t flags_{0};
  uint64_t *insn_;       // when in optimization passes, used to point to the next NDObject
  uint64_t *tail_insn_;  // when in optimization passes, used to point to the prev NDObject

  static MemPool<512, 8192> mem_pool_;
};

class NDAccess : public NDObject {
 public:
  NDAccess(void *gm, NDObject *lhs, DType type_id, ObjectType obj_id)
      : NDObject(lhs, nullptr, type_id, obj_id), addr_({gm}) {}
  RelocAddr addr_;
};

class NDLoadDummy : public NDAccess {
 public:
  NDLoadDummy(DType type_id) : NDAccess(nullptr, nullptr, type_id, ObjectType::kLoadDummy) {
    nd_.resize(1, 1);
    shape_.Resize(1);
    shape_[0] = 1;
    shape_ref_ = &shape_;
  }
  void Tile(const TileParam &tp) override {}
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeWithRef shape_;
};

class NDLoad : public NDAccess {
 public:
  NDLoad(void *src, ShapeRef *shape_ref, DType type_id = kFloat32)
      : NDAccess(src, nullptr, type_id, ObjectType::kLoad) {
    shape_ref_ = shape_ref;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  int tail_dim_{-1};
  int tail_size_{0};
  DimArray round_tile_;
};

// split input to `multi_size` parts, everytime load a piece from all parts
class NDMultiLoad : public NDLoad {
 public:
  NDMultiLoad(uint8_t *src, ShapeRef *shape_ref, DType type_id, const Communicator *comm)
      : NDLoad(src, shape_ref, type_id), comm_(comm) {
    obj_id_ = ObjectType::kMultiLoad;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  // void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  // uint32_t multi_size_{1}; // only support up to 1023(2^10)
  uint64_t gap_{0};
  uint32_t xbuf_size_{0};
  const Communicator *comm_;
};

class NDSliceLoad : public NDLoad {
 public:
  NDSliceLoad(void *src, ShapeRef *src_ref, ShapeRef *start_ref, ShapeRef *size_ref, DType type_id = kFloat32)
      : NDLoad(src, size_ref, type_id), start_ref_(start_ref), src_ref_(src_ref), size_ref_(size_ref) {}
  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int64_t CalcOffset();
  ShapeRef *start_ref_;
  ShapeRef *src_ref_;
  ShapeRef *size_ref_;
};

class NDStridedSliceLoad : public NDSliceLoad {
 public:
  NDStridedSliceLoad(void *src, ShapeRef *src_ref, ShapeRef *start_ref, ShapeRef *end_ref, ShapeRef *step_ref,
                     DType type_id = kFloat32)
      : NDSliceLoad(src, src_ref, start_ref, nullptr, type_id), end_ref_(end_ref), step_ref_(step_ref) {
    shape_ref_ = &shape_;
  }

  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeWithRef shape_;
  ShapeRef *end_ref_;
  ShapeRef *step_ref_;
};

class NDStore : public NDAccess {
 public:
  NDStore(NDObject *src) : NDAccess(nullptr, src, src->type_id_, ObjectType::kStore) { shape_ref_ = src->shape_ref_; }
  NDStore(void *dst, NDObject *src) : NDAccess(dst, src, src->type_id_, ObjectType::kStore) {
    shape_ref_ = src->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  DimArray round_tile_;

 private:
  int tail_dim_{-1};
  int tail_size_{0};
};

class NDPadStore : public NDAccess {
 public:
  NDPadStore(NDObject *src, int64_t pad_size)
      : NDAccess(nullptr, src, src->type_id_, ObjectType::kPadStore), pad_size_(pad_size) {
    shape_ref_ = &shape_;
  }
  NDPadStore(void *dst, NDObject *src, int64_t pad_size) : NDPadStore(src, pad_size) { addr_.gm = dst; }

  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeWithRef shape_;
  int64_t pad_size_;
};

class FlexOp : public NDObject {
 public:
  enum { kWsMax = 2 };
  FlexOp(NDObject *lhs, NDObject *rhs, DType type_id, ObjectType obj_id) : NDObject(lhs, rhs, type_id, obj_id) {
    flags_ |= OBJ_FLAG_WORKSPACE;
  }
  void SetXhs(NDObject *xhs) {
    xhs_ = xhs;
    flags_ |= OBJ_FLAG_XHS;
  }

  NDObject *xhs_{nullptr};
  int ws_num_{0};
  uint64_t wss_[kWsMax];
};

class CopyOp : public NDObject {
 public:
  CopyOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kCopy) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class ReshapeOp : public CopyOp {
 public:
  ReshapeOp(NDObject *input, ShapeRef *shape_ref) : CopyOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = &shape_;
    obj_id_ = ObjectType::kReshape;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeRef *dst_shape_ref_;
  ShapeWithRef shape_;
};

class UnaryOp : public NDObject {
 public:
  UnaryOp(int op_type, NDObject *input);
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static int QueryId(const std::string &op_name);

 protected:
  int op_type_;
};

class RemovePadOp : public CopyOp {
 public:
  RemovePadOp(NDObject *input) : CopyOp(input) {
    ASSERT(ITEM_SIZE[type_id_] != 1);
    obj_id_ = ObjectType::kRemovePad;
  }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class ElementAnyOp : public NDObject {
 public:
  ElementAnyOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kElementAny) {
    ASSERT(type_id_ == kFloat32);
    shape_ref_data_.data = &shape_;
    shape_ref_data_.size = 1;
    shape_ref_ = &shape_ref_data_;
  }
  int Emit(VectorKernel &k) override;
  void Tile(const TileParam &tp) override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int64_t shape_{1};
  ShapeRef shape_ref_data_;
  int tail_dim_{-1};
  int tail_size_{0};
};

class CastOp : public NDObject {
 public:
  CastOp(NDObject *input, DType type_id) : NDObject(input, nullptr, type_id, ObjectType::kCast) {
    ASSERT(type_id != lhs_->type_id_);
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

enum BinarySOpType {
  kEquals = 0,
  kNotEquals,
  kGreaters,
  kGreaterEquals,
  kLesss,
  kLessEquals,
  kAdds,
  kMuls,
  kDivs,
  kMaximums,
  kMinimums,
  kBinarySOpEnd,
};

template <typename T>
class BinaryScalarOp : public NDObject {
 public:
  BinaryScalarOp(int op_type, NDObject *input, T scalar);
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int op_type_;
  T scalar_;
};

template <typename T>
class CompareScalarOp : public FlexOp {
 public:
  CompareScalarOp(int op_type, NDObject *input, T scalar);
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int cmp_op_;
  T scalar_;
};

class _BinaryNormalizer {
 public:
  _BinaryNormalizer() = default;
  ~_BinaryNormalizer();
  void Normalize(NDObject *self, std::vector<NDObject *> &run_ops);
  std::vector<NDObject *> lhs_stuff_ops_;
  std::vector<NDObject *> rhs_stuff_ops_;
  ShapeWithRef shape_;
};

class BinaryOp : public NDObject {
 public:
  BinaryOp(int op_type, NDObject *lhs, NDObject *rhs);
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static int QueryId(const std::string &op_name);

 protected:
  int op_type_;
  _BinaryNormalizer norm_;
};

class PowerOp : public FlexOp {
 public:
  PowerOp(NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kPower) {
    ws_num_ = 2;
    shape_ref_ = &norm_.shape_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  _BinaryNormalizer norm_;
};

class CompareOp : public FlexOp {
 public:
  CompareOp(int op_type, NDObject *lhs, NDObject *rhs);
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int cmp_op_;
  _BinaryNormalizer norm_;
};

class SelectOp : public FlexOp {
 public:
  SelectOp(NDObject *cond, NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kSelect) {
    shape_ref_ = &shape_;
    ws_num_ = 1;
    SetXhs(cond);
  }
  ~SelectOp();
  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  std::vector<NDObject *> stuff_ops_[3];
  ShapeWithRef shape_;
};

class _BroadcastOp : public NDObject {
 public:
  _BroadcastOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kBroadcastTo) {}
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int64_t EmitBroadcastX(uint64_t *p, int end_dim);
  int64_t EmitBroadcastY(uint64_t *p, int start_dim, int end_dim);
};

// expect shape is align: equal rank
class BroadcastOp : public _BroadcastOp {
 public:
  BroadcastOp(NDObject *input, ShapeRef *shape_ref) : _BroadcastOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = &shape_;
  }
  ~BroadcastOp();
  void Normalize(std::vector<NDObject *> &run_ops) override;

 private:
  std::vector<NDObject *> stuff_ops_;
  ShapeRef *dst_shape_ref_;
  ShapeWithRef shape_;
};

template <typename T>
class BroadcastScalarOp : public NDObject {
 public:
  BroadcastScalarOp(T scalar, ShapeRef *shape_ref, DType type_id, NDObject *dummy_load)
      : NDObject(dummy_load, nullptr, type_id, ObjectType::kBroadcastS), scalar_(scalar) {
    shape_ref_ = shape_ref;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  T scalar_;
};

class _ReduceOp : public FlexOp {
 public:
  enum { SUM, MAX, MIN };
  _ReduceOp(NDObject *input, int red_op)
      : FlexOp(input, nullptr, input->type_id_, ObjectType::kReduce), red_op_(red_op) {}
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  void SetRange(int start, int end) {
    start_dim_ = start;
    end_dim_ = end;
    tail_dim_ = -1;
  }
  bool InRange(int dim) const { return dim >= start_dim_ && dim <= end_dim_; }

  int red_op_;
  int start_dim_{0};
  int end_dim_{0};
  int tail_dim_{-1};
  int64_t tail_size_{0};
};

class ReduceOp : public _ReduceOp {
 public:
  ReduceOp(NDObject *input, int red_op, ShapeRef *dims_ref, bool keepdims)
      : _ReduceOp(input, red_op), keepdims_(keepdims) {
    dims_ref_ = dims_ref;
    shape_ref_ = &shape_;
    ws_num_ = System::Instance().deterministic_ ? 2 : 1;
    flags_ |= OBJ_FLAG_FLEX_INPL_WS;
  }
  ~ReduceOp();
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;

  void GenClearKernel(NDAccess *store);
  NDStore *clear_store_{nullptr};
  VectorKernel *clear_kernel_{nullptr};
  void Dump(bool verbose, std::ostringstream &oss) override;

  TileVisitCoder visit_;

 private:
  int EmitDeterm(VectorKernel &k);

  std::vector<_ReduceOp *> stuff_ops_;
  ShapeWithRef shape_;
  bool keepdims_;
  ShapeRef *dims_ref_;
  DimArray round_tile_;

  RelocAddr ws_reloc_;
  ShapeRef clear_shape_;
  int64_t clear_shape_data_;
};

class CubeTuner;
class CubeOp : public NDObject {
 public:
  struct Tactics {
    bool enable_splitk{false};
    bool enable_pad{false};
    bool enable_bias_cast{false};

    int64_t k_stride;
    int64_t lhs_pad_size{0};
    int64_t rhs_pad_size{0};
  };

  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b);
  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias);

  int Emit(VectorKernel &k) override { return 0; }
  void Dump(bool verbose, std::ostringstream &oss) override;
  void CodeGen(vCubeOp *code, CubeTuner *tuner);
  void NormalizeCube();
  void NormalizeOutput();
  void InferCubeConfig();
  void GenTiling(vCubeOp *code);

  uint64_t PostFusionWorkSpace() const {
    uint64_t pingpong_size = m0_ * n0_ * ITEM_SIZE[type_id_];
    return core_loop_ < block_dim_ * 2 ? pingpong_size * core_loop_ : pingpong_size * block_dim_ * 2;
  }
  void SetRealShape(int64_t m, int64_t n, int64_t k, size_t offset_a, size_t offset_b) {
    m_real_ = m;
    n_real_ = n;
    k_real_ = k;
    offset_a_ = offset_a;
    offset_b_ = offset_b;
    NormalizeOutput();
  }
  void SetOutFp32(bool atomic_add) {
    atomic_add_ = atomic_add;
    type_id_ = kFloat32;
  }

  NDAccess *output_{nullptr};
  uint64_t block_dim_{0};
  uint64_t core_loop_{0};
  int64_t m_align_{0};
  int64_t n_align_{0};
  int64_t k_align_{0};
  int64_t m_real_{0};
  int64_t n_real_{0};
  int64_t k_real_{0};
  int64_t m0_{0};
  int64_t n0_{0};
  int64_t k0_{0};
  bool trans_a_{false};
  bool trans_b_{false};
  bool pingpong_store_{false};
  bool atomic_add_{false};
  bool batch_fold_{false};
  NDObject *bias_{nullptr};
  Tactics tactics_;

 protected:
  void ComputeBroadcastShape(NDObject *lhs, NDObject *rhs);
  float CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0);
  void Tile(vCubeOp *code);
  void GetSwizzleConfig(vCubeOp *code);

  void TileV2(vCubeOp *op);

  size_t offset_a_{0};
  size_t offset_b_{0};
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_data_;
};

class CommOp : public NDObject {
 public:
  CommOp(NDObject *input, const Communicator *comm, ObjectType obj_id)
      : NDObject(input, nullptr, input->type_id_, obj_id), comm_(comm) {
    shape_ref_ = input->shape_ref_;
  }
  // Extra space needed to store expanded instructions
  uint64_t CodeReserve() { return code_reserve_; }
  uint64_t XbufReserve() { return xbuf_reserve_; }
  void SetXbufSize(uint32_t size) { xbuf_size_ = size; }
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 public:
  std::vector<uint64_t> xbufs_;
  std::vector<uint64_t> forward_events_;
  std::vector<uint64_t> backward_events_;
  bool mix_{false};
  const Communicator *comm_;

 protected:
  uint64_t xbuf_reserve_{0};  // static
  uint64_t code_reserve_{0};
  CubeOp *cube_op_{nullptr};
  uint32_t xbuf_size_{0};
  bool store_lhs_{true};
};

// Not Support (rank_size, 1)
class ReduceScatterOp : public CommOp {
 public:
  ReduceScatterOp(NDObject *input, const Communicator *comm);
  ~ReduceScatterOp() override;
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  int MultiLoadEmit(VectorKernel &k);

  bool multi_load_;

 private:
  int tail_dim_{-1};
  int tail_size_{0};
  ShapeWithRef shape_;
  vSimdInsnID add_id_;
  DimArray round_tile_;
  NDObject *reshape_op_{nullptr};
  ShapeWithRef reshape_shape_;
};

// Design: AllReduce is used before codegen, then codegen will generate PeerLoad and PeerStore
class AllReduceOp : public CommOp {
 public:
  AllReduceOp(NDObject *input, const Communicator *comm);
  // ~AllReduceOp() = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  int MatmulEmit(VectorKernel &k);  // used when lhs_ is Matmul

 protected:
  int tail_dim_{-1};
  int tail_size_{0};
  bool use_twoshot_{false};

 private:
  vSimdInsnID add_id_;
};

class AllGatherOp : public CommOp {
 public:
  AllGatherOp(NDObject *input, const Communicator *comm);
  ~AllGatherOp() = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  DimArray round_tile_;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};

 private:
  ShapeWithRef shape_;
};

class AllGatherV2Op : public CommOp {
 public:
  AllGatherV2Op(NDObject *input, const Communicator *comm);
  ~AllGatherV2Op() = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};

 private:
  ShapeWithRef shape_;
};

class NDSStore : public NDStore {
 public:
  NDSStore(NDObject *src) : NDStore(src) { obj_id_ = kSStore; }
  NDSStore(void *dst, NDObject *src) : NDStore(dst, src) { obj_id_ = kSStore; }
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 private:
  CubeOp *cube_op_;
};

class NDSLoad : public NDLoad {
 public:
  NDSLoad(void *src, ShapeRef *shape_ref, DType type_id = kFloat32, bool is_from_cube = false)
      : NDLoad(src, shape_ref, type_id), is_from_cube_(is_from_cube) {
    obj_id_ = kSLoad;
  }
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

  bool pingpong_load_{false};
  bool is_from_cube_{false};

 private:
  CubeOp *cube_op_;
};
}  // namespace dvm
#endif  // _DVM_OPS_H_
