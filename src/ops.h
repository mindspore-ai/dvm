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

#ifndef _DVM_OPS_H_
#define _DVM_OPS_H_

#include <cstdint>
#include <vector>
#include <mutex>
#include <atomic>
#include "isa.h"
#include "system.h"
#include "code.h"

namespace dvm {
enum ObjectType {
  // Load
  kLoadDummy = 0,
  kMultiLoad,
  kLoad,

  // Store
  kPadStore,
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
  kOneHot,
  kCubeOp,
  kGmmOp,
  kObjectBulk
};

// Tile axis range: [x, y, z] -> [tile/tail, 1, 1].  x*y*z=tile*num + (tail ? tail - tile : 0)
struct TileParam {
  int start;
  int end;
  int64_t num;
  int64_t tile;
  int64_t tail;
};

struct PropRange {
  enum {
    ELEMWISE = 0,
    BROADCAST,
    REDUCE,
  };
  int base;
  int depth;
  int affine;
  int simd_dim;
  int64_t space;
};

// shard map(low axis left): [a0, a1,.. s0, s1, s2, ...] -> [a0, a1,...tile[0], tile[1], 1, 1, ..]
class DimArray;
struct ShardParam {
  enum { PARTIAL_SIZE = 2 };
  int base;
  const DimArray *dom;
  uint64_t tile[PARTIAL_SIZE];
  uint64_t tail[PARTIAL_SIZE];
  uint64_t stride[PARTIAL_SIZE];
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
  bool operator==(const DimArray &other) const {
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
  template <typename T>
  int64_t &operator[](T i) {
    return data_[i];
  }
  template <typename T>
  const int64_t &operator[](T i) const {
    return data_[i];
  }
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

std::ostream &operator<<(std::ostream &oss, const DimArray &nd);

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

class NDSpaceData {
 public:
  NDSpaceData() = default;
  ~NDSpaceData() = default;

  void UpdateStride(const DimArray &dims, uint64_t simd_width) {
    strides.resize(dims.size());
    size_t i = 0;
    while (i < dims.size() - 1 && dims[i] == 1) {
      strides[i++] = 1;
    }
    lidx = i;
    strides[i] = RoundUp<int64_t>(dims[i], simd_width);
    for (++i; i < dims.size(); ++i) {
      strides[i] = dims[i] * strides[i - 1];
    }
  }

  template <typename T>
  int64_t operator[](T i) const {
    return dims[i];
  }
  int64_t back() const { return dims.back(); }
  template <typename T>
  int64_t stride(T i) const {
    return strides[i];
  }
  int64_t stride_back() const { return strides.back(); }
  bool empty() const { return dims.empty(); }
  size_t size() const { return dims.size(); }
  int lead_idx() const { return lidx; }
  int64_t lead_stride() const { return strides[lidx]; }
  int64_t lead_dim() const { return dims[lidx]; }

  DimArray dims;
  DimArray strides;
  int lidx;
};

class NDSpace {
 public:
  NDSpace() = default;
  ~NDSpace() = default;

  NDSpace &operator=(const NDSpace &other) {
    data = other.data;
    return *this;
  }
  template <typename T>
  int64_t operator[](T i) const {
    return data->dims[i];
  }
  int64_t back() const { return data->back(); }
  template <typename T>
  int64_t stride(T i) const {
    return data->strides[i];
  }
  int64_t stride_back() const { return data->stride_back(); }
  bool empty() const { return data->empty(); }
  size_t size() const { return data->size(); }
  int lead_idx() const { return data->lidx; }
  int64_t lead_stride() const { return data->lead_stride(); }
  int64_t lead_dim() const { return data->lead_dim(); }
  const DimArray &dims() const { return data->dims; }

  const NDSpaceData *data{nullptr};
};

std::ostream &operator<<(std::ostream &oss, const NDSpace &nd);

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

// dynamic flags
#define OBJ_FLAG_FREE_LHS 1
#define OBJ_FLAG_FREE_RHS 2
#define OBJ_FLAG_REUSE_LHS 4
#define OBJ_FLAG_REUSE_RHS 8
#define OBJ_FLAG_DEAD 16

#define OBJ_FLAG_FLEX_RREE_XHS (1u << 14)
#define OBJ_FLAG_FLEX_REUSE_WS (1u << 15)
#define OBJ_FLAG_LOAD_SHARD_BCAST0 (1u << 13)
#define OBJ_FLAG_LOAD_SHARD_BCAST1 (1u << 14)
#define OBJ_FLAG_LOAD_SHARD_ROUND (1u << 15)
#define OBJ_FLAG_STORE_SHARD_BCAST0 (1u << 13)
#define OBJ_FLAG_STORE_SHARD_BCAST1 (1u << 14)
#define OBJ_FLAG_STORE_SHARD_ROUND (1u << 15)

// static flags
#define OBJ_FLAG_WORKSPACE (1u << 16)
#define OBJ_FLAG_XHS (2u << 16)
#define OBJ_FLAG_EAGER (8u << 16)
#define OBJ_FLAG_STAGE_IO (16u << 16)

#define OBJ_FLAG_FLEX_INPL_WS (1u << 31)
#define OBJ_FLAG_LOAD_PINGPONG (1u << 30)
#define OBJ_FLAG_LOAD_FROM_CUBE (1u << 31)

enum CodeGenTmpl {
  kGenSimd0 = 0,
  kGenSimd1,
  kGenSimd2,
  kGenComm,
  kGenFlex,
  kGenSimd3,
  kGenLoad,
  kGenStore,
};

struct NDObjectAttr {
  CodeGenTmpl cg_tmpl;
  bool inplace_prop;
  bool share_ndd;
};

class NDObject {
 public:
  NDObject(NDObject *lhs, NDObject *rhs, DType type_id, ObjectType obj_id) : lhs_(lhs), rhs_(rhs), obj_id_(obj_id) {
    type_id_ = type_id;
    MESS(index_, 10);
    MESS(reuse_dep_, 200);
  }
  NDObject(const NDObject &) = delete;
  NDObject &operator=(const NDObject &) = delete;
  virtual ~NDObject() = default;

  // re-infer shape(nd_) from its inputs nd_
  virtual void Normalize(std::vector<NDObject *> &run_ops) {}
  virtual void Shard(const ShardParam &sp);
  // fold axis right alignment: [base-depth+1, base]
  virtual void FoldProp(PropRange &range) {}
  // fold axis left alignment:  [0, depth-1]
  virtual void AlignProp(PropRange &range) {}
  // tile nd range
  virtual void Tile(const TileParam &tp);
  virtual uint64_t Emit(VectorKernel &k) = 0;
  virtual void Dump(bool verbose, std::ostringstream &oss);

  void *operator new(size_t size) { return mem_pool_.Get(size); }
  void operator delete(void *ptr) { std::free(ptr); }

  int64_t Size();
  uint64_t GetBlocks(int64_t size) const { return (size * ITEM_SIZE[type_id_] + 31) >> 5; }
  ObjectType GetObjectType() const { return obj_id_; }
  // Now we have comm op, which will cross different pipe. This method should be deprecated
  int Pipe() const { return obj_id_ <= kLoad ? V_PIPE_LOAD : (obj_id_ <= kStore ? V_PIPE_STORE : V_PIPE_SIMD); }
  bool IsLoad() const { return obj_id_ <= kLoad; }
  bool IsStore() const { return obj_id_ <= kStore && obj_id_ > kLoad; }
  bool IsComm() const { return obj_id_ > kStore && obj_id_ <= kAllReduce; }
  bool IsCube() const { return obj_id_ == kCubeOp || obj_id_ == kGmmOp; }
  // Comm op is considered a simd op, remember use !IsComm() to exclude comm op
  bool IsSimd() const { return obj_id_ > kStore; }
  bool NeedTailCopy() const { return obj_id_ > kReduceScatter && obj_id_ <= kAllReduce; }
  void SetFlag(uint32_t mask) { flags_ |= mask; }
  bool CheckFlag(uint32_t mask) const { return flags_ & mask; }
  NDSpaceData *Ndd() const { return attrs_[obj_id_].share_ndd ? nullptr : const_cast<NDSpaceData *>(nd_.data); }

  void Clear(int index) {
    index_ = index;
    xbuf_ = 0;
    reuse_dep_ = 0;
    flags_ &= 0xffff0000u;
  }

  NDSpace nd_;
  NDObject *lhs_;
  NDObject *rhs_;
  uint64_t xbuf_;
  ShapeRef *shape_ref_;
  NDObject *pd_next_;  // PropDomain next
  ObjectType obj_id_;
  DType type_id_;
  int reserved_;
  int index_;
  int reuse_dep_;
  uint32_t flags_{0};
  uint64_t *insn_;       // when in optimization passes, used to point to the next NDObject
  uint64_t *tail_insn_;  // when in optimization passes, used to point to the prev NDObject

  static MemPool<512, 8192> mem_pool_;
  static const NDObjectAttr attrs_[];
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
    ndd_.dims.resize(1, 1);
    shape_.Resize(1);
    shape_[0] = 1;
    shape_ref_ = &shape_;
    nd_.data = &ndd_;
  }
  void Tile(const TileParam &tp) override {}
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeWithRef shape_;
  NDSpaceData ndd_;
};

class NDLoad : public NDAccess {
 public:
  NDLoad(void *src, ShapeRef *shape_ref, DType type_id = kFloat32)
      : NDAccess(src, nullptr, type_id, ObjectType::kLoad) {
    shape_ref_ = shape_ref;
    nd_.data = &ndd_;
    MESS(tail_dim_, 100);
    MESS(tail_size_, 10000);
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Shard(const ShardParam &sp) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  int tail_dim_;
  int tail_size_;
  DimArray round_tile_;
  NDSpaceData ndd_;
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
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
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
    MESS(tail_dim_, 100);
    MESS(tail_size_, 10000);
    MESS(elem_dim_mask_, 0);
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Shard(const ShardParam &sp) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  void UpdateDimMask() {
    elem_dim_mask_ = (0x1u << nd_.size()) - 1;
    for (size_t i = 0; i < nd_.size(); ++i) {
      if (nd_[i] == 1) {
        elem_dim_mask_ ^= 1u << i;
      }
    }
  }

  DimArray round_tile_;

 private:
  int tail_dim_;
  int tail_size_;
  uint32_t elem_dim_mask_;
};

class NDPadStore : public NDAccess {
 public:
  NDPadStore(NDObject *src, int64_t pad_size)
      : NDAccess(nullptr, src, src->type_id_, ObjectType::kPadStore), pad_size_(pad_size) {
    shape_ref_ = &shape_;
  }
  NDPadStore(void *dst, NDObject *src, int64_t pad_size) : NDPadStore(src, pad_size) { addr_.gm = dst; }

  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class ReshapeOp : public CopyOp {
 public:
  ReshapeOp(NDObject *input, ShapeRef *shape_ref) : CopyOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = &shape_;
    nd_.data = &ndd_;
    obj_id_ = ObjectType::kReshape;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeRef *dst_shape_ref_;
  ShapeWithRef shape_;
  NDSpaceData ndd_;
};

class UnaryOp : public NDObject {
 public:
  UnaryOp(int op_type, NDObject *input);
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class ElementAnyOp : public NDObject {
 public:
  ElementAnyOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kElementAny) {
    ASSERT(type_id_ == kFloat32);
    shape_ref_data_.data = &shape_;
    shape_ref_data_.size = 1;
    shape_ref_ = &shape_ref_data_;
    nd_.data = &ndd_;
    MESS(tail_dim_, 100);
    MESS(tail_size_, 10000);
  }
  uint64_t Emit(VectorKernel &k) override;
  void Tile(const TileParam &tp) override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int64_t shape_{1};
  ShapeRef shape_ref_data_;
  int tail_dim_;
  int tail_size_;
  NDSpaceData ndd_;
};

class CastOp : public NDObject {
 public:
  CastOp(NDObject *input, DType type_id) : NDObject(input, nullptr, type_id, ObjectType::kCast) {
    ASSERT(type_id != lhs_->type_id_);
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  _BinaryNormalizer norm_;
};

class CompareOp : public FlexOp {
 public:
  CompareOp(int op_type, NDObject *lhs, NDObject *rhs);
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  std::vector<NDObject *> stuff_ops_[3];
  ShapeWithRef shape_;
};

class _BroadcastOp : public NDObject {
 public:
  _BroadcastOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kBroadcastTo) {
    nd_.data = &ndd_;
  }
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  NDSpaceData ndd_;

 private:
  uint64_t EmitBroadcastX(uint64_t *p, int end_dim);
  uint64_t EmitBroadcastY(uint64_t *p, int start_dim, int end_dim);
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
    nd_.data = &ndd_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  T scalar_;
  NDSpaceData ndd_;
};

class _ReduceOp : public FlexOp {
 public:
  enum { SUM, MAX, MIN };
  _ReduceOp(NDObject *input, int red_op)
      : FlexOp(input, nullptr, input->type_id_, ObjectType::kReduce), red_op_(red_op) {
    nd_.data = &ndd_;
    MESS(start_dim_, 100);
    MESS(end_dim_, 80);
    MESS(tail_dim_, 100);
    MESS(tail_size_, 10000);
  }
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  void SetRange(int start, int end) {
    start_dim_ = start;
    end_dim_ = end;
    tail_dim_ = -1;
  }
  bool InRange(int dim) const { return dim >= start_dim_ && dim <= end_dim_; }

  int red_op_;
  int start_dim_;
  int end_dim_;
  int tail_dim_;
  int64_t tail_size_;
  NDSpaceData ndd_;
};

class AtomicCleanWrap;
class ReduceOp : public _ReduceOp {
 public:
  ReduceOp(NDObject *input, int red_op, ShapeRef *dims_ref, bool keepdims)
      : _ReduceOp(input, red_op), keepdims_(keepdims) {
    dims_ref_ = dims_ref;
    shape_ref_ = &shape_;
    if (System::Instance().deterministic_) {
      ws_num_ = 2;
      visit_ = new RedVisitCoder();
    } else {
      ws_num_ = 1;
      visit_ = nullptr;
    }
    flags_ |= OBJ_FLAG_FLEX_INPL_WS;
  }
  ~ReduceOp();
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;

  void Dump(bool verbose, std::ostringstream &oss) override;

  RedVisitCoder *visit_;
  AtomicCleanWrap *clean_wrap_{nullptr};

 private:
  uint64_t EmitDeterm(VectorKernel &k);

  std::vector<_ReduceOp *> stuff_ops_;
  ShapeWithRef shape_;
  bool keepdims_;
  ShapeRef *dims_ref_;
  DimArray round_tile_;

  RelocAddr ws_reloc_;
};

template <typename T>
class OneHotOp : public NDObject {
 public:
  OneHotOp(NDObject *indices, ShapeRef *depth, int axis, T on_value, T off_value, DType type_id)
    : NDObject(indices, nullptr, type_id, kOneHot), on_value_(on_value), off_value_(off_value), axis_(axis), depth_(depth) {
    nd_.data = &ndd_;
    shape_ref_ = &shape_;
    MESS(depth_dim_, 20);
    MESS(tile_dim_, 20);
  }

  void Normalize(std::vector<NDObject *> &run_ops) override;
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  void UpdateDepthDim(int dim) { depth_dim_ = dim; }

 private:
  T on_value_;
  T off_value_;
  int axis_;
  int depth_dim_;
  int tile_dim_;
  int64_t depth_tile_;
  ShapeRef *depth_;
  NDSpaceData ndd_;
  ShapeWithRef shape_;
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

  uint64_t Emit(VectorKernel &k) override { return 0; }
  void Dump(bool verbose, std::ostringstream &oss) override;
  void NormalizeCube();
  virtual void InferCubeConfig();
  virtual void CodeGen(vCubeOp *code, CubeTuner *tuner);
  virtual void NormalizeOutput();
  virtual void GenTiling(vCubeOp *code);

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
  int64_t ka_align_{0};
  int64_t kb_align_{0};
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
  uint32_t batch_c0_{0};
  uint32_t batch_c1_{0};
  NDSpaceData ndd_;

 protected:
  float CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0);
  void Tile(vCubeOp *code);
  void GetSwizzleConfig(vCubeOp *code);

  void TileV2(vCubeOp *op);

  size_t offset_a_{0};
  size_t offset_b_{0};
  ShapeWithRef shape_;
};

class GmmOp : public CubeOp {
 public:
  GmmOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias, NDObject *group_list,
        GroupType group_type);

  void InferCubeConfig();
  void NormalizeOutput() override;
  void CodeGen(vCubeOp *code, CubeTuner *tuner) override;
  void GenTiling(vCubeOp *code) override;

  NDObject *group_list_;
  GroupType group_type_;
};

class CommIdWrap : public CodeWrap {
 public:
  int LaunchWrap(void *workspace, void *stream) override;
  std::vector<uint32_t *> ids_;  // used to ensure softsync work, not affected by last kernel
 private:
  static std::atomic<uint32_t> unique_id_;  // each kernel has a unique id
};

class CommOp : public NDObject {
 public:
  CommOp(NDObject *input, const Communicator *comm, ObjectType obj_id)
      : NDObject(input, nullptr, input->type_id_, obj_id), comm_(comm) {
    shape_ref_ = input->shape_ref_;
    max_type_ = type_id_;
    nd_.data = &ndd_;
  }
  // Extra space needed to store expanded instructions
  uint64_t CodeReserve() { return code_reserve_; }
  int XbufReserve() { return xbuf_reserve_; }
  void SetXbufSize(uint32_t size) { xbuf_size_ = size; }
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 public:
  std::vector<uint64_t> xbufs_;
  std::vector<uint64_t> forward_events_;
  std::vector<uint64_t> backward_events_;
  bool mix_{false};
  DType max_type_;
  const Communicator *comm_;
  CommIdWrap id_wrap_;

 protected:
  int xbuf_reserve_{0};  // static
  uint64_t code_reserve_{0};
  CubeOp *cube_op_{nullptr};
  uint32_t xbuf_size_{0};
  bool store_lhs_{true};
  NDSpaceData ndd_;
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
  uint64_t Emit(VectorKernel &k) override;
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

class AllReduceOpBase : public CommOp {
 public:
  AllReduceOpBase(NDObject *input, const Communicator *comm);
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};
  bool use_twoshot_{false};
  vSimdInsnID add_id_;
};

// Design: AllReduce is used before codegen, then codegen will generate PeerLoad and PeerStore
template <bool is_bf16>
class AllReduceOp : public AllReduceOpBase {
 public:
  AllReduceOp(NDObject *input, const Communicator *comm) : AllReduceOpBase(input, comm) {};
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  int MatmulEmit(VectorKernel &k);  // used when lhs_ is Matmul
};

class AllGatherOp : public CommOp {
 public:
  AllGatherOp(NDObject *input, const Communicator *comm);
  ~AllGatherOp() = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
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
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};

 private:
  ShapeWithRef shape_;
};
}  // namespace dvm
#endif  // _DVM_OPS_H_
