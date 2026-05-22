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
  kGlobalAccess,
  kGatherLoad,
  kViewLoad,
  kLoad,

  // Store
  kPadStore,
  kViewStore,
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
  int reserved;
  int64_t space;
};

struct TileInfo {
  int lead_depth;
  int lead_affine;
  uint32_t flags;
};

// shard map(low axis left): [a0, a1,.. s0, s1, s2, ...] -> [a0, a1,...tile[0], tile[1], 1, 1, ..]
class DimArray;
struct ShardParam {
  enum { PARTIAL_SIZE = 2 };
  int base;
  bool sink;
  const DimArray *dom;
  uint64_t tile[PARTIAL_SIZE];
  uint64_t tail[PARTIAL_SIZE];
  uint64_t stride[PARTIAL_SIZE];
};

std::ostream &operator<<(std::ostream &oss, const IntArrayRef &shape);
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
    if (this != &other) {
      size_ = other.size();
      if (size_ > 0) _DimCopy(data_, other.data(), size_);
    }
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
struct ShapeRefData : public IntArrayRef {
  ShapeRefData() {
    data = shape;
    size = 0;
  }
  ShapeRefData &operator=(const IntArrayRef &other) {
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

  void UpdateStride(uint64_t simd_width) {
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
    if (this != &other) {
      data = other.data;
    }
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

using scode_t = uint32_t;
void DumpScalarCode(std::ostringstream &oss, scode_t code, DataType type);

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

struct ObjectMeta {
  constexpr ObjectMeta() : flags(), tmpl(), dim_changed(), fold_prop(), tile_collect(), shape_prop() {}

  static constexpr uint32_t kNddShared = 1;
  static constexpr uint32_t kInplaceProp = 1u << 1;
  static constexpr uint32_t kLhsReuse = 1u << 2;
  static constexpr uint32_t kRhsReuse = 1u << 3;
  static constexpr uint32_t kLhsDom = 1u << 4;
  static constexpr uint32_t kDom = 1u << 5;
  static constexpr uint32_t kSimdDim = 1u << 6;
  static constexpr uint32_t kSimt = 1u << 7;

  uint32_t flags[kObjectBulk];
  CodeGenTmpl tmpl[kObjectBulk];
  void (*dim_changed[kObjectBulk])(NDObject *);
  void (*fold_prop[kObjectBulk])(NDObject *, PropRange &);
  void (*tile_collect[kObjectBulk])(NDObject *, TileInfo &);
  void (*shape_prop[kObjectBulk])(NDObject *, int64_t &);
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
#define OBJ_FLAG_LOAD_FROM_CC (1u << 11)
#define OBJ_FLAG_LOAD_FROM_CC_ONCE (1u << 12)
#define OBJ_FLAG_LOAD_SHARD_BCAST0 (1u << 13)
#define OBJ_FLAG_LOAD_SHARD_BCAST1 (1u << 14)
#define OBJ_FLAG_LOAD_SHARD_ROUND (1u << 15)
#define OBJ_FLAG_STORE_SHARD_BCAST0 (1u << 13)
#define OBJ_FLAG_STORE_SHARD_BCAST1 (1u << 14)
#define OBJ_FLAG_STORE_SHARD_ROUND (1u << 15)
#define OBJ_FLAG_BROKER_AFFINED (1u << 15)  // reshape, onehot

// static flags
#define OBJ_FLAG_WORKSPACE (1u << 16)
#define OBJ_FLAG_XHS (2u << 16)
#define OBJ_FLAG_EAGER (8u << 16)
#define OBJ_FLAG_STAGE_IO (16u << 16)

#define OBJ_FLAG_FLEX_INPL_WS (1u << 31)
#define OBJ_FLAG_LOAD_PINGPONG (1u << 30)
#define OBJ_FLAG_LOAD_FROM_CUBE (1u << 31)
#define OBJ_FLAG_REDUCE_NO_CUM (1u << 30)

class NDObject {
 public:
  NDObject(NDObject *lhs, NDObject *rhs, DataType type_id, ObjectType obj_id) : lhs_(lhs), rhs_(rhs), obj_id_(obj_id) {
    type_id_ = type_id;
    MESS(index_, 10);
    MESS(reuse_dep_, 200);
    prop_id_ = -1;  // debug
    // MESS(prop_id_, 1);
  }
  NDObject(const NDObject &) = delete;
  NDObject &operator=(const NDObject &) = delete;
  virtual ~NDObject() = default;

  // re-infer shape(nd_) from its inputs nd_
  virtual void Normalize(std::vector<NDObject *> &run_ops) {}
  virtual void Shard(const ShardParam &sp);
  // tile nd range
  virtual void Tile(const TileParam &tp);
  virtual uint64_t Emit(VectorKernel &k) = 0;
  virtual NDObject *Clone(CloneHelper &h);
  virtual void Dump(bool verbose, std::ostringstream &oss);

  void *operator new(size_t size) { return mem_pool_.Get(size); }
  void operator delete(void *ptr) { std::free(ptr); }

  int64_t Size();
  uint64_t GetBlocks(int64_t size) const { return (size * ITEM_SIZE[type_id_] + 31) >> 5; }
  ObjectType GetObjectType() const { return obj_id_; }
  bool IsLoad() const { return obj_id_ <= kLoad; }
  bool IsStore() const { return obj_id_ <= kStore && obj_id_ > kLoad; }
  bool IsComm() const { return obj_id_ > kStore && obj_id_ <= kAllReduce; }
  bool IsCube() const { return obj_id_ == kCubeOp || obj_id_ == kGmmOp; }
  // Comm op is considered a simd op, remember use !IsComm() to exclude comm op
  bool IsSimd() const { return obj_id_ > kStore; }
  bool NeedTailCopy() const { return obj_id_ > kReduceScatter && obj_id_ <= kAllReduce; }
  void SetFlag(uint32_t mask) { flags_ |= mask; }
  bool CheckFlag(uint32_t mask) const { return flags_ & mask; }
  NDSpaceData *Ndd() const { return SharedNdd() ? nullptr : const_cast<NDSpaceData *>(nd_.data); }

  uint32_t MetaFlags() const { return meta_.flags[obj_id_]; }
  bool SharedNdd() const { return meta_.flags[obj_id_] & ObjectMeta::kNddShared; }
  bool InplaceProp() const { return meta_.flags[obj_id_] & ObjectMeta::kInplaceProp; }
  CodeGenTmpl CgTmpl() const { return meta_.tmpl[obj_id_]; }

  void DimChanged() {
    if (auto func = meta_.dim_changed[obj_id_]) {
      func(this);
    }
  }
  // fold axis right alignment: [base-depth+1, base]
  void FoldProp(PropRange &range) {
    if (auto func = meta_.fold_prop[obj_id_]) {
      func(this, range);
    }
  }
  // fold axis left alignment:  [0, depth-1]
  void TileCollect(TileInfo &info) {
    if (auto func = meta_.tile_collect[obj_id_]) {
      func(this, info);
    }
  }
  // shape propagation from input shapes
  void ShapeProp(int64_t &sym_dim_next) {
    if (auto func = meta_.shape_prop[obj_id_]) {
      func(this, sym_dim_next);
    }
  }

  void Clear(int index) {
    index_ = index;
    xbuf_ = 0;
    reuse_dep_ = 0;
    flags_ &= 0xffff0000u;
  }

  template <typename T>
  void ForInput(const T &func);

  NDObject *CloneUpdate(CloneHelper &h) {
    auto op = Clone(h);
    h.SetClone(this, op);
    return op;
  }

  NDSpace nd_;
  NDObject *lhs_;
  NDObject *rhs_;
  uint64_t xbuf_;
  IntArrayRef *shape_ref_;
  ObjectType obj_id_;
  DataType type_id_;
  int prop_id_;
  int index_;
  union {
    int reuse_dep_; // simd(reuse)
    int last_ref_;  // load, simd(no_reuse)
    int first_def_; // store
  };
  uint32_t flags_{0};
  uint64_t *insn_;       // when in optimization passes, used to point to the next NDObject
  union {
    uint64_t *tail_insn_;  // when in optimization passes, used to point to the prev NDObject
    uint64_t io_reuse_mask_;
  };
  static MemPool<512, 8192> mem_pool_;
  static const ObjectMeta meta_;
};

class NDAccess : public NDObject {
 public:
  NDAccess(void *gm, NDObject *lhs, DataType type_id, ObjectType obj_id)
      : NDObject(lhs, nullptr, type_id, obj_id), addr_({gm}) {}
  RelocAddr addr_;
};

class NDGlobalAccess : public NDAccess {
 public:
  NDGlobalAccess(void *src, IntArrayRef *shape_ref, DataType type_id) : NDAccess(src, nullptr, type_id, ObjectType::kGlobalAccess) {
    shape_ref_ = shape_ref;
  }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class NDLoadDummy : public NDAccess {
 public:
  NDLoadDummy(DataType type_id) : NDAccess(nullptr, nullptr, type_id, ObjectType::kLoadDummy) {
    ndd_.dims.resize(1, 1);
    shape_.Resize(1);
    shape_[0] = 1;
    shape_ref_ = &shape_;
    nd_.data = &ndd_;
  }
  void Tile(const TileParam &tp) override {}
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  ShapeWithRef shape_;
  NDSpaceData ndd_;
};

class NDLoad : public NDAccess {
 public:
  NDLoad(void *src, IntArrayRef *shape_ref, DataType type_id = kFloat32)
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
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  int tail_dim_;
  int tail_size_;
  DimArray round_tile_;
  NDSpaceData ndd_;
};

class NDSimtLoad : public NDLoad {
 public:
  NDSimtLoad(void *src, IntArrayRef *shape_ref, DataType type_id, ObjectType obj_id)
      : NDLoad(src, shape_ref, type_id) {
    obj_id_ = obj_id;
  }
  static void TileCollect(NDObject *op, TileInfo &info) { info.flags |= ObjectMeta::kSimt; }
};

class NDGatherLoad : public NDSimtLoad {
 public:
  NDGatherLoad(void *src, IntArrayRef *src_shape_ref, NDAccess *index, int axis, DataType type_id,
               bool own_index = true)
      : NDSimtLoad(src, &shape_, type_id, ObjectType::kGatherLoad),
        src_shape_ref_(src_shape_ref),
        index_(index),
        axis_(axis),
        own_index_(own_index) {}
  ~NDGatherLoad() override;

  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  IntArrayRef *src_shape_ref_;
  NDAccess *index_;
  ShapeWithRef shape_;
  uint64_t inner_size_{1};
  uint64_t gather_size_{1};
  uint64_t gather_dim_size_{0};
  int axis_;
  bool own_index_;
};

class NDViewLoad : public NDAccess {
 public:
  NDViewLoad(void *src, IntArrayRef *shape, IntArrayRef *stride, DataType dtype)
      : NDAccess(src, nullptr, dtype, ObjectType::kViewLoad), src_stride_ref_(stride) {
    shape_ref_ = shape;
    nd_.data = &ndd_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  bool IsLeadContinuous() const { return src_stride_[0] == 1; }

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);
  static void DimChanged(NDObject *op);

 protected:
  IntArrayRef *src_stride_ref_;
  DimArray src_stride_;
  DimArray tile_;
  int tail_dim_;
  int tail_size_;
  NDSpaceData ndd_;
  uint64_t offset_bytes_{0};
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
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  void UpdateDimMask() {
    elem_dim_mask_ = (0x1u << nd_.size()) - 1;
    for (size_t i = 0; i < nd_.size(); ++i) {
      if (nd_[i] == 1) {
        elem_dim_mask_ ^= 1u << i;
      }
    }
  }

  static void DimChanged(NDObject *op);

 private:
  int tail_dim_;
  int tail_size_;
  uint32_t elem_dim_mask_;
  DimArray round_tile_;
};

class NDViewStore : public NDAccess {
 public:
  NDViewStore(void *dst, NDObject *src, IntArrayRef *stride)
      : NDAccess(dst, src, src->type_id_, ObjectType::kViewStore), dst_stride_ref_(stride) {
    shape_ref_ = src->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);
  static void DimChanged(NDObject *op);

 protected:
  IntArrayRef *dst_stride_ref_;
  DimArray dst_stride_;
  DimArray tile_;
  int tail_dim_;
  int tail_size_;
  uint64_t offset_bytes_{0};
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
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);

 private:
  ShapeWithRef shape_;
  int64_t pad_size_;
};

class NDConcatStoreM;
class NDConcatStore : public NDViewStore {
 public:
  NDConcatStore(void *dst, IntArrayRef *stride, NDObject *input, NDConcatStoreM *main)
      : NDViewStore(dst, input, stride), main_(main) {}
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  NDConcatStoreM *main_;
  friend NDConcatStoreM;
};

class NDConcatStoreM : public NDConcatStore {
 public:
  NDConcatStoreM(void *dst, NDObject *input, int concat_dim)
   : NDConcatStore(dst, &dst_stride_data_, input, this), concat_dim_(concat_dim) {
    shape_ref_ = &shape_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  NDObject *Clone(CloneHelper &h) override;

  NDConcatStore *AddSibling(NDObject *input) {
    ASSERT(input->type_id_ == type_id_);
    auto op = new NDConcatStore(addr_.gm, &dst_stride_data_, input, this);
    sibling_.push_back(op);
    return op;
  }
  void NormUpdate() {
    if (++group_sync_ == sibling_.size()) {
      group_sync_ = 0;
      _NormUpdate();
    }
  }
  void _NormUpdate();

  int concat_dim_;
  uint32_t group_sync_{0};
  std::vector<NDConcatStore *> sibling_;
  ShapeWithRef dst_stride_data_;
  ShapeWithRef shape_;
};

class FlexOp : public NDObject {
 public:
  enum { kWsMax = 2 };
  FlexOp(NDObject *lhs, NDObject *rhs, DataType type_id, ObjectType obj_id) : NDObject(lhs, rhs, type_id, obj_id) {
    flags_ |= OBJ_FLAG_WORKSPACE;
  }
  ~FlexOp() override = default;
  void SetXhs(NDObject *xhs) {
    xhs_ = xhs;
    flags_ |= OBJ_FLAG_XHS;
  }

  NDObject *xhs_{nullptr};
  int ws_num_{0};
  uint64_t wss_[kWsMax];
};

template <typename T>
void NDObject::ForInput(const T &func) {
  if (lhs_) {
    func(lhs_);
    if (rhs_) {
      func(rhs_);
      if (flags_ & OBJ_FLAG_XHS) {
        func(static_cast<FlexOp *>(this)->xhs_);
      }
    }
  }
}

class CopyOp : public NDObject {
 public:
  CopyOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kCopy) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
};

class ReshapeOp : public CopyOp {
 public:
  struct ChangeRange {
    int begin{0};
    int size{0};
    int in_size{0};
  };

  ReshapeOp(NDObject *input, IntArrayRef *shape_ref) : CopyOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = &shape_;
    nd_.data = &ndd_;
    obj_id_ = ObjectType::kReshape;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);
  bool VisitChangeRange(ChangeRange &range);

 protected:
  IntArrayRef *dst_shape_ref_;
  ShapeWithRef shape_;
  NDSpaceData ndd_;
};

class UnaryOp : public NDObject {
 public:
  UnaryOp(int op_type, NDObject *input)
      : NDObject(input, nullptr, input->type_id_, ObjectType::kUnary), op_type_(op_type) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

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
  NDObject *Clone(CloneHelper &h) override;
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
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  int64_t shape_{1};
  IntArrayRef shape_ref_data_;
  int tail_dim_;
  int tail_size_;
  NDSpaceData ndd_;
};

class CastOp : public NDObject {
 public:
  CastOp(NDObject *input, DataType type_id) : NDObject(input, nullptr, type_id, ObjectType::kCast) {
    ASSERT(type_id != lhs_->type_id_);
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
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
  ksDiv,
  kMaximums,
  kMinimums,
  kBinarySOpEnd,
};

class BinaryScalarOp : public NDObject {
 public:
  BinaryScalarOp(int op_type, NDObject *input, scode_t scalar)
      : NDObject(input, nullptr, input->type_id_, ObjectType::kBinaryS), op_type_(op_type), scalar_(scalar) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int op_type_;
  scode_t scalar_;
};

class CompareScalarOp : public FlexOp {
 public:
  CompareScalarOp(int op_type, NDObject *input, scode_t scalar)
      : FlexOp(input, nullptr, input->type_id_, ObjectType::kCompareS), scalar_(scalar) {
    if (g_system.Arch() == kAiCore_C220) {
      ws_num_ = 1;
      flags_ |= OBJ_FLAG_FLEX_INPL_WS;
    } else {
      wss_[0] = 0;
    }
    cmp_op_ = op_type;
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { nd_ = lhs_->nd_; }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int cmp_op_;
  scode_t scalar_;
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
  BinaryOp(int op_type, NDObject *lhs, NDObject *rhs)
      : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kBinary), op_type_(op_type) {
    shape_ref_ = &norm_.shape_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

 protected:
  int op_type_;
  _BinaryNormalizer norm_;
};

class PowerOp : public FlexOp {
 public:
  PowerOp(NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kPower) {
    if (g_system.Arch() == kAiCore_C220) {
      ws_num_ = 2;
    } else {
      wss_[0] = 0;
      wss_[1] = 0;
    }
    shape_ref_ = &norm_.shape_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

 protected:
  _BinaryNormalizer norm_;
};

class CompareOp : public FlexOp {
 public:
  CompareOp(int op_type, NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kCompare) {
    if (g_system.Arch() == kAiCore_C220) {
      ws_num_ = 1;
      flags_ |= OBJ_FLAG_FLEX_INPL_WS;
    } else {
      wss_[0] = 0;
    }
    cmp_op_ = op_type;
    shape_ref_ = &norm_.shape_;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override { norm_.Normalize(this, run_ops); }
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

 protected:
  int cmp_op_;
  _BinaryNormalizer norm_;
};

class SelectOp : public FlexOp {
 public:
  SelectOp(NDObject *cond, NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kSelect) {
    if (g_system.Arch() == kAiCore_C220) {
      ws_num_ = 1;
    } else {
      wss_[0] = 0;
    }
    shape_ref_ = &shape_;
    SetXhs(cond);
  }
  ~SelectOp() override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 private:
  std::vector<NDObject *> stuff_ops_[3];
  ShapeWithRef shape_;
};

class _BroadcastOp : public NDObject {
 public:
  explicit _BroadcastOp(NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kBroadcastTo) {
    nd_.data = &ndd_;
  }
  ~_BroadcastOp() override = default;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);

  NDSpaceData ndd_;

 private:
  uint64_t EmitBroadcastX(uint64_t *p, int end_dim);
  uint64_t EmitBroadcastY(uint64_t *p, int start_dim, int end_dim);
};

// expect shape is align: equal rank
class BroadcastOp : public _BroadcastOp {
 public:
  BroadcastOp(NDObject *input, IntArrayRef *shape_ref) : _BroadcastOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = &shape_;
  }
  ~BroadcastOp() override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  NDObject *Clone(CloneHelper &h) override;

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

 private:
  std::vector<NDObject *> stuff_ops_;
  IntArrayRef *dst_shape_ref_;
  ShapeWithRef shape_;
};

class BroadcastScalarOp : public NDObject {
 public:
  BroadcastScalarOp(scode_t scalar, IntArrayRef *shape_ref, DataType type_id)
      : NDObject(nullptr, nullptr, type_id, ObjectType::kBroadcastS), scalar_(scalar) {
    shape_ref_ = shape_ref;
    nd_.data = &ndd_;
  }
  ~BroadcastScalarOp() override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  scode_t scalar_;
  NDSpaceData ndd_;
  NDLoadDummy *dummy_load_{nullptr};
};

class _ReduceOp : public FlexOp {
 public:
  _ReduceOp(NDObject *input, int red_op)
      : FlexOp(input, nullptr, input->type_id_, ObjectType::kReduce), red_op_(red_op) {
    nd_.data = &ndd_;
    MESS(start_dim_, 100);
    MESS(end_dim_, 80);
    MESS(tail_dim_, 100);
    MESS(tail_size_, 10000);
  }
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);

  void SetRange(int start, int end) {
    start_dim_ = start;
    end_dim_ = end;
    tail_dim_ = -1;
  }
  bool InRange(int dim) const { return dim >= start_dim_ && dim <= end_dim_; }
  int EndDim() const { return end_dim_; }

  static void DimChanged(NDObject *op);

  NDSpaceData ndd_;
  int red_op_;

 protected:
  int start_dim_;
  int end_dim_;
  int tail_dim_;
  int64_t tail_size_;
};

class AtomicCleanWrap;
class ReduceOp : public _ReduceOp {
 public:
  ReduceOp(NDObject *input, int red_op, IntArrayRef *dims_ref, bool keepdims)
      : _ReduceOp(input, red_op), keepdims_(keepdims) {
    dims_ref_ = dims_ref;
    shape_ref_ = &shape_;
    if (g_system.deterministic_ && red_op_ == ReduceType::kSum) {
      ws_num_ = 2;
      visit_ = new RedVisitCoder();
    } else {
      ws_num_ = 1;
      visit_ = nullptr;
    }
    flags_ |= OBJ_FLAG_FLEX_INPL_WS;
  }
  ~ReduceOp() override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  bool KeepDims() const { return keepdims_; }

  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

  RedVisitCoder *visit_;
  AtomicCleanWrap *clean_wrap_{nullptr};

 private:
  uint64_t EmitDeterm(VectorKernel &k);

  std::vector<_ReduceOp *> stuff_ops_;
  ShapeWithRef shape_;
  bool keepdims_;
  IntArrayRef *dims_ref_;
  DimArray round_tile_;

  RelocAddr ws_reloc_;
};

class OneHotOp : public NDObject {
 public:
  OneHotOp(NDObject *indices, IntArrayRef *depth, int axis, scode_t on_value, scode_t off_value, DataType type_id)
      : NDObject(indices, nullptr, type_id, kOneHot),
        on_value_(on_value),
        off_value_(off_value),
        axis_(axis),
        depth_(depth) {
    nd_.data = &ndd_;
    shape_ref_ = &shape_;
    MESS(depth_dim_, 20);
    MESS(tile_dim_, 20);
  }

  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  int DepthDim() const { return depth_dim_; }

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);
  static void DimChanged(NDObject *op);
  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

 private:
  scode_t on_value_;
  scode_t off_value_;
  int axis_;
  int depth_dim_;
  int tile_dim_;
  int64_t depth_tile_;
  IntArrayRef *depth_;
  NDSpaceData ndd_;
  ShapeWithRef shape_;
};

class GraphTracker {
 public:
  GraphTracker() = default;
  ~GraphTracker() = default;
  void Record(NDObject **op_addr) { records_.emplace_back(op_addr, *op_addr); }
  void Recover() {
    for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
      *(it->addr) = it->origin;
    }
  }
  void RecoverClear() {
    if (!records_.empty()) {
      Recover();
      records_.clear();
    }
  }
  bool Empty() const { return records_.empty(); }

 protected:
  struct _Record {
    _Record(NDObject **a, NDObject *o) : addr(a), origin(o) {}
    NDObject **addr;
    NDObject *origin;
  };
  std::vector<_Record> records_;
};

template <int AFFINE>
void BroadReduceFoldProp(const DimArray &small_dim, const DimArray &big_dim, PropRange &range);
template <int AFFINE>
void BroadReduceTileCollect(const DimArray &small_dim, const DimArray &big_dim, TileInfo &info);

bool CollectRoundTile(const DimArray &nd, const TileParam &tp, DimArray &round_tile);
void BuildDimRounds(const DimArray &round_tile, uint64_t rounds[]);

vSimdInsnID GetBinaryInsnID(BinaryType op, DataType dtype);
vSimdInsnID GetCastInsnID(DataType from, DataType to);
}  // namespace dvm
#endif  // _DVM_OPS_H_
