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
#include "isa.h"
#include "code.h"

namespace dvm {
enum ObjectType {
  // Load
  kLoadDummy = 0,
  kLoad,
  // Store
  kPadStore,
  kStore,
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

class VectorKernel;

#define OBJ_FLAG_FREE_LHS   1
#define OBJ_FLAG_FREE_RHS   2
#define OBJ_FLAG_REUSE_LHS  4
#define OBJ_FLAG_REUSE_RHS  8

class NDObject {
 public:
  NDObject(NDObject *lhs, NDObject *rhs, DType type_id, ObjectType obj_id) : lhs_(lhs), rhs_(rhs), obj_id_(obj_id) {
    type_id_ = type_id;
  }
  NDObject(const NDObject&) = delete;
  NDObject &operator=(const NDObject&) = delete;
  virtual ~NDObject() = default;

  // re-infer shape(nd_) from its inputs nd_
  virtual void Normalize(std::vector<NDObject*> &run_ops) {}
  // fold axis right alignment: [base-depth+1, base]
  virtual void FoldProp(PropRange &range) {}
  // fold axis left alignment:  [0, depth-1]
  virtual void AlignProp(PropRange &range) {}
  // tile nd range
  virtual void Tile(const TileParam &tp);
  virtual int Emit(VectorKernel &k) = 0;

  void UpdateStride(uint64_t simd_width);

  int64_t Size();
  int64_t LeadAlign() const { return strides_[lead_dim_]; }
  uint64_t GetBlocks(int64_t size) const { return (size * ITEM_SIZE[type_id_] + 31) >> 5; }
  ObjectType GetObjectType() const { return obj_id_; }
  int Pipe() const { return obj_id_ <= kLoad ?  V_PIPE_LOAD : (obj_id_ <= kStore ? V_PIPE_STORE : V_PIPE_SIMD); }
  bool IsLoad() const { return obj_id_ <= kLoad; }
  bool IsStore() const { return obj_id_ <= kStore && obj_id_ > kLoad; }
  bool IsSimd() const { return obj_id_ > kStore; }

  std::vector<int64_t> nd_;
  std::vector<int64_t> strides_;
  NDObject *lhs_;
  NDObject *rhs_;
  uint64_t xbuf_{0};
  ShapeRef *shape_ref_{nullptr};
  NDObject *pd_next_{nullptr};
  int lead_dim_{0};
  ObjectType obj_id_;
  DType type_id_;

  // op info
  int index_{0};
  uint32_t flags_{0};
  uint64_t *insn_{nullptr};       // when in optimization passes, used to point to the next NDObject
  uint64_t *tail_insn_{nullptr};  // when in optimization passes, used to point to the prev NDObject
};

class NDAccess : public NDObject {
 public:
  NDAccess(uint8_t *gm, NDObject *lhs, DType type_id, ObjectType obj_id) : NDObject(lhs, nullptr, type_id, obj_id), gm_(gm) {}
  void Reloc(void *dst) {
    *reloc_addr_ = reinterpret_cast<uint64_t>(dst);
  }

  // stage store
  void SetWorkspace(int64_t offset) { gm_ = reinterpret_cast<uint8_t*>(offset); }
  int64_t GetWorkspace() const { return reinterpret_cast<int64_t>(gm_); }
  void SetOutputReuse(NDAccess *store) { gm_ = reinterpret_cast<uint8_t*>(store); }
  NDAccess* GetOutputReuse() const { return reinterpret_cast<NDAccess*>(gm_); }
  // stage load
  void SetStageStore(NDAccess* store) { gm_ = reinterpret_cast<uint8_t*>(store); }
  NDAccess* GetStageStore() const { return reinterpret_cast<NDAccess*>(gm_); }

  uint8_t *gm_;
  uint64_t *reloc_addr_{nullptr};
  bool is_stage_{false};
};

class NDLoadDummy : public NDAccess {
 public:
  NDLoadDummy(DType type_id) : NDAccess (nullptr, nullptr, type_id, ObjectType::kLoadDummy) {
    nd_ = shape_;
    shape_ref_data_ = shape_;
    shape_ref_ = &shape_ref_data_;
  }
  void Tile(const TileParam &tp) override { }
  int Emit(VectorKernel &k) override;

 private:
  std::vector<int64_t> shape_{1};
  ShapeRef shape_ref_data_;
};

class NDLoad : public NDAccess {
 public:
  NDLoad(uint8_t *src, ShapeRef *shape_ref, DType type_id = kFloat32)
      : NDAccess(src, nullptr, type_id, ObjectType::kLoad) {
    shape_ref_ = shape_ref;
  }
  void Normalize(std::vector<NDObject*> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;

  int tail_dim_{-1};
  int tail_size_{0};
  std::vector<int64_t> round_tile_;
};

class NDSliceLoad : public NDLoad {
 public:
  NDSliceLoad(uint8_t *src, ShapeRef *src_ref, ShapeRef *start_ref, ShapeRef *size_ref, DType type_id = kFloat32)
      : NDLoad(src, size_ref, type_id), start_ref_(start_ref), src_ref_(src_ref), size_ref_(size_ref) {}
  void Normalize(std::vector<NDObject *> &run_ops) override {
    ASSERT(src_ref_->size <= 3);
    NDLoad::Normalize(run_ops);
  }

  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;

 protected:
  int64_t CalcOffset();
  ShapeRef *start_ref_;
  ShapeRef *src_ref_;
  ShapeRef *size_ref_;
};

class NDStridedSliceLoad : public NDSliceLoad {
 public:
  NDStridedSliceLoad(uint8_t *src, ShapeRef *src_ref, ShapeRef *start_ref, ShapeRef *end_ref, ShapeRef *step_ref,
                     DType type_id = kFloat32)
      : NDSliceLoad(src, src_ref, start_ref, nullptr, type_id), end_ref_(end_ref), step_ref_(step_ref) {
    shape_ref_ = new ShapeRef();
  }
  ~NDStridedSliceLoad() { delete shape_ref_; }

  void Normalize(std::vector<NDObject *> &run_ops) override;

 private:
  std::vector<int64_t> shape_;
  ShapeRef *end_ref_;
  ShapeRef *step_ref_;
};

class NDStore : public NDAccess {
 public:
  NDStore(NDObject *src) : NDAccess(nullptr, src, src->type_id_, ObjectType::kStore) {
    shape_ref_ = src->shape_ref_;
  }
  NDStore(uint8_t *dst, NDObject *src) : NDAccess(dst, src, src->type_id_, ObjectType::kStore) {
    shape_ref_ = src->shape_ref_;
  }
  void Normalize(std::vector<NDObject*> &run_ops) override {
    nd_ = lhs_->nd_;
    tail_dim_ = -1;
    tail_size_ = 0;
  }
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;

 private:
  int tail_dim_{-1};
  int tail_size_{0};
};

class NDPadStore : public NDAccess {
 public:
  NDPadStore(NDObject *src, ShapeRef *pad_shape) : NDAccess(nullptr, src, src->type_id_, ObjectType::kPadStore), pad_shape_(pad_shape) {
    shape_ref_ = &shape_ref_data_;
  }
  NDPadStore(uint8_t *dst, NDObject *src, ShapeRef *pad_shape) : NDPadStore(src, pad_shape) {
    gm_ = dst;
  }

  void Normalize(std::vector<NDObject*> &run_ops) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;

 private:
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_data_;
  ShapeRef *pad_shape_;
};

class CopyOp : public NDObject {
 public:
  CopyOp(NDObject *input)
      : NDObject(input, nullptr, input->type_id_, ObjectType::kCopy) {
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject*> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
};

class ReshapeOp : public CopyOp {
 public:
  ReshapeOp(NDObject *input, ShapeRef *shape_ref)
      : CopyOp(input) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = new ShapeRef();
    obj_id_ = ObjectType::kReshape;
  }
  ~ReshapeOp() { delete shape_ref_; }
  void Normalize(std::vector<NDObject*> &run_ops) override;
  int Emit(VectorKernel &k) override;

 private:
  ShapeRef *dst_shape_ref_;
  std::vector<int64_t> shape_;
};

class UnaryOp : public NDObject {
 public:
  UnaryOp(int op_type, NDObject *input);
  void Normalize(std::vector<NDObject*> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;

 protected:
  vSimdInsnID id_;
};

class RemovePadOp : public CopyOp {
public:
  RemovePadOp(NDObject *NDObject);
  int Emit(VectorKernel &k) override;
};

class ElementAnyOp: public NDObject {
 public:
  ElementAnyOp(NDObject *input): NDObject(input, nullptr, input->type_id_, ObjectType::kElementAny) {
    ASSERT(type_id_ == kFloat32);
    shape_ref_data_ = shape_;
    shape_ref_ = &shape_ref_data_;
  }
  int Emit(VectorKernel &k) override;
  void Tile(const TileParam &tp) override;
  void Normalize(std::vector<NDObject *> &run_ops) override {
    tail_dim_ = -1;
    tail_size_ = 0;
    nd_.resize(lhs_->nd_.size(), 1);
  }

 private:
  std::vector<int64_t> shape_{1};
  ShapeRef shape_ref_data_;
  int tail_dim_{-1};
  int tail_size_{0};
};

class CastOp : public NDObject {
 public:
  CastOp(NDObject *input, DType type_id)
      : NDObject(input, nullptr, type_id, ObjectType::kCast) {
    ASSERT(type_id != lhs_->type_id_);
    shape_ref_ = input->shape_ref_;
  }
  void Normalize(std::vector<NDObject*> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;
};

enum BinarySOpType {
  kAdds = 0,
  kMuls,
  kMaximums,
  kMinimums,
  kBinarySOpEnd,
};

template <typename T>
class BinaryScalarOp : public NDObject {
 public:
  BinaryScalarOp(int op_type, NDObject *input, T scalar);
  void Normalize(std::vector<NDObject*> &run_ops) override { nd_ = lhs_->nd_; }
  int Emit(VectorKernel &k) override;

 private:
  vSimdInsnID id_;
  T scalar_;
};

class BinaryOp : public NDObject {
 public:
  BinaryOp(int op_type, NDObject *lhs, NDObject *rhs);
  ~BinaryOp();
  void Normalize(std::vector<NDObject*> &run_ops) override;
  int Emit(VectorKernel &k) override;
  vSimdInsnID id_;

 protected:
  int cmp_op_;
  std::vector<NDObject*> lhs_stuff_ops_;
  std::vector<NDObject*> rhs_stuff_ops_;
  std::vector<int64_t> shape_;
};

class SelectOp : public NDObject {
 public:
  SelectOp(NDObject *cond, NDObject *lhs, NDObject *rhs)
      : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kSelect), cond_(cond) {
    shape_ref_ = &shape_ref_data_;
  }
  ~SelectOp();
  void Normalize(std::vector<NDObject*> &run_ops) override;
  int Emit(VectorKernel &k) override;
  NDObject *cond_{nullptr};
  bool free_cond{false};

 private:
  std::vector<NDObject *> stuff_ops_[3];
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_data_;
};

class _BroadcastOp : public NDObject {
 public:
  _BroadcastOp(NDObject *input, const std::vector<int64_t> &nd)
      : NDObject(input, nullptr, input->type_id_, ObjectType::kBroadcastTo) {
    nd_ = nd;
  }
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  int Emit(VectorKernel &k) override;

 private:
  int64_t EmitBroadcastX(uint64_t *p, int end_dim, int64_t simd_width);
  int64_t EmitBroadcastY(uint64_t *p, int start_dim, int end_dim,
                         int64_t simd_width);
};

// expect shape is align: equal rank
class BroadcastOp : public _BroadcastOp {
 public:
  BroadcastOp(NDObject *input, ShapeRef *shape_ref)
      : _BroadcastOp(input, std::vector<int64_t>{1}) {
    dst_shape_ref_ = shape_ref;
    shape_ref_ = new ShapeRef();
  }
  ~BroadcastOp();
  void Normalize(std::vector<NDObject*> &run_ops) override;

 private:
  std::vector<NDObject*> stuff_ops_;
  ShapeRef *dst_shape_ref_;
  std::vector<int64_t> shape_;
};

template <typename T>
class BroadcastScalarOp : public NDObject {
 public:
  BroadcastScalarOp(T scalar, ShapeRef *shape_ref, DType type_id, NDObject *dummy_load)
  : NDObject(dummy_load, nullptr, type_id, ObjectType::kBroadcastS), scalar_(scalar) {
    shape_ref_ = shape_ref;
  }
  void Normalize(std::vector<NDObject*> &run_ops) override {
    // update nd_ from shape_ref_
    auto dims = shape_ref_->size;
    nd_.resize(dims);
    for (size_t i = 0; i < dims; ++i) {
      nd_[i] = shape_ref_->data[dims - i - 1];
    }
  }
  int Emit(VectorKernel &k) override;
 private:
  T scalar_;
  std::vector<int64_t> shape_;
};

class _ReduceOp : public NDObject {
 public:
  enum { SUM, MAX, MIN };
  _ReduceOp(NDObject *input, int red_op)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kReduce), red_op_(red_op) {
  }
  void FoldProp(PropRange &range) override;
  void AlignProp(PropRange &range) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void SetRange(int start, int end) { start_dim_ = start; end_dim_ = end; tail_dim_ = -1; }
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
    shape_ref_ = new ShapeRef();
  }
  ~ReduceOp();
  void Normalize(std::vector<NDObject*> &run_ops) override;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;

  void GenClearKernel(NDAccess *store);
  NDStore *clear_store_{nullptr};
  VectorKernel *clear_kernel_{nullptr};
  std::vector<int64_t> round_tile_;

 private:
  std::vector<int64_t> dims_;
  std::vector<_ReduceOp*> stuff_ops_;
  std::vector<int64_t> shape_dims_;
  std::vector<int64_t> shape_;
  bool keepdims_;
  ShapeRef *dims_ref_;

  ShapeRef clear_shape_;
  int64_t clear_shape_data_;
};

class CubeOp : public NDObject {
 public:
  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b);
  ~CubeOp() override;
  int Emit(VectorKernel &k) override { return 0; }
  void CodeGen(vCubeOp *code);
  void NormalizeCube();

  uint64_t PostFusionWorkSpace() const {
    uint64_t pingpong_size = m0_ * n0_ * ITEM_SIZE[type_id_];
    return core_loop_ < block_dim_ * 2 ? pingpong_size * core_loop_ : pingpong_size * block_dim_ * 2;
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
  bool pingpong_store_{false};

 protected:
  void ComputeBroadcastShape(NDObject *lhs, NDObject *rhs);
  float CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0);
  void Tile(vCubeOp *code);
  void GetSwizzleConfig(vCubeOp *code);

  bool trans_a_{false};
  bool trans_b_{false};
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_data_;
};

class NDSStore : public NDStore {
 public:
  using NDStore::NDStore;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 private:
  CubeOp *cube_op_;
};

class NDSLoad : public NDLoad {
 public:
  using NDLoad::NDLoad;
  void Tile(const TileParam &tp) override;
  int Emit(VectorKernel &k) override;
  void AlignProp(PropRange &range) override;
  void FoldProp(PropRange &range) override;
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 private:
  CubeOp *cube_op_;
};
} // namespace dvm
#endif // _DVM_OPS_H_
