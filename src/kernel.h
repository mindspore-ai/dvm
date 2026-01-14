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

#ifndef _DVM_KERNEL_H_
#define _DVM_KERNEL_H_

#include <string>
#include <vector>
#include <functional>
#include "code.h"
#include "ops.h"
#include "pass.h"

namespace dvm {

class VKernel {
 public:
  explicit VKernel(KernelType ktype, uint32_t flags) : ktype_(ktype), flags_(flags) {}
  virtual ~VKernel();

  virtual void Append(NDObject *obj);
  virtual uint64_t CodeGen();
  virtual void Dump(std::ostringstream &oss, const std::string &indent) = 0;
  virtual void Clone(VKernel *base, CloneHelper &helper);
  std::string &DumpGraph() {
    std::ostringstream oss;
    Dump(oss, "");
    dump_str_ = oss.str();
    return dump_str_;
  }
  virtual std::string &DisAssemble();
  KernelType KType() const { return ktype_; }
  uint32_t Flags() const { return flags_; }
  bool IsSplit() const { return ktype_ == KernelType::kSplit ||  ktype_ == KernelType::kEager; }
  bool IsDynamic() const { return flags_ & KernelFlag::kDynamic; }

  void UpdateIdle(const std::vector<NDObject *> &cleans);

  Code code_;

 protected:
  KernelType ktype_;
  uint32_t flags_;
  std::string dump_str_;
};

class VectorKernel : public VKernel {
 public:
  explicit VectorKernel(KernelType ktype, uint32_t flags) : VKernel(ktype, flags) {
    MESS(max_type_, 100);
    MESS(min_type_, 200);
    MESS(visit_, reinterpret_cast<VisitCoder *>(100));
  }
  ~VectorKernel() override = default;

  void Dump(std::ostringstream &oss, const std::string &indent) override;

  void BuildDomain();
  void PrepareTiling();

  uint8_t *DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, uint64_t code_reserve);
  uint64_t DoCodeGenInner(uint64_t core_limit) {
    auto code_reserve = ReserveCodeSize();
    code_.Alloc(code_reserve + code_.HeadSize());
    auto code_end = DoCodeGen(core_limit, code_.data_ + code_.HeadSize(), code_reserve);
    code_.data_size_ = code_end - code_.data_;
    if (auto visit = GetVisitor<RedVisitCoder>(); visit != nullptr) {
      code_.block_dim_ = CeilDiv<uint32_t>(visit->block_num_, 2);
      code_.UpdateVE(visit);
      return visit->ws_size_;
    }
    code_.block_dim_ = CompactBlockDim(core_limit);
    code_.UpdateV(tile_num_);
    return 0;
  }
  uint64_t DoCodeGen(uint64_t core_limit) {
    PrepareTiling();
    if (unlikely(!tile_size_)) {
      ProcessIdle();
      return 0;
    }
    return DoCodeGenInner(core_limit);
  }

  void Shard(const ShardParam &sp) {
    for (auto op : objects_) {
      op->Shard(sp);
    }
    shard_ = &sp;
  }
  void ClearShard() { shard_ = nullptr; }

  void SetTile(int start, int end, int64_t num, int64_t factor) {
    tiles_.emplace_back(DimTile{start, end, num, factor});
  }

  int MaxType() const { return max_type_; }
  int MinType() const { return min_type_; }
  uint64_t LeadAlign() const { return lead_align_; }
  const DimArray &DimSpace() const { return dom_->nd_.dims(); }

  uint64_t ReserveCodeSize() const {
    auto res = SIMD_BLOCK_SIZE + objects_.size() * V_INSN_SIZE_MAX;
    if (comm_op_) {
      res += comm_op_->CodeReserve();
    }
    return (res + 511ul) & ~511ul;  // 512B align
  }

  uint32_t CompactBlockDim(uint64_t core_limit) {
    auto tile_per_block = (tile_num_ + core_limit - 1) / core_limit;
    return (tile_num_ + tile_per_block - 1) / tile_per_block;
  }

  NDAccess *FindInplaceStore(NDAccess *load, const std::function<bool(NDAccess *)> &check) const;
  void CollectIdle(std::vector<NDObject *> &cleans);
  void ProcessIdle();

  template <typename T>
  T *GetVisitor() {
    // TODO(multi visitor):
    //   if (type_id_ != T::ID) return next_->GetCoder<T>();
    return static_cast<T *>(visit_);
  }

  void AddVisitor(VisitCoder *visit) {
    ASSERT(visit_ == nullptr);
    visit_ = visit;
  }

  std::vector<NDObject *> objects_;
  CommOp *comm_op_{nullptr};

  int64_t tile_num_{0};
  int64_t tile_size_;  // shape size of object tile size
  PropRange align_;
  NDObject *dom_;
  const ShardParam *shard_{nullptr};

  VisitCoder *visit_;

  union {
    uint64_t block_align_;  // tiling
    uint64_t lead_align_;   // codegen
  };
  int forward_event_num_;
  int backward_event_num_;

 protected:
  int64_t Analyze();
  void ShapeTiling(int64_t size_limit, int64_t core_limit);
  int64_t BodyTiling(int64_t tile_size, int64_t size_limit, int64_t core_limit, const PropRange &range, TileParam &tp);
  void LeadTiling(int64_t tile_size, int64_t size_limit, int64_t core_limit, TileParam &tp);
  void ManualTiling();
  void Optimize(std::vector<NDObject *> &build_ops, GraphTracker *tracker);

  void TileProp(const TileParam tp) {
    for (auto op : objects_) {
      op->Tile(tp);
    }
  }

  int64_t Tile(const TileParam tp, int64_t space) {
    TileProp(tp);
    tile_size_ = tile_size_ / space * tp.tile;
    tile_num_ *= tp.num;
    return tile_size_;
  }
  int64_t TileLead(const TileParam tp, int64_t lead_align) {
    TileProp(tp);
    tile_size_ = RoundUp<int64_t>(tp.tile, lead_align);
    align_.space = tile_size_;
    tile_num_ *= tp.num;
    return tile_size_;
  }

  int max_type_;
  int min_type_;

  std::vector<NDObject *> static_ops_;

  struct DimTile {
    int start;
    int end;
    int64_t num;
    int64_t factor;
  };
  std::vector<DimTile> tiles_;
  friend class CodeGenHelper;
};

class VKernelS : public VectorKernel {
 public:
  VKernelS(uint32_t flags = 0) : VectorKernel(KernelType::kVector, flags) {}
  ~VKernelS() override;
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void Clone(VKernel *base, CloneHelper &helper) override;

  virtual bool NormBuild();

  bool Normalize(bool broker_norm) {
    if (broker_norm && broker_num_ == -1) {
      BrokerInit();
    }
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      objects_.emplace_back(op);
    }
    return !broker_norm || BrokerAffine();
  }

  void StaticInit(const std::vector<NDObject *> &objects);
  void BrokerInit();
  bool BrokerAffine();
  uint64_t BrokerCodeGen(VKernel **hold_kernel);

  std::vector<NDObject *> build_ops_;

 protected:
  int broker_num_{-1};
  int last_broker_;
  VKernel *stage_kernel_{nullptr};
};

class VKernelD : public VKernelS {
 public:
  VKernelD(uint32_t flags = KernelFlag::kDynamic) : VKernelS(flags) {}
  bool NormBuild() override;

  void Recover() {
    code_.Clear();
    objects_.clear();
  }
};

class _SpecVector : public VKernelD {
 public:
  _SpecVector(uint32_t flags) : VKernelD(flags | KernelFlag::kSpeculate) {}
  ~_SpecVector() override;
  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void Next() { last_stage_++; }

 protected:
  bool use_fall_{false};
  int last_stage_{0};
  std::vector<int> stage_ids_;
  std::vector<NDObject *> post_reduces_;
  VKernel *fall_kernel_{nullptr};
};

template <bool dyn_shape>
class SpecVector : public _SpecVector {
 public:
  SpecVector() : _SpecVector(dyn_shape ? KernelFlag::kDynamic : 0) {}
  uint64_t CodeGen() override;
  uint64_t FallCodeGen();
};

class IsolateWrapVP;
class VKernelP : public VKernel {
 public:
  VKernelP() : VKernel(KernelType::kParallel, 0) { children_.push_back(new VKernelS()); }
  ~VKernelP() override;
  void AppendNext() {
    children_.push_back(new VKernelS());
    EXCEPTION_IF(children_.size() > 8, "total sub-kernels of parallel kernel exceed limit(8)");
  }
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void Clone(VKernel *base, CloneHelper &helper) override;

 protected:
  uint64_t CodeGenVE(VKernelS *kernel, RedVisitCoder *visit, uint8_t *code_begin, uint64_t code_size, uint64_t ws_size);

  std::vector<VKernelS *> children_;
  IsolateWrapVP *wrap_{nullptr};
};

class DumpRefHelper {
 public:
  explicit DumpRefHelper(std::ostringstream &oss) : oss_(oss) {}
  virtual ~DumpRefHelper() = default;
  void Dump(NDObject *op);
  virtual NDObject *GetInput(NDObject *input);

 protected:
  std::ostringstream &oss_;
  int idx_{0};
  std::unordered_map<NDObject *, int> idx_map_;
};

class KernelBuilder : public Kernel {
 public:
  KernelBuilder(VKernel *k) { kernel_ = k; }
  ~KernelBuilder() { kernel_ = nullptr; }
};
}  // namespace dvm
#endif  // _DVM_KERNEL_H_
