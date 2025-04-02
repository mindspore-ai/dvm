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
class VectorKernel;
class PropDomainBuilder;
class PropDomain {
 public:
  PropDomain(NDObject *head = nullptr) : head_(head) {}
  virtual ~PropDomain() {
    for (auto dom : subdoms_) {
      delete dom;
    }
    subdoms_.clear();
  }
  void Normalize();
  virtual void AlignProp(PropRange &range);
  virtual void FoldProp(PropRange &range);
  virtual void TileProp(const TileParam &tp);

  NDObject *DomObject() const { return dom_; }

 protected:
  NDObject *head_;
  NDObject *dom_;
  std::vector<PropDomain *> subdoms_;
  friend PropDomainBuilder;
};

class RootDomain : public PropDomain {
 public:
  RootDomain() = default;
  void SetHead(NDObject *head) { head_ = head; }
  void PrepareTiling(VectorKernel *kernel);
  int64_t Tile(const TileParam tp, int64_t space) {
    PropDomain::TileProp(tp);
    tile_size_ = tile_size_ / space * tp.tile;
    tile_num_ *= tp.num;
    return tile_size_;
  }
  int64_t TileLead(const TileParam tp, int64_t lead_align) {
    PropDomain::TileProp(tp);
    tile_size_ = RoundUp<int64_t>(tp.tile, lead_align);
    align_.space = tile_size_;
    tile_num_ *= tp.num;
    return tile_size_;
  }
  void Align(int depth, int64_t space);
  void Shard(const ShardParam &sp);

  const DimArray &DimSpace() const { return dom_->nd_.dims(); }
  int64_t TileNum() const { return tile_num_; }
  int64_t TileSize() const { return tile_size_; }

  PropRange align_;
  const ShardParam *shard_{nullptr};

 private:
  int64_t tile_size_;  // shape size of object tile size
  int64_t tile_num_;   // current tile num. multiply by tile
};

class VKernel {
 public:
  VKernel(KernelType ktype) : ktype_(ktype) {}
  virtual ~VKernel() {}

  virtual void Append(NDObject *obj) = 0;
  virtual uint64_t CodeGen() = 0;
  virtual void Dump(std::ostringstream &oss, const std::string &indent) = 0;
  std::string &DumpGraph() {
    std::ostringstream oss;
    Dump(oss, "");
    dump_str_ = oss.str();
    return dump_str_;
  }
  virtual std::string &DisAssemble();
  KernelType KType() const { return ktype_; }

  Code code_;

 protected:
  KernelType ktype_;
  std::string dump_str_;
};

class CodeGenHelper;
class VectorKernel : public VKernel {
 public:
  VectorKernel(KernelType ktype) : VKernel(ktype) {}
  virtual ~VectorKernel();

  void Dump(std::ostringstream &oss, const std::string &indent) override;

  void SetTile(int start, int end, int64_t num, int64_t factor) { tiles_.emplace_back(DimTile{start, end, num, factor}); }
  int MaxType() const { return max_type_; }
  int MinType() const { return min_type_; }
  uint64_t LeadAlign() const { return lead_align_; }
  inline uint64_t ReserveCodeSize() const {
    auto res = objects_.size() * V_INSN_SIZE_MAX;
    if (comm_op_) {
      res += comm_op_->CodeReserve();
    }
    return (res + 511ul) & ~511ul;  // 512B align
  }

  void BuildDomain(const std::vector<NDObject *> &objects);
  void NormalizeDomain() {
    root_dom_.Normalize();
    int op_index = 0;
    for (auto op : objects_) {  // clear status
      op->Clear(op_index++);
    }
    block_align_ = SIMD_BLOCK_SIZE / ITEM_SIZE[min_type_];
  }
  void Normalize() {
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      objects_.emplace_back(op);
      if (op->IsComm()) {
        ASSERT(comm_op_ == nullptr);
        comm_op_ = static_cast<CommOp *>(op);
      }
    }
  }

  int Analyze();

  uint32_t CompactBlockDim(uint64_t core_limit) {
    auto tile_per_block = (tile_num_ + core_limit - 1) / core_limit;
    return (tile_num_ + tile_per_block - 1) / tile_per_block;
  }

  uint8_t *DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, uint64_t code_reserve);
  uint64_t DoCodeGen(uint64_t core_limit) {
    auto code_reserve = ReserveCodeSize();
    code_.Alloc(code_reserve + code_.HeadSize());
    root_dom_.PrepareTiling(this);
    auto code_end = DoCodeGen(core_limit, code_.data_ + code_.HeadSize(), code_reserve);
    code_.data_size_ = code_end - code_.data_;
    if (!visit_) {
      code_.block_dim_ = CompactBlockDim(core_limit);
      code_.UpdateV(tile_num_);
      return 0;
    }
    code_.block_dim_ = CeilDiv<uint32_t>(visit_->block_num_, 2);
    code_.UpdateVE(visit_);
    return visit_->ws_size_;
  }

  NDAccess *FindInplaceStore(NDAccess *load, const std::function<bool(NDAccess *)> &check) const;

  std::vector<NDObject *> objects_;
  std::vector<NDObject *> build_ops_;
  CommOp *comm_op_{nullptr};
  RootDomain root_dom_;

  uint64_t tile_num_{0};
  TileVisitCoder *visit_{nullptr};

  union {
    uint64_t block_align_;  // tiling
    uint64_t lead_align_;   // codegen
  };
  int forward_event_num_;
  int backward_event_num_;

 protected:
  int max_type_{-1};
  int min_type_{-1};

  std::vector<NDObject *> static_ops_;

  struct DimTile {
    int start;
    int end;
    int64_t num;
    int64_t factor;
  };
  std::vector<DimTile> tiles_;
  friend CodeGenHelper;
};

class VKernelS : public VectorKernel {
 public:
  VKernelS() : VectorKernel(KernelType::kStaticShape) {}
  void Append(NDObject *obj) override;
  void Optimize();
  uint64_t CodeGen() override;

  static std::vector<pass::Pass> passes;
};

class VKernelD : public VectorKernel {
 public:
  VKernelD() : VectorKernel(KernelType::kDynShape) {}
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;

 private:
  void RecordOpRelation();
  void RecoverOpRelation();

  std::unordered_map<NDObject *, std::vector<NDObject *>> op_relations_;
  std::vector<NDObject *> pd_nexts_;
  bool elim_reshape_{false};
};

class VKernelP : public VKernel {
 public:
  VKernelP() : VKernel(KernelType::kStaticParallel) { children_.push_back(new VKernelS()); }
  ~VKernelP() override;
  void AppendNext() {
    children_.push_back(new VKernelS());
    EXCEPTION_IF(children_.size() > 8, "total sub-kernels of parallel kernel exceed limit(8)");
  }
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;

  static uint64_t UpdateSummary(VectorKernel *k, uint64_t code_offset, uint64_t code_size, uint64_t* &summaries);

 protected:
  std::vector<VKernelS *> children_;
};

class DumpRefHelper {
 public:
  DumpRefHelper(std::ostringstream &oss) : oss_(oss) {}
  void Dump(NDObject *op);
  virtual NDObject *GetInput(NDObject *input);

 protected:
  std::ostringstream &oss_;
  int idx_{0};
  std::unordered_map<NDObject *, int> idx_map_;
};
}  // namespace dvm
#endif  // _DVM_KERNEL_H_
