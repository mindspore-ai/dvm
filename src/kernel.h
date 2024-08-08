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

#ifndef _DVM_KERNEL_H_
#define _DVM_KERNEL_H_

#include <string>
#include <vector>
#include <functional>
#include "code.h"
#include "ops.h"
#include "pass.h"

namespace dvm {
template<typename T>
static inline T CeilDiv(T a, T b)  { return (a - 1) / b + 1; }

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
  virtual void Normalize();
  virtual void AlignProp(PropRange &range);
  virtual void FoldProp(PropRange &range);
  virtual void TileProp(const TileParam &tp);

  NDObject* DomObject() const { return dom_; }

 protected:
  NDObject* head_;
  NDObject* dom_;
  std::vector<PropDomain*> subdoms_;
  friend PropDomainBuilder;
};

class RootDomain : public PropDomain {
 public:
  RootDomain() = default;
  void SetHead(NDObject *head) { head_ = head; }
  void Normalize(VectorKernel *kernel);
  int64_t Tile(int start, int end, int64_t space, int64_t num);
  void GroupTile(int dim, int64_t space, int64_t tile);
  void Align(int depth, int64_t space);

  std::vector<int64_t>& DimSpace() const { return dom_->nd_; }
  int64_t TileNum() const { return tile_num_; }
  int64_t TileSize() const { return tile_size_; }

  PropRange align_;

 private:
  int block_align_;    // mim block align
  int64_t tile_size_;  // shape size of object tile size
  int64_t tile_num_;   // current tile num. multiply by tile
};

class VKernel {
 public:
  VKernel(KernelType ktype) : ktype_(ktype) {}
  virtual ~VKernel() {}

  virtual void Append(NDObject *obj) = 0;
  virtual uint64_t CodeGen() = 0;
  virtual void DumpKernel(std::ostringstream &oss, const std::string &indent) = 0;

  std::string& DumpGraph() {
    std::ostringstream oss;
    DumpKernel(oss, "");
    dump_str_ = oss.str();
    return dump_str_;
  }
  std::string& DisAssemble();
  KernelType KType() const { return ktype_; }

  Code code_;

 protected:
  KernelType ktype_;
  std::string dump_str_;
};

struct Metrics {
  float mem_usage{0.0f};  // total_use_ub / ub_mem_size
  float core_usage{0.0f}; // load * tile_num / (per_core_load * core_num)
  float simd_usage{0.0f}; // tiled_shape_size / (repeat_num * max_simd_width)
};

class CodeGenHelper;
class VectorKernel : public VKernel {
 public:
  VectorKernel(KernelType ktype) : VKernel(ktype) {}
  virtual ~VectorKernel();

  void DumpKernel(std::ostringstream &oss, const std::string &indent) override;
  void CollectMetrics(Metrics &metrics) const;

  void Reserve(size_t size) {
    build_ops_.reserve(size);
    objects_.reserve(size * 2);
  }

  void SetTile(int start, int end, int64_t num) {
    tiles_.emplace_back(DimTile{start, end, num});
  }
  int MaxType() const { return max_type_; }
  int MinType() const { return min_type_; }
  uint64_t BlockAlign() const { return SIMD_BLOCK_SIZE / ITEM_SIZE[min_type_]; }
  uint64_t ReserveCodeSize() const { return (objects_.size() * V_INSN_SIZE_MAX + 511ul) & ~511ul; } // 512B align

  void BuildDomain(const std::vector<NDObject *> &objects);
  void NormalizeDomain() { root_dom_.Normalize(this); }
  void Normalize() {
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      objects_.emplace_back(op);
    }
  }

  int Analyze();
  void DoCodeGen(uint64_t core_limit);

  NDAccess* FindInplaceStore(NDAccess *load, const std::function<bool(NDAccess*)> &check) const;

  std::vector<NDObject *> objects_;
  std::vector<NDObject *> build_ops_;
  RootDomain root_dom_;

  uint64_t tile_num_{0};
  uint64_t simd_width_{0};

 protected:
  int max_type_{-1};
  int min_type_{-1};

  std::vector<NDObject*> static_ops_;

  struct DimTile {
    int start;
    int end;
    int64_t num;
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
  void Append(NDObject *obj) override {
    build_ops_.push_back(obj);
    if (obj->obj_id_ == ObjectType::kReshape) {
      elim_reshape_ = true;
    }
  }
  uint64_t CodeGen() override;

 private:
  void RecordOpRelation();
  void RecoverOpRelation();

  std::unordered_map<NDObject*, std::vector<NDObject*>> op_relations_;
  std::vector<NDObject*> pd_nexts_;
  bool elim_reshape_{false};
};

class VKernelP : public VKernel {
 public:
  VKernelP() : VKernel(KernelType::kStaticParallel) {
    children_.push_back(new VKernelS());
  }
  ~VKernelP() {
    for (auto k : children_) {
      delete k;
    }
  }
  void AppendNext() {
    children_.push_back(new VKernelS());
    EXCEPTION_IF(children_.size() > 8, "total sub-kernels of parallel kernel exceed limit(8)");
  }
  void Append(NDObject *obj) override { children_.back()->Append(obj); }
  void Reserve(size_t size) { children_.back()->Reserve(size); }

  uint64_t CodeGen() override;
  void DumpKernel(std::ostringstream &oss, const std::string &indent) override;

 protected:
  std::vector<VKernelS*> children_;
};

class MixKernel : public VKernel {
 public:
  MixKernel() : VKernel(KernelType::kStaticMix) {}
  ~MixKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void DumpKernel(std::ostringstream &oss, const std::string &indent) override;

 protected:
  void EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node);
  uint64_t SplitKCodeGen();
  VKernelS *post_fusion_{nullptr};
  CubeOp *cube_op_{nullptr};
  NDAccess *sload_{nullptr};

  Kernel *stage_kernel_{nullptr};
};

class StagesKernel : public VKernel {
 public:
  StagesKernel() : VKernel(KernelType::kStaticStages) {}
  ~StagesKernel() override;

  void StageSwitch(KernelType type) {
    VKernel *kernel = nullptr;
    if (type == KernelType::kStaticShape) {
      kernel = new VKernelS();
    } else if (type == KernelType::kStaticMix) {
      kernel = new MixKernel();
    } else if (type == KernelType::kStaticParallel) {
      kernel = new VKernelP();
    } else {
      ASSERT(0);
    }
    stages_.push_back(new Stage(kernel));
  }

  void ParallelSwitch() {
    auto current = stages_.back()->kernel;
    ASSERT(current->KType() != KernelType::kStaticParallel);
    static_cast<VKernelP*>(current)->AppendNext();
  }

  void StageStore(NDAccess *store) {
    store->is_stage_ = true;
    stages_.back()->kernel->Append(store);
    stages_.back()->ios.push_back(store);
  }

  void StageLoad(NDAccess *load, NDAccess *store) {
    load->is_stage_ = true;
    load->SetStageStore(store);
    stages_.back()->kernel->Append(load);
    stages_.back()->ios.push_back(load);
  }

  VKernel* Current() const { return stages_.back()->kernel; }

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void DumpKernel(std::ostringstream &oss, const std::string &indent) override;

 protected:
  uint64_t AllocWorkspace();

  struct Stage {
    Stage(VKernel *k) : kernel(k) {}
    VKernel* kernel;
    int64_t ws_size{-1};
    int64_t ws_offset{-1};
    std::vector<NDAccess*> ios;
  };
  std::vector<Stage*> stages_;
};
} // namespace dvm
#endif // _DVM_KERNEL_H_
