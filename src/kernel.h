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
#include "code.h"
#include "ops.h"
#include "pass.h"

namespace dvm {
template<typename T>
static inline T CeilDiv(T a, T b)  { return (a - 1) / b + 1; }

class VKernelBase;
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
  void Normalize(VKernelBase *kernel);
  int64_t Tile(int start, int end, int64_t space, int64_t num);

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
  VKernel(CodeBase* code_ptr, KernelType ktype) : code_ptr_(code_ptr), ktype_(ktype) {}
  virtual ~VKernel() {}

  virtual void Append(NDObject *obj) = 0;
  virtual void CodeGen() = 0;
  virtual void DumpKernel(std::ostringstream &oss) = 0;

  CodeBase *GetCode() const { return code_ptr_; }
  std::string& DumpGraph() {
    std::ostringstream oss;
    DumpKernel(oss);
    dump_str_ = oss.str();
    return dump_str_;
  }
  std::string& DisAssemble();
  KernelType KType() const { return ktype_; }

 protected:
  CodeBase* code_ptr_{nullptr};
  KernelType ktype_;
  std::string dump_str_;
};

struct Metrics {
  float mem_usage{0.0f};  // total_use_ub / ub_mem_size
  float core_usage{0.0f}; // load * tile_num / (per_core_load * core_num)
  float simd_usage{0.0f}; // tiled_shape_size / (repeat_num * max_simd_width)
};

class CodeGenHelper;
class VKernelBase : public VKernel {
 public:
  VKernelBase(KernelType ktype) : VKernel(&code_, ktype) {}
  virtual ~VKernelBase();

  void DumpKernel(std::ostringstream &oss) override;
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

  int Analyze();
  void DoCodeGen(uint64_t core_limit);

  std::vector<NDObject *> objects_;
  std::vector<NDObject *> build_ops_;
  Code code_;
  RootDomain root_dom_;

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

class VKernelS : public VKernelBase {
 public:
  VKernelS() : VKernelBase(KernelType::kStaticShape) {}
  void Append(NDObject *obj) override;
  void Optimize();
  void CodeGen() override;

  static std::vector<pass::Pass> passes;
};

class VKernelD : public VKernelBase {
 public:
  VKernelD() : VKernelBase(KernelType::kDynShape) {}
  void Append(NDObject *obj) override {
    build_ops_.push_back(obj);
    if (obj->obj_id_ == ObjectType::kReshape) {
      elim_reshape_ = true;
    }
  }
  void CodeGen() override;

 private:
  std::vector<NDObject*> pd_nexts_;
  bool elim_reshape_{false};
};

class VKernelP : public VKernel {
 public:
  VKernelP() : VKernel(&code_, KernelType::kStaticParallel) {
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

  void CodeGen() override;
  void DumpKernel(std::ostringstream &oss) override;

 protected:
  std::vector<VKernelS*> children_;
  CodeP code_;
};

class CubeOp : public NDObject {
 public:
  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b);
  int Emit(Code &code) override { return 0; }
  float CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0);
  void Tile(vCubeOp *code);
  void CodeGen(vCubeOp *code);
  void GetSwizzleConfig(vCubeOp *code);

  NDObject *output_{nullptr};
  uint64_t block_dim_{0};
  uint64_t core_loop_{0};

 protected:
  bool trans_a_{false};
  bool trans_b_{false};
  int64_t m_{0};
  int64_t n_{0};
  int64_t k_{0};
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_data_;
};

class MixKernel : public VKernel {
 public:
  MixKernel() : VKernel(&code_, KernelType::kStaticMix) {}
  ~MixKernel() override;

  void Append(NDObject *obj) override;
  void CodeGen() override;
  void DumpKernel(std::ostringstream &oss) override;

 protected:
  MixCode code_;
  VKernel *pre_fusion_{nullptr};
  VKernel *post_fusion_{nullptr};
  CubeOp *cube_op_{nullptr};
};
} // namespace dvm
#endif // _DVM_KERNEL_H_
