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

namespace dvm {
class VKernelBase;
class PropDomainBuilder;
class PropDomain {
 public:
  PropDomain(NDObject *head = nullptr) : head_(head) {}
  virtual ~PropDomain() { Clear(); }
  void Clear() {
    for (auto dom : subdoms_) {
      delete dom;
    }
    subdoms_.clear();
  }
  virtual void Normalize();
  virtual void AlignProp(PropRange &range);
  virtual void FoldProp(PropRange &range);
  virtual void TileProp(const TileParam &tp);
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
  CodeBase *GetCode() const { return code_ptr_; }

  virtual std::string DumpGraph() = 0;
  std::string& DisAssemble();
  KernelType KType() const { return ktype_; }

 protected:
  CodeBase* code_ptr_{nullptr};
  std::string das_str_;
  KernelType ktype_;
};

class CodeGenHelper;
class VKernelBase : public VKernel {
 public:
  VKernelBase(KernelType ktype) : VKernel(&code_, ktype) {}
  virtual ~VKernelBase();

  std::string DumpGraph() override;
  void Reserve(size_t size) {
    build_ops_.reserve(size);
    objects_.reserve(size * 2);
  }

  void SetTile(int start, int end, int64_t num) {
    tiles_.emplace_back(DimTile{start, end, num});
  }
  uint64_t MaxTypeSize() const { return ITEM_SIZE[max_type_]; }
  uint64_t MinTypeSize() const { return ITEM_SIZE[max_type_]; }
  uint64_t BlockAlign() const { return SIMD_BLOCK_SIZE / ITEM_SIZE[min_type_]; }
  uint64_t ReserveCodeSize() const { return (objects_.size() * V_INSN_SIZE_MAX + 511ul) & ~511ul; } // 512B align

  void BuildDomain();
  void NormalizeDomain() { root_dom_.Normalize(this); }

  int Analyze(CodeGenHelper &codegen);
  void DoCodeGen(uint64_t core_limit);

  std::vector<NDObject *> objects_;
  std::vector<NDObject *> build_ops_;
  Code code_;
  RootDomain root_dom_;

 protected:
  int max_type_{-1};
  int min_type_{-1};

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
};

class VKernelD : public VKernelBase {
 public:
  VKernelD() : VKernelBase(KernelType::kDynShape) {}
  void Append(NDObject *obj) override { build_ops_.push_back(obj); }
  void CodeGen() override;
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
  void AppendNext() { children_.push_back(new VKernelS()); }
  void Append(NDObject *obj) override { children_.back()->Append(obj); }
  void Reserve(size_t size) { children_.back()->Reserve(size); }

  void CodeGen() override;
  std::string DumpGraph() override;

 protected:
  std::vector<VKernelS*> children_;
  CodeP code_;
};
} // namespace dvm
#endif // _DVM_KERNEL_H_
