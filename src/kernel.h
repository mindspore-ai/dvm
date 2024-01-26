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
class VKernel;
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
  virtual int AlignProp(int depth);
  struct FoldRange {
    int base;
    int depth;
    int64_t space;
  };
  virtual void FoldProp(FoldRange &range);
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
  void Normalize(VKernel *kernel);
  int64_t Tile(int start, int end, int64_t space, int64_t num);

  std::vector<int64_t>& DimSpace() const { return dom_->nd_; }
  int AlignDepth() const { return align_depth_; }
  int64_t TileNum() const { return tile_num_; }
  int64_t TileSize() const { return tile_size_; }

 private:
  int align_depth_;    // max align axis depth
  int block_align_;    // mim block align
  int64_t align_size_; // align size
  int64_t tile_size_;  // shape size of object tile size
  int64_t tile_num_;   // current tile num. multiply by tile
};

class CodeGenHelper;
class VKernel {
 public:
  VKernel() = default;
  virtual ~VKernel();

  virtual void Append(NDObject *obj) = 0;
  virtual void CodeGen();

  void SetTile(int start, int end, int64_t num) {
    tiles_.emplace_back(DimTile{start, end, num});
  }

  std::string DumpGraph();
  std::string& DisAssemble();

  Code *GetCode() { return &code_; }

  uint64_t MaxTypeSize() const { return ITEM_SIZE[max_type_]; }
  uint64_t BlockAlign() const { return SIMD_BLOCK_SIZE / ITEM_SIZE[min_type_]; }
  uint64_t ReserveCodeSize() const { return (objects_.size() * V_INSN_SIZE_MAX + 511ul) & ~511ul; } // 512B align

 protected:
  int Analyze(CodeGenHelper &codegen);
  void BuildDomain();

  std::vector<NDObject *> objects_;
  std::vector<NDObject *> build_ops_;
  Code code_;

  RootDomain root_dom_;
  int max_type_{-1};
  int min_type_{-1};

  struct DimTile {
    int start;
    int end;
    int64_t num;
  };
  std::vector<DimTile> tiles_;
  std::string das_str_;
  friend CodeGenHelper;
};

class VKernelS : public VKernel {
 public:
  void Append(NDObject *obj) override {
    obj->Normalize(objects_);
    obj->index_ = objects_.size();
    objects_.emplace_back(obj);
    build_ops_.push_back(obj);
  }
};

class VKernelD : public VKernel {
 public:
  void Append(NDObject *obj) override {
    build_ops_.push_back(obj);
  }
  void CodeGen() override {
    objects_.clear();
    root_dom_.Clear();
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      op->index_ = objects_.size();
      objects_.emplace_back(op);
    }
    VKernel::CodeGen();
  }
};

} // namespace dvm
#endif // _DVM_KERNEL_H_
