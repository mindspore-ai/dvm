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

#include <queue>
#include <unordered_map>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include "kernel.h"
#include "pass.h"

namespace dvm {
class CodeGenHelper {
 public:
  CodeGenHelper() {}
  bool Generate(VKernelBase *kernel) {
    auto &code = kernel->code_;
    auto code_reserved = kernel->ReserveCodeSize();
    code.Alloc(code_reserved + code.HeadSize());
    uint64_t *code_ptr = reinterpret_cast<uint64_t*>(code.data_ + code.HeadSize());
    static_xbuf_ = DeviceInfo::Instance().UbWorkspaceSize() + code_reserved;
    for (auto op : static_ops_) {
      op->xbuf_ = static_xbuf_;
      static_xbuf_ += xbuf_size_;
    }
    for (auto op: kernel->objects_) {
      auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
      if (op->flags_ & OBJ_FLAG_FREE_LHS) {
        free_xbuf_.emplace(op->lhs_, op);
      }
      if (op->flags_ & OBJ_FLAG_FREE_RHS) {
        free_xbuf_.emplace(op->rhs_, op);
      }
      op->UpdateStride(code.simd_width_);
      op->tail_insn_ = op->insn_ = code_ptr;
      code_ptr += op->Emit(code);
      if (anti_dep) {
        InjectBar(anti_dep, op);
      }
      if (op->GetObjectType() == kSelect) {
        SelectOpPostProc(op);
      } else if (op->lhs_) {
        if (op->rhs_ == nullptr)  {
          InjectSync(op->lhs_, op);
        } else {
          if (op->rhs_->index_ > op->lhs_->index_) {
            InjectSync(op->rhs_, op);
            InjectSync(op->lhs_, op);
          } else {
            InjectSync(op->lhs_, op);
            InjectSync(op->rhs_, op);
          }
        }
      }
    }
    *(back_set_->tail_insn_) |= 0x1ul << V_HEAD_BACK_SET_OFFSET;
    *(back_wait_->insn_) |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET;
    code.data_size_ = reinterpret_cast<uint8_t*>(code_ptr) - code.data_;
    if (DeviceInfo::Instance().Arch() == kAiCore_C100) {
      OverWriteCoreLimit(code);
    }
    code.FillHead();
    return true;
  }

 private:
  void OverWriteCoreLimit(Code &code) {
    for (auto op : static_ops_) {
      if (op->obj_id_ <= kLoad || op->obj_id_ == kElementAny || (op->obj_id_ == kReduce && static_cast<ReduceOp*>(op)->factor_ > 1)) {
        continue;
      }
      // producer node for Store
      uint64_t size = op->strides_.back() / op->LeadAlign() * op->nd_[op->lead_dim_] * ITEM_SIZE[op->type_id_];
      if (size < SIMD_BLOCK_SIZE) {
        code.ApplyTileLimit((SIMD_BLOCK_SIZE + size - 1) / size);
      }
    }
  }

  void SelectOpPostProc(NDObject *op) {
    SelectOp *select_op = reinterpret_cast<SelectOp*>(op);
    if (select_op->free_cond) {
      free_xbuf_.emplace(select_op->cond_, select_op);
    }
    NDObject *inputs[3] = {op->lhs_, op->rhs_, select_op->cond_};
    if(inputs[0]->index_ < inputs[1]->index_) {
      std::swap(inputs[0],inputs[1]);
    }
    if(inputs[0]->index_ < inputs[2]->index_) {
      std::swap(inputs[0],inputs[2]);
    }
    if(inputs[1]->index_ < inputs[2]->index_) {
      std::swap(inputs[1],inputs[2]);
    }
    for (size_t i = 0; i < 3; i++) {
      InjectSync(inputs[i], op);
    }
  }

  NDObject* AllocDynXBuf(NDObject *obj) {
    if (obj->flags_ & OBJ_FLAG_REUSE_LHS) {
      obj->xbuf_ = obj->lhs_->xbuf_;
      return nullptr;
    }
    if (obj->flags_ & OBJ_FLAG_REUSE_RHS) {
      obj->xbuf_ = obj->rhs_->xbuf_;
      return nullptr;
    }
    if (static_xbuf_ + xbuf_size_ <= DeviceInfo::Instance().LocalMemSize()) {
      obj->xbuf_ = static_xbuf_;
      static_xbuf_ += xbuf_size_;
      return nullptr;
    }
    ASSERT(!free_xbuf_.empty());
    auto &op = free_xbuf_.front();
    obj->xbuf_ = op.first->xbuf_;
    auto anti_dep = op.second;
    free_xbuf_.pop();
    return  anti_dep;
  }

  void InjectBar(NDObject *from, NDObject *to) {
    if (from->pipe_idx >= vector_vector_sync) {
      *(to->insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      vector_vector_sync = to->pipe_idx;
    }
  }

  void InjectSync(NDObject *from, NDObject *to) {
    int from_pipe_idx = from->pipe_idx;
    auto from_insn = from->tail_insn_;
    auto to_insn = to->insn_;
    uint64_t event;
    if (from->Pipe() == V_PIPE_LOAD) { // load -vector
      if (from_pipe_idx <= load_vector_sync) {
        return; 
      } 
      event = load_vector_event;
      load_vector_event = (load_vector_event + 1) % DeviceInfo::Instance().EventNum();
      load_vector_sync = from_pipe_idx;
    } else if (to->Pipe() == V_PIPE_SIMD) {  // vector - vector
      InjectBar(from, to);
      return;
    } else { // vector - store
      if (from_pipe_idx <= vector_store_sync) return;
      event = vector_store_event;
      vector_store_event = (vector_store_event + 1) % DeviceInfo::Instance().EventNum();
      vector_store_sync = from_pipe_idx;
    }
    *from_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET;
    *from_insn |= event << V_HEAD_SET_EVENT_OFFSET;
    *to_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET;
    *to_insn |= event << V_HEAD_WAIT_EVENT_OFFSET;
  }

  std::vector<NDObject*> static_ops_;
  NDObject* back_set_{nullptr};
  NDObject* back_wait_{nullptr};

  uint32_t xbuf_size_{0};
  uint64_t static_xbuf_{0};
  std::queue<std::pair<NDObject*, NDObject*>> free_xbuf_;

  int load_vector_sync = -1;
  int vector_vector_sync = 0;
  int vector_store_sync = -1;
  uint64_t load_vector_event = 0;
  uint64_t vector_store_event = 0;

  friend VKernelBase;
};

void PropDomain::Normalize() {
  auto nd_size = head_->nd_.size();
  if (nd_size == 0) {
    nd_size = 1;
  }
  dom_ = nullptr;
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    auto size = op->nd_.size();
    if (nd_size > size) {
      op->nd_.resize(nd_size, 1);
    } else if (nd_size > size) {
      nd_size = size;
      for (auto op2 = head_; op2 != op; op2 = op2->pd_next_) {
        op->nd_.resize(nd_size, 1);
      }
    }
    auto obj_type = op->GetObjectType();
    if (obj_type != kLoadDummy && obj_type != kStore) {
      if (obj_type == kReduce || obj_type == kElementAny) {
        dom_ = op->lhs_;
      } else if (dom_ == nullptr) {
        dom_ = op;
      }
    }
  }
  if (!subdoms_.empty()) {
    for (auto sd: subdoms_) {
      sd->Normalize();
    }
  }
}

void PropDomain::AlignProp(PropRange &range) {
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    op->AlignProp(range);
  }
  if (!subdoms_.empty()) {
    for (auto sd: subdoms_) {
      sd->AlignProp(range);
    }
  }
}

void PropDomain::FoldProp(PropRange &range) {
  int old_depth = range.depth;
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    op->FoldProp(range);
  }
  if (range.depth < old_depth || range.space == -1) {
    int64_t space = dom_->nd_[range.base];
    for (int i = 1; i < range.depth; ++i) {
      space *= dom_->nd_[range.base - i];
    }
    range.space = space;
  }
  if (!subdoms_.empty()) {
    for (auto sd: subdoms_) {
      sd->FoldProp(range);
    }
  }
}

void PropDomain::TileProp(const TileParam &tp) {
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    op->Tile(tp);
  }
  if (!subdoms_.empty()) {
    for (auto sd: subdoms_) {
      sd->TileProp(tp);
    }
  }
}

void RootDomain::Normalize(VKernelBase *kernel) {
  tile_num_ = 1;
  PropDomain::Normalize();
  auto &nd = dom_->nd_;
  align_.base = 0;
  align_.depth = nd.size();
  PropDomain::AlignProp(align_);
  tile_size_ = 1;
  for (int i = 0; i < align_.depth; ++i) {
    tile_size_ *= nd[i];
  }
  align_.space = tile_size_;
  block_align_ = kernel->BlockAlign();
  if (tile_size_ % block_align_) {
    tile_size_ += block_align_ - tile_size_ % block_align_;
  }
  for (size_t i = align_.depth; i < nd.size(); ++i) {
    tile_size_ *= nd[i];
  }
}

int64_t RootDomain::Tile(int start, int end, int64_t space, int64_t num) {
  TileParam tp;
  tp.start = start;
  tp.end = end;
  tp.num = num;
  tp.tile = (space + num - 1) / num;
  tp.tail = space % tp.tile;
  PropDomain::TileProp(tp);
  if (start > 0) {
    tile_size_ = tile_size_ / space * ((space + num - 1) / num);
  } else {
    auto align_size = (tp.tile + block_align_ - 1) / block_align_ * block_align_;
    tile_size_ = tile_size_ / align_.space * align_size;
    align_.space = align_size;
  }
  tile_num_ *= num;
  return tile_size_;
}

class ReshapeDomain : public PropDomain {
 public:
  ReshapeDomain(NDObject *head, const std::vector<int64_t> &src, const std::vector<int64_t> &dst)
   : PropDomain(head), src_(src), dst_(dst) {}
  void FoldProp(PropRange &range) override {
    PropRange dst_range;
    dst_range.base = 0;
    for (int i = dst_.size() - 1; i >= 0; --i) {
      if (dst_[i] > 1) {
        dst_range.base = i;
        break;
      }
    }
    int start = dst_range.base;
    int64_t space = dst_[start];
    while (space < range.space && start > 0) {
      space *= dst_[--start];
    }
    dst_range.depth = dst_range.base - start + 1;
    dst_range.space = range.space;
    PropDomain::FoldProp(dst_range);
    if (range.space > dst_range.space) {
      int src_start = range.base - range.depth;
      while (range.space > dst_range.space) {
        range.space /= src_[++src_start];
      }
      range.depth = range.base - src_start;
    }
    if (dst_range.affine > range.affine) {
      range.affine = dst_range.affine;
    }
    prop_base = dst_range.base;
  }
  void AlignProp(PropRange &range) override {
    int64_t space = src_[0];
    for (int i = 1; i < range.depth; ++i) {
      space *= src_[i];
    }
    int dst_init_depth = 0;
    while (space > 1) {
      space /= dst_[dst_init_depth++];
    }
    PropRange dst_range;
    dst_range.depth = dst_init_depth;
    PropDomain::AlignProp(dst_range);
    if (dst_range.depth < dst_init_depth) {
      int64_t delta_space = dst_[dst_range.depth];
      for (int i = dst_range.depth + 1; i < dst_init_depth; ++i) {
        delta_space *= dst_[i];
      }
      while (delta_space > 1) {
        delta_space /= src_[--range.depth];
      }
    }
    if (dst_range.affine > range.affine) {
      range.affine = dst_range.affine;
    }
  }
  void TileProp(const TileParam &tp) override {
    int64_t dim_space = tp.tile * tp.num;
    if (tp.tail) {
      dim_space -= tp.tile - tp.tail;
    }
    TileParam t;
    if (tp.start == 0) {
      t.start = 0;
      t.end = 0;
      while (true) {
        dim_space /= dst_[t.end];
        if (dim_space <= 1 || t.end + 1 == static_cast<int>(dst_.size())) break;
        t.end++;
      }
    } else {
      t.end = prop_base;
      t.start = t.end;
      while (true) {
        dim_space /= dst_[t.start];
        if (dim_space <= 1 || t.start == 0) break;
        t.start--;
      }
    }
    t.num = tp.num;
    t.tile = tp.tile;
    t.tail = tp.tail;
    PropDomain::TileProp(t);
  }

 private:
  const std::vector<int64_t> &src_;
  const std::vector<int64_t> &dst_;
  int prop_base = -1;
};

class ShapeTiling {
 public:
  ShapeTiling(VKernelBase *kernel, RootDomain &prim_dom, int64_t core_limit)
  : prim_dom_(prim_dom), core_limit_(core_limit) {
    repeat_size_ = SIMD_REPEAT_SIZE / kernel->MinTypeSize();
  }
  ~ShapeTiling() = default;
  void Run(int64_t tile_size_limit) {
    tile_size_limit_ = tile_size_limit;
    int align_depth = prim_dom_.align_.depth;
    PropRange fold;
    fold.base = prim_dom_.DimSpace().size() - 1;
    int64_t tile_size = prim_dom_.TileSize();
    do  {
      if (fold.base + 1 > align_depth) {
        fold.depth = fold.base + 1;
        fold.space = -1;
        prim_dom_.FoldProp(fold);
        int start_dim = fold.base + 1 - fold.depth;
        ASSERT(start_dim > 0);
        if (start_dim < align_depth) { // axis of 1
          start_dim = align_depth;
        }
        int64_t num = CalcTile(tile_size, fold);
        if (num > 1) {
          tile_size = prim_dom_.Tile(start_dim, fold.base, fold.space, num);
        }
        fold.base = start_dim - 1;
      } else {
        int64_t num = CalcTileAlign(tile_size, prim_dom_.align_);
        if (num > 1) {
          tile_size = prim_dom_.Tile(0, fold.base, prim_dom_.align_.space, num);
        }
        return;
      }
    } while(tile_size > tile_size_limit_);
    if (align_depth > 1) {
      prim_dom_.Tile(0, align_depth - 1, prim_dom_.align_.space, 1);
    }
  }
 protected:
  int64_t GetDivision(int64_t val, int64_t min) {
    int64_t div = min;
    int64_t factor = val / div;
    while (div <= factor) {
      if (div * factor == val) {
        return div;
      }
      div++;
      factor = val / div;
    }
    while (div <= val) {
      div = val / (val / div);
      if (val % div == 0) {
        return div;
      }
      div++;
    }
    ASSERT(0);
    return -1;
  }
  int64_t CostMeasure(int64_t space, int64_t tile_num) {
    return ((tile_num - 1) / core_limit_ + 1) * (std::max((space - 1) / repeat_size_, 3L) + 2);
  }
  int64_t CalcTile(int64_t tile_size, const PropRange &range) {
    // ceil(a/b) <= c --> b >= ceil(a/(c+1))+1
    // floor(a/b) = ceil((a-1)/b)  <= c-1 --> b >= ceil((a-1)/(c-1 + 1))+1
    // floor(a/b) <= c --> b >= floor(a/c)
    // floor(tile_size_/tile_num <= tile_size_limit_) --> tile_num >= floor(tile_size_/tile_size_limit_)
    // (tile_size / space) * floor(space /tile_num) <= tile_size_limit_) --> tile_num >= floor(space/max_factor)
    // EQUAL TO:
    //  int64_t tile_num = (space + max_factor - 1) / max_factor;
    //  while (tile_num < space && ((space + tile_num - 1) / tile_num) > max_factor) {
    //    tile_num++;
    //  }
    int64_t space = range.space;
    int64_t tile_num, tile_factor;
    if (tile_size > tile_size_limit_) {
      int64_t max_factor = tile_size_limit_ / (tile_size / space);
      if (max_factor <= 1) {
       return space;
      }
      tile_num = std::max((space + max_factor - 1) / max_factor, (tile_size_limit_ + tile_size - 1) / tile_size);
      tile_factor = (space - 1) / tile_num + 1;
    } else if (range.affine >= PropRange::REDUCE) {
      return 1;
    } else {
      tile_num = 1;
      tile_factor = space;
    }
    int64_t space_unit = prim_dom_.TileSize() / space;
    int64_t tile_num_base = prim_dom_.TileNum();
    int64_t best_cost = CostMeasure(space_unit * tile_factor, tile_num * tile_num_base);
    int64_t max_tile_num = std::min(int64_t(tile_num + DeviceInfo::Instance().CoreNum() - 1), space);
    for (int64_t t_num = tile_num + 1; t_num <= max_tile_num; ++t_num) {
      tile_factor = (space - 1) / t_num + 1;
      int64_t cost = CostMeasure(space_unit * tile_factor, tile_num * tile_num_base);
      if (cost < best_cost) {
        tile_num = t_num;
        best_cost = cost;
      }
    }
    return tile_num;
  }

  int64_t CalcTileAlign(int64_t tile_size, const PropRange &range) {
    int64_t space = range.space;
    int64_t tile_num = GetDivision(space, (tile_size - 1) / tile_size_limit_ + 1);
    if (tile_num < space && range.affine < PropRange::REDUCE) {
      int64_t tile_num_base = prim_dom_.TileNum();
      int64_t cost = CostMeasure(space / tile_num, tile_num * tile_num_base);
      int64_t div_tile = tile_num;
      while (div_tile < space) {
        div_tile = GetDivision(space, div_tile + 1);
        int64_t div_cost = CostMeasure(space / div_tile, div_tile * tile_num_base);
        if (div_cost >= cost) break;
        cost = div_cost;
        tile_num = div_tile;
      }
    }
    ASSERT(space % tile_num == 0);
    return tile_num;
  }
  VKernelBase* kernel_;
  RootDomain &prim_dom_;
  int64_t tile_size_limit_;
  int64_t repeat_size_;
  int64_t core_limit_;
};

std::string& VKernel::DisAssemble() {
  std::ostringstream oss;
  code_ptr_->DisAssemble(oss);
  das_str_ = oss.str();
  return das_str_;
}

VKernelBase::~VKernelBase() {
  for (auto &op: build_ops_) {
    delete op;
  }
}

static const uint64_t ITEM_SIMD_WIDTH_MAX[kTypeEnd] = {128, 128, 64, 64};
void VKernelBase::DoCodeGen(uint64_t core_limit) {
  code_.Reset();
  CodeGenHelper helper;
  int peak_live = Analyze(helper);
  uint64_t free_mem = DeviceInfo::Instance().LocalMemSize() - DeviceInfo::Instance().UbWorkspaceSize() - ReserveCodeSize();
  uint64_t tile_size_limit = free_mem / (MaxTypeSize() * peak_live);
  // tiling
  if (tiles_.empty()) {
    ShapeTiling tiling(this, root_dom_, core_limit);
    tiling.Run(tile_size_limit);
  } else {
    auto &dims = root_dom_.DimSpace();
    for (auto &t: tiles_) {
      int64_t space = dims[t.start];
      for (int i = t.start + 1; i <= t.end; ++i) {
        space *= dims[i];
      }
      root_dom_.Tile(t.start, t.end, space, t.num);
    }
  }
  // simd_width
  auto lead_dim = root_dom_.DimSpace().front();
  uint64_t tile_outer = root_dom_.TileSize() / lead_dim;
  uint64_t max_sw = ITEM_SIMD_WIDTH_MAX[max_type_];
  uint64_t block_sw = BlockAlign();
  uint64_t best_repeat = 0xfffffffful;
  for (uint64_t sw = max_sw; sw > 0; sw -= block_sw) {
    uint64_t repeat = (lead_dim + sw - 1) / sw * tile_outer;
    if (repeat * sw <= tile_size_limit && repeat <= best_repeat) {
      code_.simd_width_ = sw;
      best_repeat = repeat;
    }
  }
  code_.tile_num_ = root_dom_.TileNum();
  code_.UpdateBlockDim(core_limit);
  // codegen
  helper.xbuf_size_ = best_repeat * code_.simd_width_ * MaxTypeSize();
  helper.Generate(this);
}

std::string VKernelBase::DumpGraph() {
  static std::unordered_map<ObjectType, std::string> obj_name = {
    {kLoad, "Load"},
    {kLoadDummy, "LoadDummy"},
    {kStore, "Store"},
    {kCopy, "Copy"},
    {kReshape, "Reshape"},
    {kUnary, "Unary"},
    {kBinary, "Binary"},
    {kBinaryS, "BinaryS"},
    {kBroadcastTo, "BroadcastTo"},
    {kBroadcastS, "BroadcastS"},
    {kReduce, "Reduce"},
    {kSelect, "Select"},
    {kCast, "Cast"},
  };
  std::ostringstream oss;
  auto dump_op = [&oss](NDObject *op) {
    oss << "%" << op->index_<< "[";
    if (!op->nd_.empty()) {
      for (size_t i = 0; i < op->nd_.size() - 1; ++i) {
        oss << op->nd_[i] << ",";
      }
      oss << op->nd_.back();
    }
    oss << "]";
  };
  oss << "vkernel.graph(tile_num=" << code_.tile_num_ << ", simd_width="<<code_.simd_width_ << ", insn_num="
      << code_.insn_num_ << ") {" << std::endl;
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    oss << "  ";
    dump_op(op);
    oss << " = " << obj_name[op->GetObjectType()] << "(";
    if (op->GetObjectType() == kSelect) {
      dump_op(static_cast<SelectOp*>(op)->cond_);
      oss << ", ";
    }
    if (op->lhs_) {
      dump_op(op->lhs_);
      if (op->rhs_) {
        oss << ", ";
        dump_op(op->rhs_);
      }
    }
    oss << ") // stride = [";
    if (!op->strides_.empty()) {
      for (size_t i = 0; i < op->strides_.size() - 1; ++i) {
        oss << op->strides_[i] << ",";
      }
      oss << op->strides_.back();
    }
    oss << "]" << std::endl;
  }
  oss << "}";
  return oss.str();
}

// lead_dim_ is used only in codegen phase. so we reuse it for liveness analyze
#define OP_GEN_S(op) do { op->lead_dim_ = 2; } while(0)
#define OP_GEN_D(op) do { op->lead_dim_ = 1; } while(0)
#define OP_KILL(op) do { op->lead_dim_ = 0; } while(0)
#define OP_LIVE(op) (op->lead_dim_)
#define OP_LIVE_D(op) (op->lead_dim_ == 1)
int VKernelBase::Analyze(CodeGenHelper &codegen) {
  int cur_live = 0;
  int load_pipe_idx = 0;
  int store_pipe_idx = 0;
  int vector_pipe_idx = 0;
  int back_wait_idx = INT_MAX;
  int op_index_ = 0;
  for (auto op : objects_) {
    op->index_ = op_index_++;
    if (op->Pipe() == V_PIPE_LOAD) {
      cur_live++;
      codegen.static_ops_.push_back(op);
      op->pipe_idx = load_pipe_idx++;
    } else if (op->Pipe() == V_PIPE_STORE) {
      int prod_idx = op->lhs_->index_;
      if (prod_idx < back_wait_idx) {
        back_wait_idx = prod_idx;
      }
      OP_GEN_S(op->lhs_);
      codegen.static_ops_.push_back(op->lhs_);
      op->xbuf_ = -1; // donot need
      cur_live++;
      op->pipe_idx = store_pipe_idx++;
    } else {
      op->pipe_idx = vector_pipe_idx++;
    }
  }
  int live_peak = cur_live;
  ASSERT(back_wait_idx > 0);
  codegen.back_wait_ = objects_[back_wait_idx];
  int back_set_idx = -1;
  auto LivenessEnd = [&back_set_idx, &cur_live](NDObject *op, NDObject *end) {
    if (end->Pipe() == V_PIPE_SIMD && !OP_LIVE(end)) {
      OP_GEN_D(end);
      return true;
    }
    if (back_set_idx == -1 && end->Pipe() == V_PIPE_LOAD) {
      back_set_idx = op->index_;
    }
    return false;
  };
  for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
    auto op = *it;
    if (op->Pipe() == V_PIPE_SIMD) {
     auto kill = op->lhs_;
      if (kill && LivenessEnd(op, kill)) {
        if (OP_LIVE_D(op) && (op->obj_id_ == kUnary || op->obj_id_ == kBinary || op->obj_id_ == kBinaryS)) {
          op->flags_ |= OBJ_FLAG_REUSE_LHS;
        } else {
          op->flags_ |= OBJ_FLAG_FREE_LHS;
          cur_live++;
        }
      }
      kill = op->rhs_;
      if (kill && LivenessEnd(op, kill)) {
        if (OP_LIVE_D(op) && !(op->flags_ & OBJ_FLAG_REUSE_LHS) && op->obj_id_ == kBinary) {
          op->flags_ |= OBJ_FLAG_REUSE_RHS;
        } else {
          op->flags_ |=  OBJ_FLAG_FREE_RHS;
          cur_live++;
        }
      }
      if (op->GetObjectType() == kSelect) {
        SelectOp *select_op = reinterpret_cast<SelectOp*>(op);
        if (LivenessEnd(op, select_op->cond_)) {
          cur_live++;
          select_op->free_cond = true;
        }
      }
      if (cur_live > live_peak) {
        live_peak = cur_live;
      }
      if (OP_LIVE_D(op) && !(op->flags_ & (OBJ_FLAG_REUSE_LHS | OBJ_FLAG_REUSE_RHS))) {
        cur_live--;
      }
      OP_KILL(op);
      op->xbuf_ = 0;
    }
  }
  ASSERT(back_set_idx > 0);
  codegen.back_set_ = objects_[back_set_idx];
  return live_peak;
}

class PropDomainBuilder {
 public:
  void Build(const std::vector<NDObject*> &objects, RootDomain &root) {
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
      NDObject* op = *it;
      if (GetHead(op) == nullptr) {
        SetHead(op, op);
      }
      if (op->lhs_) {
        BuildOp(op, op->lhs_);
        if (op->rhs_) {
          BuildOp(op, op->rhs_);
          if (op->GetObjectType() == kSelect) {
            SelectOp *select = reinterpret_cast<SelectOp*>(op);
            BuildOp(op, select->cond_);
          }
        }
      }
    } // end for
    auto dom_op = GetHead(objects.back());
    root.SetHead(dom_op);
    link_num_ = link_ops_.size();
    if (link_num_) {
      BuildSumDomain(&root, dom_op);
    }
  }
 private:
  NDObject* GetHead(NDObject *op)  { return reinterpret_cast<NDObject*>(op->insn_); }
  void SetHead(NDObject *op, NDObject* head)  { op->insn_ = reinterpret_cast<uint64_t*>(head); }
  void BuildOp(NDObject* op, NDObject* next) {
    auto op_head = GetHead(op);
    auto next_head = GetHead(next);
    if (op_head == next_head) return;
    if (op->GetObjectType() == kReshape) {
      SetHead(next, next);
      link_ops_.push_back(op);
      return;
    }
    if (next_head == nullptr) {
      next->pd_next_ = op->pd_next_;
      op->pd_next_ = next;
      SetHead(next, op_head);
    } else {
      auto old_next = op->pd_next_;
      op->pd_next_ = next_head;
      while (true) {
        SetHead(next_head, op_head);
        if (next_head->pd_next_ == nullptr) {
          next_head->pd_next_ = old_next;
          break;
        }
        next_head = next_head->pd_next_;
      }
    }
  }
  void BuildSumDomain(PropDomain *dom, NDObject *head) {
    for (size_t i = 0; i < link_ops_.size(); ++i) {
      auto op = link_ops_[i];
      if (op == nullptr) continue;
      auto head1 = GetHead(op);
      auto head2 = GetHead(op->lhs_);
      if (head1 == head2) {
        link_ops_[i] = nullptr;
        link_num_--;
        continue;
      }
      if (head1 == head) {
        auto sdom = new ReshapeDomain(head2, op->nd_, op->lhs_->nd_);
        dom->subdoms_.push_back(sdom);
        link_ops_[i] = nullptr;
        if (--link_num_) {
          BuildSumDomain(sdom, head2);
        }
      } else if (head2 == head) {
        auto sdom = new ReshapeDomain(head1, op->lhs_->nd_, op->nd_);
        dom->subdoms_.push_back(sdom);
        link_ops_[i] = nullptr;
        if (--link_num_) {
          BuildSumDomain(sdom, head1);
        }
      }
    }
  }
  std::vector<NDObject*> link_ops_;
  int link_num_;
};

void VKernelBase::BuildDomain() {
  max_type_ = objects_.front()->type_id_;
  min_type_ = objects_.front()->type_id_;
  bool slow_build_path = false;
  for (auto op : objects_) {
    auto type = op->GetObjectType();
    if (type == kReshape) {
      slow_build_path = true;
    } else if (type == kCast || type == kLoad) {
      int type = op->type_id_;
      if (type > max_type_) {
        max_type_ = type;
      } else if (type < min_type_) {
        min_type_ = type;
      }
    }
  }
  if (slow_build_path) {
    PropDomainBuilder builder;
    builder.Build(objects_, root_dom_);
  } else {
    NDObject *next = nullptr;
    for (auto obj: objects_) {
      obj->pd_next_ = next;
      next = obj;
    }
    root_dom_.SetHead(next);
  }
}

void VKernelS::Append(NDObject *obj) {
  build_ops_.push_back(obj);
  obj->Normalize(objects_);
  objects_.emplace_back(obj);
}

static std::vector<pass::Pass> passes = {&pass::ReorderStore};
void VKernelS::Optimize() {
  auto bb = pass::BasicBlock(objects_);
  for (auto pass : passes) {
    pass(bb);
  }
  objects_ = bb.ToVector();
}

void VKernelS::CodeGen() {
  Optimize();
  BuildDomain();
  NormalizeDomain();
  DoCodeGen(DeviceInfo::Instance().CoreNum());
}

void VKernelD::CodeGen() {
  objects_.clear();
  root_dom_.Clear();
  for (auto op : build_ops_) {
    op->Normalize(objects_);
    objects_.emplace_back(op);
  }
  BuildDomain();
  NormalizeDomain();
  DoCodeGen(DeviceInfo::Instance().CoreNum());
}

void VKernelP::CodeGen() {
  auto WorkLoad = [](VKernelS *k) -> uint64_t { return k->root_dom_.TileSize() * k->objects_.size(); };
  uint64_t total_workload = 0;
  for (auto k : children_) {
    k->BuildDomain();
    k->NormalizeDomain();
    total_workload += WorkLoad(k);
  }
  std::sort(children_.begin(), children_.end(), [&WorkLoad](VKernelS *a, VKernelS *b) -> bool { return WorkLoad(a) < WorkLoad(b); });
  uint64_t core_num = DeviceInfo::Instance().CoreNum();
  for (size_t i = 0; i < children_.size(); ++i) {
    auto k = children_[i];
    auto workload = WorkLoad(k);
    uint64_t core_limit = core_num * workload / total_workload;
    k->DoCodeGen(core_limit);
    code_.children_.push_back(&k->code_);
    total_workload -= workload;
    core_num -= k->GetCode()->block_dim_;
  }
  std::vector<uint64_t> offsets;
  code_.LinkAll(offsets);
  for (size_t i = 0; i < children_.size(); ++i) {
    uint64_t *new_base = reinterpret_cast<uint64_t*>(code_.data_ + offsets[i]);
    uint64_t *old_base = reinterpret_cast<uint64_t*>(code_.children_[i]->data_);
    for (auto op :  children_[i]->objects_) {
      if (op->obj_id_ == kLoad) {
        auto load = static_cast<NDLoad*>(op);
        load->insn_ = new_base + (load->insn_ - old_base);
      } else if (op->obj_id_ == kStore) {
        auto store = static_cast<NDStore*>(op);
        store->insn_ = new_base + (store->insn_ - old_base);
      }
    }
  }
}

std::string VKernelP::DumpGraph() {
  std::ostringstream oss;
  oss << "vgraph.parallel() {" << std::endl;
  for (auto k : children_) {
    oss << k->DumpGraph() << std::endl;
  }
  oss << "}";
  return oss.str();
}
} // namespace dvm
