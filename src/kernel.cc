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
#include <cstring>
#include <climits>
#include <algorithm>
#include "kernel.h"
#include "pass.h"

namespace dvm {
static const uint64_t ITEM_SIMD_WIDTH_MAX[kTypeEnd] = {128, 128, 128, 64, 64};

class CodeGenHelper {
 public:
  enum CodeGenType {
    kGenSimd0 = 0,
    kGenSimd1,
    kGenSimd2,
    kGenSimd3,
    kGenLoad,
    kGenStore,
  };
  struct EventManager {
    enum { MAX_EVENT_NUM = 8 };
    int hold_idx[MAX_EVENT_NUM]{0};
    int hold_event{-1};
    int sync_idx{-1};
  };

  CodeGenHelper(VectorKernel &kernel): kernel_(kernel) {}
  bool Generate() {
    static const CodeGenType codegen_types[ObjectType::kObjectBulk] = {
      kGenLoad,  // loaddummy
      kGenLoad,  // load
      kGenStore, // padstore
      kGenStore, // store
      kGenSimd1, // reshape
      kGenSimd1, // copy
      kGenSimd1, // unary
      kGenSimd2, // binary
      kGenSimd1, // cast
      kGenSimd1, // binarys
      kGenSimd1, // broadcastto
      kGenSimd0, // broadcasts
      kGenSimd1, // reduce
      kGenSimd3, // select
      kGenSimd1, // elementany
      kGenSimd1, // RemovePad
    };
    auto &code = kernel_.code_;
    auto code_reserved = kernel_.ReserveCodeSize();
    code.Alloc(code_reserved + code.HeadSize());
    uint64_t *code_ptr = reinterpret_cast<uint64_t*>(code.data_ + code.HeadSize());
    static_xbuf_ = System::Instance().UbWorkspaceSize() + code_reserved;
    for (auto op : kernel_.static_ops_) {
      op->xbuf_ = static_xbuf_;
      static_xbuf_ += xbuf_size_;
    }
    auto simd_width = kernel_.simd_width_;
    for (auto op: kernel_.objects_) {
      op->UpdateStride(simd_width);
      op->tail_insn_ = op->insn_ = code_ptr;
      switch (codegen_types[op->obj_id_]) {
        case kGenSimd0: {
          auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          break;
        }
        case kGenSimd1: {
          auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.emplace(op->lhs_, op);
          }
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          SimdSync(op->lhs_, op);
          break;
        }
        case kGenSimd2: {
          auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.emplace(op->lhs_, op);
          }
          if (op->flags_ & OBJ_FLAG_FREE_RHS) {
            free_xbuf_.emplace(op->rhs_, op);
          }
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          if (op->rhs_->index_ > op->lhs_->index_) {
            SimdSync(op->rhs_, op);
            SimdSync(op->lhs_, op);
          } else {
            SimdSync(op->lhs_, op);
            SimdSync(op->rhs_, op);
          }
          break;
        }
        case kGenSimd3: {
          auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.emplace(op->lhs_, op);
          }
          if (op->flags_ & OBJ_FLAG_FREE_RHS) {
            free_xbuf_.emplace(op->rhs_, op);
          }
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          SelectOpPostProc(op);
          break;
        }
        case kGenLoad: {
          code_ptr += op->Emit(kernel_);
          break;
        }
        case kGenStore: {
          code_ptr += op->Emit(kernel_);
          StoreSync(op->lhs_, op);
#ifdef DEBUG
          auto head = *(op->tail_insn_);
          ASSERT((head & 1ul << V_HEAD_SIMD_FLAG_OFFSET) == 0);
          *code_ptr++ = op->Size();
          auto new_head_size = ((head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK) + 1;
          vClrBitRange(head, V_M_HEAD_SIZE_OFFSET, 4);
          *(op->tail_insn_) = head | new_head_size << V_M_HEAD_SIZE_OFFSET;
#endif
          break;
        }
        default:
          ASSERT(0);
          break;
      } // end switch
    } // end for op
    *code_ptr++ = vMakeHead(vLoadInsnID::V_LOAD_NONE, 0, 0, V_PIPE_LOAD);
    BackwardSync();
    code.data_size_ = reinterpret_cast<uint8_t*>(code_ptr) - code.data_;
    ASSERT(code.data_size_ <= code_reserved + code.HeadSize());
    code.UpdateHead(kernel_.tile_num_, simd_width, 0);
    return true;
  }

 private:
  void BackwardSync() {
    auto &objects = kernel_.objects_;
    EventManager vl_event, sv_event;
    auto alloc_event = [](EventManager &m, uint64_t &event) -> bool {
      event = m.hold_event + 1;
      if (event >= System::Instance().EventNum()) {
        event = m.hold_event;
        return false;
      }
      m.hold_event = event;
      return true;
    };
    vl_event.sync_idx = sv_event.sync_idx = static_cast<int>(objects.size());
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
      auto op = *it;
      auto pipe = op->Pipe();
      if (pipe == V_PIPE_STORE) {  // STORE -> SIMD
        auto simd = op->lhs_;
        if (simd->index_ < sv_event.sync_idx) {
          uint64_t event;
          if (alloc_event(sv_event, event)) {
            *(op->tail_insn_) |= 1ul << V_M_HEAD_SET_FLAG_OFFSET | event << V_M_HEAD_SET_EVENT_OFFSET;
          } else {
            auto to_sync = kernel_.objects_[sv_event.sync_idx]->insn_;
            *to_sync &= ~(0x1ul << V_HEAD_BACK_WAIT_OFFSET);
          }
          *(simd->insn_) |= 1ul << V_HEAD_BACK_WAIT_OFFSET | event << V_HEAD_B_WAIT_EVENT_OFFSET;
          sv_event.sync_idx = simd->index_;
        }
      } else if (pipe == V_PIPE_SIMD && op->lhs_) { // SIMD -> LOAD
        NDObject *load = nullptr;
        if (op->lhs_->IsLoad()) load = op->lhs_;
        auto rhs = op->rhs_;
        if (rhs) {
          if (rhs->IsLoad() && (load == nullptr || rhs->index_ < load->index_)) load = rhs;
          if (op->obj_id_ == ObjectType::kSelect) {
            NDObject *cond = reinterpret_cast<SelectOp*>(op)->cond_;
            if (cond->IsLoad() && (load == nullptr || cond->index_ < load->index_)) load = cond;
          }
        }
        if (load != nullptr && load->index_ < vl_event.sync_idx) {
          uint64_t event;
          if (alloc_event(vl_event, event)) {
            *(op->tail_insn_) |= 1ul << V_HEAD_BACK_SET_OFFSET | event << V_HEAD_B_SET_EVENT_OFFSET;
          } else {
            auto to_sync = kernel_.objects_[vl_event.sync_idx]->insn_;
            *to_sync &= ~(0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET);
          }
          *(load->insn_) |= 1ul << V_M_HEAD_WAIT_FLAG_OFFSET | event << V_M_HEAD_WAIT_EVENT_OFFSET;
          vl_event.sync_idx = load->index_;
        }
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
      SimdSync(inputs[i], op);
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
    if (!free_xbuf_.empty() && free_xbuf_.front().second->index_ < vector_vector_sync) {
      // roughly reuse for simplify: ignore inputs barrier to be inserted
      obj->xbuf_ = free_xbuf_.front().first->xbuf_;
      free_xbuf_.pop();
      return nullptr;
    }
    if (static_xbuf_ + xbuf_size_ <= System::Instance().LocalMemSize()) {
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

  inline void SimdBarrier(NDObject *from, NDObject *to) {
    if (from->index_ >= vector_vector_sync) {
      *(to->insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      vector_vector_sync = to->index_;
    }
  }

  void SimdSync(NDObject *from, NDObject *to) {
    if (from->IsSimd()) {
      SimdBarrier(from, to);
      return;
    }
    int from_pipe_idx = from->index_;
    if (from_pipe_idx <= lv_event_.sync_idx) return;
    auto from_insn = from->tail_insn_;
    auto to_insn = to->insn_;
    uint64_t event;
    if (AllocForwardEvent(lv_event_, from_pipe_idx, to->index_, event)) {
      *to_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | event << V_HEAD_WAIT_EVENT_OFFSET;
    } else {
      auto from_sync = kernel_.objects_[lv_event_.sync_idx]->tail_insn_;
      *from_sync &= ~(0x1ul << V_M_HEAD_SET_FLAG_OFFSET);
    }
    *from_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | event << V_M_HEAD_SET_EVENT_OFFSET;
    lv_event_.sync_idx = from_pipe_idx;
  }

  inline void StoreSync(NDObject *from, NDObject *to) {
    int from_pipe_idx = from->index_;
    if (from_pipe_idx <= vs_event_.sync_idx) return;
    auto from_insn = from->tail_insn_;
    auto to_insn = to->insn_;
    uint64_t event;
    if (AllocForwardEvent(vs_event_, from_pipe_idx, to->index_, event)) {
      *to_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | event << V_M_HEAD_WAIT_EVENT_OFFSET;
    } else {
      auto from_sync = kernel_.objects_[vs_event_.sync_idx]->tail_insn_;
      *from_sync &= ~(0x1ul << V_HEAD_SET_FLAG_OFFSET);
    }
    *from_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | event << V_HEAD_SET_EVENT_OFFSET;
    vs_event_.sync_idx = from_pipe_idx;
  }

  inline bool AllocForwardEvent(EventManager &m, int from_idx, int to_idx, uint64_t &event) {
    int total = System::Instance().EventNum();
    for (int i = 1; i <= total; ++i) {
      event = (m.hold_event + i) % total;
      if (from_idx >= m.hold_idx[event]) {
        m.hold_idx[event] = to_idx;
        m.hold_event = event;
        return true;
      }
    }
    event = m.hold_event;
    return false;
  }

  uint32_t xbuf_size_{0};
  uint64_t static_xbuf_{0};
  std::queue<std::pair<NDObject*, NDObject*>> free_xbuf_;

  int vector_vector_sync = 0;
  EventManager lv_event_;
  EventManager vs_event_;

  VectorKernel &kernel_;
  friend VectorKernel;
};

void PropDomain::Normalize() {
  size_t nd_size = 1;
  dom_ = nullptr;
  auto select_dom = [this](NDObject *cand) -> bool {
    auto &dom_nd = dom_->nd_;
    auto &cand_nd = cand->nd_;
    if (cand_nd.size() != dom_nd.size()) {
      return cand_nd.size() > dom_nd.size();
    }
    for (size_t i = 0; i < dom_nd.size(); ++i) {
      if (cand_nd[i] > dom_nd[i]) return true;
    }
    return false;
  };
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    auto size = op->nd_.size();
    if (size > nd_size) {
      nd_size = size;
    }
    NDObject *cand = nullptr;
    auto obj_type = op->GetObjectType();
    if (op->IsStore() || obj_type == kReduce || obj_type == kElementAny) {
      if (op->lhs_ != dom_) cand = op->lhs_;
    } else if (obj_type == kBroadcastTo) {
      if (op != dom_) cand = op;
    }
    if (cand) {
      if (dom_ == nullptr || select_dom(cand)) dom_ = cand;
    } else if (dom_ == nullptr && !op->IsStore() && obj_type != kLoadDummy) {
      dom_ = op;
    }
  }
  ASSERT(dom_ != nullptr);
  for (auto op = head_; op != nullptr; op = op->pd_next_) {
    if (op->nd_.size() < nd_size) {
      op->nd_.resize(nd_size, 1);
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

void RootDomain::Normalize(VectorKernel *kernel) {
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
  tp.tile = CeilDiv(space, num);
  tp.tail = space % tp.tile;
  tp.group_tile = false;
  PropDomain::TileProp(tp);
  if (start > 0) {
    tile_size_ = tile_size_ / space * CeilDiv(space, num);
  } else {
    auto align_size = CeilDiv<int64_t>(tp.tile, block_align_) * block_align_;
    tile_size_ = tile_size_ / (CeilDiv<int64_t>(align_.space, block_align_) * block_align_) * align_size;
    align_.space = align_size;
  }
  tile_num_ *= num;
  return tile_size_;
}

void RootDomain::GroupTile(int dim, int64_t space, int64_t tile) {
  TileParam tp;
  tp.start = dim;
  tp.end = dim;
  tp.num = CeilDiv(space, tile);
  tp.tile = tile;
  tp.tail = space % tile;
  tp.group_tile = true;
  PropDomain::TileProp(tp);
}

void RootDomain::Align(int depth, int64_t space) {
  TileParam tp;
  tp.start = 0;
  tp.end = depth - 1;
  tp.num = 1;
  tp.tile = space;
  tp.tail = 0;
  tp.group_tile = false;
  PropDomain::TileProp(tp);
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
    int64_t dst_space = 1;
    if (tp.start == 0) {
      t.start = 0;
      t.end = 0;
      while (true) {
        dst_space *= dst_[t.end];
        if (dst_space > dim_space || t.end + 1 == static_cast<int>(dst_.size())) break;
        t.end++;
      }
    } else {
      t.end = prop_base;
      t.start = t.end;
      while (true) {
        dst_space *= dst_[t.start];
        if (dst_space >= dim_space || t.start == 0) break;
        t.start--;
      }
    }
    t.num = tp.num;
    t.tile = CeilDiv(dst_space, t.num);
    t.tail = dst_space % t.tile;
    PropDomain::TileProp(t);
  }

 private:
  const std::vector<int64_t> &src_;
  const std::vector<int64_t> &dst_;
  int prop_base = -1;
};

class ShapeTiling {
 public:
  ShapeTiling(VectorKernel *kernel, RootDomain &prim_dom, int64_t core_limit)
  : kernel_(kernel), prim_dom_(prim_dom), core_limit_(core_limit) {
    repeat_size_ = ITEM_SIMD_WIDTH_MAX[kernel->MaxType()];
  }
  ~ShapeTiling() = default;
  void Run(int64_t tile_size_limit) {
    tile_size_limit_ = tile_size_limit;
    int align_depth = prim_dom_.align_.depth;
    PropRange fold;
    fold.base = prim_dom_.DimSpace().size() - 1;
    int64_t tile_size = prim_dom_.TileSize();
    int64_t num;
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
        num = CalcTile(tile_size, fold);
        if (num > 1) {
          tile_size = prim_dom_.Tile(start_dim, fold.base, fold.space, num);
        }
        fold.base = start_dim - 1;
      } else {
        num = CalcLeadTile(tile_size, prim_dom_.align_);
        tile_size = prim_dom_.Tile(0, fold.base, prim_dom_.align_.space, num);
        return;
      }
    } while(tile_size > tile_size_limit_ || (num > 1 && num == fold.space));
    if (align_depth > 1) {
      prim_dom_.Align(align_depth, prim_dom_.align_.space);
    }
  }

  int64_t proposal_sw_{0};

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
  int64_t CostMeasure(int64_t factor, int64_t tile_num) {
    return CeilDiv(tile_num, core_limit_) * (factor + 2);
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
    int64_t tile_num;
    if (tile_size > tile_size_limit_) {
      int64_t max_factor = tile_size_limit_ / (tile_size / space);
      if (max_factor <= 1) {
       return space;
      }
      tile_num = std::max(CeilDiv(space, max_factor), CeilDiv(tile_size_limit_, tile_size));
    } else {
      tile_num = 1;
    }
    if (prim_dom_.TileNum() > 1) { // avoid tile range pad
      tile_num = GetDivision(space, tile_num);
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
    } else {
      int64_t init_factor = CeilDiv(space, tile_num);
      int64_t best_cost = CostMeasure(init_factor, tile_num);
      int64_t start_num = tile_num + 1;
      while (true) {
        int64_t align_tile = CeilDiv(start_num, core_limit_) * core_limit_;
        int64_t factor = CeilDiv(space, align_tile);
        int64_t t_num = CeilDiv(space, factor);
        int64_t cost = CostMeasure(factor, t_num);
        if (cost < best_cost) {
          best_cost = cost;
          tile_num = t_num;
          if (t_num % core_limit_ == 0) break;
        }
        if (factor == 1 || cost > best_cost * 2) break;
        start_num = std::max(align_tile + core_limit_, CeilDiv(space, factor - 1));
      }
    }
    return tile_num;
  }

  int64_t CostMeasure2(int64_t repeat_num, int64_t tile_num) {
    int64_t core_tile = CeilDiv(tile_num, core_limit_);
    return core_tile * (repeat_num + 2);
  }

  int64_t CalcLeadDivision(int64_t tile_size, const PropRange &range) {
    int64_t block_size = kernel_->BlockAlign();
    int64_t space = range.space;
    auto ProposalSimdWidth = [this, block_size, space](int64_t tile_num, int64_t &best_cost) -> bool {
      int64_t factor = space / tile_num;
      int64_t last_cost = INT_MAX;
      bool selected = false;
      for (int64_t sw = repeat_size_; sw > 0; sw -= block_size) {
        int64_t align_repeat = CeilDiv(factor, sw);
        int64_t align_factor = align_repeat * sw;
        if (align_factor > tile_size_limit_) continue;
        int64_t cost = CostMeasure2(align_repeat, tile_num * prim_dom_.TileNum());
        if (cost > last_cost) break;
        last_cost = cost;
        if (cost <= best_cost) {
          proposal_sw_ = sw;
          best_cost = cost;
          selected = true;
        }
      }
      return selected;
    };
    int64_t tile_num = GetDivision(space, CeilDiv(tile_size, tile_size_limit_));
    if (tile_num < space && range.affine < PropRange::REDUCE) {
      int64_t best_cost = INT_MAX;
      ProposalSimdWidth(tile_num, best_cost);
      int64_t div_tile = tile_num;
      while (div_tile < space) {
        div_tile = GetDivision(space, div_tile + 1);
        if (!ProposalSimdWidth(div_tile, best_cost)) break;
        tile_num = div_tile;
      }
    }
    return tile_num;
  }

  int64_t CalcLeadTile(int64_t tile_size, const PropRange &range) {
    if (prim_dom_.TileNum() > 1) { // avoid tile range pad
      return CalcLeadDivision(tile_size, range);
    }
    int64_t space = range.space;
    int64_t init_repeat = space > repeat_size_ ? std::min(tile_size_limit_, space) / repeat_size_ : 1L;
    int64_t best_tile = (space -1) / (init_repeat * repeat_size_)+ 1;
    int64_t best_cost = CostMeasure2(init_repeat, best_tile);
    int64_t start_num = best_tile + 1;
    int64_t block_size = kernel_->BlockAlign();
    while (true) {
      int64_t align_tile = CeilDiv(start_num, core_limit_) * core_limit_;
      int64_t factor = CeilDiv(space, align_tile);
      if (factor < repeat_size_ * 4) break;
      start_num = align_tile + core_limit_;
      int64_t last_cost = INT_MAX;
      int64_t selected = 0;
      for (int64_t sw = repeat_size_; sw > 0; sw -= block_size) {
        int64_t repeat = CeilDiv(factor, sw);
        int64_t align_factor = repeat * sw;
        if (align_factor > tile_size_limit_) continue;
        int64_t tile_num = CeilDiv(space, align_factor);
        int64_t cost = CostMeasure2(repeat, tile_num);
        if (cost > last_cost) break;
        last_cost = cost;
        if (cost <= best_cost) {
          best_cost = cost;
          best_tile = tile_num;
          selected = sw;
        }
      }
      if (selected > 0) {
        proposal_sw_ = selected;
        if (best_tile % core_limit_ == 0) break;
      }
    }
    return best_tile;
  }

  VectorKernel* kernel_;
  RootDomain &prim_dom_;
  int64_t tile_size_limit_;
  int64_t repeat_size_;
  int64_t core_limit_;
};

std::string& VKernel::DisAssemble() {
  std::ostringstream oss;
  code_.DisAssemble(oss);
  dump_str_ = oss.str();
  return dump_str_;
}

VectorKernel::~VectorKernel() {
  for (auto &op: build_ops_) {
    delete op;
  }
}

void VectorKernel::DoCodeGen(uint64_t core_limit) {
  int peak_live = Analyze();
  int64_t free_mem = System::Instance().LocalMemSize() - System::Instance().UbWorkspaceSize() - ReserveCodeSize();
  int64_t tile_size_limit = free_mem / (ITEM_SIZE[max_type_] * peak_live);
  // tiling
  ShapeTiling tiling(this, root_dom_, core_limit);
  if (tiles_.empty()) {
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
  tile_num_ = root_dom_.TileNum();
  auto tile_per_block = (tile_num_ + core_limit - 1) / core_limit;
  code_.block_dim_ = (tile_num_ + tile_per_block - 1) / tile_per_block;
  // simd_width
  int64_t lead_dim = root_dom_.DimSpace().front();
  int64_t block_sw = BlockAlign();
  int64_t align_lead_dim = CeilDiv(lead_dim, block_sw) *  block_sw;
  int64_t tile_outer = root_dom_.TileSize() / align_lead_dim;
  int64_t best_repeat;
  if (tiling.proposal_sw_) {
    best_repeat = CeilDiv(lead_dim, tiling.proposal_sw_) * tile_outer;
    simd_width_ = tiling.proposal_sw_;
  } else {
    simd_width_ = block_sw;
    best_repeat = CeilDiv(lead_dim, block_sw) * tile_outer;
    for (int64_t sw = ITEM_SIMD_WIDTH_MAX[max_type_]; sw > block_sw; sw -= block_sw) {
      int64_t repeat = CeilDiv(lead_dim, sw) * tile_outer;
      if (repeat * sw <= tile_size_limit && repeat <= best_repeat) {
        simd_width_ = sw;
        best_repeat = repeat;
      }
    }
  }
  // codegen
  CodeGenHelper helper(*this);
  helper.xbuf_size_ = best_repeat * simd_width_ * ITEM_SIZE[max_type_];
  helper.Generate();
}

void VectorKernel::DumpKernel(std::ostringstream &oss, const std::string &indent) {
  static const char* obj_names[ObjectType::kObjectBulk] = {
    "LoadDummy",
    "Load",
    "PadStore",
    "Store",
    "Reshape",
    "Copy",
    "Unary",
    "Binary",
    "Cast",
    "BinaryS",
    "BroadcastTo",
    "BroadcastS",
    "Reduce",
    "Select",
    "ElemAny",
    "RemovePad"
  };
  static const char* dtype_names[DType::kTypeEnd] = {
    "Bool",
    "Float16",
    "BFloat16",
    "Float32",
    "Int32"
  };
  if (code_.data_ == nullptr) {
    for (size_t i = 0; i < objects_.size(); ++i) {
      objects_[i]->index_ = i;
    }
  }
  auto dump_op = [&oss](NDObject *op) {
    oss << "%" << op->index_<< "[";
    if (!op->nd_.empty()) {
      for (size_t i = 0; i < op->nd_.size() - 1; ++i) {
        oss << op->nd_[i] << ",";
      }
      oss << op->nd_.back();
    }
    oss << "]<" << dtype_names[op->type_id_] << ">";
  };
  oss << indent << "vgraph(tile_num=" << tile_num_ << ", simd_width="<< simd_width_ << ") {" << std::endl;
  std::string body_indent = indent + "  ";
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    oss << body_indent;
    dump_op(op);
    oss << " = " << obj_names[op->GetObjectType()] << "(";
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
    oss << ") // stride=[";
    if (!op->strides_.empty()) {
      for (size_t i = 0; i < op->strides_.size() - 1; ++i) {
        oss << op->strides_[i] << ",";
      }
      oss << op->strides_.back();
    }
    if (op->shape_ref_ != nullptr && op->shape_ref_->size > 0) {
      oss << "], shape_ref=[";
      auto last_idx = op->shape_ref_->size - 1;
      for (size_t i = 0; i < last_idx; ++i) {
        oss << op->shape_ref_->data[i] << ",";
      }
      oss << op->shape_ref_->data[last_idx];
    }
    oss << "]" << std::endl;
  }
  oss << indent << "}";
}

void VectorKernel::CollectMetrics(Metrics &metrics) const {
  ASSERT(code_.data_ != nullptr);
  uint64_t max_xbuf_ = 0;
  for (auto op : objects_) {
    if (op->xbuf_ > max_xbuf_) {
      max_xbuf_ = op->xbuf_;
    }
  }
  NDObject *dom = root_dom_.DomObject();
  metrics.mem_usage = float(max_xbuf_ + dom->strides_.back() * ITEM_SIZE[max_type_]) / float(System::Instance().LocalMemSize()) - ReserveCodeSize();
  uint64_t tile_per_block = CeilDiv(tile_num_, static_cast<uint64_t>(code_.block_dim_));
  metrics.core_usage = float(tile_num_) / float(tile_per_block  * System::Instance().CoreNum());
  uint64_t tiled_shape_size = 1;
  for (auto d : dom->nd_) {
    tiled_shape_size *= d;
  }
  metrics.simd_usage = float(tiled_shape_size) / float(dom->strides_.back() / simd_width_ * ITEM_SIMD_WIDTH_MAX[max_type_]);
}

// lead_dim_ is used only in codegen phase. so we reuse it for liveness analyze
#define OP_GEN_S(op) do { op->lead_dim_ = 2; } while(0)
#define OP_GEN_D(op) do { op->lead_dim_ = 1; } while(0)
#define OP_KILL(op) do { op->lead_dim_ = 0; } while(0)
#define OP_LIVE(op) (op->lead_dim_)
#define OP_LIVE_D(op) (op->lead_dim_ == 1)

static inline bool BinaryInplaceCheck(NDObject *obj) {
  if (obj->obj_id_ == kBinary) {
    auto id = static_cast<BinaryOp *>(obj)->id_;
    if (id != V_POW && id != V_POW_FP16) {
      return true;
    }
  }
  return false;
}

static inline bool LhsInplaceCheck(NDObject *obj) {
  if (obj->obj_id_ == kUnary || obj->obj_id_ == kBinaryS || BinaryInplaceCheck(obj)) {
    return true;
  }
  if (obj->obj_id_ == kCast && obj->type_id_ <= obj->lhs_->type_id_) {
    return true;
  }
  return false;
}

int VectorKernel::Analyze() {
  int op_index = objects_.size();
  int cur_live = static_ops_.size();
  int live_peak = cur_live;
  auto LivenessEnd = [&cur_live](NDObject *op, NDObject *end) {
    if (end->IsSimd() && !OP_LIVE(end)) {
      OP_GEN_D(end);
      return true;
    }
    return false;
  };
  for (auto op : objects_) {  // clear status
    op->lead_dim_ = 0;
  }
  for (auto op : static_ops_) {
    if (op->IsSimd()) {
      OP_GEN_S(op);
    }
  }
  for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
    auto op = *it;
    op->index_ = --op_index;
    op->flags_ = 0;
    if (op->IsSimd()) {
     auto kill = op->lhs_;
      if (kill && LivenessEnd(op, kill)) {
        if (OP_LIVE_D(op) && LhsInplaceCheck(op)) {
          op->flags_ |= OBJ_FLAG_REUSE_LHS;
        } else {
          op->flags_ |= OBJ_FLAG_FREE_LHS;
          cur_live++;
        }
      }
      kill = op->rhs_;
      if (kill && LivenessEnd(op, kill)) {
        if (OP_LIVE_D(op) && !(op->flags_ & OBJ_FLAG_REUSE_LHS) && BinaryInplaceCheck(op)) {
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

void VectorKernel::BuildDomain(const std::vector<NDObject *> &objects) {
  static_ops_.clear();
  max_type_ = objects.front()->type_id_;
  min_type_ = objects.front()->type_id_;
  bool slow_build_path = false;
  for (auto op : objects) {
    auto type = op->GetObjectType();
    if (type == kReshape) {
      slow_build_path = true;
    } else if (type == kCast || op->IsLoad()) {
      int type = op->type_id_;
      if (type > max_type_) {
        max_type_ = type;
      } else if (type < min_type_) {
        min_type_ = type;
      }
    }
    if (op->IsLoad()) {
      static_ops_.push_back(op);
    } else if (op->IsStore()) {
      static_ops_.push_back(op->lhs_);
      op->xbuf_ = -1;
    }
  }
  if (slow_build_path) {
    PropDomainBuilder builder;
    builder.Build(objects, root_dom_);
  } else {
    NDObject *next = nullptr;
    for (auto obj: objects) {
      obj->pd_next_ = next;
      next = obj;
    }
    root_dom_.SetHead(next);
  }
}

NDAccess* VectorKernel::FindInplaceStore(NDAccess *load, const std::function<bool(NDAccess*)> &check) const {
  const static bool elem_objects[ObjectType::kObjectBulk] = {
    true,  // loaddummy
    true,  // load
    false, // padstore
    true,  // store
    true,  // reshape
    true,  // copy
    true,  // unary
    true,  // binary
    true,  // cast
    true,  // binarys
    false, // broadcastto
    true,  // broadcasts
    false, // reduce
    true,  // select
    false, // elementany
    true,  // RemovePad
  };
  auto update_flag = [](int input_flag, bool elem_type, int &flag) {
    // undetermined -> elemwise -> no-elemwise
    //          |___________________|
    if (input_flag != 0 && flag != -1) {
      if (flag == 0) {
        flag = input_flag == 1 && elem_type ? 1 : -1;
      } else if (input_flag == -1 || !elem_type) { // 1
        flag = -1;
      }
    }
  };
  std::vector<int> elem_flags(objects_.size(), 0); // 1: elemwise, 0: undetermined, -1: no-elemwise
  elem_flags[load->index_] = 1;
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    auto &flag = elem_flags[op->index_];
    if (op->lhs_) {
      auto elem_type = elem_objects[op->obj_id_];
      update_flag(elem_flags[op->lhs_->index_], elem_type, flag);
      if (op->rhs_) {
        update_flag(elem_flags[op->rhs_->index_], elem_type, flag);
        if (op->GetObjectType() == kSelect) {
          update_flag(elem_flags[reinterpret_cast<SelectOp*>(op)->cond_->index_], elem_type, flag);
        }
      }
    }
    if (op->IsStore() && flag == 1 && op->type_id_ == load->type_id_ &&
        (check == nullptr || check(static_cast<NDAccess*>(op)))) {
      return static_cast<NDAccess*>(op);
    }
  }
  return nullptr;
}

void VKernelS::Append(NDObject *obj) {
  build_ops_.push_back(obj);
  obj->Normalize(objects_);
  objects_.emplace_back(obj);
}

void VKernelS::Optimize() {
  auto bb = pass::BasicBlock(objects_, build_ops_);
  for (auto pass : pass::passes) {
    pass(bb);
  }
  bb.Export(objects_);
}

uint64_t VKernelS::CodeGen() {
  Optimize();
  BuildDomain(objects_);
  NormalizeDomain();
  DoCodeGen(System::Instance().CoreNum());
  return 0;
}


void VKernelD::RecordOpRelation() {
  for (auto op : objects_) {
    if (op_relations_.find(op) != op_relations_.end()) {
      // already recorded during the first iteration
      continue;
    }
    auto is_select = op->obj_id_ == ObjectType::kSelect;
    bool has_reshape = op->obj_id_ == ObjectType::kReshape ||
                       (op->lhs_ != nullptr && op->lhs_->obj_id_ == ObjectType::kReshape) ||
                       (op->rhs_ != nullptr && op->rhs_->obj_id_ == ObjectType::kReshape) ||
                       (is_select && reinterpret_cast<SelectOp *>(op)->cond_->obj_id_ == ObjectType::kReshape);
    if (has_reshape) {
      auto input_num = is_select ? 3 : 2;
      op_relations_[op].resize(input_num);
      op_relations_[op][0] = op->lhs_;
      op_relations_[op][1] = op->rhs_;
      if (is_select) {
        op_relations_[op][2] = reinterpret_cast<SelectOp *>(op)->cond_;
      }
    }
  }
}

void VKernelD::RecoverOpRelation() {
  for (const auto &item : op_relations_) {
    auto op = item.first;
    op->lhs_ = item.second[0];
    op->rhs_ = item.second[1];
    if (op->obj_id_ == ObjectType::kSelect) {
      reinterpret_cast<SelectOp *>(op)->cond_ = item.second[2];
    }
  }
}

uint64_t VKernelD::CodeGen() {
  objects_.clear();
  code_.Clear();
  if (elim_reshape_) {
    RecoverOpRelation();
    for (auto op : build_ops_) {
      op->Normalize(objects_);
      objects_.emplace_back(op);
    }
    RecordOpRelation();
    pass::BasicBlock bb(objects_, build_ops_); // build_ops_ is not used
    pass::EliminateReshape(bb);
    bb.Export(objects_);
    BuildDomain(objects_);
    NormalizeDomain();
    DoCodeGen(System::Instance().CoreNum());
    return 0;
  }
  if (pd_nexts_.empty()) { // first
    BuildDomain(build_ops_);
    for (auto op : build_ops_) {
      pd_nexts_.push_back(op->pd_next_);
    }
  }
  size_t start = 0;
  for (size_t i = 0; i < build_ops_.size(); ++i) {
    auto op = build_ops_[i];
    op->Normalize(objects_);
    auto pd_next = pd_nexts_[i];
    size_t size = objects_.size();
    if (size > start) {
      for (size_t j = start; j < size; ++j) {
        objects_[j]->pd_next_ = pd_next;
        pd_next = objects_[j];
      }
    }
    objects_.emplace_back(op);
    op->pd_next_ = pd_next;
    start = size + 1;
  }
  NormalizeDomain();
  DoCodeGen(System::Instance().CoreNum());
  return 0;
}

uint64_t VKernelP::CodeGen() {
  auto WorkLoad = [](VKernelS *k) -> uint64_t { return k->root_dom_.TileSize() * k->objects_.size(); };
  uint64_t total_workload = 0;
  for (auto k : children_) {
    k->Optimize();
    k->BuildDomain(k->objects_);
    k->NormalizeDomain();
    total_workload += WorkLoad(k);
  }
  std::sort(children_.begin(), children_.end(), [&WorkLoad](VKernelS *a, VKernelS *b) -> bool { return WorkLoad(a) < WorkLoad(b); });
  uint64_t core_num = System::Instance().CoreNum();
  for (size_t i = 0; i < children_.size(); ++i) {
    auto k = children_[i];
    auto workload = WorkLoad(k);
    // If workload is inbalanced, make sure that each workload occupies at least one core
    uint64_t core_limit = std::max(core_num * workload / total_workload, 1ul);
    k->DoCodeGen(core_limit);
    total_workload -= workload;
    core_num -= k->code_.block_dim_;
  }
  code_.block_dim_ = 0;
  uint64_t code_size = 0;
  for (auto c : children_) {
    code_.block_dim_ += c->code_.block_dim_;
    code_size += ((c->code_.data_size_ - c->code_.HeadSize() + 31) >> 5) << 5;
  }
  uint64_t summary_size = ((code_.block_dim_ * sizeof(uint64_t) + 31) >> 5) << 5;
  code_.data_size_ = code_.HeadSize() + summary_size + code_size;
  code_.Alloc(code_.data_size_);
  uint64_t offset = summary_size;
  uint64_t *summaries = reinterpret_cast<uint64_t*>(code_.data_ + code_.HeadSize());
  uint64_t summary_idx = 0;
  for (size_t k = 0; k < children_.size(); ++k) {
    Code &code = children_[k]->code_;
    auto tile_num = children_[k]->tile_num_;
    ASSERT(tile_num <= 0xffffful);
    // summary
    uint64_t lenburst = (code.data_size_ - code.HeadSize() + 31) / 32;
    uint64_t summary = lenburst << 58 | (offset >> 5) << 49 | children_[k]->simd_width_ << 41;
    uint64_t tile_per_block = (tile_num - 1) / code.block_dim_ + 1;
    uint64_t start_idx = 0;
    for (uint64_t i = 0; i < code.block_dim_ - 1; ++i) {
      summaries[summary_idx++] = summary | tile_per_block << 20 | start_idx;
      start_idx += tile_per_block;
    }
    summaries[summary_idx++] = summary | (tile_num - start_idx) << 20 | start_idx | 1ul << 40;
    // data
    std::vector<NDAccess*> ios;
    for (auto op :  children_[k]->objects_) {
      if (!op->IsSimd()) ios.push_back(static_cast<NDAccess*>(op));
    }
    code_.LinkBody(code_.HeadSize() + offset, code, ios, 0);
    offset += ((code.data_size_ - code.HeadSize() + 31) >> 5) << 5;
  }
  code_.UpdateParallelHead();
  return 0;
}

void VKernelP::DumpKernel(std::ostringstream &oss, const std::string &indent) {
  oss << indent << "vgraph.parallel() {" << std::endl;
  std::string body_indent = indent + "  ";
  for (auto k : children_) {
    k->DumpKernel(oss, body_indent);
    oss << std::endl;
  }
  oss << indent << "}";
}

MixKernel::~MixKernel() {
  if (post_fusion_) delete post_fusion_;
  if (cube_op_) delete cube_op_;
}

void MixKernel::Append(NDObject *obj) {
  const uint32_t LOAD_PENDING = 1;
  if (obj->IsLoad()) {
    obj->flags_ = LOAD_PENDING;
  } else if (obj->obj_id_ == kCubeOp) {
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    std::vector<NDObject*> empty_run_ops;
    obj->lhs_->Normalize(empty_run_ops);
    obj->rhs_->Normalize(empty_run_ops);
    cube_op_ = static_cast<CubeOp*>(obj);
    cube_op_->NormalizeCube();
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    obj->nd_ = cube_op_->nd_;
    cube_op_->output_ = static_cast<NDAccess*>(obj);
  } else {
    if (post_fusion_ == nullptr) {
      post_fusion_ = new VKernelS();
    }
    auto WorkLoad = [this](NDObject *&op) {
      if (op == cube_op_) {
        if (sload_ == nullptr) {
          sload_ = new NDSLoad(nullptr, cube_op_->shape_ref_, cube_op_->type_id_);
          post_fusion_->Append(sload_);
        }
        op = sload_;
      } else if (op->IsLoad() && op->flags_ == LOAD_PENDING) {
        op->flags_ = 0;
        post_fusion_->Append(op);
      }
    };
    if (obj->lhs_) {
      WorkLoad(obj->lhs_);
      if (obj->rhs_) {
        WorkLoad(obj->rhs_);
        if (obj->obj_id_ == kSelect) {
          WorkLoad(static_cast<SelectOp*>(obj)->cond_);
        }
      }
    }
    post_fusion_->Append(obj);
  }
}

uint64_t MixKernel::CodeGen() {
  if (sload_ && cube_op_->output_ == nullptr) {
    cube_op_->output_ = sload_;
  }
  size_t size = code_.HeadSize() + sizeof(vCubeOp);
  vCubeOp cube_code;
  cube_op_->CodeGen(&cube_code);
  code_.block_dim_ = cube_op_->block_dim_;
  cube_code.subtilenum = 0;
  uint64_t head_flags = V_ENTRY_FLAG_MIX;
  uint64_t head_simd = 0;
  NDAccess *inplace_store = nullptr;
  if (post_fusion_) {
    post_fusion_->Optimize();
    post_fusion_->BuildDomain(post_fusion_->objects_);
    post_fusion_->NormalizeDomain();
    if (cube_op_->output_ == sload_) {
      inplace_store = post_fusion_->FindInplaceStore(sload_, nullptr);
      if (inplace_store == nullptr) {
        cube_op_->pingpong_store_ = true;
        cube_code.flags |= V_CUBE_FLAG_PINGPONG_STORE;
      }
    }
    for (auto op : post_fusion_->objects_) {
      if (op->IsLoad()) {
        static_cast<NDSLoad*>(op)->SetCubeOp(cube_op_);
      } else if (op->IsStore()) {
        static_cast<NDSStore*>(op)->SetCubeOp(cube_op_);
      }
    }
    auto m = cube_op_->output_->nd_[1];
    auto n = cube_op_->output_->nd_[0];
    post_fusion_->root_dom_.GroupTile(1, m, cube_code.m0);
    post_fusion_->root_dom_.GroupTile(0, n, cube_code.n0);
    for (size_t i = 2; i < cube_op_->output_->nd_.size(); i++) {
      post_fusion_->root_dom_.GroupTile(i, cube_op_->output_->nd_[i], 1);
    }
    post_fusion_->NormalizeDomain();
    post_fusion_->DoCodeGen(2);
    size += post_fusion_->code_.data_size_;
    uint64_t subtile_0 = (post_fusion_->tile_num_ + 1) / 2;
    uint64_t subtile_1 = post_fusion_->tile_num_ - subtile_0;
    cube_code.subtilenum = subtile_1 << 32 | subtile_0;
    cube_code.flags |= V_CUBE_FLAG_GROUP_SET;
    head_flags |= V_ENTRY_FLAG_PRE_WAIT;
    head_simd = post_fusion_->simd_width_;
  }
  code_.target_ = post_fusion_ ? Code::kTargetMix : Code::kTargetCube;
  code_.data_size_ = size;
  code_.Alloc(size);
  code_.UpdateHead(cube_op_->core_loop_, head_simd, head_flags);
  std::memcpy(code_.data_ + code_.HeadSize(), &cube_code, sizeof(vCubeOp));
  // only support post fusion
  vCubeOp *link_cube = reinterpret_cast<vCubeOp*>(code_.data_ + code_.HeadSize());
  static_cast<NDAccess*>(cube_op_->lhs_)->reloc_addr_ = &link_cube->gm_a;
  static_cast<NDAccess*>(cube_op_->rhs_)->reloc_addr_ = &link_cube->gm_b;
  if (!post_fusion_) {
    cube_op_->output_->reloc_addr_ = &link_cube->gm_c;
    return 0;
  }
  std::vector<NDAccess*> ios;
  for (auto op :  post_fusion_->objects_) {
    if (!op->IsSimd()) ios.push_back(static_cast<NDAccess*>(op));
  }
  code_.LinkBody(code_.HeadSize() + sizeof(vCubeOp), post_fusion_->code_, ios, 0);
  if (inplace_store) {
    code_.reloc_reuse_.emplace_back(&link_cube->gm_c, inplace_store->reloc_addr_);
    code_.reloc_reuse_.emplace_back(cube_op_->output_->reloc_addr_, inplace_store->reloc_addr_);
    return 0;
  } else if (cube_op_->output_->IsStore()) {
    cube_op_->output_->reloc_addr_ = &link_cube->gm_c;
    code_.reloc_reuse_.emplace_back(sload_->reloc_addr_, &link_cube->gm_c);
    return 0;
  } else {
    code_.reloc_workspaces_.emplace_back(&link_cube->gm_c, 0);
    code_.reloc_reuse_.emplace_back(cube_op_->output_->reloc_addr_, &link_cube->gm_c);
    return cube_op_->PostFusionWorkSpace();
  }
  return 0;
}

void MixKernel::DumpKernel(std::ostringstream &oss, const std::string &indent) {
  auto dump_nd = [&oss](const std::vector<int64_t> &nd) {
    oss << "[";
    if (!nd.empty()) {
      for (size_t i = 0; i < nd.size() - 1; ++i) {
        oss << nd[i] << ",";
      }
      oss << nd.back();
    }
    oss << "]";
  };
  oss << indent << "vgraph.mix(tile_num=" << cube_op_->core_loop_ << ") {\n";
  std::string body_indent = indent + "  ";
  oss << body_indent << "// cube" << std::endl;
  oss << body_indent << "%" << cube_op_->index_;
  dump_nd(cube_op_->nd_);
  oss << " = MatMul(%" << cube_op_->lhs_->index_;
  dump_nd(cube_op_->lhs_->nd_);
  oss << ", %" << cube_op_->rhs_->index_;
  dump_nd(cube_op_->rhs_->nd_);
  oss << ")\n";
  if (post_fusion_) {
    oss << body_indent << "// post_fusion" << std::endl;
    post_fusion_->DumpKernel(oss, body_indent);
    oss << std::endl;
  }
  oss << indent << "}";
}

StagesKernel::~StagesKernel() {
  for (auto s : stages_) {
    delete s->kernel;
    delete s;
  }
}

void StagesKernel::Append(NDObject *obj) {
  stages_.back()->kernel->Append(obj);
  if (!obj->IsSimd()) {
    stages_.back()->ios.push_back(static_cast<NDAccess*>(obj));
  }
}

#define STAGE_FLAG_WORKSPACE  1
#define STAGE_FLAG_REUSE      2

uint64_t StagesKernel::CodeGen() {
  constexpr int64_t ffts_size = sizeof(uint64_t);
  uint64_t code_size = ffts_size;
  code_.target_ = Code::kTargetMix;
  code_.block_dim_ = 0;
  for (auto &s : stages_) {
    s->ws_size = s->kernel->CodeGen();
    s->code_offset = code_size;
    auto &stage_code = s->kernel->code_;
    code_size += stage_code.data_size_ - ffts_size;
    if (stage_code.target_ == Code::kTargetVec) {
      auto group_num = (stage_code.block_dim_ + 1) / 2;
      if (group_num > code_.block_dim_) code_.block_dim_ = group_num;
    } else if (stage_code.block_dim_ > code_.block_dim_) {
      code_.block_dim_= stage_code.block_dim_;
    }
    if (!stage_code.atomic_clean_.empty()) {
      for (auto ac : stage_code.atomic_clean_) code_.atomic_clean_.push_back(ac);
    }
  }
  uint64_t ws_size = AllocWorkspace();
  // link
  code_.Alloc(code_size);
  code_.data_size_ = code_size;
  *reinterpret_cast<uint64_t*>(code_.data_) = 0; //ffts
  for (size_t sidx = 0; sidx < stages_.size(); ++sidx) {
    auto stage = stages_[sidx];
    auto &src_code = stage->kernel->code_;
    code_.LinkBody(stage->code_offset + sizeof(uint64_t), src_code, stage->ios, stage->ws_offset);
    auto cur_entry = *reinterpret_cast<uint64_t*>(src_code.data_ + ffts_size);
    if (sidx > 0) { // add sync
      auto pre_code = code_.data_ + stages_[sidx - 1]->code_offset;
      auto pre_entry = *reinterpret_cast<uint64_t*>(pre_code);
      pre_entry |= V_ENTRY_FLAG_NEXT_STAGE;
      if ((pre_entry & V_ENTRY_FLAG_MIX) &&
         !(reinterpret_cast<vCubeOp*>(pre_code + sizeof(uint64_t))->flags & V_CUBE_FLAG_GROUP_SET)) { // cube->vector/cube/mix
        if (!(cur_entry & V_ENTRY_FLAG_MIX)) {
          cur_entry |= V_ENTRY_FLAG_PRE_WAIT;
        }
      } else if (cur_entry & V_ENTRY_FLAG_MIX) { // vector/mix->cube/mix
        auto cube = reinterpret_cast<vCubeOp*>(code_.data_ + stage->code_offset + sizeof(uint64_t));
        cube->flags |= V_CUBE_FLAG_PRE_WAIT;
      }
      *reinterpret_cast<uint64_t*>(pre_code) = pre_entry;
    }
    *reinterpret_cast<uint64_t*>(code_.data_ + stage->code_offset) = cur_entry;
  }
  for (auto stage : stages_) {
    for (auto op : stage->ios) {
      if (op->is_stage_) {
        if (op->IsStore()) {
          if (op->flags_ == STAGE_FLAG_REUSE) {
            auto reuse = op->GetOutputReuse();
            code_.reloc_reuse_.emplace_back(op->reloc_addr_, reuse->reloc_addr_);
          } else {
            auto offset = op->GetWorkspace();
            code_.reloc_workspaces_.emplace_back(op->reloc_addr_, offset);
          }
        } else {
          code_.reloc_reuse_.emplace_back(op->reloc_addr_, op->GetStageStore()->reloc_addr_);
        }
      }
    }
  }
  return ws_size;
}

uint64_t StagesKernel::AllocWorkspace() {
  struct Group {
    Group(uint64_t s, bool l) : size(s), live(l) {}
    std::vector<NDAccess*> ops;
    std::vector<Stage*> wss;
    uint64_t size;
    bool live;
  };
  std::vector<Group> groups;
  groups.reserve(stages_.size() * 8);
  std::unordered_map<NDAccess*, int> lives; // >= 0: group_idx. -1: reused
  auto select_group = [&groups](uint64_t size) -> int {
    int up = -1, down = -1;
    for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
      auto &g = groups[i];
      if (g.live) continue;
      if (g.size >= size) {
        if (up == -1 || g.size < groups[up].size) {
          up = i;
        }
      } else if (up == -1 && (down == -1|| g.size > groups[down].size)) {
        down = i;
      }
    }
    return up >= 0 ? up : down;
  };
  for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) {
    auto stage = *it;
    // stage buffer gen
    for (auto io : stage->ios) {
      if (io->IsLoad() && io->is_stage_) {
        auto store = io->GetStageStore();
        if (lives.find(store) != lives.end()) continue;
        if (stage->kernel->KType() == kStaticShape) { // TODO: parallel fusion
          NDAccess *inplace_stage = nullptr;
          auto inplace_out = static_cast<VectorKernel*>(stage->kernel)->FindInplaceStore(io,
            [&lives, &inplace_stage](NDAccess *op) -> bool {
              if (!op->is_stage_ || op->flags_ == STAGE_FLAG_REUSE) {
                return true;
              }
              if (inplace_stage == nullptr && lives[op] >= 0) {
                inplace_stage = op;
              }
              return false;
          });
          if (inplace_out) {
            store->flags_ = STAGE_FLAG_REUSE;
            store->SetOutputReuse(inplace_out->is_stage_ ? inplace_out->GetOutputReuse() : inplace_out);
            lives[store] = -1;
            continue;
          }
          if (inplace_stage) {
            auto inplace_it = lives.find(inplace_stage);
            groups[inplace_it->second].ops.push_back(store);
            lives[store] = inplace_it->second;
            inplace_it->second = -1;
            continue;
          }
        }
        auto size = store->Size();
        int index = select_group(size);
        if (index == -1) {
          index = groups.size();
          groups.emplace_back(size, true);
        }
        groups[index].ops.emplace_back(store);
        lives[store] = index;
      }
    }
    if (stage->ws_size > 0) {
      auto index = select_group(stage->ws_size);
      if (index == -1) {
        index = groups.size();
        groups.emplace_back(stage->ws_size, false);
      }
      groups[index].wss.push_back(stage);
    }
    // stage buffer kill
    for (auto io : stage->ios) {
      if (io->IsStore() && io->is_stage_) {
        auto live_it = lives.find(io);
        if (live_it->second >= 0) {
          groups[live_it->second].live = false;
        }
        lives.erase(live_it);
      }
    }
  }
  uint64_t workspace_size = 0;
  for (auto &g : groups) {
    for (auto op : g.ops) {
      op->flags_ = STAGE_FLAG_WORKSPACE;
      op->SetWorkspace(workspace_size);
    }
    for (auto stage : g.wss) {
      stage->ws_offset = workspace_size;
    }
    workspace_size += g.size;
  }
  return workspace_size;
}

void StagesKernel::DumpKernel(std::ostringstream &oss, const std::string &indent) {
  oss << indent << "vgraph.stages() {\n";
  int stage_idx = 0;
  std::string body_indent = indent + "  ";
  for (auto &s : stages_) {
    oss << body_indent << "// stage " << stage_idx << std::endl;
    stage_idx++;
    s->kernel->DumpKernel(oss, body_indent);
    oss << std::endl;
  }
  oss << indent << "}";
}

} // namespace dvm
