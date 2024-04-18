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
static const uint64_t ITEM_SIMD_WIDTH_MAX[kTypeEnd] = {128, 128, 128, 64, 64};

constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t CONST_512 = 512;

inline __attribute__((always_inline)) uint32_t RoundUp(uint32_t num, uint32_t rnd) {
  if (rnd == 0) {
      return 0;
  }
  return (num + rnd - 1) / rnd * rnd;
}

inline __attribute__((always_inline)) uint32_t RoundDown(uint32_t num, uint32_t rnd) {
  if (rnd == 0) {
    return 0;
  }
  return num / rnd * rnd;
}

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

  CodeGenHelper(VKernelBase *kernel): kernel_(kernel) {}
  bool Generate() {
    static const CodeGenType codegen_types[ObjectType::kObjectBulk] = {
      kGenLoad,  // loaddummy
      kGenLoad,  // load
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
    auto &code = kernel_->code_;
    auto code_reserved = kernel_->ReserveCodeSize();
    code.Alloc(code_reserved + code.HeadSize());
    uint64_t *code_ptr = reinterpret_cast<uint64_t*>(code.data_ + code.HeadSize());
    static_xbuf_ = DeviceInfo::Instance().UbWorkspaceSize() + code_reserved;
    for (auto op : kernel_->static_ops_) {
      op->xbuf_ = static_xbuf_;
      static_xbuf_ += xbuf_size_;
    }
    for (auto op: kernel_->objects_) {
      op->UpdateStride(code.simd_width_);
      op->tail_insn_ = op->insn_ = code_ptr;
      switch (codegen_types[op->obj_id_]) {
        case kGenSimd0: {
          auto anti_dep = op->xbuf_ == 0 ? AllocDynXBuf(op) : nullptr;
          code_ptr += op->Emit(code);
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
          code_ptr += op->Emit(code);
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
          code_ptr += op->Emit(code);
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
          code_ptr += op->Emit(code);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          SelectOpPostProc(op);
          break;
        }
        case kGenLoad: {
          code_ptr += op->Emit(code);
          break;
        }
        case kGenStore: {
          code_ptr += op->Emit(code);
          StoreSync(op->lhs_, op);
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
    if (DeviceInfo::Instance().Arch() == kAiCore_C100) {
      OverWriteCoreLimit();
    }
    code.FillHead();
    return true;
  }

 private:
  void BackwardSync() {
    auto &objects = kernel_->objects_;
    EventManager vl_event, sv_event;
    auto alloc_event = [](EventManager &m, uint64_t &event) -> bool {
      event = m.hold_event + 1;
      if (event >= DeviceInfo::Instance().EventNum()) {
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
            auto to_sync = kernel_->objects_[sv_event.sync_idx]->insn_;
            *to_sync &= ~(0x1ul << V_HEAD_BACK_WAIT_OFFSET);
          }
          *(simd->insn_) |= 1ul << V_HEAD_BACK_WAIT_OFFSET | event << V_HEAD_B_WAIT_EVENT_OFFSET;
          sv_event.sync_idx = simd->index_;
        }
      } else if (pipe == V_PIPE_SIMD && op->lhs_) { // SIMD -> LOAD
        NDObject *load = nullptr;
        if (op->lhs_->Pipe() == V_PIPE_LOAD) load = op->lhs_;
        auto rhs = op->rhs_;
        if (rhs) {
          if (rhs->Pipe() == V_PIPE_LOAD && (load == nullptr || rhs->index_ < load->index_)) load = rhs;
          if (op->obj_id_ == ObjectType::kSelect) {
            NDObject *cond = reinterpret_cast<SelectOp*>(op)->cond_;
            if (cond->Pipe() == V_PIPE_LOAD && (load == nullptr || cond->index_ < load->index_)) load = cond;
          }
        }
        if (load != nullptr && load->index_ < vl_event.sync_idx) {
          uint64_t event;
          if (alloc_event(vl_event, event)) {
            *(op->tail_insn_) |= 1ul << V_HEAD_BACK_SET_OFFSET | event << V_HEAD_B_SET_EVENT_OFFSET;
          } else {
            auto to_sync = kernel_->objects_[vl_event.sync_idx]->insn_;
            *to_sync &= ~(0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET);
          }
          *(load->insn_) |= 1ul << V_M_HEAD_WAIT_FLAG_OFFSET | event << V_M_HEAD_WAIT_EVENT_OFFSET;
          vl_event.sync_idx = load->index_;
        }
      }
    }
  }

  void OverWriteCoreLimit() {
    for (auto op : kernel_->static_ops_) {
      if (op->obj_id_ <= kLoad || op->obj_id_ == kElementAny || (op->obj_id_ == kReduce && static_cast<ReduceOp*>(op)->factor_ > 1)) {
        continue;
      }
      // producer node for Store
      uint64_t size = op->strides_.back() / op->LeadAlign() * op->nd_[op->lead_dim_] * ITEM_SIZE[op->type_id_];
      if (size < SIMD_BLOCK_SIZE) {
        kernel_->code_.ApplyTileLimit(CeilDiv(SIMD_BLOCK_SIZE, size));
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

  inline void SimdBarrier(NDObject *from, NDObject *to) {
    if (from->index_ >= vector_vector_sync) {
      *(to->insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      vector_vector_sync = to->index_;
    }
  }

  void SimdSync(NDObject *from, NDObject *to) {
    if (from->Pipe() == V_PIPE_SIMD) {
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
      auto from_sync = kernel_->objects_[lv_event_.sync_idx]->tail_insn_;
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
      auto from_sync = kernel_->objects_[vs_event_.sync_idx]->tail_insn_;
      *from_sync &= ~(0x1ul << V_HEAD_SET_FLAG_OFFSET);
    }
    *from_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | event << V_HEAD_SET_EVENT_OFFSET;
    vs_event_.sync_idx = from_pipe_idx;
  }

  inline bool AllocForwardEvent(EventManager &m, int from_idx, int to_idx, uint64_t &event) {
    int total = DeviceInfo::Instance().EventNum();
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

  VKernelBase *kernel_;
  friend VKernelBase;
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
    if (obj_type == kStore || obj_type == kReduce || obj_type == kElementAny) {
      if (op->lhs_ != dom_) cand = op->lhs_;
    } else if (obj_type == kBroadcastTo) {
      if (op != dom_) cand = op;
    }
    if (cand) {
      if (dom_ == nullptr || select_dom(cand)) dom_ = cand;
    } else if (dom_ == nullptr && obj_type != kStore && obj_type != kLoadDummy) {
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
  tp.tile = CeilDiv(space, num);
  tp.tail = space % tp.tile;
  PropDomain::TileProp(tp);
  if (start > 0) {
    tile_size_ = tile_size_ / space * CeilDiv(space, num);
  } else {
    auto align_size = CeilDiv<int64_t>(tp.tile, block_align_) * block_align_;
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
        int64_t num = CalcLeadTile(tile_size, prim_dom_.align_);
        if (num > 1) {
          tile_size = prim_dom_.Tile(0, fold.base, prim_dom_.align_.space, num);
          return;
        }
      }
    } while(tile_size > tile_size_limit_);
    if (align_depth > 1) {
      prim_dom_.Tile(0, align_depth - 1, prim_dom_.align_.space, 1);
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

  VKernelBase* kernel_;
  RootDomain &prim_dom_;
  int64_t tile_size_limit_;
  int64_t repeat_size_;
  int64_t core_limit_;
};

std::string& VKernel::DisAssemble() {
  std::ostringstream oss;
  code_ptr_->DisAssemble(oss);
  dump_str_ = oss.str();
  return dump_str_;
}

VKernelBase::~VKernelBase() {
  for (auto &op: build_ops_) {
    delete op;
  }
}

void VKernelBase::DoCodeGen(uint64_t core_limit) {
  int peak_live = Analyze();
  int64_t free_mem = DeviceInfo::Instance().LocalMemSize() - DeviceInfo::Instance().UbWorkspaceSize() - ReserveCodeSize();
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
  code_.Reset();
  code_.tile_num_ = root_dom_.TileNum();
  code_.UpdateBlockDim(core_limit);
  // simd_width
  int64_t lead_dim = root_dom_.DimSpace().front();
  int64_t block_sw = BlockAlign();
  int64_t align_lead_dim = CeilDiv(lead_dim, block_sw) *  block_sw;
  int64_t tile_outer = root_dom_.TileSize() / align_lead_dim;
  int64_t best_repeat;
  if (tiling.proposal_sw_) {
    best_repeat = CeilDiv(lead_dim, tiling.proposal_sw_) * tile_outer;
    code_.simd_width_ = tiling.proposal_sw_;
  } else {
    code_.simd_width_ = block_sw;
    best_repeat = CeilDiv(lead_dim, block_sw) * tile_outer;
    for (int64_t sw = ITEM_SIMD_WIDTH_MAX[max_type_]; sw > block_sw; sw -= block_sw) {
      int64_t repeat = CeilDiv(lead_dim, sw) * tile_outer;
      if (repeat * sw <= tile_size_limit && repeat <= best_repeat) {
        code_.simd_width_ = sw;
        best_repeat = repeat;
      }
    }
  }
  // codegen
  CodeGenHelper helper(this);
  helper.xbuf_size_ = best_repeat * code_.simd_width_ * ITEM_SIZE[max_type_];
  helper.Generate();
}

void VKernelBase::DumpKernel(std::ostringstream &oss) {
  static const char* obj_names[ObjectType::kObjectBulk] = {
    "LoadDummy",
    "Load",
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
  oss << "vgraph(tile_num=" << code_.tile_num_ << ", simd_width="<<code_.simd_width_ << ") {" << std::endl;
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    oss << "  ";
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
  oss << "}";
}

void VKernelBase::CollectMetrics(Metrics &metrics) const {
  ASSERT(code_.data_ != nullptr);
  uint64_t max_xbuf_ = 0;
  for (auto op : objects_) {
    if (op->xbuf_ > max_xbuf_) {
      max_xbuf_ = op->xbuf_;
    }
  }
  NDObject *dom = root_dom_.DomObject();
  metrics.mem_usage = float(max_xbuf_ + dom->strides_.back() * ITEM_SIZE[max_type_]) / float(DeviceInfo::Instance().LocalMemSize()) - ReserveCodeSize();
  uint64_t tile_per_block = CeilDiv(code_.tile_num_, code_.block_dim_);
  metrics.core_usage = float(code_.tile_num_) / float(tile_per_block  * DeviceInfo::Instance().CoreNum());
  uint64_t tiled_shape_size = 1;
  for (auto d : dom->nd_) {
    tiled_shape_size *= d;
  }
  metrics.simd_usage = float(tiled_shape_size) / float(dom->strides_.back() / code_.simd_width_ * ITEM_SIMD_WIDTH_MAX[max_type_]);
}

// lead_dim_ is used only in codegen phase. so we reuse it for liveness analyze
#define OP_GEN_S(op) do { op->lead_dim_ = 2; } while(0)
#define OP_GEN_D(op) do { op->lead_dim_ = 1; } while(0)
#define OP_KILL(op) do { op->lead_dim_ = 0; } while(0)
#define OP_LIVE(op) (op->lead_dim_)
#define OP_LIVE_D(op) (op->lead_dim_ == 1)
int VKernelBase::Analyze() {
  int op_index = objects_.size();
  int cur_live = static_ops_.size();
  int live_peak = cur_live;
  auto LivenessEnd = [&cur_live](NDObject *op, NDObject *end) {
    if (end->Pipe() == V_PIPE_SIMD && !OP_LIVE(end)) {
      OP_GEN_D(end);
      return true;
    }
    return false;
  };
  for (auto op : static_ops_) {
    if (op->Pipe() == V_PIPE_SIMD) {
      OP_GEN_S(op);
    }
  }
  for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
    auto op = *it;
    op->index_ = --op_index;
    op->flags_ = 0;
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

void VKernelBase::BuildDomain(const std::vector<NDObject *> &objects) {
  max_type_ = objects.front()->type_id_;
  min_type_ = objects.front()->type_id_;
  bool slow_build_path = false;
  for (auto op : objects) {
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
    if (op->Pipe() == V_PIPE_LOAD) {
      static_ops_.push_back(op);
    } else if (op->Pipe() == V_PIPE_STORE) {
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
  objects_ = bb.ToVector();
  bb.Clear();
}

void VKernelS::CodeGen() {
  Optimize();
  BuildDomain(objects_);
  NormalizeDomain();
  DoCodeGen(DeviceInfo::Instance().CoreNum());
  EXCEPTION_IF(code_.data_size_ > 4096, "kernel code size exceed limit(4096)");
}

void VKernelD::CodeGen() {
  if (pd_nexts_.empty()) { // first
    BuildDomain(build_ops_);
    for (auto op : build_ops_) {
      pd_nexts_.push_back(op->pd_next_);
    }
  }
  objects_.clear();
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
  DoCodeGen(DeviceInfo::Instance().CoreNum());
}

void VKernelP::CodeGen() {
  auto WorkLoad = [](VKernelS *k) -> uint64_t { return k->root_dom_.TileSize() * k->objects_.size(); };
  uint64_t total_workload = 0;
  for (auto k : children_) {
    k->Optimize();
    k->BuildDomain(k->objects_);
    k->NormalizeDomain();
    total_workload += WorkLoad(k);
  }
  std::sort(children_.begin(), children_.end(), [&WorkLoad](VKernelS *a, VKernelS *b) -> bool { return WorkLoad(a) < WorkLoad(b); });
  uint64_t core_num = DeviceInfo::Instance().CoreNum();
  for (size_t i = 0; i < children_.size(); ++i) {
    auto k = children_[i];
    auto workload = WorkLoad(k);
    // If workload is inbalanced, make sure that each workload occupies at least one core
    uint64_t core_limit = std::max(core_num * workload / total_workload, 1ul);
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
        load->reloc_addr_ = new_base + (load->reloc_addr_ - old_base);
      } else if (op->obj_id_ == kStore) {
        auto store = static_cast<NDStore*>(op);
        store->reloc_addr_ = new_base + (store->reloc_addr_ - old_base);
      }
    }
  }
  EXCEPTION_IF(code_.data_size_ > 4096, "kernel code size exceed limit(4096)");
}

void VKernelP::DumpKernel(std::ostringstream &oss) {
  oss << "vgraph.parallel() {" << std::endl;
  for (auto k : children_) {
    k->DumpKernel(oss);
    oss << std::endl;
  }
  oss << "}";
}

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b)
  : NDObject(lhs, rhs, lhs->type_id_, kCubeOp), trans_a_(trans_a), trans_b_(trans_b) {
  m_ = trans_a ? lhs->nd_[0] : lhs->nd_[1];
  k_ = trans_a ? lhs->nd_[1] : lhs->nd_[0];
  n_ = trans_b ? rhs->nd_[1] : rhs->nd_[0];
  if (lhs->nd_.size() == 2 && rhs->nd_.size() == 2) {
    nd_ = {n_, m_};
    shape_ = {m_, n_};
  } else {
    int64_t batch = lhs->nd_.size() == 3 ? lhs->nd_[2] : rhs->nd_[2];
    nd_ = {n_, m_, batch};
    shape_ = {batch, m_, n_};
  }
  shape_ref_data_ = shape_;
}

float CubeOp::CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0) {
  float a_coef = 1.0f;
  float b_coef = 1.0f;
  float bw_coef = 5.0f;
  auto m_loop = CeilDiv(op->m, m0);
  auto n_loop = CeilDiv(op->n, n0);
  if (m_loop == 0 || n_loop == 0) {
    return 1.0f;
  }
  auto core_need = m_loop * n_loop;
  auto core_num = DeviceInfo::Instance().CoreNum(CoreType::kCube);
  auto l2_num = DeviceInfo::Instance().L2Size() / ITEM_SIZE[type_id_];
  uint32_t block_dim = core_need < core_num ? core_need : core_num;
  uint32_t m_once = block_dim < n_loop ? m0 : block_dim / n_loop * m0;
  uint32_t n_once = block_dim < n_loop ? core_num * n0 : op->n;
  if (m_once * op->k > l2_num) {
      a_coef = bw_coef;
  }
  if (n_once * op->k > l2_num) {
      b_coef = bw_coef;
  }
  return 1.0f / (a_coef * static_cast<float>(n0)) + 1.0f / (b_coef * static_cast<float>(m0));
}

void CubeOp::Tile(vCubeOp *op) {
  auto pri_flag = m_ < n_ ? false : true;
  auto m_round = RoundUp(static_cast<uint32_t>(m_), BLOCK_SIZE);
  auto n_round = RoundUp(static_cast<uint32_t>(n_), BLOCK_SIZE);
  auto pri_axis = pri_flag ? m_round : n_round;
  auto axis = pri_flag ? n_round : m_round;
  auto axis_max = AXES_ALIGN_SIZE / ITEM_SIZE[type_id_];
  auto pri_axis0_max = pri_axis < axis_max ? pri_axis : axis_max;
  auto axis0_max = axis < axis_max ? axis : axis_max;
  auto l0c_num = DeviceInfo::Instance().L0CSize() / FP32_SIZE;
  uint32_t pri_axis0_init = BLOCK_SIZE;
  uint32_t axis0_init = BLOCK_SIZE;
  float min_cost = 1.0f;
  // m0, n0
  for (uint32_t pri_axis0 = pri_axis0_init; pri_axis0 <= pri_axis0_max; pri_axis0 *= 2) {
    for (uint32_t axis0 = axis0_init; axis0 <= axis0_max; axis0 *= 2) {
      if (pri_axis0 * axis0 > l0c_num) {
        break;
      }
      auto m0 = pri_flag ? pri_axis0 : axis0;
      auto n0 = pri_flag ? axis0 : pri_axis0;
      auto cost = CostFunc(op, m0, n0);
      if (cost < min_cost) {
        min_cost = cost;
        op->m0 = m0;
        op->n0 = n0;
      }
    }
  }
  // k0
  uint32_t cubeBlockSize = CUBE_BLOCK_SIZE;
  uint32_t kBlockSize = BLOCK_SIZE;
  auto l1_ping_pong_num = DeviceInfo::Instance().L1Size() / 2 / ITEM_SIZE[type_id_];
  auto k0_max = l1_ping_pong_num / (op->m0 + op->n0);
  op->k0 = k0_max < cubeBlockSize ? RoundDown(k0_max, kBlockSize) : RoundDown(k0_max, cubeBlockSize);
  if (op->k0 > CONST_512) {
    op->k0 = RoundDown(op->k0, CONST_512);
  }
  if (op->k0 > op->k) {
    op->k0 = op->k;
  }
}

void CubeOp::GetSwizzleConfig(vCubeOp *op) {
  uint32_t swizzle_cnt = 1;
  uint32_t swizzle_dir = 0;
  float mincost = op->m * op->k + op->k * op->n;
  for (size_t i = 1; i <= block_dim_; i++) {
    uint32_t c = (block_dim_ + i - 1) / i;
    uint32_t mem_a_zN = c * op->m0 * op->k;
    uint32_t mem_b_zN = i * op->n0 * op->k;
    uint32_t mem_a_nZ = c * op->n0 * op->k;
    uint32_t mem_b_nZ = i * op->m0 * op->k;
    float cost;
    if (mem_a_zN + mem_b_zN < mem_a_nZ + mem_b_nZ) {
      swizzle_dir = 1; // zN
      cost = mem_a_zN + mem_b_zN;
      if (cost <= mincost) {
          mincost = cost;
          swizzle_cnt = i;
      }
    } else {
      swizzle_dir = 0; // nZ
      cost = mem_a_nZ + mem_b_nZ;
      if (cost < mincost) {
          mincost = cost;
          swizzle_cnt = i;
      }
    }
  }
  op->swizzle = swizzle_dir << 16 | swizzle_cnt;
}

void CubeOp::CodeGen(vCubeOp *op) {
  op->m = m_;
  op->n = n_;
  op->k = k_;
  op->gm_a = reinterpret_cast<uint64_t>(static_cast<NDLoad*>(lhs_)->src_);
  op->gm_b = reinterpret_cast<uint64_t>(static_cast<NDLoad*>(rhs_)->src_);
  op->gm_c = reinterpret_cast<uint64_t>(static_cast<NDStore*>(output_)->dst_);
  op->transpose = trans_a_ << 16 | trans_b_;
  Tile(op);
  auto m_loop = CeilDiv(op->m, op->m0);
  auto n_loop = CeilDiv(op->n, op->n0);
  core_loop_ = m_loop * n_loop;
  auto core_num = DeviceInfo::Instance().CoreNum(CoreType::kCube);
  block_dim_ = core_loop_ < core_num ? core_loop_ : core_num;
  GetSwizzleConfig(op);
}

MixKernel::~MixKernel() {
  if (pre_fusion_) delete pre_fusion_;
  if (post_fusion_) delete post_fusion_;
  if (cube_op_) delete cube_op_;
}

void MixKernel::Append(NDObject *obj) {
  if (obj->obj_id_ == kCubeOp) {
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    cube_op_ = static_cast<CubeOp*>(obj);
    return;
  }
  if (cube_op_ == nullptr) {
    if (pre_fusion_ == nullptr) {
      pre_fusion_ = new VKernelS();
    }
    pre_fusion_->Append(obj);
  } else {
    if (post_fusion_ == nullptr) {
      post_fusion_ = new VKernelS();
    }
    post_fusion_->Append(obj);
    if (obj->obj_id_ == kStore && obj->lhs_ == cube_op_) {
      cube_op_->output_ = obj;
    }
  }
}

void MixKernel::CodeGen() {
  size_t size = sizeof(uint64_t) + sizeof(vCubeOp);
  code_.Alloc(size);
  code_.target_ = CodeBase::kTargetCube;
  code_.data_size_ = size;
  cube_op_->CodeGen(reinterpret_cast<vCubeOp*>(code_.data_ + sizeof(uint64_t)));
  code_.block_dim_ = cube_op_->block_dim_;
  auto *ptr = reinterpret_cast<uint64_t*>(code_.data_);
  *ptr = (cube_op_->core_loop_ - 1) << 40;
}

void MixKernel::DumpKernel(std::ostringstream &oss) {
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
  oss << "vgraph.mix() {\n// pre_fusion" << std::endl;
  pre_fusion_->DumpKernel(oss);
  oss << std::endl;
  oss << "// cube\n%" << cube_op_->output_->index_;
  dump_nd(cube_op_->output_->nd_);
  oss << " = MatMul(%" << cube_op_->lhs_->index_;
  dump_nd(cube_op_->lhs_->nd_);
  oss << ", %" << cube_op_->rhs_->index_;
  dump_nd(cube_op_->rhs_->nd_);
  oss << ")\n// post_fusion" << std::endl;
  post_fusion_->DumpKernel(oss);
  oss << std::endl;
  oss << "}";
}
} // namespace dvm
