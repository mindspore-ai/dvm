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

#include <unordered_map>
#include <climits>
#include <memory>
#include <cstring>
#include "kernel.h"
#include "xkernel.h"
#include "msprof.h"

namespace dvm {
class IdleCleanWrap : public CodeWrap {
 public:
  ~IdleCleanWrap() override { Clear(); }

  void CodeGen(const std::vector<NDObject *> &cleans) {
    Clear();
    for (auto op : cleans) {
      auto store = static_cast<NDAccess *>(op);
      auto k = new VKernelD();
      KernelBuilder b(k);
      auto broadcast = b.Broadcast(0, store->shape_ref_, store->type_id_);
      auto out = static_cast<NDAccess *>(b.Store(store->addr_.gm, broadcast));
      k->CodeGen();
      auto &addr = store->addr_;
      addr.Update(&addr.data);
      k->code_.BindOpFast(out->addr_, addr);
      kernels_.push_back(k);
    }
  }

  void Clear() {
    if (!kernels_.empty()) {
      for (auto k : kernels_) {
        delete k;
      }
      kernels_.clear();
    }
  }

  int LaunchWrap(void *workspace, void *stream) override {
    for (auto k : kernels_) {
      k->code_.RelocBinds(nullptr);
      k->code_.Launch(nullptr, stream);
    }
    return 0;
  }

  void DasWrap(std::ostringstream &oss) override {
    for (auto k : kernels_) {
      k->code_.DisAssemble(oss);
    }
    oss << "\nvmain.idle() {}";
  }

  void CollectWrap(std::vector<Code *> &codes) override {
    for (auto k : kernels_) {
      k->code_.Collect(codes);
    }
  }

 private:
  std::vector<VKernelD *> kernels_;
};

class IdleCodeWrap : public CodeWrap {
 public:
  int LaunchWrap(void *workspace, void *stream) override { return 0; }
  void DasWrap(std::ostringstream &oss) override {
    oss << "vmain.idle() {}";
  }
};


VKernel::~VKernel() {
  delete idle_clean_wrap_;
  delete msprof_;
}

void VKernel::UpdateIdle(const std::vector<NDObject *> &cleans) {
  static IdleCodeWrap idle_code_wrap;
  if (cleans.empty()) {
    code_.InsertWrap(&idle_code_wrap);
  } else {
    if (idle_clean_wrap_ == nullptr) {
      idle_clean_wrap_ = new IdleCleanWrap();
    }
    idle_clean_wrap_->CodeGen(cleans);
    code_.InsertWrap(idle_clean_wrap_);
  }
}

void VKernel::Normalize() { pre_ws_size_ = CodeGen(); }

void VKernel::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  auto reloc = relocs;
  for (size_t i = 0; i < reloc_size; ++i, ++reloc) {
    static_cast<NDAccess *>(reloc->io)->addr_.Reloc(reloc->addr);
  }
  if (g_system.enable_profile_) {
    if (msprof_ == nullptr) {
      msprof_ = new MsprofHelper();
      auto &info = msprof_->info_;
      info.op_name = op_name_ ? op_name_ : "UnnamedDvmOp";
      info.op_fullname = op_fullname_ ? op_fullname_ : "UnnamedDvmOp";
      info.block_dim =code_.block_dim_;
      for (size_t i = 0; i < reloc_size; ++i) {
        auto op = relocs[i].io;
        if (op->IsLoad()) {
          info.AppendInput(op);
        } else {
          info.AppendOutput(op);
        }
      }
      msprof_->InitReportNode();
    } else if (IsDynamic()) {
      msprof_->UpdateReportNode(code_.block_dim_);
    }
  }
  if (ws_alloc) {
    pre_ws_mem_ = ws_alloc->Alloc(pre_ws_size_);
  }
}

int VKernel::Launch(void *stream) {
  code_.RelocBinds(pre_ws_mem_);
  if (likely(!g_system.enable_profile_ || msprof_ == nullptr)) {
    return code_.Launch(pre_ws_mem_, stream);
  } else {
    struct _MsprofLaunchGuard : public CodeLaunchGuard {
      _MsprofLaunchGuard(Code &code, MsprofHelper *helper) : CodeLaunchGuard(code), helper_(helper) {}


      int CodeLaunch(Code *code, void *workspace, void *stream) {
        helper_->Update(code->target_);
        int ret = code->DoLaunch(workspace, stream);
        helper_->ReportTask();
        return ret;
      }
      MsprofHelper *helper_;
    };
    _MsprofLaunchGuard guard(code_, msprof_);
    return code_.Launch(pre_ws_mem_, stream);
  }
}

void VKernel::Clone(VKernel *base, CloneHelper &helper) { DvmException("unsupport clone kernel"); }

struct XbufPool {
  using Node = std::pair<uint64_t, NDObject *>;
  enum { BUILTIN_POOL_SIZE = 16 };

  XbufPool(uint32_t size) {
    if (likely(size <= BUILTIN_POOL_SIZE)) {
      pool_ = builtin_pool_;
      size_mask_ = BUILTIN_POOL_SIZE - 1;
    } else {
      size = 1u << (32 - __builtin_clz(size - 1));
      pool_ = new Node[size];
      size_mask_ = size - 1;
    }
  }
  ~XbufPool() {
    if (size_mask_ > BUILTIN_POOL_SIZE) {
      delete[] pool_;
    }
  }
  void Push(uint64_t xbuf, NDObject *user) {
    auto &node = pool_[(prod_++) & size_mask_];
    node.first = xbuf;
    node.second = user;
  }
  Node &Pop() { return pool_[(cons_++) & size_mask_]; }
  Node &Front() const { return pool_[cons_ & size_mask_]; }
  bool Empty() const { return prod_ == cons_; }

  Node *pool_;
  uint32_t size_mask_;
  uint32_t prod_{0};
  uint32_t cons_{0};
  Node builtin_pool_[BUILTIN_POOL_SIZE];
};

class CodeGenHelper {
 public:
  struct EventManager {
    enum { MAX_EVENT_NUM = 8 };
    int hold_idx[MAX_EVENT_NUM]{0};
    int hold_event{-1};
    int sync_idx{-1};
  };

  CodeGenHelper(VectorKernel &kernel, uint32_t xbuf_size, uint32_t pool_size)
      : free_xbuf_(pool_size), xbuf_size_(xbuf_size), kernel_(kernel) {}
  uint8_t *Generate(uint8_t *code_begin, uint64_t code_reserve, uint32_t tile_size) {
    uint64_t *code_ptr = reinterpret_cast<uint64_t *>(code_begin);
    static_xbuf_ = code_reserve;
    for (size_t i = 0; i < kernel_.static_ops_.size(); ++i) {
      auto op = kernel_.static_ops_[i];
      if (i < kernel_.load_num_) {
        op->xbuf_ = static_xbuf_;
        if (op->obj_id_ == kMultiLoad) {
          static_xbuf_ += xbuf_size_ + xbuf_size_;
          static_cast<NDMultiLoad *>(op)->xbuf_size_ = xbuf_size_;
        }
      } else {
        op->lhs_->xbuf_ = static_xbuf_;
      }
      if (op->type_id_ != kernel_.max_type_) {
        static_xbuf_ += tile_size * ITEM_SIZE[op->type_id_];
      } else {
        static_xbuf_ += xbuf_size_;
      }
    }
    if (auto comm = kernel_.comm_op_) {
      for (int i = 0; i < comm->XbufReserve(); ++i) {
        comm->xbufs_.push_back(static_xbuf_);
        static_xbuf_ += xbuf_size_;
      }
      comm->SetXbufSize(xbuf_size_);
    }
    for (auto op : kernel_.objects_) {
      if (op->flags_ & OBJ_FLAG_DEAD) {
        continue;
      }
      op->tail_insn_ = op->insn_ = code_ptr;
      switch (op->CgTmpl()) {
        case kGenSimd0: {
          auto anti_dep = op->xbuf_ == 0 ? AllocOutXBuf(op) : nullptr;
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          break;
        }
        case kGenSimd1: {
          auto anti_dep = op->xbuf_ == 0 ? AllocOutXBuf(op) : nullptr;
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.Push(op->lhs_->xbuf_, op);
          }
          code_ptr += op->Emit(kernel_);
          if (anti_dep) {
            SimdBarrier(anti_dep, op);
          }
          SimdSync(op->lhs_, op);
          break;
        }
        case kGenSimd2: {
          auto anti_dep = op->xbuf_ == 0 ? AllocOutXBuf(op) : nullptr;
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.Push(op->lhs_->xbuf_, op);
          }
          if (op->flags_ & OBJ_FLAG_FREE_RHS) {
            free_xbuf_.Push(op->rhs_->xbuf_, op);
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
        case kGenFlex: {
          code_ptr += GenFlexOpCommon(static_cast<FlexOp *>(op));
          if (op->rhs_ == nullptr) {
            ASSERT(op->lhs_);
            SimdSync(op->lhs_, op);
          } else if (op->rhs_->index_ > op->lhs_->index_) {
            SimdSync(op->rhs_, op);
            SimdSync(op->lhs_, op);
          } else {
            SimdSync(op->lhs_, op);
            SimdSync(op->rhs_, op);
          }
          break;
        };
        case kGenSimd3: {
          auto flex = static_cast<FlexOp *>(op);
          code_ptr += GenFlexOpCommon(flex);
          if (op->flags_ & OBJ_FLAG_FLEX_RREE_XHS) {
            free_xbuf_.Push(flex->xhs_->xbuf_, op);
          }
          NDObject *inputs[3] = {flex->lhs_, flex->rhs_, flex->xhs_};
          if (inputs[0]->index_ < inputs[1]->index_) {
            std::swap(inputs[0], inputs[1]);
          }
          if (inputs[0]->index_ < inputs[2]->index_) {
            std::swap(inputs[0], inputs[2]);
          }
          if (inputs[1]->index_ < inputs[2]->index_) {
            std::swap(inputs[1], inputs[2]);
          }
          for (size_t i = 0; i < 3; i++) {
            SimdSync(inputs[i], op);
          }
          break;
        };
        case kGenLoad: {
          code_ptr += op->Emit(kernel_);
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
        case kGenComm: {
          if (op->xbuf_ == 0) {
            AllocOutXBuf(op);
          }
          if (op->flags_ & OBJ_FLAG_FREE_LHS) {
            free_xbuf_.Push(op->lhs_->xbuf_, op);
          }
          auto code_size = op->Emit(kernel_);
          code_ptr += code_size;
          SimdSync(op->lhs_, op);
          break;
        }
        default:
          ASSERT(0);
          break;
      }  // end switch
    }    // end for op
    *code_ptr++ = 0;
    BackwardSync();
    return reinterpret_cast<uint8_t *>(code_ptr);
  }

 private:
  void BackwardSync() {
    auto &objects = kernel_.objects_;
    uint64_t max_event = kernel_.backward_event_num_ - 1;
    uint64_t cur_event = 0;
    int sync_idx = 0;
    for (size_t i = 0; i < kernel_.load_num_; ++i) {
      auto load = kernel_.static_ops_[i];
      if (load->flags_ & OBJ_FLAG_DEAD) {
        continue;
      }
      auto simd = objects[load->last_ref_];
      if (simd->index_ > sync_idx) {
        uint64_t event;
        if (cur_event <= max_event) {
          event = cur_event++;
          *(load->insn_) |= 1ul << V_M_HEAD_WAIT_FLAG_OFFSET | event << V_M_HEAD_WAIT_EVENT_OFFSET;
        } else {
          event = max_event;
          auto from_sync = objects[sync_idx]->tail_insn_;
          *from_sync &= ~(0x1ul << V_HEAD_BACK_SET_OFFSET);
        }
        *(simd->tail_insn_) |= 1ul << V_HEAD_BACK_SET_OFFSET | event << V_HEAD_B_SET_EVENT_OFFSET;
        sync_idx = simd->index_;
      }
    }
    cur_event = 0;
    sync_idx = objects.size();
    for (size_t i = kernel_.static_ops_.size(); i > kernel_.load_num_; --i) {
      auto store = kernel_.static_ops_[i - 1];
      auto simd = objects[store->first_def_];
      if (simd->index_ < sync_idx) {
        uint64_t event;
        if (cur_event <= max_event) {
          event = cur_event++;
          *(store->tail_insn_) |= 1ul << V_M_HEAD_SET_FLAG_OFFSET | event << V_M_HEAD_SET_EVENT_OFFSET;
        } else {
          event = max_event;
          auto to_sync = objects[sync_idx]->insn_;
          *to_sync &= ~(0x1ul << V_HEAD_BACK_WAIT_OFFSET);
        }
        *(simd->insn_) |= 1ul << V_HEAD_BACK_WAIT_OFFSET | event << V_HEAD_B_WAIT_EVENT_OFFSET;
        sync_idx = simd->index_;
      }
    }
  }

  int GenFlexOpCommon(FlexOp *op) {
    int anti_num = 0;
    NDObject *anti_ops[FlexOp::kWsMax + 1];
    if (op->xbuf_ == 0) {
      auto anti = AllocOutXBuf(op);
      if (anti) anti_ops[anti_num++] = anti;
    }
    ASSERT(op->ws_num_ <= 2);
    if (op->ws_num_ > 0) {
      if (op->flags_ & OBJ_FLAG_FLEX_REUSE_WS) {
        auto reuse = op->wss_[0] == 0 ? op->lhs_ : op->rhs_;
        op->wss_[0] = reuse->xbuf_;
        if (op->reuse_dep_) {
          anti_ops[anti_num++] = kernel_.objects_[op->reuse_dep_];
        }
      } else {
        auto anti = AllocDynXBuf(op, op->wss_[0]);
        if (anti) anti_ops[anti_num++] = anti;
      }
      if (op->ws_num_ > 1) {
        auto anti = AllocDynXBuf(op, op->wss_[1]);
        if (anti) anti_ops[anti_num++] = anti;
        free_xbuf_.Push(op->wss_[1], op);
      }
      free_xbuf_.Push(op->wss_[0], op);
    }
    if (op->flags_ & OBJ_FLAG_FREE_LHS) {
      free_xbuf_.Push(op->lhs_->xbuf_, op);
    }
    if (op->flags_ & OBJ_FLAG_FREE_RHS) {
      free_xbuf_.Push(op->rhs_->xbuf_, op);
    }
    int size = op->Emit(kernel_);
    for (int i = 0; i < anti_num; ++i) {
      SimdBarrier(anti_ops[i], op);
    }
    return size;
  }

  NDObject *AllocDynXBuf(NDObject *obj, uint64_t &xbuf) {
    NDObject *anti = nullptr;
    if (!free_xbuf_.Empty() && free_xbuf_.Front().second->index_ < vector_vector_sync) {
      // roughly reuse for simplify: ignore inputs barrier to be inserted
      xbuf = free_xbuf_.Pop().first;
    } else if (static_xbuf_ + xbuf_size_ <= g_system.LocalMemSize()) {
      xbuf = static_xbuf_;
      static_xbuf_ += xbuf_size_;
    } else {
      ASSERT(!free_xbuf_.Empty());
      auto &op = free_xbuf_.Pop();
      xbuf = op.first;
      anti = op.second;
    }
    return anti;
  }

  NDObject *AllocOutXBuf(NDObject *obj) {
    if (obj->flags_ & OBJ_FLAG_REUSE_LHS) {
      obj->xbuf_ = obj->lhs_->xbuf_;
      return obj->reuse_dep_ ? kernel_.objects_[obj->reuse_dep_] : nullptr;
    }
    if (obj->flags_ & OBJ_FLAG_REUSE_RHS) {
      obj->xbuf_ = obj->rhs_->xbuf_;
      return obj->reuse_dep_ ? kernel_.objects_[obj->reuse_dep_] : nullptr;
    }
    return AllocDynXBuf(obj, obj->xbuf_);
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
    int total = kernel_.forward_event_num_;
    for (int i = 1; i <= total; ++i) {
      if (int next = (m.hold_event + i) % total; from_idx >= m.hold_idx[next]) {
        event = static_cast<uint64_t>(next);
        m.hold_idx[next] = to_idx;
        m.hold_event = next;
        return true;
      }
    }
    event = static_cast<uint64_t>(m.hold_event);
    return false;
  }

  uint64_t static_xbuf_;
  XbufPool free_xbuf_;

  uint32_t xbuf_size_;

  int vector_vector_sync = 0;
  EventManager lv_event_;
  EventManager vs_event_;

  VectorKernel &kernel_;
};

void VectorKernel::BuildDomain() {
  uint32_t one_bits = 0;
  auto select_dom = [this, &one_bits](NDObject *cand) -> bool {
    auto &dom_nd = dom_->nd_;
    auto &cand_nd = cand->nd_;
    if (cand_nd.size() != dom_nd.size()) {
      return cand_nd.size() > dom_nd.size();
    }
    uint32_t bits = one_bits;
    while (bits) {
      size_t idx = 31 - __builtin_clz(bits);
      uint32_t unmask = 1u << idx;
      bits &= ~unmask;
      if (cand_nd[idx] == 1) continue;
      for (size_t n_idx = 0; n_idx < idx; ++n_idx) {
        if (cand_nd[n_idx] == 1) {
          if ((one_bits & (1u << n_idx)) == 0) {
            return false;
          }
        } else {
          unmask |= 1u << n_idx;
        }
      }
      one_bits &= ~unmask;
      return true;
    }
    return false;
  };
  dom_ = *objects_.rbegin();
  for (size_t i = 0; i < dom_->nd_.size(); ++i) {
    if (dom_->nd_[i] == 1) {
      one_bits |= 1u << i;
    }
  }
  for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
    auto op = *it;
    if (op->MetaFlags() & (ObjectMeta::kLhsDom | ObjectMeta::kDom)) {
      auto cand = op->MetaFlags() & ObjectMeta::kLhsDom ? op->lhs_ : op;
      if (dom_ != cand && select_dom(cand)) {
        dom_ = cand;
      }
    }
  }
  auto nd_size = std::max(dom_->nd_.size(), 1ul);
  int op_index = 0;
  for (auto op : objects_) {
    // Make rank of all ops equal by broadcast to (..., 1, 1, .., 1)
    if (auto ndd = op->Ndd(); ndd != nullptr && ndd->dims.size() < nd_size) {
      ndd->dims.resize(nd_size, 1);
    }
    op->Clear(op_index++);
  }
  block_align_ = SIMD_BLOCK_SIZE / ITEM_SIZE[min_type_];
  visit_ = nullptr;
}

void VectorKernel::PrepareTiling() {
  tile_num_ = 1;
  auto &nd = dom_->nd_;
  align_.base = 0;
  align_.depth = shard_ ? shard_->base + 1 : nd.size();
  align_.affine = PropRange::ELEMWISE;
  align_.simd_dim = -1;
  for (auto op : objects_) {
    op->AlignProp(align_);
  }
  tile_size_ = 1;
  for (int i = 0; i < align_.depth; ++i) {
    tile_size_ *= nd[i];
  }
  align_.space = tile_size_;
  tile_size_ = RoundUp<int64_t>(tile_size_, block_align_);
  for (size_t i = align_.depth; i < nd.size(); ++i) {
    tile_size_ *= nd[i];
  }
}

void VectorKernel::ManualTiling() {
  auto &dims = DimSpace();
  TileParam tp;
  for (auto &t : tiles_) {
    int64_t space = dims[t.start];
    for (int i = t.start + 1; i <= t.end; ++i) {
      space *= dims[i];
    }
    tp.start = t.start;
    tp.end = t.end;
    tp.num = t.num;
    tp.tile = t.factor ? t.factor : CeilDiv(space, t.num);
    tp.tail = space % tp.tile;
    if (t.start > 0) {
      Tile(tp, space);
    } else {
      TileLead(tp, LeadAlign());
    }
  }
}

void VectorKernel::ShapeTiling(int64_t size_limit, int64_t core_limit) {
  int align_depth = align_.depth;
  PropRange fold;
  fold.base = DimSpace().size() - 1;
  int64_t tile_size = tile_size_;
  TileParam tp;
  do {
    if (fold.base + 1 > align_depth) {
      if (shard_ && fold.base > shard_->base) {
        int partial_end = shard_->base + ShardParam::PARTIAL_SIZE;
        fold.depth = fold.base >= partial_end ? fold.base - partial_end + 2 : 1;
        fold.space = DimSpace()[fold.base + 1 - fold.depth];
      } else {
        fold.depth = fold.base + 1;
        fold.affine = PropRange::ELEMWISE;
        for (auto op : objects_) {
          op->FoldProp(fold);
        }
        fold.space = dom_->nd_[fold.base];
        for (int i = 1; i < fold.depth; ++i) {
          fold.space *= dom_->nd_[fold.base - i];
        }
      }
      tp.start = fold.base + 1 - fold.depth;
      ASSERT(tp.start > 0);
      if (tp.start < align_depth) {  // axis of 1
        tp.start = align_depth;
      }
      if (fold.space > 1) {
        tp.end = fold.base;
        tile_size = BodyTiling(tile_size, size_limit, core_limit, fold, tp);
      }
      fold.base = tp.start - 1;
    } else {
      tp.start = 0;
      tp.end = fold.base;
      LeadTiling(tile_size, size_limit, core_limit, tp);
      return;
    }
  } while (tile_size > size_limit || (tp.num > 1 && tp.num == fold.space));
  if (align_depth > 1) {
    TileParam tp;
    tp.start = 0;
    tp.end = align_depth - 1;
    tp.num = 1;
    tp.tile = align_.space;
    tp.tail = 0;
    TileProp(tp);
  }
}

static int64_t GetDivision(int64_t val, int64_t min) {
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

int64_t VectorKernel::BodyTiling(int64_t tile_size, int64_t size_limit, int64_t core_limit, const PropRange &range,
                                 TileParam &tp) {
  auto cost_measure = [core_limit](int64_t factor, int64_t tile_num) -> int64_t {
    return CeilDiv(tile_num, core_limit) * (factor + 2);
  };
  // ceil(a/b) <= c --> b >= ceil(a/(c+1))+1
  // floor(a/b) = ceil((a-1)/b)  <= c-1 --> b >= ceil((a-1)/(c-1 + 1))+1
  // floor(a/b) <= c --> b >= floor(a/c)
  // floor(tile_size_/tile_num) <= tile_size_limit_ --> tile_num >= floor(tile_size_/tile_size_limit_)
  // (tile_size / space) * floor(space /tile_num) <= tile_size_limit_) --> tile_num >= floor(space/max_factor)
  // EQUAL TO:
  //  int64_t tile_num = (space + max_factor - 1) / max_factor;
  //  while (tile_num < space && ((space + tile_num - 1) / tile_num) > max_factor) {
  //    tile_num++;
  //  }
  int64_t space = range.space;
  int64_t init_tile_num;
  if (tile_size > size_limit) {
    int64_t max_factor = size_limit / (tile_size / space);
    if (max_factor <= 1) {
      tp.num = space;
      tp.tile = 1;
      tp.tail = 0;
      return Tile(tp, space);
    }
    init_tile_num = std::max(CeilDiv(space, max_factor), CeilDiv(size_limit, tile_size));
  } else {
    init_tile_num = 1;
  }
  if (tile_num_ > 1) {  // avoid tile range pad
    tp.num = GetDivision(space, init_tile_num);
    tp.tile = space / tp.num;
    if (tp.num < space && range.affine < PropRange::REDUCE) {
      int64_t tile_num_base = tile_num_;
      int64_t cost = cost_measure(tp.tile, tp.num * tile_num_base);
      int64_t div_tile = tp.num;
      while (div_tile < space) {
        div_tile = GetDivision(space, div_tile + 1);
        int64_t factor = space / div_tile;
        int64_t div_cost = cost_measure(factor, div_tile * tile_num_base);
        if (div_cost >= cost) break;
        cost = div_cost;
        tp.num = div_tile;
        tp.tile = factor;
      }
    }
    tp.tail = 0;
  } else {
    int64_t factor = CeilDiv(space, init_tile_num);
    tp.num = init_tile_num;
    tp.tile = factor;
    int64_t best_cost = cost_measure(factor, init_tile_num);
    int64_t start_num = init_tile_num + 1;
    while (factor > 1) {
      int64_t align_tile = RoundUp(start_num, core_limit);
      start_num = align_tile + core_limit;
      int64_t align_factor = CeilDiv(space, align_tile);
      if (align_factor >= factor) continue;
      factor = align_factor;
      int64_t t_num = CeilDiv(space, factor);
      int64_t cost = cost_measure(factor, t_num);
      if (cost < best_cost) {
        best_cost = cost;
        tp.num = t_num;
        tp.tile = factor;
      }
      if (align_tile > core_limit * 4) break;
    }
    tp.tail = space % tp.tile;
  }
  return tp.num > 1 ? Tile(tp, space) : tile_size;
}

void VectorKernel::LeadTiling(int64_t tile_size, int64_t size_limit, int64_t core_limit, TileParam &tp) {
  auto cost_measure = [this, core_limit](int64_t block_num, int64_t tile_num) -> int64_t {
    int64_t core_tile = CeilDiv(tile_num * tile_num_, core_limit);
    return core_tile * (CeilDiv(block_num, 8L) + 2);
  };
  int64_t block_size = block_align_;
  int64_t space = align_.space;
  if (tile_num_ > 1) {  // avoid tile range pad
    tp.num = GetDivision(space, CeilDiv(tile_size, size_limit));
    tp.tile = space / tp.num;
    if (tp.num < space && align_.affine < PropRange::REDUCE) {
      int64_t best_cost = cost_measure(CeilDiv(tp.tile, block_size), tp.num);
      int64_t div_tile = tp.num;
      while (div_tile < space) {
        div_tile = GetDivision(space, div_tile + 1);
        int64_t factor = space / div_tile;
        int64_t cost = cost_measure(CeilDiv(factor, block_size), div_tile);
        if (cost >= best_cost) break;
        best_cost = cost;
        tp.num = div_tile;
        tp.tile = factor;
      }
    }
    tp.tail = 0;
  } else {
    constexpr int64_t min_factor = 512L;
    int64_t align_width = space > min_factor ? block_size : space;
    auto factor = RoundDown(std::min(space, size_limit), align_width);
    auto tile_num = CeilDiv(space, factor);
    auto best_cost = cost_measure(factor / block_size, tile_num);
    tp.num = tile_num;
    tp.tile = factor;
    auto align_num = RoundUp(tile_num + 1, core_limit);
    while (align_num < core_limit * 5) {
      auto factor_min = CeilDiv(space, align_num);
      if (auto align_factor = RoundUp(factor_min, align_width); align_factor < factor) {
        if (align_factor < min_factor) break;
        factor = align_factor;
        tile_num = CeilDiv(space, factor);
        auto cost = cost_measure(factor / block_size, tile_num);
        if (cost < best_cost) {
          best_cost = cost;
          tp.num = tile_num;
          tp.tile = factor;
        }
      }
      align_num = RoundUp(align_num + core_limit, core_limit);
    }
    tp.tail = space % tp.tile;
  }
  TileLead(tp, block_size);
}

void DumpRefHelper::Dump(NDObject *op) {
  int idx = idx_++;
  idx_map_[op] = idx;
  auto dump_var = [this](NDObject *obj) {
    if (obj == nullptr) {
      oss_ << "%?[]";
      return;
    }
    oss_ << "%" << idx_map_[obj];
    if (obj->shape_ref_) {
      oss_ << *obj->shape_ref_;
    }
    oss_ << "<" << DTYPE_NAMES[obj->type_id_] << ">";
  };
  dump_var(op);
  oss_ << " = ";
  op->Dump(true, oss_);
  oss_ << "(";
  if (op->lhs_) {
    dump_var(GetInput(op->lhs_));
    if (op->rhs_) {
      oss_ << ", ";
      dump_var(GetInput(op->rhs_));
      if (op->flags_ & OBJ_FLAG_XHS) {
        oss_ << ", ";
        dump_var(GetInput(static_cast<FlexOp *>(op)->xhs_));
      } else if (op->IsCube()) {
        auto cube = static_cast<CubeOp *>(op);
        if (cube->bias_) {
          oss_ << ", ";
          dump_var(GetInput(cube->bias_));
        }
      }
    }
  }
  oss_ << ")";
}

NDObject *DumpRefHelper::GetInput(NDObject *input) {
  while (input && idx_map_.count(input) == 0) {
    input = input->lhs_;
  }
  return input;
}

void VKernel::Append(NDObject *obj) {}
uint64_t VKernel::CodeGen() { return 0; }

std::string &VKernel::DisAssemble() {
  std::ostringstream oss;
  code_.DisAssemble(oss);
  dump_str_ = oss.str();
  return dump_str_;
}

uint8_t *VectorKernel::DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, uint64_t code_reserve) {
  int64_t live_peak = Analyze();
  int64_t tile_size_limit = TileSizeLimit(live_peak);
  // tiling
  if (likely(tiles_.empty())) {
    ShapeTiling(tile_size_limit, core_limit);
  } else {
    ManualTiling();
  }
  // simd_width
  uint64_t tile_size = tile_size_;
  if (align_.simd_dim >= 0) {
    int64_t lead_dim = DimSpace()[0];
    int64_t block_sw = block_align_;
    int64_t block_lead = RoundUp(lead_dim, block_sw);
    int64_t tile_outer = tile_size / block_lead;
    int64_t lead_limit = tile_size_limit / tile_outer;
    int64_t simd_width = std::min(static_cast<int64_t>(ITEM_SIMD_WIDTH_MAX[max_type_]), block_lead);
    while (simd_width > block_sw && RoundUp(lead_dim, simd_width) > lead_limit) {
      simd_width -= block_sw;
    }
    lead_align_ = simd_width;
    tile_size = RoundUp(lead_dim, simd_width) * tile_outer;
  }
  // codegen
  code_.block_dim_ = core_limit;
  forward_event_num_ = backward_event_num_ = g_system.EventNum();
  CodeGenHelper helper(*this, tile_size * ITEM_SIZE[max_type_], live_peak);
  auto code_end = helper.Generate(code_ptr, code_reserve, tile_size);
  ASSERT(static_cast<uint64_t>(code_end - code_ptr) <= code_reserve);
  return code_end;
}

void VectorKernel::Dump(std::ostringstream &oss, const std::string &indent) {
  auto dump_op = [&oss](NDObject *op) {
    oss << "%" << op->index_ << op->nd_ << "<" << DTYPE_NAMES[op->type_id_] << ">";
  };
  oss << indent << "vgraph(tile_num=" << tile_num_ << ") {" << std::endl;
  std::string body_indent = indent + "  ";
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    oss << body_indent;
    dump_op(op);
    oss << " = ";
    op->Dump(true, oss);
    oss << "(";
    if (op->lhs_) {
      dump_op(op->lhs_);
      if (op->rhs_) {
        oss << ", ";
        dump_op(op->rhs_);
        if (op->flags_ & OBJ_FLAG_XHS) {
          dump_op(static_cast<FlexOp *>(op)->xhs_);
          oss << ", ";
        }
      }
    }
    oss << ") // stride=[";
    if (const auto &strides = op->nd_.data->strides; !strides.empty()) {
      for (size_t i = 0; i < strides.size() - 1; ++i) {
        oss << strides[i] << ",";
      }
      oss << strides.back();
    }
    oss << "]" << std::endl;
  }
  oss << indent << "}";
}

// lead_dim_ is used only in codegen phase. so we reuse it for liveness analyze
#define OP_GEN_S(op) \
  do {               \
    op->xbuf_ = 2;   \
  } while (0)
#define OP_GEN_D(op) \
  do {               \
    op->xbuf_ = 1;   \
  } while (0)
#define OP_KILL(op) \
  do {              \
    op->xbuf_ = 0;  \
  } while (0)
#define OP_LIVE(op) (op->xbuf_)
#define OP_LIVE_D(op) (op->xbuf_ == 1)

static inline bool RhsInplaceCheck(NDObject *obj) { return obj->MetaFlags() & ObjectMeta::kRhsReuse; }

static inline bool LhsInplaceCheck(NDObject *obj) {
  if (obj->obj_id_ == kCast) {
    return (obj->type_id_ <= obj->lhs_->type_id_);
  }
  return obj->MetaFlags() & ObjectMeta::kLhsReuse;
}

int64_t VectorKernel::Analyze() {
  constexpr int REUSE_REJECT = -1;
  constexpr int REUSE_READY = 0;
  constexpr int REUSE_SUCC = 1;

  auto LivenessEnd = [this](NDObject *op, NDObject *end) {
    // TODO: check if other comm op can also spare 1 xbuf(like AllReduce)
    if (end->IsSimd()) {
      if (!OP_LIVE(end)) {
        OP_GEN_D(end);
        return true;
      }
      if (end->reuse_dep_) {
        objects_[end->reuse_dep_]->reuse_dep_ = op->index_;
        end->reuse_dep_ = 0;
      }
    } else if (!OP_LIVE(end)) {
      ASSERT(end->IsLoad());
      OP_GEN_S(end);
      end->last_ref_ = op->index_;
    }
    return false;
  };
  for (size_t i = load_num_; i < static_ops_.size(); ++i) {
    auto store = static_ops_[i];
    store->first_def_ = store->lhs_->index_;
    OP_GEN_S(store->lhs_);
  }
  int cur_live = 0;
  int live_peak = 0;
  for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
    auto op = *it;
    if (op->IsSimd()) {
      if (!OP_LIVE(op)) {
        op->flags_ |= OBJ_FLAG_DEAD;
        continue;
      }
      int reuse_flag = OP_LIVE_D(op) ? REUSE_READY : REUSE_REJECT;
      if (!(op->flags_ & OBJ_FLAG_WORKSPACE)) {
        if (auto kill = op->lhs_; kill && LivenessEnd(op, kill)) {
          if (reuse_flag == REUSE_READY && LhsInplaceCheck(op)) {
            op->flags_ |= OBJ_FLAG_REUSE_LHS;
            reuse_flag = REUSE_SUCC;
            op->reuse_dep_ = 0;
            kill->reuse_dep_ = op->index_;
          } else {
            op->flags_ |= OBJ_FLAG_FREE_LHS;
            cur_live++;
          }
        }
        if (auto kill = op->rhs_; kill && LivenessEnd(op, kill)) {
          if (reuse_flag == REUSE_READY && RhsInplaceCheck(op)) {
            op->flags_ |= OBJ_FLAG_REUSE_RHS;
            reuse_flag = REUSE_SUCC;
            op->reuse_dep_ = 0;
            kill->reuse_dep_ = op->index_;
          } else {
            op->flags_ |= OBJ_FLAG_FREE_RHS;
            cur_live++;
          }
        }
        if (cur_live > live_peak) {
          live_peak = cur_live;
        }
      } else {
        auto flex = static_cast<FlexOp *>(op);
        int ws_num = flex->ws_num_;
        if (auto kill = op->lhs_; kill && LivenessEnd(op, kill)) {
          if (reuse_flag == REUSE_READY && LhsInplaceCheck(op)) {
            op->flags_ |= OBJ_FLAG_REUSE_LHS;
            reuse_flag = REUSE_SUCC;
            op->reuse_dep_ = 0;
            kill->reuse_dep_ = op->index_;
          } else {
            if (op->flags_ & OBJ_FLAG_FLEX_INPL_WS) {
              op->flags_ |= OBJ_FLAG_FLEX_REUSE_WS;
              op->reuse_dep_ = 0;
              op->lhs_->reuse_dep_ = op->index_;
              flex->wss_[0] = 0;
              ws_num--;
            } else {
              op->flags_ |= OBJ_FLAG_FREE_LHS;
            }
            cur_live++;
          }
        }
        if (auto kill = op->rhs_; kill && LivenessEnd(op, kill)) {
          if (reuse_flag == REUSE_READY && RhsInplaceCheck(op)) {
            op->flags_ |= OBJ_FLAG_REUSE_RHS;
            reuse_flag = REUSE_SUCC;
            op->reuse_dep_ = 0;
            kill->reuse_dep_ = op->index_;
          } else {
            if ((op->flags_ & (OBJ_FLAG_FLEX_INPL_WS | OBJ_FLAG_FLEX_REUSE_WS)) == OBJ_FLAG_FLEX_INPL_WS) {
              op->flags_ |= OBJ_FLAG_FLEX_REUSE_WS;
              op->reuse_dep_ = 0;
              op->lhs_->reuse_dep_ = op->index_;
              flex->wss_[0] = 1;
              ws_num--;
            } else {
              op->flags_ |= OBJ_FLAG_FREE_RHS;
            }
            cur_live++;
          }
        }
        if ((op->flags_ & OBJ_FLAG_XHS) && LivenessEnd(op, flex->xhs_)) {
          cur_live++;
          flex->flags_ |= OBJ_FLAG_FLEX_RREE_XHS;
        }
        if (cur_live + ws_num > live_peak) {
          live_peak = cur_live + ws_num;
        }
      }  // end flexop
      if (reuse_flag == REUSE_READY) {
        cur_live--;
      }
      OP_KILL(op);
    } else if (op->IsLoad()) {
      if (!OP_LIVE(op)) {
        op->flags_ |= OBJ_FLAG_DEAD;
      }
    }
  }
  return live_peak;
}

void VKernelS::StaticInit(const std::vector<NDObject *> &objects) {
  max_type_ = min_type_ = objects.front()->type_id_;
  for (auto op : objects) {
    StaticAppend(op);
    if (op->IsComm()) {
      ASSERT(comm_op_ == nullptr);
      comm_op_ = static_cast<CommOp *>(op);
      if (comm_op_->max_type_ > max_type_) {
        max_type_ = comm_op_->max_type_;
      }
    }
  }
}

static inline void SetPdHead(NDObject *op, NDObject *head) { op->insn_ = reinterpret_cast<uint64_t *>(head); }
static inline NDObject *GetPdHead(NDObject *op) { return reinterpret_cast<NDObject *>(op->insn_); }
static inline bool IsBroker(NDObject *op) { return op->obj_id_ == kReshape; }

void VKernelS::BrokerInit() {
  broker_num_ = 0;
  for (auto op : build_ops_) {
    SetPdHead(op, nullptr);
    op->prop_id_ = 0;
    if (IsBroker(op)) {
      broker_num_++;
    }
  }
  if (broker_num_ == 0) {
    return;
  }
  last_broker_ = 0;
  int idx = build_ops_.size() - 1;
  for (auto it = build_ops_.rbegin(); it != build_ops_.rend(); ++it) {
    auto op = *it;
    op->index_ = idx--;
    auto head = GetPdHead(op);
    if (head == nullptr) {
      head = op;
      SetPdHead(op, head);
    }
    if (IsBroker(op)) {
      if (GetPdHead(op->lhs_) == nullptr) {
        SetPdHead(op->lhs_, op->lhs_);
      }
      if (!last_broker_) last_broker_ = op->index_;
    } else {
      op->ForInput([head, op](NDObject *input) {
        auto input_head = GetPdHead(input);
        if (input_head == nullptr) {
          SetPdHead(input, head);
        } else {
          auto root = GetPdHead(head);
          auto input_root = GetPdHead(input_head);
          if (root->index_ > input_root->index_) {
            SetPdHead(input_root, root);
          } else {
            SetPdHead(root, input_root);
          }
        }
      });
    }
  }
  int prop_id = 0;
  for (auto it = build_ops_.rbegin(); it != build_ops_.rend(); ++it) {
    auto op = *it;
    if (GetPdHead(op) == op) {
      op->prop_id_ = prop_id++;
    } else {
      op->prop_id_ = GetPdHead(op)->prop_id_;
    }
  }
}

class DomainUnifier {
 public:
  DomainUnifier(std::vector<NDObject *> &objects, bool no_stuff = false) : objects_(objects) {
    if (no_stuff) {
      return;
    }
    // TODO: move to stuff ops..
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
      auto op = *it;
      if (auto lhs = op->lhs_) {
        if (lhs->prop_id_ == -1) lhs->prop_id_ = op->prop_id_;
        if (auto rhs = op->rhs_) {
          if (rhs->prop_id_ == -1) rhs->prop_id_ = op->prop_id_;
        }
      }
    }
  }

  bool Process(ReshapeOp *op, bool forward_only = false) {
    ReshapeOp::ChangeRange range;
    DimArray update;
    auto gen_update = [&update](const DimArray &src, int begin, int size) {
      update.resize(size);
      for (int i = 0; i < size; ++i) {
        update[i] = src[begin + i];
      }
    };
    while (op->VisitChangeRange(range)) {
      if (range.size == 0) {
        ExpandDim(op->prop_id_, range.begin, range.in_size);
        range.begin += range.in_size;
      } else if (range.in_size == 0) {
        ExpandDim(op->lhs_->prop_id_, range.begin, range.size);
        range.begin += range.size;
      } else if (auto prop = op->lhs_->prop_id_; !forward_only && AffineCheck(prop, range.begin, range.in_size)) {
        gen_update(op->nd_.data->dims, range.begin, range.size);
        ReshapeRange(prop, range.begin, range.in_size, update);
        range.begin += range.size;
      } else if (auto prop = op->prop_id_; AffineCheck(prop, range.begin, range.size)) {
        gen_update(op->lhs_->nd_.data->dims, range.begin, range.in_size);
        ReshapeRange(prop, range.begin, range.size, update);
        range.begin += range.in_size;
      } else {
        return false;
      }
    }
    op->SetFlag(OBJ_FLAG_BROKER_AFFINED);
    return true;
  }

 private:
  int BrokerRemap(NDObject *op, int dim) {
    return IsBroker(op) && op->CheckFlag(OBJ_FLAG_BROKER_AFFINED) ? dim : -1;
  }

  bool AffineCheck(int prop, int range_begin, int range_size) {
    PropRange range;
    range.base = range_begin + range_size - 1;
    range.depth = range_size;
    range.affine = PropRange::ELEMWISE;
    for (auto op : objects_) {
      if (op->prop_id_ != prop || op->SharedNdd()) continue;
      if (op->obj_id_ == kLoad && op->CheckFlag(OBJ_FLAG_LOAD_FROM_CUBE)) {
        return false;
      }
      op->FoldProp(range);
      if (range.depth != range_size) {
        return false;
      }
      if (auto rmap = BrokerRemap(op, range_begin); rmap >= 0 && !AffineCheck(op->lhs_->prop_id_, rmap, range_size)) {
        return false;
      }
    }
    return true;
  }

  void ReshapeRange(int prop, int range_begin, int range_size, const DimArray &update) {
    for (auto op : objects_) {
      if (op->prop_id_ != prop) continue;
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        auto &dims = ndd->dims;
        int dim_size = static_cast<int>(dims.size());
        int dim_end = range_begin + range_size;
        if (dim_end <= dim_size) {
          int update_size = static_cast<int>(update.size());
          bool strict_elemwise = false;
          for (int i = range_begin; i < dim_end; ++i) {
            if (dims[i] != 1) {
              strict_elemwise = true;
              break;
            }
          }
          if (int delta = update_size - range_size; delta != 0) {
            dims.resize(dim_size + delta);
            if (delta > 0) {
              for (int i = dim_size - 1; i >= dim_end; --i) {
                dims[i + delta] = dims[i];
              }
            } else {
              for (int i = dim_end; i < dim_size; ++i) {
                dims[i + delta] = dims[i];
              }
            }
          }
          for (int i = 0; i < update_size; ++i) {
            dims[range_begin + i] = strict_elemwise ? update[i] : 1;
          }
        }
      }
      if (auto rmap = BrokerRemap(op, range_begin); rmap >= 0) {
        ReshapeRange(op->lhs_->prop_id_, rmap, range_size, update);
      }
      op->DimChanged();
    }
  }

  void ExpandDim(int prop, int idx, int expand_size) {
    for (auto op : objects_) {
      if (op->prop_id_ != prop) continue;
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        auto &dims = ndd->dims;
        if (int dim_size = dims.size(); idx < dim_size) {
          for (int i = dim_size - 1; i >= idx; --i) {
            dims[i + expand_size] = dims[i];
          }
          for (int i = 0; i < expand_size; ++i) {
            dims[idx + i] = 1;
          }
          dims.resize(dim_size + expand_size);
        }
      }
      if (auto rmap = BrokerRemap(op, idx); rmap >= 0) {
        ExpandDim(op->lhs_->prop_id_, rmap, expand_size);
      }
      op->DimChanged();
    }
  }

 private:
  std::vector<NDObject *> &objects_;
};

bool VKernelS::BrokerAffine() {
  if (auto broker_num = broker_num_; broker_num > 0) {
    DomainUnifier unifier(objects_);
    for (int idx = last_broker_; broker_num > 0; --idx) {
      auto op = build_ops_[idx];
      ASSERT(!IsBroker(op) || op->prop_id_ != op->lhs_->prop_id_);
      if (op->obj_id_ == kReshape) {
        if (!unifier.Process(static_cast<ReshapeOp *>(op))) return false;
      } else {
        continue;
      }
      broker_num--;
    }
  }
  return true;
}

class SplitVector : public VectorKernel {
 public:
  SplitVector() : VectorKernel(KernelType::kVector, 0) {}
  ~SplitVector() override {
    for (auto op : build_ops_) {
      delete op;
    }
  }
  uint64_t CodeGen() override {
    max_type_ = comm_op_ ? comm_op_->max_type_ : objects_.front()->type_id_;
    min_type_ = objects_.front()->type_id_;
    for (auto op : objects_) {
      StaticAppend(op);
    }
    BuildDomain();
    return DoCodeGen(g_system.CoreNum());
  }
  std::vector<NDObject *> build_ops_;
};

uint64_t VKernelS::BrokerCodeGen(VKernel **hold_kernel) {
  auto set_sstore = [](NDObject *op, NDStore *st) { op->tail_insn_ = reinterpret_cast<uint64_t *>(st); };
  auto get_sstore = [](NDObject *op) { return reinterpret_cast<NDStore *>(op->tail_insn_); };
  StagesKernel *stage = new StagesKernel(0);
  GraphTracker tracker;
  std::vector<SplitVector *> children;
  children.resize(broker_num_ * 2, nullptr);
  for (auto op : objects_) {
    set_sstore(op, nullptr);
    auto k = children[op->prop_id_];
    if (k == nullptr) {
      k = new SplitVector();
      stage->AddStage(k);
      children[op->prop_id_] = k;
    }
    if (IsBroker(op)) {
      auto input_k = children[op->lhs_->prop_id_];
      auto store = get_sstore(op->lhs_);
      if (store == nullptr) {
        store = new NDStore(op->lhs_);
        set_sstore(op->lhs_, store);
        store->Normalize(input_k->objects_);
        input_k->objects_.push_back(store);
        stage->StageStore(input_k, store);
        input_k->build_ops_.push_back(store);
      }
      auto load = new NDLoad(nullptr, op->shape_ref_, op->type_id_);
      load->Normalize(k->objects_);
      k->objects_.push_back(load);
      k->build_ops_.push_back(load);
      stage->StageLoad(k, load, store);
      tracker.Record(&op->lhs_);
      op->lhs_ = load;
    }
    k->objects_.push_back(op);
  }
  auto ws_size = stage->CodeGen();
  tracker.RecoverClear();
  code_ = std::move(stage->code_);
  *hold_kernel = stage;
  return ws_size;
}

NDAccess *VectorKernel::FindInplaceStore(NDAccess *load, const std::function<bool(NDAccess *)> &check) const {
  auto update_flag = [](int input_flag, bool elem_type, int &flag) {
    // undetermined -> elemwise -> no-elemwise
    //          |___________________|
    if (input_flag != 0 && flag != -1) {
      if (flag == 0) {
        flag = input_flag == 1 && elem_type ? 1 : -1;
      } else if (input_flag == -1 || !elem_type) {  // 1
        flag = -1;
      }
    }
  };
  std::vector<int> elem_flags(objects_.size(), 0);  // 1: elemwise, 0: undetermined, -1: no-elemwise
  elem_flags[load->index_] = 1;
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    auto &flag = elem_flags[op->index_];
    if (op->lhs_) {
      auto elem_type = op->InplaceProp();
      update_flag(elem_flags[op->lhs_->index_], elem_type, flag);
      if (op->rhs_) {
        update_flag(elem_flags[op->rhs_->index_], elem_type, flag);
        if (op->flags_ & OBJ_FLAG_XHS) {
          update_flag(elem_flags[static_cast<FlexOp *>(op)->xhs_->index_], elem_type, flag);
        }
      }
    }
    if (op->IsStore() && flag == 1 && op->type_id_ == load->type_id_ &&
        (check == nullptr || check(static_cast<NDAccess *>(op)))) {
      return static_cast<NDAccess *>(op);
    }
  }
  return nullptr;
}

void VectorKernel::CollectIdle(std::vector<NDObject *> &cleans) {
  for (auto op : objects_) {
    if (op->IsStore()) {
      auto shape = op->shape_ref_->data;
      auto size = op->shape_ref_->size;
      bool zero = false;
      for (size_t i = 0; i < size; ++i) {
        if (auto dim = *shape++; dim == 0) {
          zero = true;
          break;
        }
      }
      if (!zero) {
        cleans.push_back(op);
      }
    }
  }
}

void VectorKernel::ProcessIdle() {
  std::vector<NDObject *> cleans;
  CollectIdle(cleans);
  UpdateIdle(cleans);
}

void VectorKernel::Optimize(std::vector<NDObject *> &build_ops, GraphTracker *tracker) {
  auto bb = pass::BasicBlock(objects_, build_ops, tracker);
  for (auto pass : pass::passes) {
    pass(bb);
  }
  bb.Export(objects_);
}

VKernelS::~VKernelS() {
  delete stage_kernel_;
  for (auto op : build_ops_) {
    delete op;
  }
}

uint64_t VKernelS::CodeGen() {
  auto broker_succ = NormBuild();
  if (!broker_succ) {
    delete stage_kernel_;
    return BrokerCodeGen(&stage_kernel_);
  }
  return DoCodeGen(g_system.CoreNum());
}

void VKernelS::Append(NDObject *obj) { build_ops_.push_back(obj); }

bool VKernelS::NormBuild() {
  if (IsDynamic()) {
    Clear();
    if (static_ops_.empty()) {
      StaticInit(build_ops_);
    }
    if (!Normalize(true)) {
      return false;
    }
  } else {
    if (!Normalize(true)) {
      return false;
    }
    Optimize(build_ops_, nullptr);
    StaticInit(objects_);
  }
  BuildDomain();
  return true;
}

void VKernelS::Clone(VKernel *base, CloneHelper &helper) {
  auto k = static_cast<VKernelS *>(base);
  for (auto op : k->build_ops_) {
    Append(op->CloneUpdate(helper));
  }
}

void VKernelS::Dump(std::ostringstream &oss, const std::string &indent) {
  if (tile_num_ > 0) {
    return VectorKernel::Dump(oss, indent);
  }
  DumpRefHelper helper(oss);
  oss << indent << "rgraph.vec() {" << std::endl;
  std::string body_indent = indent + "  ";
  for (auto op : build_ops_) {
    oss << body_indent;
    helper.Dump(op);
    oss << std::endl;
  }
  oss << indent << "}";
}

_SpecVector::~_SpecVector() {
  if (fall_kernel_) {
    delete fall_kernel_;
  }
}

void _SpecVector::Append(NDObject *obj) {
  obj->ForInput([this](NDObject *&in) {
    if (in->obj_id_ == ObjectType::kReduce) {
      auto red = static_cast<ReduceOp *>(in);
      if (red->insn_ == nullptr) {
        post_reduces_.push_back(in);
        if (!red->KeepDims()) {
          in = new ReshapeOp(red, red->shape_ref_);
          VKernelS::Append(in);
          in->index_ = stage_ids_.size();
          stage_ids_.push_back(last_stage_);
        }
        red->insn_ = reinterpret_cast<uint64_t *>(in);
      } else if (!red->KeepDims()) {
        in = reinterpret_cast<NDObject *>(red->insn_);
      }
    } else if (in->IsLoad() && stage_ids_[in->index_] < 0) {
      stage_ids_[in->index_] = last_stage_;
    }
  });
  VKernelS::Append(obj);
  obj->index_ = stage_ids_.size();
  int sid;
  if (obj->IsStore()) {
    sid = stage_ids_[obj->lhs_->index_];
  } else if (obj->IsLoad()) {
    sid = -1;
  } else {
    sid = last_stage_;
    if (obj->obj_id_ == ObjectType::kReduce) {
      obj->insn_ = nullptr;
    }
  }
  stage_ids_.push_back(sid);
}

void _SpecVector::Clone(VKernel *base, CloneHelper &helper) {
  auto k = static_cast<_SpecVector *>(base);
  for (size_t i = 0; i < k->build_ops_.size(); ++i) {
    auto op = k->build_ops_[i];
    if (op->IsSimd() && k->stage_ids_[i] != last_stage_) {
      ASSERT(k->stage_ids_[i] == last_stage_ + 1);
      Next();
    }
    Append(op->CloneUpdate(helper));
  }
}

void _SpecVector::Dump(std::ostringstream &oss, const std::string &indent) {
  if (use_fall_) {
    fall_kernel_->Dump(oss, indent);
  } else {
    VKernelS::Dump(oss, indent);
  }
}

template <bool dyn_shape>
uint64_t SpecVector<dyn_shape>::CodeGen() {
  auto reduce_fall_check = [this]() ->  bool {
    if (!post_reduces_.empty()) {
      int tile_size_limit = TileSizeLimit(Analyze());
      auto ndd = dom_->nd_.data;
      const_cast<NDSpaceData *>(ndd)->UpdateStride(lead_align_);
      for (auto op : post_reduces_) {
        auto end_dim = static_cast<ReduceOp *>(op)->EndDim();
        auto size = ndd->stride(end_dim);
        if (size > tile_size_limit || size == ndd->stride_back()) {
          return true;
        }
      }
    }
    return false;
  };
  if constexpr (dyn_shape) {
    Clear();
    if (static_ops_.empty()) {
      StaticInit(build_ops_);
    }
    if (!Normalize(true)) {
      return FallCodeGen();
    }
    BuildDomain();
    PrepareTiling();
    if (reduce_fall_check()) {
      return FallCodeGen();
    }
  } else {
    if (!Normalize(true)) {
      return FallCodeGen();
    }
    GraphTracker tracker;
    Optimize(build_ops_, &tracker);
    StaticInit(objects_);
    BuildDomain();
    PrepareTiling();
    if (reduce_fall_check()) {
      tracker.Recover();
      return FallCodeGen();
    }
  }
  use_fall_ = false;
  if (unlikely(!tile_size_)) {
    ProcessIdle();
    return 0;
  }
  return DoCodeGenInner(g_system.CoreNum());
}

template <typename T>
class RemapKernel : public T {
 public:
  void Remap(NDAccess *op, NDAccess *orig) {
    remap_.emplace_back(op, orig);
    orig->addr_.Update(&orig->addr_.data);
  }
  uint64_t CodeGen() override {
    auto ws_size = T::CodeGen();
    for (auto &r : remap_) {
      this->code_.BindOpFast(r.first->addr_, r.second->addr_);
    }
    return ws_size;
  }
  void SetDynamic() { this->flags_ = KernelFlag::kDynamic; }
 protected:
  std::vector<std::pair<NDAccess *, NDAccess *>> remap_;
};

#define INIT_SSTORE(op) do { (op)->insn_ = nullptr; } while (0)
#define SET_SSTORE(op, st) do { (op)->insn_ = reinterpret_cast<uint64_t *>(st); } while (0)
#define GET_SSTORE(op) reinterpret_cast<NDStore *>((op)->insn_)

template <bool dyn_shape>
uint64_t SpecVector<dyn_shape>::FallCodeGen() {
  struct _CloneHelper : public CloneHelper {
    IntArrayRef *GetClone(IntArrayRef *shape) override { return shape; }
    ScalarRef *GetClone(ScalarRef *scalar) override { return scalar; }
    NDObject *GetClone(NDObject *op) override { return clones_[op->index_]; }
    void SetClone(NDObject *op, NDObject *clone) {}
    std::vector<NDObject *> clones_;
  };
  if (fall_kernel_ == nullptr) {
    auto stage_kernel = new RemapKernel<StagesKernel>();
    if constexpr (dyn_shape) {
      stage_kernel->SetDynamic();
    }
    for (int i = 0; i <= last_stage_; ++i) {
      stage_kernel->AddStage(new std::conditional_t<dyn_shape, VKernelD, VKernelS>());
    }
    _CloneHelper  helper;
    helper.clones_.reserve(stage_ids_.size());
    for (size_t i = 0; i < stage_ids_.size(); ++i) {
      auto out_sid = stage_ids_[i];
      if (out_sid == -1) continue; // load only
      auto src_op = build_ops_[i];
      src_op->index_ = i;
      auto clone_op = src_op->Clone(helper);
      INIT_SSTORE(clone_op);
      clone_op->index_ = i;
      helper.clones_.push_back(clone_op);
      if (!clone_op->IsSimd()) {
        stage_kernel->Remap(static_cast<NDAccess *>(clone_op), static_cast<NDAccess *>(src_op));
      }
      clone_op->ForInput([this, out_sid, stage_kernel, &helper](NDObject *&in) {
        auto in_sid = stage_ids_[in->index_];
        if (in_sid == out_sid) {
          return;
        }
        auto out_stage = stage_kernel->KernelAt(out_sid);
        NDAccess *load;
        if (in->IsLoad()) {
          load = static_cast<NDAccess *>(in->Clone(helper));
          stage_kernel->Remap(load, static_cast<NDAccess *>(build_ops_[in->index_]));
        } else {
          auto store = GET_SSTORE(in);
          if (store == nullptr) {
            store = new NDStore(in);
            SET_SSTORE(in, store);
            auto in_stage = stage_kernel->KernelAt(in_sid);
            in_stage->Append(store);
            stage_kernel->StageStore(in_stage, store);
          }
          load = new NDLoad(nullptr, in->shape_ref_, in->type_id_);
          stage_kernel->StageLoad(out_stage, load, store);
        }
        out_stage->Append(load);
        in = load;
      });
      stage_kernel->KernelAt(out_sid)->Append(clone_op);
    }
    fall_kernel_ = stage_kernel;
  }
  use_fall_ = true;
  auto ws_size = fall_kernel_->CodeGen();
  code_ = std::move(fall_kernel_->code_);
  return ws_size;
}

template class SpecVector<false>;
template class SpecVector<true>;

struct SpecVecStage : public StagesKernel::Stage {
  SpecVecStage(uint32_t flags, SpecVecContext &ctx) : StagesKernel::Stage(&spec_k_), spec_k_(flags, ctx) {}
  SpecVecBase spec_k_;
};

SpecVecContext::~SpecVecContext() {
  if (stage_k_) {
    stage_k_->Reset();
    delete stage_k_;
  }
  for (auto s : stage_pool_) {
    delete s;
  }
  for (auto op : spec_ops_) {
    delete op;
  }
}

uint64_t SpecVecBase::CodeGen() { return DoCodeGenInner(g_system.CoreNum()); }

bool SpecVecBase::SpecBuild() {
  if (fall_opt_ & FALL_RESHAPE) {
    fall_opt_ &= ~FALL_RESHAPE;
    if (ReshapeSpec()) {
      SplitBuild();
      return false;
    }
  }
  // TODO: Optimize
  BuildDomain();
  PrepareTiling();
  tile_limit_ = -1;
  if (fall_opt_ & FALL_REDUCE) {
    fall_opt_ &= ~FALL_REDUCE;
    if (ReduceSpec()) {
      SplitBuild();
      return false;
    }
  }
  if (fall_opt_ & FALL_BROADCAST) {
    fall_opt_ &= ~FALL_BROADCAST;
    if (BroadcastSpec()) {
      SplitBuild();
      return false;
    }
  }
  return true;
}

void SpecVecBase::SplitPlan(size_t cut_begin) {
  auto &stack = ctx_.spec_ops_;
  uint64_t cut_area_mask = 0;
  size_t cut_end = stack.size();
  for (size_t c = cut_end; c > cut_begin; --c) {
    auto cut_op = stack[c - 1];
    auto meta = GetMeta(cut_op);
    if (meta->aid != -1) {
      meta->UnCut();
      continue;
    }
    auto aid = ctx_.AssignArea();
    stack.push_back(cut_op);
    meta->aid = aid;
    cut_area_mask |= 1ull << aid;
    while (stack.size() > cut_end) {
      auto top = stack.back();
      stack.pop_back();
      top->ForInput([this, aid, &stack](NDObject *in) {
        auto in_meta = GetMeta(in);
        if (in_meta->aid == -1) {
          stack.push_back(in);
          in_meta->aid = aid;
        } else if (auto in_aid = ctx_.RootArea(in_meta->aid); in_aid != aid) {
          ctx_.MergeArea(aid, in_aid);
        }
      });
    }
  }
  for (size_t store_idx = load_num_; store_idx < static_ops_.size(); ++store_idx) {
    auto store = static_ops_[store_idx];
    auto meta = GetMeta(store);
    ASSERT(meta->aid == -1);
    if (auto input = GetMeta(store->lhs_); input->IsCut()) {
      meta->aid = input->aid;
      continue;
    }
    auto aid = ctx_.AssignArea();
    stack.push_back(store);
    meta->aid = aid;
    uint64_t merge_mask = 0;
    uint64_t unmerge_mask = 0;
    while (stack.size() > cut_end) {
      auto top = stack.back();
      stack.pop_back();
      top->ForInput([this, aid, cut_area_mask, &unmerge_mask, &merge_mask, &stack](NDObject *in) {
        auto in_meta = GetMeta(in);
        if (in_meta->aid == -1) {
          stack.push_back(in);
          in_meta->aid = aid;
        } else if (auto in_aid = ctx_.RootArea(in_meta->aid); in_aid != aid) {
          if ((cut_area_mask & (1ull << in_aid)) == 0) {
            ctx_.MergeArea(aid, in_aid);
          } else if (in_meta->IsCut()) {
            unmerge_mask |= 1ull << in_aid;
          } else {
            merge_mask |= 1ull << in_aid;
          }
        }
      });
    }
    merge_mask &= ~unmerge_mask;
    if (merge_mask) {
      cut_area_mask |= 1ull << aid;
      while (merge_mask) {
        auto cut_aid = 63 - __builtin_clzl(merge_mask);
        merge_mask &= ~(1ull << cut_aid);
        ctx_.MergeArea(cut_aid, aid);
        aid = cut_aid;
      }
    }
  }
}

bool SpecVecBase::BroadcastSpec() {
  constexpr int64_t MIN_TILE_NUM = 8192;
  constexpr int64_t MIN_IO_NUM = 3;
  ctx_.ResetSpec();
  size_t broadcast_begin = ctx_.spec_ops_.size();
  for (auto op : objects_) {
    InitMeta(op);
    if (op->obj_id_ == ObjectType::kBroadcastTo && !op->lhs_->IsLoad() && !GetMeta(op->lhs_)->IsCut()) {
      ctx_.spec_ops_.push_back(op);
      GetMeta(op->lhs_)->SetCut();
    }
  }
  if (ctx_.spec_ops_.size() == broadcast_begin || LazyTileLimit() * MIN_TILE_NUM < tile_size_) {
    return false;
  }
  size_t cut_begin = ctx_.spec_ops_.size();
  for (size_t i = broadcast_begin; i < cut_begin; ++i) {
    ctx_.spec_ops_.push_back(ctx_.spec_ops_[i]->lhs_);
  }
  SplitPlan(cut_begin);
  for (auto op : static_ops_) {
    if (auto aid = GetMeta(op)->aid; aid >= 0) {
      ctx_.areas_[ctx_.RootArea(aid)].u32 += 1;
    }
  }
  bool split = false;
  for (size_t i = broadcast_begin; i < cut_begin; ++i) {
    auto op = ctx_.spec_ops_[i];
    auto aid = ctx_.RootArea(GetMeta(op)->aid);
    auto cut_aid = ctx_.RootArea(GetMeta(op->lhs_)->aid);
    if (cut_aid != aid) {
      if (ctx_.areas_[cut_aid].u32 < MIN_IO_NUM) {
        ctx_.MergeArea(cut_aid, aid);
      } else {
        split = true;
      }
    }
  }
  ctx_.spec_ops_.resize(broadcast_begin);
  return split;
}

bool SpecVecBase::ReduceSpec() {
  ctx_.ResetSpec();
  const_cast<NDSpaceData *>(dom_->nd_.data)->UpdateStride(lead_align_);
  size_t cut_begin = ctx_.spec_ops_.size();
  for (auto op : objects_) {
    InitMeta(op);
    if (op->IsSimd() && op->obj_id_ != ObjectType::kReduce) {
      op->ForInput([this](NDObject *in) {
        if (in->obj_id_ == ObjectType::kReduce && !GetMeta(in)->IsCut()) {
          auto size = dom_->nd_.stride(static_cast<ReduceOp *>(in)->EndDim());
          if (size > LazyTileLimit() || size == dom_->nd_.stride_back()) {
            ctx_.spec_ops_.push_back(in);
            GetMeta(in)->SetCut();
          }
        }
      });
    }
  }
  if (ctx_.spec_ops_.size() == cut_begin) {
    return false;
  }
  SplitPlan(cut_begin);
  for (auto i = cut_begin; i < ctx_.spec_ops_.size(); ++i) {
    if (auto meta = GetMeta(ctx_.spec_ops_[i]); !meta->IsCut()) {
      ctx_.areas_[ctx_.RootArea(meta->aid)].ext_opt = FALL_REDUCE;
    }
  }
  ctx_.spec_ops_.resize(cut_begin);
  return true;
}

bool SpecVecBase::ReshapeSpec() {
  ctx_.ResetSpec();
  auto &spec_ops = ctx_.spec_ops_;
  size_t reshape_begin = spec_ops.size();
  for (auto op : objects_) {
    InitMeta(op);
    if (auto input = op->lhs_;
        op->obj_id_ == ObjectType::kReshape && !GetMeta(input)->IsCut() && !(input->nd_.dims() == op->nd_.dims())) {
      spec_ops.push_back(op);
      GetMeta(input)->SetCut();
    }
  }
  size_t cut_begin = spec_ops.size();
  if (cut_begin == reshape_begin) {
    return false;
  }
  for (size_t i = reshape_begin; i < cut_begin; ++i) {
    spec_ops.push_back(spec_ops[i]->lhs_);
  }
  SplitPlan(cut_begin);
  for (auto op : objects_) {
    op->prop_id_ = ctx_.RootArea(GetMeta(op)->aid);
  }
  DomainUnifier affine(objects_, true);
  auto &areas = ctx_.areas_;
  bool fail_touch = false;
  uint64_t fail_affine = false;
  for (size_t i = reshape_begin; i < cut_begin; ++i) {
    auto op = static_cast<ReshapeOp *>(spec_ops[i]);
    if (op->prop_id_ == op->lhs_->prop_id_) {
      areas[op->prop_id_].u32 = 1;
      fail_touch = true;
    } else if (!affine.Process(op, areas[op->lhs_->prop_id_].u32)) {
      fail_affine = true;
    }
  }
  if (fail_affine) {
    for (size_t i = reshape_begin; i < cut_begin; ++i) {
      auto op = spec_ops[i];
      auto aid = ctx_.RootArea(GetMeta(op)->aid);
      if (op->lhs_->nd_.dims() == op->nd_.dims()) {
        if (auto in_aid = ctx_.RootArea(GetMeta(op->lhs_)->aid); in_aid != aid) {
          ctx_.MergeArea(in_aid, aid);
        }
      } else {
        ctx_.areas_[aid].ext_opt = FALL_RESHAPE;
      }
    }
    spec_ops.resize(reshape_begin);
    return true;
  }
  spec_ops.resize(reshape_begin);
  return fail_touch ? ReshapeSpec() : false;
}

namespace {
struct _SpecSwapLoad : public NDLoad {
  _SpecSwapLoad(NDAccess *store) : NDLoad(store->addr_.gm, store->shape_ref_, store->type_id_), store_(store) {}
  void Normalize(std::vector<NDObject *> &run_ops) override {
    ndd_.dims = store_->nd_.dims();
    tail_dim_ = -1;
    tail_size_ = 0;
    round_tile_.resize(0);
  }
  NDAccess *store_;
};

struct _ReloadCloner : public CloneHelper {
  IntArrayRef *GetClone(IntArrayRef *shape) override { return shape; }
  ScalarRef *GetClone(ScalarRef *scalar) override { return scalar; }
  NDObject *GetClone(NDObject *op) override { return op; }
  void SetClone(NDObject *op, NDObject *clone) override {}
};
}

void SpecVecBase::SplitBuild() {
  if (ctx_.stage_k_ == nullptr) {
    ctx_.stage_k_ = new StagesKernel();
  } else if (ctx_.stage_size_ == 0) {
    ctx_.stage_k_->Reset();
  }
  size_t stage_begin = ctx_.stage_size_;
  for (int aid = 0; aid < ctx_.area_size_; ++aid) {
    if (auto &area = ctx_.areas_[aid]; area.parent == aid) {
      SpecVecStage *stage;
      if (auto stage_size = ctx_.stage_size_++; stage_size < ctx_.stage_pool_.size()) {
        stage = ctx_.stage_pool_[stage_size];
        stage->Reset();
      } else {
        stage = new SpecVecStage(flags_, ctx_);
        ctx_.stage_pool_.push_back(stage);
      }
      stage->spec_k_.Reset();
      stage->spec_k_.fall_opt_ = fall_opt_ | area.ext_opt;
      area.stage = stage;
    }
  }
  for (auto op : objects_) {
    auto out_aid = GetMeta(op)->aid;
    out_aid = ctx_.RootArea(out_aid);
    auto stage = ctx_.areas_[out_aid].stage;
    if (op->IsLoad()) {
      stage->AddIO(static_cast<NDAccess *>(op),
                   op->CheckFlag(OBJ_FLAG_STAGE_IO) ? static_cast<_SpecSwapLoad *>(op)->store_ : nullptr);
    } else if (op->IsStore()) {
      ASSERT(ctx_.RootArea(GetMeta(op->lhs_)->aid) == out_aid);
      stage->AddIO(static_cast<NDAccess *>(op));
    } else {
        op->ForInput([this, out_aid, stage](NDObject *&in) {
        if (auto in_aid = ctx_.RootArea(GetMeta(in)->aid); in_aid != out_aid) {
          if (auto recent = GetMeta(in)->recent_load; recent < OpMeta::IO_END) {
            auto load = static_cast<NDAccess *>(ctx_.spec_ops_[recent]);
            if (GetMeta(load)->aid == out_aid) {
              in = load;
              return;
            }
          }
          if (in->IsLoad()) {
            _ReloadCloner cloner;
            auto clone = in->Clone(cloner);
            GetMeta(in)->recent_load = ctx_.spec_ops_.size();
            ctx_.spec_ops_.push_back(clone);
            clone->Normalize(stage->spec_k_.objects_);
            stage->spec_k_.SplitAppend(clone);
            stage->StageLoad(static_cast<NDAccess *>(clone), static_cast<NDAccess *>(in));
            GetMeta(clone)->aid = out_aid;
            in = clone;
          } else {
            auto in_stage = ctx_.areas_[in_aid].stage;
            auto store_idx = GetMeta(in)->store;
            auto store = store_idx < OpMeta::IO_END ? static_cast<NDAccess *>(ctx_.spec_ops_[store_idx]) : nullptr;
            if (store == nullptr) {
              store = new NDStore(in);
              GetMeta(in)->store = ctx_.spec_ops_.size();
              ctx_.spec_ops_.push_back(store);
              store->Normalize(in_stage->spec_k_.objects_);
              in_stage->spec_k_.SplitAppend(store);
              in_stage->StageStore(store);
            }
            auto load = new _SpecSwapLoad(store);
            GetMeta(in)->recent_load = ctx_.spec_ops_.size();
            ctx_.spec_ops_.push_back(load);
            load->Normalize(stage->spec_k_.objects_);
            stage->spec_k_.SplitAppend(load);
            stage->StageLoad(load, store);
            ctx_.tracker_.Record(&in);
            GetMeta(load)->aid = out_aid;
            in = load;
          }
        }
      });
    }
    if (op->SharedNdd() && op->nd_.data != op->lhs_->nd_.data) {
      ctx_.tracker_.Record((NDObject **)(&op->nd_.data));
      op->nd_.data = op->lhs_->nd_.data;
    }
    stage->spec_k_.SplitAppend(op);
  }
  auto stage_end = ctx_.stage_size_;
  for (size_t i = stage_begin; i < stage_end; ++i) {
    auto s = ctx_.stage_pool_[i];
    if (s->spec_k_.SpecBuild()) {
      ctx_.stage_k_->AppendStage(s);
    }
  }
}

void SpecVecBase::Dump(std::ostringstream &oss, const std::string &indent) {
  if (!tile_num_) {
    tile_num_ = 1;
    VKernelS::Dump(oss, indent);
    tile_num_ = 0;
  } else {
    VKernelS::Dump(oss, indent);
  }
}

void SpecVecKernel::Append(NDObject *obj) {
  if (obj->obj_id_ == ObjectType::kReshape) {
    fall_opt_init_ |= FALL_RESHAPE;
  } else if (obj->obj_id_ == ObjectType::kReduce) {
    obj->insn_ = nullptr;
  } else if (obj->rhs_ != nullptr || obj->obj_id_ == ObjectType::kBroadcastTo) {
    fall_opt_init_ |= FALL_BROADCAST;
  }
  if (obj->IsSimd()) {
    obj->ForInput([this](NDObject *&in) {
      if (in->obj_id_ == ObjectType::kReduce) {
        fall_opt_init_ |= FALL_REDUCE;
        if (auto red = static_cast<ReduceOp *>(in); !red->KeepDims()) {
          if (red->insn_ == nullptr) {
            in = new ReshapeOp(red, red->shape_ref_);
            VKernelS::Append(in);
            red->insn_ = reinterpret_cast<uint64_t *>(in);
            fall_opt_init_ |= FALL_RESHAPE;
          } else {
            in = reinterpret_cast<NDObject *>(red->insn_);
          }
        }
      }
    });
  }
  VKernelS::Append(obj);
}

uint64_t SpecVecKernel::CodeGen() {
  context_.Reset();
  Clear();
  fall_opt_ = fall_opt_init_;
  if (static_ops_.empty()) {
    StaticInit(build_ops_);
    // TODO: dead code elim
    if ((fall_opt_init_ & FALL_BROADCAST) && static_ops_.size() < 4) {
      fall_opt_init_ &= ~FALL_BROADCAST;
    }
  }
  for (auto op : build_ops_) {
    op->Normalize(objects_);
    objects_.push_back(op);
  }
  if (SpecBuild()) {
    return SpecVecBase::CodeGen();
  }
  auto ws = context_.stage_k_->CodeGen();
  code_ = std::move(context_.stage_k_->code_);
  return ws;
}

void SpecVecKernel::Dump(std::ostringstream &oss, const std::string &indent) {
  if (context_.stage_size_) {
    context_.stage_k_->Dump(oss, indent);
  } else {
    SpecVecBase::Dump(oss, indent);
  }
}

void SpecVecKernel::Clone(VKernel *base, CloneHelper &helper) {
  context_.tracker_.Recover();
  static_cast<SpecVecKernel*>(base)->fall_opt_init_ = fall_opt_init_;
  VKernelS::Clone(base, helper);
}
}  // namespace dvm
