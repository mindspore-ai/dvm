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
    forward_event_num_ = g_system.EventNum() - kernel_.tile_info_.event_reserve;
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
          if (flex->xhs_->free_mask & 1u) {
            free_xbuf_.Push(flex->xhs_->data[0]->xbuf_, op);
          }
          NDObject *inputs[3] = {flex->lhs_, flex->rhs_, flex->xhs_->data[0]};
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
        case kGenCustom: {
          code_ptr += CustomOpGen(static_cast<CustomOp *>(op));
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
    uint64_t max_event = g_system.EventNum() - kernel_.tile_info_.event_reserve - 1;
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

  int CustomOpGen(CustomOp *op) {
    uint32_t dead_mask = 0;
    if (auto xout = op->xout_; xout != nullptr) {
      for (int i = 0; i < xout->out_num; ++i) {
        if (auto item = xout->data[i]; item->xbuf_ == 0) {
          AllocDynXBuf(item, item->xbuf_);
          if (item->flags_ & OBJ_FLAG_DEAD) {
            dead_mask |= 1u << i;
          }
        }
      }
    }
    int size = GenFlexOpCommon(op);
    if (dead_mask) {
      while (dead_mask) {
        auto idx = 31 - __builtin_clz(dead_mask);
        dead_mask &= ~(1ul << idx);
        free_xbuf_.Push(op->xout_->data[idx]->xbuf_, op);
      }
    }
    if (op->rhs_ == nullptr) {
      ASSERT(op->lhs_);
      SimdSync(op->lhs_, op);
    } else if (op->xhs_ == nullptr) {
      if (op->rhs_->index_ > op->lhs_->index_) {
        SimdSync(op->rhs_, op);
        SimdSync(op->lhs_, op);
      } else {
        SimdSync(op->lhs_, op);
        SimdSync(op->rhs_, op);
      }
    } else {
      auto xhs = op->xhs_;
      int n = xhs->in_num;
      for (int i = 0; i < n; ++i) {
        if (xhs->free_mask & (1u << i)) {
          free_xbuf_.Push(xhs->data[i]->xbuf_, op);
        }
      }
      constexpr int kMaxXhsInputs = 4;
      NDObject *inputs[2 + kMaxXhsInputs];
      inputs[0] = op->lhs_;
      inputs[1] = op->rhs_;
      for (int i = 0; i < n; ++i) {
        inputs[2 + i] = xhs->data[i];
      }
      int total = 2 + n;
      for (int i = 1; i < total; ++i) {
        auto key = inputs[i];
        int j = i - 1;
        while (j >= 0 && inputs[j]->index_ < key->index_) {
          inputs[j + 1] = inputs[j];
          --j;
        }
        inputs[j + 1] = key;
      }
      for (int i = 0; i < total; ++i) {
        SimdSync(inputs[i], op);
      }
    }
    return size;
  }

  NDObject *AllocDynXBuf(NDObject *obj, uint64_t &xbuf) {
    NDObject *anti = nullptr;
    if (!free_xbuf_.Empty() && free_xbuf_.Front().second->index_ < vector_vector_sync) {
      // roughly reuse for simplify: ignore inputs barrier to be inserted
      xbuf = free_xbuf_.Pop().first;
    } else if (static_xbuf_ + xbuf_size_ <= kernel_.local_mem_size_) {
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
    int total = forward_event_num_;
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
  int forward_event_num_;
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
  tile_info_.Reset(shard_ ? shard_->base + 1 : nd.size());
  for (auto op : objects_) {
    op->TileCollect(tile_info_);
  }
  local_mem_size_ = g_system.LocalMemSize() - g_system.UbWorkspaceSize() - tile_info_.ext_ws;
  if (g_system.Arch() == AiCoreArch::kAiCore_C310) {
    if (tile_info_.flags & ObjectMeta::kSimt) {
      flags_ |= K_FLAG_SIMT;
      local_mem_size_ -= g_system.SimtWorkspace();
    } else {
      flags_ &= ~K_FLAG_SIMT;
    }
  }
  align_space_ = 1;
  for (int i = 0; i < tile_info_.lead_depth; ++i) {
    align_space_ *= nd[i];
  }
  tile_size_ = RoundUp<int64_t>(align_space_, block_align_);
  for (size_t i = tile_info_.lead_depth; i < nd.size(); ++i) {
    tile_size_ *= nd[i];
  }
}

class ShapeTilingHelper {
 public:
  ShapeTilingHelper(int64_t size_limit, int64_t core_limit, int64_t tile_size, VectorKernel::TileRegion &range)
      : size_limit_(size_limit), core_limit_(core_limit), tile_size_(tile_size), range_(range) {}
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

  bool BodyTiling(int start, const PropRange &range) {
    auto cost_measure = [this](int64_t factor, int64_t tile_num) -> int64_t {
      return CeilDiv(tile_num, core_limit_) * (factor + 2);
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
    int64_t init_tile_num = 1;
    if (tile_size_ > size_limit_) {
      int64_t min_tile_size = tile_size_ / space;
      int64_t max_factor = size_limit_ / min_tile_size;
      if (max_factor <= 1) {
        tile_size_ = min_tile_size;
        tile_num_ *= space;
        range_.Add(start, space, space, 1);
        return true;
      }
      init_tile_num = std::max(CeilDiv(space, max_factor), CeilDiv(size_limit_, tile_size_));
    }
    int64_t num, tile;
    if (tile_num_ > 1) {  // avoid tile range pad
      num = GetDivision(space, init_tile_num);
      tile = space / num;
      if (num < space && range.affine < PropRange::REDUCE) {
        int64_t tile_num_base = tile_num_;
        int64_t cost = cost_measure(tile, num * tile_num_base);
        int64_t div_tile = num;
        while (div_tile < space) {
          div_tile = GetDivision(space, div_tile + 1);
          int64_t factor = space / div_tile;
          int64_t div_cost = cost_measure(factor, div_tile * tile_num_base);
          if (div_cost >= cost) break;
          cost = div_cost;
          num = div_tile;
          tile = factor;
        }
      }
    } else {
      int64_t factor = CeilDiv(space, init_tile_num);
      num = init_tile_num;
      tile = factor;
      int64_t best_cost = cost_measure(factor, init_tile_num);
      int64_t start_num = init_tile_num + 1;
      while (factor > 1) {
        int64_t align_tile = RoundUp(start_num, core_limit_);
        start_num = align_tile + core_limit_;
        int64_t align_factor = CeilDiv(space, align_tile);
        if (align_factor >= factor) continue;
        factor = align_factor;
        int64_t t_num = CeilDiv(space, factor);
        int64_t cost = cost_measure(factor, t_num);
        if (cost < best_cost) {
          best_cost = cost;
          num = t_num;
          tile = factor;
        }
        if (align_tile > core_limit_ * 4) break;
      }
    }
    if (num > 1) {
      tile_size_ = tile_size_ / space * tile;
      tile_num_ *= num;
      range_.Add(start, space, num, tile);
      if (tile == 1) {
        return true;
      }
    }
    return false;
  }

  bool LeadTiling(int64_t block_size, int64_t space, int affine) {
    auto cost_measure = [this](int64_t block_num, int64_t tile_num_inner) -> int64_t {
      int64_t core_tile = CeilDiv(tile_num_inner * tile_num_, core_limit_);
      return core_tile * (CeilDiv(block_num, 8L) + 2);
    };
    int64_t num, tile;
    if (tile_num_ > 1) {  // avoid tile range pad
      num = GetDivision(space, CeilDiv(tile_size_, size_limit_));
      tile = space / num;
      if (num < space && affine < PropRange::REDUCE) {
        int64_t best_cost = cost_measure(CeilDiv(tile, block_size), num);
        int64_t div_tile = num;
        while (div_tile < space) {
          div_tile = GetDivision(space, div_tile + 1);
          int64_t factor = space / div_tile;
          int64_t cost = cost_measure(CeilDiv(factor, block_size), div_tile);
          if (cost >= best_cost) break;
          best_cost = cost;
          num = div_tile;
          tile = factor;
        }
      }
    } else {
      constexpr int64_t min_factor = 512L;
      int64_t align_width = space > min_factor ? block_size : space;
      auto factor = RoundDown(std::min(space, size_limit_), align_width);
      auto tile_num = CeilDiv(space, factor);
      auto best_cost = cost_measure(factor / block_size, tile_num);
      num = tile_num;
      tile = factor;
      auto align_num = RoundUp(tile_num + 1, core_limit_);
      while (align_num < core_limit_ * 5) {
        auto factor_min = CeilDiv(space, align_num);
        if (auto align_factor = RoundUp(factor_min, align_width); align_factor < factor) {
          if (align_factor < min_factor) break;
          factor = align_factor;
          tile_num = CeilDiv(space, factor);
          auto cost = cost_measure(factor / block_size, tile_num);
          if (cost < best_cost) {
            best_cost = cost;
            num = tile_num;
            tile = factor;
          }
        }
        align_num = RoundUp(align_num + core_limit_, core_limit_);
      }
    }
    if (num > 1) {
      tile_size_ = RoundUp<int64_t>(tile, block_size);
      tile_num_ *= num;
      range_.Add(0, space, num, tile);
    }
    return false;
  }

  int64_t size_limit_;
  int64_t core_limit_;
  int64_t tile_size_;
  int64_t tile_num_{1};
  VectorKernel::TileRegion &range_;
};

void VectorKernel::ShapeTiling(int64_t size_limit, int64_t core_limit, TileUpdate &update) {
  TileRegion &range = tile_region_;
  range.Reset();
  int align_depth = tile_info_.lead_depth;
  PropRange fold;
  fold.base = DimSpace().size() - 1;
  ShapeTilingHelper helper(size_limit, core_limit, tile_size_, range);
  bool running = true;
  while (running) {
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
      int start = fold.base + 1 - fold.depth;
      ASSERT(start > 0);
      if (start < align_depth) {  // fold crosses into lead region, clamp to lead boundary
        start = align_depth;
      }
      if (fold.space > 1) {
        running = helper.BodyTiling(start, fold);
      }
      fold.base = start - 1;
    } else {
      running = helper.LeadTiling(block_align_, align_space_, tile_info_.lead_affine);
    }
  }
  update.tile_size = helper.tile_size_;
  update.tile_num = helper.tile_num_;
  if (range.depth == 1) {
    int64_t full_space = range.last_num * range.last_tile;
    int64_t real_space = range.spaces[0];
    if (full_space != real_space) {
      range.tail_size = real_space + range.last_tile - full_space;
      range.tail_dim = range.starts[0];
    }
  }
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
        auto xhs = static_cast<FlexOp *>(op)->xhs_;
        for (int i = 0; i < xhs->in_num; ++i) {
          oss_ << ", ";
          dump_var(GetInput(xhs->data[i]));
        }
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

void DumpRefHelper::DumpGraph(const std::string &indent, const std::string &name, const std::vector<NDObject *> &build_ops) {
  oss_ << indent << "rgraph." << name << "() {" << std::endl;
  std::string body_indent = indent + "  ";
  for (auto op : build_ops) {
    oss_ << body_indent;
    Dump(op);
    oss_ << std::endl;
  }
  oss_ << indent << "}";
}

void VKernel::Append(NDObject *obj) {}
uint64_t VKernel::CodeGen() { return 0; }

std::string &VKernel::DisAssemble() {
  std::ostringstream oss;
  code_.DisAssemble(oss);
  dump_str_ = oss.str();
  return dump_str_;
}

void VectorKernel::ApplyTiling(const TileUpdate &update) {
  TileRegion &tr = tile_region_;
  TileParam tp;
  bool lead_fold = tile_info_.lead_depth > 1;
  if (tr.depth) {
    tp.end = DimSpace().size() - 1;
    if (tr.depth > 1) {
      tp.tile = 1;
      tp.tail = 0;
      for (int i = 0; i < tr.depth - 1; ++i) {
        tp.start = tr.starts[i];
        tp.num = tr.spaces[i];
        TileProp(tp);
        tp.end = tp.start - 1;
      }
    } else {
      tp.tail = tr.tail_size;
    }
    tp.start = tr.starts[tr.depth - 1];
    tp.num = tr.last_num;
    tp.tile = tr.last_tile;
    TileProp(tp);
    lead_fold = lead_fold && tp.start > 1;
  }
  if (lead_fold) {
    tp.start = 0;
    tp.end = tile_info_.lead_depth - 1;
    tp.num = 1;
    tp.tile = align_space_;
    tp.tail = 0;
    TileProp(tp);
  }
  tile_size_ = update.tile_size;
  tile_num_ = update.tile_num;
}

void VectorKernel::AlignSimd(int64_t tile_size_limit) {
  int64_t lead_dim = DimSpace()[0];
  int64_t block_sw = block_align_;
  int64_t block_lead = RoundUp(lead_dim, block_sw);
  int64_t tile_outer = tile_size_ / block_lead;
  int64_t lead_limit = tile_size_limit / tile_outer;
  int64_t simd_width = std::min(static_cast<int64_t>(ITEM_SIMD_WIDTH_MAX[max_type_]), block_lead);
  while (simd_width > block_sw && RoundUp(lead_dim, simd_width) > lead_limit) {
    simd_width -= block_sw;
  }
  lead_align_ = simd_width;
  tile_size_ = RoundUp(lead_dim, simd_width) * tile_outer;
}

uint8_t *VectorKernel::DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, uint64_t code_reserve) {
  int64_t live_peak = Analyze();
  int64_t tile_size_limit = TileSizeLimit(live_peak);
  TileUpdate update;
  ShapeTiling(tile_size_limit, core_limit, update);
  ApplyTiling(update);
  // simd_width
  if (tile_info_.flags & ObjectMeta::kSimdDim) {
    AlignSimd(tile_size_limit);
  }
  // codegen
  code_.block_dim_ = core_limit;
  CodeGenHelper helper(*this, tile_size_ * ITEM_SIZE[max_type_], live_peak);
  auto code_end = helper.Generate(code_ptr, code_reserve, tile_size_);
  ASSERT(static_cast<uint64_t>(code_end - code_ptr) <= code_reserve);
  return code_end;
}

uint8_t *VectorKernel::DoTileGen(int64_t live_peak, uint8_t *code_ptr, uint64_t code_reserve) {
  CodeGenHelper helper(*this, tile_size_ * ITEM_SIZE[max_type_], live_peak);
  auto code_end = helper.Generate(code_ptr, code_reserve, tile_size_);
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
          auto xhs = static_cast<FlexOp *>(op)->xhs_;
          for (int i = 0; i < xhs->in_num; ++i) {
            oss << ", ";
            dump_op(xhs->data[i]);
          }
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
        if (op->flags_ & OBJ_FLAG_XHS) {
          auto xhs = static_cast<FlexOp *>(op)->xhs_;
          xhs->free_mask = 0;
          for (int i = 0; i < xhs->in_num; ++i) {
            if (LivenessEnd(op, xhs->data[i])) {
              cur_live++;
              xhs->free_mask |= (1u << i);
            }
          }
        }
        int flex_live = cur_live;
        if (op->flags_ & OBJ_FLAG_XOUT) {
          auto xout = static_cast<CustomOp *>(op)->xout_;
          for (int i = 0; i < xout->out_num; ++i) {
            bool is_static = false;
            for (size_t j = load_num_; j < static_ops_.size(); ++j) {
              if (xout->data[i] == static_ops_[j]->lhs_) {
                is_static = true;
                static_ops_[j]->first_def_ = op->index_; // extend live
                break;
              }
            }
            if (!is_static) flex_live++;
          }
        }
        if (flex_live + ws_num > live_peak) {
          live_peak = flex_live + ws_num;
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

void VKernelS::SchInit(const std::vector<NDObject *> &objects) {
  ASSERT(sch_gen_ == nullptr);
  if (flags_ & KernelFlag::kOptFractalTrans) {
    sch_gen_ = new FractalSchGen(this);
  } else {
    for (auto op : build_ops_) {
      if (op->obj_id_ == kConcat) {
        sch_gen_ = new ConcatSchGen(this, static_cast<ConcatOp *>(op), objects);
        break;
      }
      if (op->obj_id_ == kSplitOp) {
        ASSERT(static_cast<SplitOpM *>(op)->main_ == op); // first is splitm
        sch_gen_ = new SplitSchGen(this, static_cast<SplitOpM *>(op), objects);
        break;
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
  for (size_t bidx = 0; bidx < build_ops_.size(); ++ bidx) {
    auto op = build_ops_[bidx];
    if (!IsBroker(op) || op->prop_id_ != op->lhs_->prop_id_) continue;
    int merged_prop = op->prop_id_;
    int broker_prop = prop_id++;
    op->prop_id_ = broker_prop;
    for (size_t midx = bidx + 1; midx < build_ops_.size(); ++midx) {
      auto cand = build_ops_[midx];
      if (cand->prop_id_ != merged_prop) continue;
      bool is_after = false;
      bool is_cross = false;
      cand->ForInput([&](NDObject *in) {
        if (in->prop_id_ == broker_prop) {
          is_after = true;
        } else if (in->prop_id_ == merged_prop) {
          is_cross = true;
        }
      });
      if (is_after) {
        if (is_cross) {
          cand->ForInput([&](NDObject *&in) {
            if (in->prop_id_ == merged_prop) {
              in = new ReshapeOp(in, in->shape_ref_);
              build_ops_.insert(build_ops_.begin() + midx, in);
              if (midx > static_cast<size_t>(last_broker_)) {
                last_broker_ = midx;
              } else {
                last_broker_++;
              }
              broker_num_++;
              midx++;
            }
          });
        }
        cand->prop_id_ = broker_prop;
      }
    } // it2
  } // it1
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
      op->ForInput([op](NDObject *in) {
        if (in->prop_id_ == -1) in->prop_id_ = op->prop_id_;
      });
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
      if (!forward_only) {
        visited_mask_ = 0;
        if (auto prop = op->lhs_->prop_id_; AffineCheck(prop, range.begin, range.in_size, range.size, op->nd_.dims())) {
          gen_update(op->nd_.data->dims, range.begin, range.size);
          visited_mask_ = 0;
          ReshapeRange(prop, range.begin, range.in_size, update);
          range.begin += range.size;
          continue;
        }
      }
      visited_mask_ = 0;
      if (auto prop = op->prop_id_; AffineCheck(prop, range.begin, range.size, range.in_size, op->lhs_->nd_.dims())) {
        gen_update(op->lhs_->nd_.data->dims, range.begin, range.in_size);
        visited_mask_ = 0;
        ReshapeRange(prop, range.begin, range.size, update);
        range.begin += range.in_size;
      } else {
        return false;
      }
    }
    size_t in_size = op->lhs_->nd_.size();
    size_t out_size = op->nd_.size();
    if (in_size > out_size) {
      ExpandDim(op->prop_id_, in_size);
    } else if (in_size < out_size) {
      ExpandDim(op->lhs_->prop_id_, out_size);
    }
    op->SetFlag(OBJ_FLAG_BROKER_AFFINED);
    return true;
  }

 private:
  int BrokerRemap(NDObject *op, int dim) {
    return IsBroker(op) && op->CheckFlag(OBJ_FLAG_BROKER_AFFINED) ? dim : -1;
  }
  bool IsVisited(int prop) const { return (visited_mask_ >> prop) & 0x1ul; }
  void SetVisited(int prop) { visited_mask_ |= 1ul << prop; }
  bool DimRangeEqual(const DimArray &dim, const DimArray &base, int begin, int size) {
    if (dim.size() < static_cast<size_t>(begin + size)) return false;
    for (int i = begin; i < begin + size; ++i) {
      if (dim[i] != base[i]) return false;
    }
    return true;
  }

  bool AffineCheck(int prop, int range_begin, int range_size, int out_size, const DimArray &nd) {
    SetVisited(prop);
    PropRange range;
    range.base = range_begin + range_size - 1;
    range.depth = range_size;
    range.affine = PropRange::ELEMWISE;
    for (auto op : objects_) {
      if (op->prop_id_ != prop) {
        if (IsBroker(op) && op->lhs_->prop_id_ == prop) {
          if (op->CheckFlag(OBJ_FLAG_BROKER_AFFINED)) {
            if (!IsVisited(op->prop_id_) && !AffineCheck(op->prop_id_, range_begin, range_size, out_size, nd)) {
              return false;
            }
          } else {
            if (IsVisited(op->prop_id_) && !DimRangeEqual(op->nd_.dims(), nd, range_begin, range_size)) {
              return false;
            }
          }
        }
        continue;
      }
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        int dim_size = ndd->dims.size();
        if (range_begin >= dim_size) {
          continue;
        }
        if ((op->obj_id_ == kLoad && op->CheckFlag(OBJ_FLAG_LOAD_FROM_CUBE)) ||
            (op->obj_id_ == kPermute && !op->CheckFlag(OBJ_FLAG_BROKER_AFFINED))) {
          return false;
        }
        if (range_size > 0) {
          if (out_size > 0) {
            op->FoldProp(range);
            if (range.depth != range_size) {
              return false;
            }
          } else {
            for (auto d = range_begin; d < std::min(range_begin + range_size, dim_size); ++d) {
              if (ndd->dims[d] != 1) return false;
            }
          }
        }
      }
      if (IsBroker(op)) {
        if (op->CheckFlag(OBJ_FLAG_BROKER_AFFINED)) {
          if (!IsVisited(op->lhs_->prop_id_) && !AffineCheck(op->lhs_->prop_id_, range_begin, range_size, out_size, nd)) {
            return false;
          }
        } else {
          if (IsVisited(op->lhs_->prop_id_) && !DimRangeEqual(op->lhs_->nd_.dims(), nd, range_begin, range_size)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  void ReshapeRange(int prop, int range_begin, int range_size, const DimArray &update) {
    SetVisited(prop);
    for (auto op : objects_) {
      if (op->prop_id_ != prop) {
        auto rmap = BrokerRemap(op, range_begin);
        if (rmap >= 0 && op->lhs_->prop_id_ == prop && !IsVisited(op->prop_id_)) {
          ReshapeRange(op->prop_id_, rmap, range_size, update);
        }
        continue;
      }
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
      if (auto rmap = BrokerRemap(op, range_begin); rmap >= 0 && !IsVisited(op->lhs_->prop_id_)) {
        ReshapeRange(op->lhs_->prop_id_, rmap, range_size, update);
      }
      op->DimChanged();
    }
  }

  void ExpandDim(int prop, size_t expand_size) {
    SetVisited(prop);
    for (auto op : objects_) {
      if (op->prop_id_ != prop) {
        if (IsBroker(op) && op->CheckFlag(OBJ_FLAG_BROKER_AFFINED) && op->lhs_->prop_id_ == prop && !IsVisited(op->prop_id_)) {
          ExpandDim(op->prop_id_, expand_size);
        }
        continue;
      }
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        auto &dims = ndd->dims;
        if (auto size = dims.size(); size < expand_size) {
          dims.resize(expand_size);
          for (auto i = size; i < expand_size; ++i) {
            dims[i]= 1;
          }
        }
        if (IsBroker(op) && op->CheckFlag(OBJ_FLAG_BROKER_AFFINED) && !IsVisited(op->lhs_->prop_id_)) {
          ExpandDim(op->lhs_->prop_id_, expand_size);
        }
        op->DimChanged();
      }
    }
  }

 private:
  std::vector<NDObject *> &objects_;
  uint64_t visited_mask_;
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
  std::vector<StagesKernel::Stage *> children;
  children.resize(broker_num_ * 2, nullptr);
  uint64_t append_mask = 0;
  for (auto op : objects_) {
    set_sstore(op, nullptr);
    auto s = children[op->prop_id_];
    if (s == nullptr) {
      s = new StagesKernel::Stage(new SplitVector());
      children[op->prop_id_] = s;
    }
    auto k = static_cast<SplitVector *>(s->kernel);
    if (IsBroker(op)) {
      auto store = get_sstore(op->lhs_);
      if (store == nullptr) {
        int prop_id = op->lhs_->prop_id_;
        auto input_s = children[prop_id];
        if (uint64_t prop_mask = 0x1ull << prop_id; !(append_mask & prop_mask)) {
          append_mask |= prop_mask;
          stage->AppendStage(input_s);
        }
        auto input_k = static_cast<SplitVector *>(input_s->kernel);
        auto input = op->lhs_;
        if (input->IsLoad()) {
          input = new CopyOp(input);
          input->Normalize(input_k->objects_);
          input_k->objects_.push_back(input);
          input_k->build_ops_.push_back(input);
        }
        store = new NDStore(input);
        set_sstore(op->lhs_, store);
        store->Normalize(input_k->objects_);
        input_s->StageStore(store);
        input_k->objects_.push_back(store);
        input_k->build_ops_.push_back(store);
      }
      auto load = new NDLoad(nullptr, op->shape_ref_, op->type_id_);
      load->Normalize(k->objects_);
      k->objects_.push_back(load);
      k->build_ops_.push_back(load);
      s->StageLoad(load, store);
      tracker.Record(&op->lhs_);
      op->lhs_ = load;
    }
    k->objects_.push_back(op);
  }
  for (size_t i = 0; i < children.size(); ++i) {
    auto s = children[i];
    if (s == nullptr) {
      break;
    }
    if (!((append_mask >> i) & 0x1ull)) {
      stage->AppendStage(s);
    }
  }
  auto ws_size = stage->CodeGen();
  tracker.RecoverClear();
  code_ = std::move(stage->code_);
  *hold_kernel = stage;
  return ws_size;
}

void VectorKernel::InOutReusePlan(const DimArray *dom) {
  for (size_t i = 0; i < objects_.size(); ++i) {
    auto op = objects_[i];
    if (!op->InplaceProp()) {
      op->io_reuse_mask_ = 0;
    } else if (op->obj_id_ == ObjectType::kLoad) {
      bool flatten = dom != nullptr
                       ? *dom == op->nd_.dims()
                       : static_cast<NDLoad *>(op)->ndd_.pointwise_tile_mask == dom_->nd_.data->pointwise_tile_mask;
      op->io_reuse_mask_ = i < 64 && flatten ? 1ull << i : 0;
    } else {
      uint64_t mask = 0;
      op->ForInput([&mask](NDObject *in) { mask |= in->io_reuse_mask_; });
      op->io_reuse_mask_ = mask;
    }
  }
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

void VectorSchedule::SpaceInit() {
  space_records_.clear();
  for (auto op : kernel_->objects_) {
    if (auto ndd = op->Ndd()) {
      space_records_.push_back({0, ndd});
    }
    if (op->IsSimd() && NDObject::meta_.dim_changed[op->obj_id_]) {
      auto &r = space_records_.emplace_back();
      r.bcast_mask = SpaceRecord::OP_MASK;
      r.change_op = op;
    }
  }
}

void VectorSchedule::SpaceSplit(int dim, int64_t npart, int64_t nfactor) {
  ASSERT(npart > 0 && nfactor > 0);
  auto &space = kernel_->DimSpace();
  size_t space_size = space.size();
  ASSERT(dim >= 0 && static_cast<size_t>(dim) < space_size);
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      size_t n = dims.size();
      int64_t orig_dim = dims[dim];
      dims.resize(n + 1);
      for (size_t i = n; i > static_cast<size_t>(dim) + 1; --i) {
        dims[i] = dims[i - 1];
      }
      if (orig_dim != 1) {
        dims[dim] = nfactor;
        dims[dim + 1] = npart;
      } else {
        dims[dim] = 1;
        dims[dim + 1] = 1;
      }
    }
  }
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < space_size) stride.resize(space_size, 0);
    int64_t orig_stride = stride[dim];
    size_t n = stride.size();
    stride.resize(n + 1);
    for (size_t j = n; j > static_cast<size_t>(dim) + 1; --j) {
      stride[j] = stride[j - 1];
    }
    stride[dim] = orig_stride;
    stride[dim + 1] = orig_stride * nfactor;
  }
}

void VectorSchedule::SpaceTrans(int dim1, int dim2) {
  auto &space = kernel_->DimSpace();
  ASSERT(dim1 >= 0 && static_cast<size_t>(dim1) < space.size());
  ASSERT(dim2 >= 0 && static_cast<size_t>(dim2) < space.size());
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      std::swap(r.ndd->dims[dim1], r.ndd->dims[dim2]);
    }
  }
  size_t nd_size = space.size();
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < nd_size) stride.resize(nd_size, 0);
    std::swap(stride[dim1], stride[dim2]);
  }
}

void VectorSchedule::SaveSpace() {
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 1) {
          r.bcast_mask |= (1u << i);
        }
      }
    }
  }
}

class VectorDupHelper {
 public:
  VectorDupHelper(VectorKernel *kernel, int dup_num, RelocAddr *relocs, uint64_t total_quota)
      : kernel_(kernel), dup_num_(dup_num), dup_idx_(0), block_begin_(0), relocs_(relocs),
        free_core_(RoundUp(static_cast<uint64_t>(dup_num), g_system.CoreNum())), remain_quota_(total_quota) {
    uint64_t code_reserve = dup_num * kernel_->ReserveCodeSize();
    encoder_.Reset(&kernel_->code_, Code::kTargetVec, dup_num, code_reserve, free_core_);
  }

  void Reset() {
    int op_index = 0;
    for (auto op : kernel_->objects_) {
      op->Clear(op_index++);
    }
  }

  void DoAppend(uint64_t quota, uint64_t cap_core = 0) {
    ASSERT(dup_idx_ < dup_num_);
    dup_idx_++;
    kernel_->PrepareTiling();
    ASSERT(kernel_->tile_size_);
    auto code_begin = encoder_.ProgData();
    uint64_t code_reserve = kernel_->ReserveCodeSize();
    uint64_t core_limit = CeilDiv(quota * free_core_, remain_quota_);
    if (cap_core != 0 && core_limit > cap_core) {
      core_limit = cap_core;
    }
    remain_quota_ -= quota;
    auto code_end = kernel_->DoCodeGen(core_limit, code_begin, code_reserve);
    uint64_t code_size = code_end - code_begin;
    uint64_t block_dim = kernel_->CompactBlockDim(core_limit);
    ASSERT(kernel_->visit_ == nullptr);
    auto prog = encoder_.Append(Code::GenEntryV(kernel_->tile_num_, block_dim, code_size), code_size);
    encoder_.AssignAiv(block_begin_, block_dim, prog);
    block_begin_ += block_dim;
    for (auto op : kernel_->static_ops_) {
      if (!(op->flags_ & OBJ_FLAG_DEAD)) {
        auto &r = static_cast<NDAccess *>(op)->addr_;
        auto bind = relocs_++;
        bind->Update(r.reloc_);
        kernel_->code_.BindOpFast(*bind, r);
      }
    }
    free_core_ -= block_dim;
  }

  void Append(uint64_t quota, uint64_t cap_core = 0) {
    Reset();
    DoAppend(quota, cap_core);
  }

  void Submit() {
    ASSERT(dup_idx_ == dup_num_);
    encoder_.Submit(block_begin_);
  }

 protected:
  VectorKernel *kernel_;
  PCodeEncoder encoder_;
  int dup_num_;
  int dup_idx_;
  uint64_t block_begin_;
  RelocAddr *relocs_;
  uint64_t free_core_;
  uint64_t remain_quota_;
};

SchGenHelper::~SchGenHelper() { delete []reloc_array_; }
int64_t SchGenHelper::CodeGen() { return -1; }

void SchGenHelper::AllocStride(NDAccess *acc) {
  if (ext_strides_.empty()) {
    ext_strides_.resize(kernel_->static_ops_.size());
  }
  auto &ext = ext_strides_[ext_stride_used_++];
  ext.acc = acc;
  acc->stride_ = &ext.stride;
  auto &dims = acc->nd_.dims();
  auto n = dims.size();
  ext.stride.resize(n);
  int64_t cur_stride = ITEM_SIZE[acc->type_id_];
  for (size_t i = 0; i < n; ++i) {
    ext.stride[i] = cur_stride;
    cur_stride *= dims[i];
  }
}

FractalSchGen::FractalSchGen(VectorKernel *kernel) : SchGenHelper(kernel) {
  for (auto op : kernel->static_ops_) {
    EXCEPTION_IF(!static_cast<NDAccess *>(op)->IsSupportView(), "unsupport view op");
  }
}

int64_t FractalSchGen::CodeGen() {
  constexpr int64_t min_dim_limit = 6;
  int h_idx = -1;
  int64_t item_size = 0;
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto load = static_cast<NDAccess *>(kernel_->static_ops_[i]);
    if (load->stride_ == nullptr) {
      continue;
    }
    auto &stride = *load->stride_;
    auto &dims = load->Ndd()->dims;
    int64_t type_size = ITEM_SIZE[load->type_id_];
    if (stride[0] != type_size && dims[0] >= min_dim_limit) {
      if (h_idx == -1) {
        if (type_size != 2 && type_size != 4) {
          continue;
        }
        for (size_t j = 1; j < stride.size(); ++j) {
          if (stride[j] == type_size) {
            if (dims[j] >= min_dim_limit) {
              h_idx = static_cast<int>(j);
              item_size = type_size;
              load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
            }
            break;
          }
        }
      } else if (static_cast<size_t>(h_idx) < stride.size() && type_size == item_size && stride[h_idx] == type_size && dims[h_idx] >= min_dim_limit) {
        load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
      }
    }
  }
  if (h_idx < 0) {
    return -1;
  }

  class FractalDunGen {
   public:
    FractalDunGen(SchGenHelper &gen, int h_idx, uint64_t item_size)
        : gen_(gen), space_(gen.kernel_->DimSpace()), h_idx_(h_idx) {
      w_fractal_ = g_system.Arch() == AiCoreArch::kAiCore_C310 ? 128 : 16;
      h_fractal_ = item_size == 2 ? 16 : 8;
      int64_t w_size = space_[0];
      w_body_ = w_size / w_fractal_;
      w_tail_ = w_size - w_body_ * w_fractal_;
      int64_t h_size = space_[h_idx];
      h_body_ = h_size / h_fractal_;
      h_tail_ = h_size - h_body_ * h_fractal_;
    }
    void CodeGen() {
      int64_t w_npart = w_body_;
      int64_t h_npart = h_body_;
      int dup_num;
      if (h_tail_ && w_tail_) {
        dup_num = h_body_ && w_body_ ? 4 : (h_body_ || w_body_ ? 2 : 1);
        w_npart++;
        h_npart++;
      } else if (h_tail_) {
        dup_num = h_body_ ? 2 : 1;
        h_npart++;
      } else if (w_tail_) {
        dup_num = w_body_ ? 2 : 1;
        w_npart++;
      } else {
        dup_num = 1;
      }
      gen_.SpaceInit();
      gen_.SpaceSplit(0, w_npart, w_fractal_);
      h_idx_ += 1;
      gen_.SpaceTrans(1, h_idx_);
      gen_.SpaceSplit(1, h_npart, h_fractal_);
      gen_.SaveSpace();
      part_base_ = 1;
      size_t w_part_dim = h_idx_ + 1;
      size_.resize(space_.size());
      size_[2] = space_[2];
      for (size_t i = 3; i < space_.size(); ++i) {
        size_[i] = space_[i];
        if (i != w_part_dim) {
          part_base_ *= space_[i];
        }
      }
      VectorDupHelper helper(gen_.kernel_, dup_num, gen_.ReserveReloc(gen_.kernel_->static_ops_.size() * 4),
                             h_npart * w_npart);
      if (w_tail_ && h_body_) {
        GenDup(helper, 0, w_body_, w_tail_, h_fractal_, h_body_, 1);
      }
      if (h_tail_ && w_body_) {
        GenDup(helper, h_body_, 0, w_fractal_, h_tail_, 1, w_body_);
      }
      if (w_tail_ && h_tail_) {
        GenDup(helper, h_body_, w_body_, w_tail_, h_tail_, 1, 1);
      }
      if (w_body_ && h_body_) {
        GenDup(helper, 0, 0, w_fractal_, h_fractal_, h_body_, w_body_);
      }
      helper.Submit();
    }

    void GenDup(VectorDupHelper &helper, int64_t h_part_off, int64_t w_part_off, int64_t w_fac_size, int64_t h_fac_size,
                int64_t h_part_size, int64_t w_part_size) {
      constexpr int w_factor_dim = 0;
      constexpr int h_factor_dim = 1;
      constexpr int h_part_dim = 2;
      int w_part_dim = h_idx_ + 1;
      size_[w_factor_dim] = w_fac_size;
      size_[h_factor_dim] = h_fac_size;
      size_[h_part_dim] = h_part_size;
      size_[w_part_dim] = w_part_size;
      gen_.ApplySubSpace(size_);
      // TODO: consider broadcast
      for (auto op : gen_.kernel_->static_ops_) {
        auto acc = static_cast<NDAccess *>(op);
        auto &stride = *acc->stride_;
        acc->ViewUpdate(stride[h_part_dim] * h_part_off + stride[w_part_dim] * w_part_off);
      }
      uint64_t part_num = h_part_size * w_part_size;
      helper.Append(part_num, part_base_ * part_num);
    };

    SchGenHelper &gen_;
    const DimArray &space_;
    int64_t h_fractal_;
    int64_t w_fractal_;
    int64_t w_body_;
    int64_t w_tail_;
    int64_t h_body_;
    int64_t h_tail_;
    DimArray size_;
    uint64_t part_base_;
    int h_idx_;
  };

  int64_t result = 0;
  for (auto op : kernel_->objects_) {
    if ((op->obj_id_ == ObjectType::kBroadcastTo || op->obj_id_ == ObjectType::kReduce) &&
        op->lhs_->nd_[0] != op->nd_[0]) {
      result = -1;
      break;
    }
  }
  if (!result) {
    for (auto op : kernel_->static_ops_) {
      auto acc = static_cast<NDAccess *>(op);
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
    FractalDunGen gen(*this, h_idx, item_size);
    gen.CodeGen();
    ResetStrides();
  }
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto op = kernel_->static_ops_[i];
    if (op->CheckFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL)) {
      op->flags_ &= ~OBJ_FLAG_VIEW_LOAD_FRACTAL;
    }
  }
  return result;
}

ConcatSchGen::ConcatSchGen(VectorKernel *kernel, ConcatOp *concat, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), concat_(concat) {
  for (auto op : objects) {
    op->reuse_dep_ = -1;
  }
  auto &slices = concat->slices_;
  for (int i = 0; i < static_cast<int>(slices.size()); ++i) {
    slices[i].input->reuse_dep_ = i;
  }
  for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
    auto op = *it;
    if (int slice_idx = op->reuse_dep_; slice_idx >= 0) {
      op->ForInput([slice_idx](NDObject *in) {
        in->reuse_dep_ = slice_idx;
      });
    }
  }
  // TODO: side away load/store
  slice_ios_.resize(kernel->static_ops_.size());
  for (size_t i = 0; i < slice_ios_.size(); ++i) {
    auto op = kernel->static_ops_[i];
    slice_ios_[i].op = op;
    slice_ios_[i].slice = op->reuse_dep_;
    EXCEPTION_IF(op->reuse_dep_ < 0 && !static_cast<NDAccess *>(op)->IsSupportView(),
                 "concat output expect viewload/viewstore or load/store");
  }
  load_num_ = kernel->load_num_;
  // Reject shared inputs: concat slice loads consumed by non-concat ops (e.g. mul)
  // would be referenced by programs of other slices without being loaded.
  for (auto &slice : slices) {
    auto *load = slice.input;
    if (!load->IsLoad()) continue;
    for (auto *op : kernel->objects_) {
      if (op == concat_ || op->IsLoad() || op->IsStore() || op->reuse_dep_ >= 0) continue;
      bool shared = false;
      op->ForInput([&](NDObject *in) { if (in == load) shared = true; });
      EXCEPTION_IF(shared, "concat input shared by non-concat op is unsupported");
    }
  }
}

int64_t ConcatSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  DimArray size = concat_->nd_.dims();
  int cat_dim = concat_->CatDim();
  auto &static_ops = kernel_->static_ops_;
  for (auto &s : slice_ios_) {
    if (s.slice < 0) {
      s.slice = s.op->nd_[cat_dim] == 1 ? -2 : -1;
      if (auto acc = static_cast<NDAccess *>(s.op); acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int dup_num = concat_->slices_.size();
  VectorDupHelper helper(kernel_, dup_num, ReserveReloc(static_ops.size() * dup_num), size[cat_dim]);
  size_t cat_offset = 0;
  auto ctx = concat_->PartialInit();
  for (int cat_idx = 0; cat_idx < dup_num; ++cat_idx) {
    auto &slice = concat_->slices_[cat_idx];
    auto cat_size = slice.size;
    size[cat_dim] = cat_size;
    concat_->PartialSet(slice.input);
    ApplySubSpace(size);
    static_ops.clear();
    for (size_t i = 0; i < load_num_; ++i) {
      auto &s = slice_ios_[i];
      if (s.slice < 0) {
        auto load = static_cast<NDAccess *>(s.op);
        load->ViewUpdate(s.slice == -1 ? (*load->stride_)[cat_dim] * static_cast<uint64_t>(cat_offset) : 0);
      } else if (s.slice != cat_idx) {
        continue;
      }
      static_ops.push_back(s.op);
    }
    kernel_->load_num_ = static_ops.size();
    for (size_t i = load_num_; i < slice_ios_.size(); ++i) {
      auto &s = slice_ios_[i];
      if (s.slice < 0) {
        auto store = static_cast<NDAccess *>(s.op);
        store->ViewUpdate(s.slice == -1 ? (*store->stride_)[cat_dim] * static_cast<uint64_t>(cat_offset) : 0);
      } else if (s.slice != cat_idx) {
        continue;
      }
      static_ops.push_back(s.op);
    }
    cat_offset += cat_size;
    helper.Append(cat_size);
  }
  concat_->PartialRecover(ctx);
  helper.Submit();
  ResetStrides();
  return 0;
}

SplitSchGen::SplitSchGen(VectorKernel *kernel, SplitOpM *split, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), split_(split) {
  ASSERT(split_->slice_idx_ == 0);
  for (auto op : objects) {
    if (op->IsLoad()) {
      op->reuse_dep_ = -1;
    } else if (op->obj_id_ == kSplitOp) {
      op->reuse_dep_ = static_cast<SplitOp *>(op)->slice_idx_;
    } else {
      int input_idx = -1;
      op->ForInput([&input_idx](NDObject *in) {
        if (in->reuse_dep_ >= 0) {
          input_idx = in->reuse_dep_;
        }
      });
      op->reuse_dep_ = input_idx;
    }
  }
  // TODO: side away load/store
  slice_ios_.resize(kernel->static_ops_.size());
  for (size_t i = 0; i < slice_ios_.size(); ++i) {
    auto op = kernel->static_ops_[i];
    slice_ios_[i].op = op;
    slice_ios_[i].slice = op->reuse_dep_;
    EXCEPTION_IF(op->reuse_dep_ < 0 && !static_cast<NDAccess *>(op)->IsSupportView(),
                 "split input expect viewload/viewstore or load/store");
  }
  // Reject shared/cross-slice consumption in one pass:
  // - shared input: an op consuming both the split input load and a split output
  //   block (directly or via intermediate ops) reads a ViewUpdate-polluted load;
  //   ops reading only loads (e.g. compare) run before split and are safe
  // - cross-slice: an op (e.g. mul) consuming split blocks from different slices
  //   would be assigned to one slice, leaving the other blocks unloaded
  // reuse_dep_ is already propagated by slice above: -1 for loads, slice idx for
  // split blocks, inherited from inputs for intermediate ops, so in->reuse_dep_ >= 0
  // identifies split-chain nodes at any depth.
  auto *load = split_->lhs_;
  for (auto *op : kernel->objects_) {
    if (op == split_ || op->IsLoad() || op->IsStore() || op->obj_id_ == kSplitOp) continue;
    bool uses_load = false;
    bool uses_split_out = false;
    int first_slice = -1;
    bool cross_slice = false;
    op->ForInput([&](NDObject *in) {
      if (load->IsLoad() && in == load) {
        uses_load = true;
      }
      if (in->reuse_dep_ >= 0) {  // split-chain node (block or intermediate)
        uses_split_out = true;
        if (first_slice == -1) {
          first_slice = in->reuse_dep_;
        } else if (in->reuse_dep_ != first_slice) {
          cross_slice = true;
        }
      }
    });
    EXCEPTION_IF(uses_load && uses_split_out,
                 "split input shared by op that also consumes split output is unsupported");
    EXCEPTION_IF(cross_slice,
                 "op consuming split blocks from different slices is unsupported");
  }
}

int64_t SplitSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  int split_dim = split_->split_dim_;
  for (auto &s : slice_ios_) {
    if (s.slice < 0) {
      s.slice = s.op->nd_[split_dim] == 1 ? -2 : -1;
      if (auto acc = static_cast<NDAccess *>(s.op); acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int slice_num = static_cast<int>(split_->siblings_.size());
  int64_t split_dim_size = split_->lhs_->nd_[split_dim];
  VectorDupHelper helper(kernel_, slice_num, ReserveReloc(slice_ios_.size() * slice_num), split_dim_size);
  size_t load_num = kernel_->load_num_;
  auto &static_ops = kernel_->static_ops_;
  int64_t split_offset = 0;
  DimArray size = split_->nd_.dims();
  for (int slice_idx = 0; slice_idx < slice_num; ++slice_idx) {
    int64_t split_size = slice_idx < slice_num - 1 ? split_->split_size_ : split_dim_size - split_->split_size_ * (slice_num - 1);
    size[split_dim] = split_size;
    ApplySubSpace(size);
    helper.Reset();
    static_ops.resize(load_num);
    for (size_t i = 0; i < load_num; ++i) {
      if (auto &s = slice_ios_[i]; s.slice < 0) {
        auto load = static_cast<NDAccess *>(s.op);
        load->ViewUpdate(s.slice == -1 ? (*load->stride_)[split_dim] * static_cast<uint64_t>(split_offset) : 0);
      }
    }
    for (size_t i = load_num; i < slice_ios_.size(); ++i) {
      auto &s = slice_ios_[i];
      if (s.slice < 0) {
        static_ops.push_back(s.op);
        auto store = static_cast<NDAccess *>(s.op);
        store->ViewUpdate(s.slice == -1 ? (*store->stride_)[split_dim] * static_cast<uint64_t>(split_offset) : 0);
      } else if (s.slice == slice_idx) {
        static_ops.push_back(s.op);
        s.op->flags_ &= ~OBJ_FLAG_DEAD;
      } else {
        s.op->flags_ |= OBJ_FLAG_DEAD;
      }
    }
    split_offset += split_size;
    helper.DoAppend(split_size);
  }
  helper.Submit();
  static_ops.resize(load_num);
  for (size_t i = load_num; i < slice_ios_.size(); ++i) {
    static_ops.push_back(slice_ios_[i].op);
  }
  ResetStrides();
  return 0;
}

int64_t DupTilingSchGen::DupCodeGen(int split_dim, int64_t truck_size) {
  DimArray dim_space = kernel_->DimSpace();
  int64_t split_size = dim_space[split_dim];
  int64_t body_size = split_size / truck_size * truck_size;
  int64_t tail_size = split_size - body_size;
  if (tail_size == 1) { // avoid x view
    body_size = split_size / 2;
    tail_size = split_size  - body_size;
  }
  ASSERT(body_size > 1 && tail_size > 1);
  auto &static_ops = kernel_->static_ops_;
  uint64_t bcast_mask = 0;
  for (size_t i = 0; i < static_ops.size(); ++i) {
    auto acc = static_cast<NDAccess *>(static_ops[i]);
    if (acc->nd_[split_dim] == 1) {
      bcast_mask |= 1ull << i;
    }
    if (acc->stride_ == nullptr) {
      AllocStride(acc);
    }
  }
  auto reloc_ios = [&static_ops, split_dim, bcast_mask](int64_t offset) {
    for (size_t i = 0; i < static_ops.size(); ++i) {
      auto acc = static_cast<NDAccess *>(static_ops[i]);
      acc->ViewUpdate(bcast_mask >> i & 1ull ? 0 : (*acc->stride_)[split_dim] * offset);
    }
  };
  SpaceInit();
  SaveSpace();
  VectorDupHelper helper(kernel_, 2, ReserveReloc(static_ops.size() * 2), split_size);
  dim_space[split_dim] = tail_size;
  ApplySubSpace(dim_space);
  reloc_ios(body_size);
  helper.Append(tail_size);
  dim_space[split_dim] = body_size;
  ApplySubSpace(dim_space);
  reloc_ios(0);
  helper.Append(body_size);
  helper.Submit();
  ResetStrides();
  return 0;
}

VKernelS::~VKernelS() {
  delete stage_kernel_;
  delete sch_gen_;
  delete dup_gen_;
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
  PrepareTiling();
  if (unlikely(!tile_size_)) {
    ProcessIdle();
    return 0;
  }
  if (sch_gen_) {
    if (auto ret = sch_gen_->CodeGen(); ret != -1) {
      return ret;
    }
  }
  uint64_t core_limit = g_system.CoreNum();
  int64_t live_peak = Analyze();
  int64_t tile_size_limit = TileSizeLimit(live_peak);
  if (likely(tiles_.empty())) {
    TileUpdate update;
    ShapeTiling(tile_size_limit, core_limit, update);
    if (update.tile_size * 8 < tile_size_limit && update.tile_num > 256 && tile_region_.depth > 1 &&
        !(flags_ & K_FLAG_DIS_DUP_TILING)) {
      if (auto ret = DupTilingGen(tile_region_, update.tile_size, tile_size_limit); ret != -1) {
        return ret;
      }
    }
    ApplyTiling(update);
  } else {
    ManualTiling();
  }
  // simd_width
  if (tile_info_.flags & ObjectMeta::kSimdDim) {
    AlignSimd(tile_size_limit);
  }
  return TileGen(live_peak, core_limit);
}

int64_t VKernelS::DupTilingGen(const TileRegion &region, int64_t tile_size, int64_t tile_size_limit) {
  if (dup_gen_ == nullptr) {
    for (auto op : static_ops_) {
      if (!static_cast<NDAccess *>(op)->IsSupportView()) {
        flags_ |= K_FLAG_DIS_DUP_TILING;
        return -1;
      }
    }
    if (comm_op_ || shard_) {
      flags_ |= K_FLAG_DIS_DUP_TILING;
      return -1;
    }
    dup_gen_ = new DupTilingSchGen(this);
  }
  int split_dim = region.starts[region.depth - 1];
  if (int end = region.starts[region.depth - 2] - 1; end > split_dim) {
    TileParam tp;
    tp.start = split_dim;
    tp.end = end;
    tp.num = 1;
    tp.tile = region.spaces[region.depth - 1];
    tp.tail = 0;
    TileProp(tp);
  }
  int64_t truck_size;
  if (split_dim == 0) {
    truck_size = tile_size_limit / block_align_ * block_align_;
  } else {
    truck_size = tile_size_limit / (tile_size / region.last_tile);
  }
  return dup_gen_->DupCodeGen(split_dim, truck_size);
}

void VKernelS::Append(NDObject *obj) { build_ops_.push_back(obj); }

bool VKernelS::NormBuild() {
  if (IsDynamic()) {
    Clear();
    if (static_ops_.empty()) {
      StaticInit(build_ops_);
      SchInit(build_ops_);
      Optimize<true>(build_ops_, nullptr);
    }
    if (!Normalize(true)) {
      return false;
    }
  } else {
    if (!Normalize(true)) {
      return false;
    }
    Optimize<false>(build_ops_, nullptr);
    StaticInit(objects_);
    SchInit(objects_);
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
  helper.DumpGraph(indent, "vec", build_ops_);
}

void VKernelS::ManualTiling() {
  auto &dims = DimSpace();
  TileParam tp;
  tile_region_.Reset();
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
    TileProp(tp);
    if (t.start > 0) {
      tile_size_ = tile_size_ / space * tp.tile;
      tile_num_ *= tp.num;
    } else {
      tile_size_ = RoundUp<int64_t>(tp.tile, LeadAlign());
      align_space_ = tile_size_;
      tile_num_ *= tp.num;
    }
    tile_region_.Add(t.start, space, t.num, tp.tile);
    if (tp.tail && tiles_.size() == 1) {
      tile_region_.tail_size = tp.tail;
      tile_region_.tail_dim = t.start;
    }
  }
}

_SpecVector::~_SpecVector() {
  if (fall_kernel_) {
    delete fall_kernel_;
  }
}

void _SpecVector::Append(NDObject *obj) {
  if (obj->IsSimd()) {
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
  }
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
  auto reduce_fall_check = [this](int64_t tile_size_limit) -> bool {
    if (!post_reduces_.empty()) {
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
  int64_t live_peak = 0;
  int64_t tile_size_limit = 0;
  if constexpr (dyn_shape) {
    Clear();
    if (static_ops_.empty()) {
      StaticInit(build_ops_);
      Optimize<true>(build_ops_, nullptr);
    }
    if (!Normalize(true)) {
      return FallCodeGen();
    }
    BuildDomain();
    PrepareTiling();
    live_peak = Analyze();
    tile_size_limit = TileSizeLimit(live_peak);
    if (reduce_fall_check(tile_size_limit)) {
      return FallCodeGen();
    }
  } else {
    if (!Normalize(true)) {
      return FallCodeGen();
    }
    GraphTracker tracker;
    Optimize<false>(build_ops_, &tracker);
    StaticInit(objects_);
    BuildDomain();
    PrepareTiling();
    live_peak = Analyze();
    tile_size_limit = TileSizeLimit(live_peak);
    if (reduce_fall_check(tile_size_limit)) {
      tracker.Recover();
      return FallCodeGen();
    }
  }
  use_fall_ = false;
  if (unlikely(!tile_size_)) {
    ProcessIdle();
    return 0;
  }
  uint64_t core_limit = g_system.CoreNum();
  TileUpdate update;
  ShapeTiling(tile_size_limit, core_limit, update);
  ApplyTiling(update);
  if (tile_info_.flags & ObjectMeta::kSimdDim) {
    AlignSimd(tile_size_limit);
  }
  return TileGen(live_peak, core_limit);
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
    {
      std::vector<int> uf(stage_ids_.size());
      for (size_t i = 0; i < uf.size(); ++i) uf[i] = i;
      auto Find = [&uf](int i) -> int {
        while (uf[i] != i) { uf[i] = uf[uf[i]]; i = uf[i]; }
        return i;
      };
      for (size_t i = 0; i < stage_ids_.size(); ++i) {
        build_ops_[i]->index_ = i;
      }
      for (size_t i = 0; i < stage_ids_.size(); ++i) {
        build_ops_[i]->ForInput([&](NDObject *in) {
          auto in_idx = in->index_;
          while (static_cast<size_t>(in_idx) >= stage_ids_.size() || in != build_ops_[in_idx]) {
            in = in->lhs_;
            in_idx = in->index_;
          }
          if (stage_ids_[in_idx] == stage_ids_[i]) {
            int a = Find(i), b = Find(in->index_);
            if (a != b) uf[a] = b;
          }
        });
      }
      std::vector<int> comp_id(stage_ids_.size(), -1);
      int new_sid = 0;
      for (int s = 0; s <= last_stage_; ++s) {
        std::vector<int> roots;
        for (size_t i = 0; i < stage_ids_.size(); ++i) {
          if (stage_ids_[i] != s) continue;
          int root = Find(i);
          int cid = 0;
          for (; cid < (int)roots.size(); ++cid)
            if (roots[cid] == root) break;
          if (cid == (int)roots.size()) roots.push_back(root);
          comp_id[i] = new_sid + cid;
        }
        new_sid += roots.size();
      }
      for (size_t i = 0; i < stage_ids_.size(); ++i) {
        if (comp_id[i] >= 0) stage_ids_[i] = comp_id[i];
      }
      last_stage_ = new_sid - 1;
    }
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
      if (src_op->obj_id_ == ObjectType::kStore) {
        if (auto sstore = GET_SSTORE(helper.GetClone(src_op->lhs_))) {
          stage_kernel->Remap(static_cast<NDAccess *>(sstore), static_cast<NDAccess *>(src_op));
          helper.clones_.push_back(sstore);
          auto stage = stage_kernel->StageAt(out_sid);
          for (auto &ss : stage->sstores_) {
            if (ss.store == sstore) {
              ss.is_out = true;
              break;
            }
          }
          continue;
        }
      }
      auto clone_op = src_op->Clone(helper);
      INIT_SSTORE(clone_op);
      clone_op->index_ = i;
      helper.clones_.push_back(clone_op);
      if (!clone_op->IsSimd()) {
        stage_kernel->Remap(static_cast<NDAccess *>(clone_op), static_cast<NDAccess *>(src_op));
        if (clone_op->obj_id_ == ObjectType::kStore) {
          SET_SSTORE(clone_op->lhs_, clone_op);
        }
      }
      clone_op->ForInput([this, out_sid, stage_kernel, &helper](NDObject *&in) {
        auto in_sid = stage_ids_[in->index_];
        if (in_sid == out_sid) {
          return;
        }
        auto out_stage = stage_kernel->StageAt(out_sid);
        NDAccess *load;
        if (in->IsLoad()) {
          load = static_cast<NDAccess *>(in->Clone(helper));
          stage_kernel->Remap(load, static_cast<NDAccess *>(build_ops_[in->index_]));
        } else {
          auto store = GET_SSTORE(in);
          if (store == nullptr) {
            store = new NDStore(in);
            SET_SSTORE(in, store);
            auto in_stage = stage_kernel->StageAt(in_sid);
            in_stage->kernel->Append(store);
            in_stage->StageStore(store);
          }
          load = new NDLoad(nullptr, in->shape_ref_, in->type_id_);
          out_stage->StageLoad(load, store);
        }
        out_stage->kernel->Append(load);
        in = load;
      });
      stage_kernel->StageAt(out_sid)->kernel->Append(clone_op);
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
  ~SpecVecStage() override { kernel = nullptr; }
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

uint64_t SpecVecBase::CodeGen() {
  auto tile_size_limit = LazyTileLimit();
  uint64_t core_limit = g_system.CoreNum();
  TileUpdate update;
  ShapeTiling(tile_size_limit, core_limit, update);
  ApplyTiling(update);
  if (tile_info_.flags & ObjectMeta::kSimdDim) {
    AlignSimd(tile_size_limit);
  }
  return TileGen(live_peak_, core_limit);
}

bool SpecVecBase::SpecBuild() {
  if (fall_opt_ & FALL_CUSTOM_SPLIT) {
    fall_opt_ &= ~FALL_CUSTOM_SPLIT;
    if (CustomSpec()) {
      SplitBuild();
      return false;
    }
  }
  if (fall_opt_ & FALL_RESHAPE) {
    fall_opt_ &= ~FALL_RESHAPE;
    if (ReshapeSpec()) {
      SplitBuild();
      return false;
    }
  }
  if (fall_opt_ & FALL_PERMUTE) {
    fall_opt_ &= ~FALL_PERMUTE;
    if (PermuteSpec()) {
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
  int cut_area_end = ctx_.area_size_;
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
    while (stack.size() > cut_end) {
      auto top = stack.back();
      stack.pop_back();
      top->ForInput([this, aid, cut_area_end, &stack](NDObject *in) {
        auto in_meta = GetMeta(in);
        if (in_meta->aid == -1) {
          stack.push_back(in);
          in_meta->aid = aid;
        } else if (auto in_aid = ctx_.RootArea(in_meta->aid); in_aid != aid) {
          auto &area = ctx_.areas_[aid];
          if (in_aid >= cut_area_end) {
            ctx_.MergeArea(aid, in_aid);
            area.MergeMask() |= ctx_.areas_[in_aid].MergeMask();
            area.UnMergeMask() |= ctx_.areas_[in_aid].UnMergeMask();
          } else if (in_meta->IsCut()) {
            area.UnMergeMask() |= 1ul << in_aid;
          } else {
            area.MergeMask() |= 1ul << in_aid;
          }
        }
      });
    }
  }
  for (int aid = cut_area_end; aid < ctx_.area_size_; ++aid) {
    if (auto &area = ctx_.areas_[aid]; area.parent == aid) {
      uint32_t merge_mask = area.MergeMask();
      auto unmerge_mask = area.UnMergeMask();
      int rid = aid;
      while (merge_mask) {
        auto cut_aid = 31 - __builtin_clz(merge_mask);
        merge_mask &= ~(1ul << cut_aid);
        auto cut_rid = ctx_.RootArea(cut_aid);
        if (cut_rid == rid) continue;
        if (unmerge_mask & static_cast<uint32_t>((1ull << cut_rid) | ctx_.areas_[cut_rid].child_mask)) continue;
        if (cut_rid < rid) {
          ctx_.MergeArea(cut_rid, rid);
          rid = cut_rid;
        } else {
          ctx_.MergeArea(rid, cut_rid);
        }
      }
      area.ClearMask();
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
    if (op->obj_id_ == ObjectType::kBroadcastTo) {
      auto in = op->lhs_;
      if (in->obj_id_ != ObjectType::kBroadcastTo && in->IsLoad() && !GetMeta(in)->IsCut()) {
        ctx_.spec_ops_.push_back(op);
        GetMeta(in)->SetCut();
      }
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

bool SpecVecBase::CustomSpec() {
  ctx_.ResetSpec();
  auto &spec_ops = ctx_.spec_ops_;
  size_t custom_begin = spec_ops.size();
  for (auto op : objects_) {
    InitMeta(op);
  }
  for (size_t i = 0; i < load_num_; ++i) {
    auto op = static_ops_[i];
    if (op->CheckFlag(OBJ_FLAG_LOAD_BIND)) {
      auto store = static_cast<NDAccess *>(op)->LoadBind();
      if (!GetMeta(store)->IsCut()) {
        spec_ops.push_back(store);
        GetMeta(store)->SetCut();
      }
    }
  }
  size_t cut_begin = spec_ops.size();
  if (cut_begin == custom_begin) {
    return false;
  }
  for (size_t i = custom_begin; i < cut_begin; ++i) {
    spec_ops.push_back(spec_ops[i]->lhs_);
  }
  SplitPlan(cut_begin);
  for (size_t i = 0; i < load_num_; ++i) {
    auto op = static_ops_[i];
    if (op->CheckFlag(OBJ_FLAG_LOAD_BIND)) {
      auto load = static_cast<NDAccess *>(op);
      auto store = load->LoadBind();
      auto store_aid = ctx_.RootArea(GetMeta(store)->aid);
      auto load_aid = ctx_.RootArea(GetMeta(load)->aid);
      if (store_aid == load_aid) {
        ctx_.areas_[store_aid].ext_opt = FALL_CUSTOM_SPLIT;
      } else {
        ctx_.tracker_.Record(reinterpret_cast<NDObject **>(&load->addr_.gm));
      }
    }
  }
  spec_ops.resize(custom_begin);
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
  uint64_t side_mask = 0;
  for (auto op : objects_) {
    op->prop_id_ = ctx_.RootArea(GetMeta(op)->aid);
    if (op->obj_id_ != ObjectType::kReshape) {
      op->ForInput([&side_mask, op](NDObject *in) {
        if (in->prop_id_ != op->prop_id_) {
          side_mask |= 1ull << in->prop_id_;
        }
      });
    }
  }
  DomainUnifier affine(objects_, true);
  auto &areas = ctx_.areas_;
  bool fail_touch = false;
  uint64_t fail_affine = 0;
  for (size_t i = reshape_begin; i < cut_begin; ++i) {
    auto op = static_cast<ReshapeOp *>(spec_ops[i]);
    auto lhs_prop = op->lhs_->prop_id_;
    if (op->prop_id_ == lhs_prop) {
      areas[op->prop_id_].u32 = 1;
      fail_touch = true;
    } else if (((side_mask >> lhs_prop) & 1ull) || !affine.Process(op, areas[lhs_prop].u32)) {
      fail_affine |= 1ull << op->prop_id_;
    }
  }
  if (fail_affine) {
    for (size_t i = reshape_begin; i < cut_begin; ++i) {
      auto op = spec_ops[i];
      auto aid = ctx_.RootArea(GetMeta(op)->aid);
      if (op->lhs_->nd_.dims() == op->nd_.dims() && !((fail_affine >> op->prop_id_) & 1ull)) {
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

template <int64_t fill>
void PermuteDimArray(DimArray &dims, const DimArray &perm) {
  size_t P = perm.size();
  size_t N = dims.size();
  if (N < P) {
    dims.resize(P);
    for (size_t k = N; k < P; ++k) dims[k] = fill;
  }
  DimArray old_dims = dims;
  for (size_t k = 0; k < P; ++k) {
    dims[k] = old_dims[perm[k]];
  }
}

static bool IsRangeSplited(const DimArray &inv_perm, const DimArray &small, const DimArray &big) {
  int64_t small_size = small.size();
  int64_t big_size = big.size();
  size_t i = 0;
  while (i < inv_perm.size()) {
    auto e1 = inv_perm[i] < small_size ? small[inv_perm[i]] : 1;
    auto e2 = inv_perm[i] < big_size ? big[inv_perm[i]] : 1;
    i++;
    if (e1 != e2) {
      break;
    }
  }
  while (i < inv_perm.size()) {
    auto e1 = inv_perm[i] < small_size ? small[inv_perm[i]] : 1;
    i++;
    if (e1 > 1) {
      break;
    }
  }
  while (i < inv_perm.size()) {
    auto e1 = inv_perm[i] < small_size ? small[inv_perm[i]] : 1;
    auto e2 = inv_perm[i] < big_size ? big[inv_perm[i]] : 1;
    if (e1 != e2) {
      return true;
    }
    i++;
  }
  return false;
};

bool SpecVecBase::PermPropCheck(int prop, const DimArray &inv_perm) {
  constexpr uint64_t black_mask = 1ull << kGatherLoad | 1ull << kLoad | 1ull << kStore | 1ull << kPermute;
  for (auto obj : objects_) {
    if (obj->prop_id_ != prop) continue;
    if ((1ull << obj->obj_id_) & black_mask) {
      return false;
    } else if (obj->obj_id_ == ObjectType::kBroadcastTo) {
      if (IsRangeSplited(inv_perm, obj->lhs_->nd_.dims(), obj->nd_.dims())) {
        return false;
      }
    } else if (obj->obj_id_ == ObjectType::kReduce) {
      if (IsRangeSplited(inv_perm, obj->nd_.dims(), obj->lhs_->nd_.dims())) {
        return false;
      }
    }
  }
  return true;
}

void SpecVecBase::PermPropUpdate(int prop, const DimArray &perm) {
  for (auto obj : objects_) {
    if (obj->prop_id_ != prop) continue;
    if (auto ndd = obj->Ndd(); ndd != nullptr) {
      PermuteDimArray<1>(ndd->dims, perm);
    }
    if (obj->obj_id_ == ObjectType::kViewLoad) {
      PermuteDimArray<0>(static_cast<NDViewLoad *>(obj)->Stride(), perm);
    } else if (obj->obj_id_ == ObjectType::kViewStore) {
      PermuteDimArray<0>(static_cast<NDViewStore *>(obj)->Stride(), perm);
    } else {
      obj->DimChanged();
    }
  }
}

bool SpecVecBase::PermuteSpec() {
  ctx_.ResetSpec();
  auto &spec_ops = ctx_.spec_ops_;
  size_t permute_begin = spec_ops.size();
  for (auto op : objects_) {
    InitMeta(op);
    if (op->obj_id_ == ObjectType::kPermute && !op->CheckFlag(OBJ_FLAG_BROKER_AFFINED) && !GetMeta(op->lhs_)->IsCut()) {
      spec_ops.push_back(op);
      GetMeta(op->lhs_)->SetCut();
    }
  }
  size_t cut_begin = spec_ops.size();
  if (cut_begin == permute_begin) return false;
  for (size_t i = permute_begin; i < cut_begin; ++i) {
    spec_ops.push_back(spec_ops[i]->lhs_);
  }
  SplitPlan(cut_begin);
  uint64_t side_mask = 0;
  for (auto op : objects_) {
    op->prop_id_ = ctx_.RootArea(GetMeta(op)->aid);
    if (op->obj_id_ != ObjectType::kPermute) {
      op->ForInput([&side_mask, op](NDObject *in) {
        if (in->prop_id_ != op->prop_id_) {
          side_mask |= 1ull << op->prop_id_;
        }
      });
    }
  }
  bool fail = false;
  for (size_t i = permute_begin; i < cut_begin; ++i) {
    auto perm_op = static_cast<PermuteOp *>(spec_ops[i]);
    int out_prop = perm_op->prop_id_;
    int in_prop = perm_op->lhs_->prop_id_;
    bool fall_back = (out_prop == in_prop) || ((side_mask >> out_prop) & 1ull);
    if (!fall_back) {
      auto &perm = perm_op->GetNddPerm();
      DimArray inv_perm;
      inv_perm.resize(perm.size());
      for (size_t k = 0; k < perm.size(); ++k) {
        inv_perm[perm[k]] = k;
      }
      if (PermPropCheck(in_prop, inv_perm)) {
        PermPropUpdate(in_prop, perm);
        perm_op->SetFlag(OBJ_FLAG_BROKER_AFFINED);
      } else if (PermPropCheck(out_prop, perm)) {
        PermPropUpdate(out_prop, inv_perm);
        perm_op->SetFlag(OBJ_FLAG_BROKER_AFFINED);
      } else {
        fall_back = true;
      }
    }
    if (fall_back) {
      auto aid = ctx_.RootArea(GetMeta(perm_op)->aid);
      ctx_.areas_[aid].ext_opt = FALL_PERMUTE;
      fail = true;
    }
  }
  spec_ops.resize(permute_begin);
  return fail;
}

namespace {
struct _SpecSwapLoad : public NDLoad {
  _SpecSwapLoad(NDAccess *store)
      : NDLoad(store->addr_.gm, store->shape_ref_, store->type_id_), store_(store) {}
  void Normalize(std::vector<NDObject *> &run_ops) override {
    ndd_.Reset(store_->nd_.dims());
  }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "SwapLoad"; }
  NDAccess *store_;
};

struct _SpecSwapViewLoad : public NDViewLoad {
  _SpecSwapViewLoad(NDAccess *store)
      : NDViewLoad(store->addr_.gm, store->shape_ref_, &stride_data_, store->type_id_), store_(store) {
    auto shape = store->shape_ref_;
    stride_data_.Resize(shape->size);
    int64_t stride = 1;
    for (int i = shape->size - 1; i >= 0; --i) {
      stride_data_[i] = stride;
      stride *= shape->data[i];
    }
  }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "SwapViewLoad"; }
  NDAccess *store_;
  ShapeWithRef stride_data_;
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
  for (size_t i = 0; i < load_num_; ++i) {
    auto load = static_cast<NDAccess *>(static_ops_[i]);
    auto stage = ctx_.areas_[ctx_.RootArea(GetMeta(load)->aid)].stage;
    if (load->CheckFlag(OBJ_FLAG_STAGE_IO)) {
      stage->StageLoad(load, load->obj_id_ == ObjectType::kViewLoad ? static_cast<_SpecSwapViewLoad *>(load)->store_
                                                                    : static_cast<_SpecSwapLoad *>(load)->store_);
    } else if (load->CheckFlag(OBJ_FLAG_LOAD_BIND)) {
      auto store = static_cast<NDAccess *>(load->addr_.gm);
      stage->StageLoad(load, store);
    }
  }
  for (size_t i = load_num_; i < static_ops_.size(); ++i) {
    auto store = static_ops_[i];
    auto out_aid = ctx_.RootArea(GetMeta(store)->aid);
    ASSERT(ctx_.RootArea(GetMeta(store->lhs_)->aid) == out_aid);
    auto stage = ctx_.areas_[out_aid].stage;
    if (store->obj_id_ == ObjectType::kStore) {
      GetMeta(store->lhs_)->store = static_cast<uint16_t>(stage->sstores_.size());
    }
    stage->StageStore(static_cast<NDAccess *>(store), !(store->flags_ & (OBJ_FLAG_STAGE_IO | OBJ_FLAG_STORE_TEMP)));
  }
  for (auto op : objects_) {
    auto out_aid = ctx_.RootArea(GetMeta(op)->aid);
    auto stage = ctx_.areas_[out_aid].stage;
    if (op->IsSimd()) {
        op->ForInput([this, out_aid, stage, op](NDObject *&in) {
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
            if (op->obj_id_ != kReshape) {
              GetMeta(in)->recent_load = ctx_.spec_ops_.size();
            }
            ctx_.spec_ops_.push_back(clone);
            clone->Normalize(stage->spec_k_.objects_);
            stage->spec_k_.SplitAppend(clone);
            stage->StageLoad(static_cast<NDAccess *>(clone), static_cast<NDAccess *>(in));
            GetMeta(clone)->aid = out_aid;
            ctx_.tracker_.Record(&in);
            in = clone;
          } else {
            auto in_stage = ctx_.areas_[in_aid].stage;
            auto store_idx = GetMeta(in)->store;
            auto store = store_idx < OpMeta::IO_END ? in_stage->sstores_[store_idx].store : nullptr;
            if (store == nullptr) {
              store = new NDStore(in);
              store->SetFlag(OBJ_FLAG_STAGE_IO);
              GetMeta(in)->store = static_cast<uint16_t>(in_stage->sstores_.size());
              ctx_.spec_ops_.push_back(store);
              store->Normalize(in_stage->spec_k_.objects_);
              in_stage->spec_k_.SplitAppend(store);
              in_stage->StageStore(store);
            }
            NDAccess *load;
            if (op->obj_id_ != kPermute) {
              load = new _SpecSwapLoad(store);
              if (op->obj_id_ != kReshape) {
                GetMeta(in)->recent_load = ctx_.spec_ops_.size();
              }
            } else {
              load = new _SpecSwapViewLoad(store);
            }
            load->SetFlag(OBJ_FLAG_STAGE_IO);
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
    auto lhs = obj->lhs_;
    if (lhs->obj_id_ == ObjectType::kReshape) {
      obj->lhs_ = lhs->lhs_;
    } else if (lhs->IsLoad()) {
      // TODO: support load split
      obj->lhs_ = new CopyOp(lhs);
      VKernelS::Append(obj->lhs_);
    }
  } else if (obj->obj_id_ == ObjectType::kPermute) {
    if (auto lhs = obj->lhs_; lhs->IsLoad() && lhs->obj_id_ != ObjectType::kViewLoad) {
      obj->lhs_ = new CopyOp(lhs);
      VKernelS::Append(obj->lhs_);
    }
    fall_opt_init_ |= FALL_PERMUTE;
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

void SpecVecKernel::SpecInit() {
  // dead code elim
  for (auto op : build_ops_) {
    op->reuse_dep_ = op->IsStore() ?  1 : 0;
  }
  for (auto it = build_ops_.rbegin(); it != build_ops_.rend(); ++it) {
    if (auto op = *it; op->reuse_dep_) {
      op->ForInput([](NDObject *in) { in->reuse_dep_ = 1;});
    }
  }
  size_t index = 0;
  for (auto op : build_ops_) {
    if (!op->reuse_dep_) {
      if (!op->IsSimd()) {
        auto &addr = static_cast<NDAccess *>(op)->addr_;
        addr.Update(&addr.data);
      }
      context_.spec_ops_.push_back(op);
      context_.spec_begin_++;
    } else {
      build_ops_[index++] = op;
    }
  }
  build_ops_.resize(index);
  StaticInit(build_ops_);
  if ((fall_opt_init_ & FALL_BROADCAST) && static_ops_.size() < 4) {
    fall_opt_init_ &= ~FALL_BROADCAST;
  }
  for (size_t i = 0; i < load_num_; ++i) {
    if (static_ops_[i]->CheckFlag(OBJ_FLAG_LOAD_BIND)) {
      fall_opt_init_ |= FALL_CUSTOM_SPLIT;
    }
  }
}

uint64_t SpecVecKernel::CodeGen() {
  if (static_ops_.empty()) {
    SpecInit();
  }
  context_.Reset();
  Clear();
  fall_opt_ = fall_opt_init_;
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
  } else if (objects_.empty()) {
    DumpRefHelper helper(oss);
    helper.DumpGraph(indent, "spec", build_ops_);
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
