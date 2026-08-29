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

#include <algorithm>
#include <queue>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "pass.h"

namespace dvm::pass {

namespace {
inline std::vector<NDObject *> GetPreds(NDObject *obj) {
  std::vector<NDObject *> res;
  obj->ForInput([&res](NDObject *op) { res.push_back(op); });
  return res;
}

size_t GetInputsNum(NDObject *obj) {
  if (obj->lhs_ == nullptr) {
    return 0;
  }
  if (obj->rhs_ == nullptr) {
    return 1;
  }
  return 2 + (obj->CheckFlag(OBJ_FLAG_XHS) ? static_cast<FlexOp *>(obj)->xhs_->in_num : 0);
}

constexpr int STATIC_LIVE = 2;
constexpr int DYN_LIVE = 1;

size_t LivenessAnalyze(BasicBlock &bb, std::unordered_map<NDObject *, int> &lives) {
  for (NDObject *obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    if (obj->IsLoad()) {
      lives[obj] = STATIC_LIVE;
    } else if (obj->IsStore()) {
      lives[obj->lhs_] = STATIC_LIVE;
    }
  }
  size_t current = lives.size();
  size_t peak = current;
  for (auto obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    if (!lives.count(obj)) {
      continue;
    }
    obj->ForInput([&lives, &current](NDObject *in) {
      if (auto it = lives.find(in); it == lives.end()) {
        current++;
        lives[in] = DYN_LIVE;
      }
    });
    if (current > peak) {
      peak = current;
    }
    if (lives[obj] == DYN_LIVE) {
      current--;
    }
  }
  return peak;
}

std::vector<NDObject *> ReorderObjectsHeuristic(BasicBlock &bb, std::unordered_map<NDObject *, int> &lives) {
  // Preprocess, calculate points of all objects
  std::unordered_map<NDObject *, int64_t> points;
  std::unordered_map<NDObject *, int64_t> heights;
  constexpr int64_t B1 = -1;
  constexpr int64_t B2 = 1000;
  constexpr int64_t B3 = 1000000;
  for (NDObject *obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    int64_t height = 0;
    for (auto user : bb.GetUsers(obj)) {
      if (auto h = heights[user] + 1; h > height) {
        height = h;
      }
    }
    heights[obj] = height;
    int64_t point = height * B1;
    if (!obj->IsSimd()) {
      point += B3;
    }
    obj->ForInput([&lives, &point](NDObject *in) {
      if (lives[in] == DYN_LIVE) {
        point += B2;
      }
    });
    points[obj] = point;
  }
  auto compare = [&points](NDObject *a, NDObject *b) { return points[a] < points[b]; };
  std::priority_queue<NDObject *, std::vector<NDObject *>, decltype(compare)> pq(compare);
  std::unordered_map<NDObject *, uint32_t> in_degrees;
  for (NDObject *obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    in_degrees[obj] = GetInputsNum(obj);
    if (in_degrees[obj] == 0) {
      pq.push(obj);
    }
  }
  std::vector<NDObject *> res;
  res.reserve(bb.size());
  while (!pq.empty()) {
    auto obj = pq.top();
    pq.pop();
    res.emplace_back(obj);
    for (auto user : bb.GetUsers(obj)) {
      --in_degrees[user];
      if (in_degrees[user] == 0) {
        pq.push(user);
      }
    }
  }
  return res;
}
}  // namespace

void ObjectList::Build(const std::vector<NDObject *> &objects, bool reindex) {
  // build linked list from objects
  NDObject *last_object = &sentinel_;
  int index = 0;
  for (auto object : objects) {
    SetNext(last_object, object);
    SetPrev(object, last_object);
    last_object = object;
    if (reindex) {
      last_object->index_ = index;
      ++index;
    }
  }
  SetNext(last_object, &sentinel_);
  SetPrev(&sentinel_, last_object);
  if (reindex) {
    size_ = capacity_ = objects.size();
  }
}

BasicBlock::BasicBlock(const std::vector<NDObject *> &objects, GraphTracker *tracker) : tracker_(tracker) {
  // build linked list from objects
  ObjectList::Build(objects, true);
  for (auto obj : objects) {
    SetHead(obj, -1);
  }
  edges_.reserve(objects.size() * 2);
  for (auto obj : objects) {
    obj->ForInput([this, obj](NDObject *pred) { this->AddUser(pred, obj); });
  }
}

template <bool if_update_index>
std::vector<NDObject *> BasicBlock::ToVector() {
  std::vector<NDObject *> res;
  res.reserve(size());
  int i = 0;
  for (NDObject *iter = Begin(); iter != End(); iter = Next(iter)) {
    if constexpr (if_update_index) {
      iter->index_ = i++;
    }
    res.push_back(iter);
  }
  return res;
}

// ToVector is defined in this translation unit.  Keep the non-reindexing
// specialization available to pass implementations in other translation
// units as well.
template std::vector<NDObject *> BasicBlock::ToVector<false>();

void BasicBlock::Export(std::vector<NDObject *> &objects) {
  objects.clear();
  objects.reserve(size());
  int i = 0;
  for (NDObject *iter = Begin(); iter != End(); iter = Next(iter)) {
    iter->index_ = i++;
    ObjectList::Prev(iter)->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    objects.push_back(iter);
  }
  ObjectList::Prev(End())->insn_ = nullptr;
}

NDObject *BasicBlock::Insert(NDObject *pos, NDObject *object) {
  if (pos == object) {
    return object;
  }
  ObjectList::Insert(pos, object);
  SetHead(object, -1);
  for (auto pred : GetPreds(object)) {
    AddUser(pred, object);
  }
  // object should be deleted by owner
  news_.push_back(object);
  return object;
}

void BasicBlock::Erase(NDObject *object) {
  ObjectList::Erase(object);
  SetHead(object, -1);
  for (auto pred : GetPreds(object)) {
    auto idx = GetHead(pred);
    auto last = idx;
    while (idx != -1) {
      if (edges_[idx].user == object) {
        if (idx == GetHead(pred)) {
          SetHead(pred, edges_[idx].next);
        } else {
          edges_[last].next = edges_[idx].next;
        }
        break;
      }
      last = idx;
      idx = edges_[idx].next;
    }
    ASSERT(idx != -1);
  }
  dels_.push_back(object);
}

NDObject *BasicBlock::Move(NDObject *pos, NDObject *obj) {
  if (pos == obj || Prev(pos) == obj) {
    return obj;
  }
  ObjectList::Erase(obj);
  ObjectList::Insert(pos, obj);
  return obj;
}

void BasicBlock::UpdateInput(NDObject *obj, NDObject *old, NDObject *update) {
  if (obj->GetObjectType() == kConcat) {
    for (auto &s : static_cast<ConcatOp *>(obj)->slices_) {
      if (s.input == old) {
        s.input = update;
      }
    }
  }
  if (obj->lhs_ == old) {
    if (tracker_) {
      tracker_->Record(&obj->lhs_);
    }
    obj->lhs_ = update;
    if (obj->SharedNdd()) {
      auto old_ndd = obj->nd_.data;
      auto new_ndd = update->nd_.data;
      if (old_ndd != new_ndd) {
        std::vector<NDObject *> stack = {obj};
        while (!stack.empty()) {
          auto top = stack.back();
          stack.pop_back();
          top->nd_.data = new_ndd;
          auto users = GetUsers(top);
          for (auto op : users) {
            if (op->nd_.data == old_ndd) {
              stack.push_back(op);
            }
          }
        }
      }
    }
  } else if (obj->rhs_ == old) {
    if (tracker_) {
      tracker_->Record(&obj->rhs_);
    }
    obj->rhs_ = update;
  } else {
    // ops with extended inputs (xhs_)
    ASSERT(obj->flags_ & OBJ_FLAG_XHS);
    auto xhs = static_cast<FlexOp *>(obj)->xhs_;
    for (int i = 0; i < xhs->in_num; ++i) {
      if (xhs->data[i] == old) {
        if (tracker_) {
          tracker_->Record(&xhs->data[i]);
        }
        xhs->data[i] = update;
        return;
      }
    }
    ASSERT(false);
  }
}

void ReorderStore(BasicBlock &block) {
  for (NDObject *iter = block.Begin(); iter != block.End();) {
    if (iter->IsStore()) {
      auto obj = iter->lhs_;
      auto store = iter;
      iter = block.Next(iter);
      // If store is in right position, Move will do nothing
      block.Move(block.Next(obj), store);
    } else {
      iter = block.Next(iter);
    }
  }
}

void ReorderLoad(BasicBlock &block) {
  uint32_t idx = 0;
  std::unordered_map<NDObject *, std::pair<uint32_t, uint32_t>> user_idx;
  // Get order of Load by usage
  for (NDObject *iter = block.Begin(); iter != block.End(); iter = block.Next(iter)) {
    iter->ForInput([&user_idx, idx](NDObject *in) {
      if (in->IsLoad()) {
        auto it = user_idx.find(in);
        if (it == user_idx.end()) {
          user_idx[in] = {idx, idx};
        } else {
          it->second.second = idx;
        }
      }
    });
    idx++;
  }
  using LoadOrder = std::pair<NDObject *, uint32_t>;
  std::vector<LoadOrder> loads;
  loads.reserve(user_idx.size());
  for (const auto &kv : user_idx) {
    loads.push_back({kv.first, kv.second.first << 16 | kv.second.second});
  }
  std::sort(loads.begin(), loads.end(), [](const LoadOrder &a, const LoadOrder &b) { return a.second > b.second; });
  for (auto &load : loads) {
    block.Move(block.Begin(), load.first);
  }
}

void InsertRemovePad(BasicBlock &block) {
  size_t max_depth = 1;
  auto min_type_id = kDataTypeEnd;
  for (NDObject *op = block.Begin(); op != block.End(); op = block.Next(op)) {
    auto obj_type = op->GetObjectType();
    if (obj_type == kReshape || obj_type == kConcat || obj_type == kSplitOp) {
      return;
    }
    max_depth = std::max(max_depth, op->nd_.size());
    min_type_id = std::min(min_type_id, op->type_id_);
  }

  TileInfo info;
  info.Reset(max_depth);
  for (NDObject *op = block.Begin(); op != block.End(); op = block.Next(op)) {
    if (auto ndd = op->Ndd(); ndd != nullptr && ndd->dims.size() != max_depth) {
      ndd->dims.resize(max_depth, 1);
    }
    op->TileCollect(info);
  }
  for (NDObject *iter = block.Begin(); iter != block.End(); iter = block.Next(iter)) {
    if (iter->GetObjectType() == kStore) {
      auto obj_id = iter->lhs_->obj_id_;
      if (obj_id == kElementAny || static_cast<int>(iter->nd_.size()) == info.lead_depth) {
        continue;
      }
      if (obj_id == kReduce && g_system.deterministic_) {
        continue;
      }
      uint64_t item_size = ITEM_SIZE[iter->lhs_->type_id_];
      if (item_size != 2 && item_size != 4) {
        continue;
      }
      uint64_t iter_size = ITEM_SIZE[min_type_id];
      for (int i = 0; i < info.lead_depth; i++) {
        iter_size *= iter->nd_[i];
      }
      const uint64_t store_threshold = g_system.CoreNum() * iter_size;
      if (static_cast<uint64_t>(iter->Size()) <= store_threshold) {  // Below this threshold, the store can be split without requiring repeat.
        continue;
      }
      if (iter_size % SIMD_BLOCK_SIZE && iter_size < SIMD_REPEAT_SIZE) {
        if (obj_id == kReduce) {
          iter->lhs_->SetFlag(OBJ_FLAG_REDUCE_NO_CUM);
        }
        auto remove_pad = new RemovePadOp(iter->lhs_);
        remove_pad->nd_ = iter->lhs_->nd_;
        if (auto tracker = block.Tracker()) {
          tracker->Record(&iter->lhs_);
        }
        iter->lhs_ = remove_pad;
        block.Insert(iter, remove_pad);
      }
    }
  }
}

void PrintPeakLive(BasicBlock &bb) {
  std::unordered_map<NDObject *, int> lives;
  auto maxlive = LivenessAnalyze(bb, lives);
  std::cout << "peak live: " << maxlive << std::endl;
}

void CompactPeakLiveness(BasicBlock &bb) {
  std::vector<NDObject *> backup = bb.ToVector<false>();
  std::unordered_map<NDObject *, int> lives;
  auto old_peak = LivenessAnalyze(bb, lives);
  auto new_order = ReorderObjectsHeuristic(bb, lives);
  bb.Build(new_order, false);
  lives.clear();
  auto new_peak = LivenessAnalyze(bb, lives);
  if (new_peak >= old_peak) {
    // Reorder cause a bad result, rollback
    bb.Build(backup, false);
  }
}

void EliminateReshape(BasicBlock &bb) {
  for (NDObject *it = bb.Begin(); it != bb.End(); it = bb.Next(it)) {
    auto op = it;
    if (op->GetObjectType() != kReshape) continue;
    auto lhs = op->lhs_;
    auto small = &lhs->nd_.dims();
    auto big = &op->nd_.dims();
    if (small->size() > big->size()) {
      std::swap(small, big);
    }
    for (size_t i = 0; i < small->size(); ++i) {
      if (small->operator[](i) != big->operator[](i)) continue;
    }
    for (size_t i = small->size(); i < big->size(); ++i) {
      if (big->operator[](i) != 1) continue;
    }
    if (lhs->IsLoad()) {
      for (auto succ : bb.GetUsers(op)) {
        if (succ->IsStore()) {
          auto copy = new CopyOp(lhs);
          std::vector<NDObject *> stuff_ops;
          copy->Normalize(stuff_ops);
          ASSERT(stuff_ops.empty());
          bb.Insert(op, copy);
          lhs = copy;
          break;
        }
      }
    }
    for (auto succ : bb.GetUsers(op)) {
      bb.UpdateInput(succ, op, lhs);
      bb.AddUser(lhs, succ);
    }
    bb.Erase(op);
  }
}

void VectorDoubleBuffer(BasicBlock &bb) {
  struct Span {
    NDObject *first_op;
    int first_ref;
    int last_ref;
  };
  std::unordered_map<NDObject *, Span> load_span;
  int index = 0;
  for (auto obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    obj->ForInput([&load_span, index, obj](NDObject *in) {
      if (in->IsLoad()) {
        auto it = load_span.find(in);
        if (it == load_span.end()) {
          load_span[in] = {obj, index, index};
        } else {
          it->second.first_op = obj;
          it->second.first_ref = index;
        }
      }
    });
    index++;
  }
  std::vector<NDObject *> stuff_ops;
  constexpr int MIN_DB_SPAN = 3;
  for (auto &p : load_span) {
    auto &span = p.second;
    if (span.first_ref - span.last_ref >= MIN_DB_SPAN) {
      auto copy = new CopyOp(p.first);
      copy->Normalize(stuff_ops);
      for (auto succ : bb.GetUsers(p.first)) {
        bb.UpdateInput(succ, p.first, copy);
      }
      bb.Insert(span.first_op, copy);
    }
  }
}

void DeadCodeEliminate(BasicBlock &bb) {
  for (auto obj = bb.Begin(); obj != bb.End(); obj = bb.Next(obj)) {
    obj->reuse_dep_ = obj->IsStore() ? 1 : 0;
  }
  for (auto obj = bb.ReverseBegin(); obj != bb.ReverseEnd(); obj = bb.Prev(obj)) {
    if (obj->reuse_dep_) {
      obj->ForInput([](NDObject *in) { in->reuse_dep_ = 1; });
    }
  }
  for (auto obj = bb.Begin(); obj != bb.End();) {
    auto next = bb.Next(obj);
    if (obj->reuse_dep_ == 0) {
      bb.Erase(obj);
    }
    obj = next;
  }
}

class PassOptimizerC220 : public PassOptimizer {
 public:
  void RunPass(BasicBlock &bb, bool dyn_shape) override {
    DeadCodeEliminate(bb);
    if (dyn_shape) {
      CompactPeakLiveness(bb);
      VectorDoubleBuffer(bb);
      ReorderLoad(bb);
      ReorderStore(bb);
    } else {
      EliminateReshape(bb);
      CompactPeakLiveness(bb);
      VectorDoubleBuffer(bb);
      ReorderLoad(bb);
      ReorderStore(bb);
      InsertRemovePad(bb);
    }
  }
};

class PassOptimizerC310 : public PassOptimizer {
 public:
  void RunPass(BasicBlock &bb, bool dyn_shape) override {
    DeadCodeEliminate(bb);
    if (dyn_shape) {
      CompactPeakLiveness(bb);
      VectorDoubleBuffer(bb);
      ReorderLoad(bb);
      ReorderStore(bb);
    } else {
      EliminateReshape(bb);
      CompactPeakLiveness(bb);
      VfFusion(bb);
      VectorDoubleBuffer(bb);
      ReorderLoad(bb);
      ReorderStore(bb);
    }
  }
};

PassOptimizer *CreateOptimizer(AiCoreArch arch) {
  if (arch == kAiCore_C220) {
    return new PassOptimizerC220();
  }
  return new PassOptimizerC310();
}
}  // namespace dvm::pass
