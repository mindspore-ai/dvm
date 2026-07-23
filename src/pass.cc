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

#include "pass.h"
#include <algorithm>
#include <queue>
#include <cstdint>
#include <vector>
#include <stack>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace dvm::pass {

namespace {
constexpr int kNumUsersBig = 100;
constexpr int kNumUsers1 = 1;
constexpr int kNumUsers2 = 2;

inline std::vector<NDObject *> GetPreds(NDObject *obj) {
  std::vector<NDObject *> res;
  obj->ForInput([&res](NDObject *op) { res.push_back(op); });
  return res;
}

inline void ItePreds(NDObject *obj, std::function<void(NDObject *)> fun) { obj->ForInput(fun); }

size_t GetInputsNum(NDObject *obj) {
  if (obj->lhs_ == nullptr) {
    return 0;
  }
  if (obj->rhs_ == nullptr) {
    return 1;
  }
  if (obj->GetObjectType() == ObjectType::kSelect) {
    return 3;
  }
  return 2;
}

size_t MaxLive(BasicBlock &bb) {
  size_t peak = 0;
  size_t current_live = 0;
  std::unordered_map<NDObject *, int> num_users;
  auto try_deallcate = [&num_users, &current_live](NDObject *obj) {
    if (--num_users[obj] == 0) {
      current_live--;
      num_users.erase(obj);
    }
  };
  // Store and Load always occupy a variable
  for (auto &obj : bb) {
    if (obj.IsLoad()) {
      current_live++;
      num_users[&obj] = kNumUsersBig;
    }
    if (obj.IsStore()) {
      current_live++;
      num_users[obj.lhs_] = kNumUsersBig;
    }
  }

  peak = current_live;
  for (auto &obj : bb) {
    // Store doesn't introduce new variable
    if (!obj.IsSimd()) {
      continue;
    }
    if (num_users.find(&obj) == num_users.end()) {
      num_users[&obj] = bb.GetUserNum(&obj);
      ++current_live;
    }

    // unary, binary and binaryS can inplace
    auto type = obj.GetObjectType();
    bool need_update = true;
    if (type == kUnary || type == kBinary || type == kBinaryS) {
      if (need_update && obj.lhs_ != nullptr && num_users[obj.lhs_] == kNumUsers1) {
        need_update = false;
      }
      if (need_update && obj.rhs_ != nullptr && num_users[obj.rhs_] == kNumUsers1) {
        need_update = false;
      }
      if (need_update && obj.lhs_ != nullptr && obj.rhs_ == obj.lhs_ && num_users[obj.rhs_] == kNumUsers2) {
        need_update = false;
      }
    }
    if (need_update) {
      peak = std::max(peak, current_live);
    }

    // deallocate variable
    obj.ForInput(try_deallcate);
  }
  return peak;
}

// This encode method support basic block with size less than 64
template <class InputIt>
uint64_t Encode(InputIt first, InputIt last) {
  uint64_t res = 0;
  for (; first != last; ++first) {
    res |= 1 << (*first)->index_;
  }
  return res;
}

std::vector<NDObject *> Decode(const std::vector<NDObject *> objects, uint64_t code) {
  std::vector<NDObject *> res;
  for (size_t i = 0; i < objects.size() && code != 0; ++i) {
    if (code & 0x1) {
      res.emplace_back(objects[i]);
    }
    code >>= 1;
  }
  return res;
}

std::vector<NDObject *> ReorderObjectsHeuristic(BasicBlock &bb) {
  // Preprocess, calculate points of all objects
  std::unordered_map<NDObject *, int64_t> points;
  std::unordered_map<NDObject *, uint32_t> heights;
  constexpr int32_t B1 = 1;
  constexpr int32_t B2 = 1000;
  constexpr int64_t B3 = 1000000;
  auto get_reducable_inputs_num = [](NDObject *obj) -> size_t {
    size_t res = 0;
    if (obj->lhs_ == nullptr) {
      return 0;
    }
    if (!obj->lhs_->IsLoad()) {
      ++res;
    }
    if (obj->rhs_ == nullptr) {
      return res;
    }
    if (!obj->rhs_->IsLoad()) {
      ++res;
    }
    if ((obj->flags_ & OBJ_FLAG_XHS) && !static_cast<FlexOp *>(obj)->xhs_->IsLoad()) {
      ++res;
    }
    return res;
  };
  for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
    auto obj = iter.get();
    heights[obj] = 0;
    for (auto user : bb.GetUsers(iter.get())) {
      heights[obj] = std::max(heights[obj], heights[user] + 1);
    }
    if (!obj->IsSimd()) {
      points[obj] = B3;
    }
    points[obj] += get_reducable_inputs_num(obj) * B2;
    points[obj] -= heights[obj] * B1;
  }
  auto compare = [&points](NDObject *a, NDObject *b) { return points[a] < points[b]; };
  std::priority_queue<NDObject *, std::vector<NDObject *>, decltype(compare)> pq(compare);
  std::unordered_map<NDObject *, uint32_t> in_degrees;
  for (auto &obj : bb) {
    in_degrees[&obj] = GetInputsNum(&obj);
    if (in_degrees[&obj] == 0) {
      pq.push(&obj);
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

// Ref: ORDERING CHAOS: MEMORY-AWARE SCHEDULING OF IRREGULARLY WIRED NEURAL NETWORKS FOR EDGE DEVICES
__attribute__((unused)) std::vector<NDObject *> ReorderObjectsDP(BasicBlock &bb) {
  struct DpStruct {
    uint32_t cur_live;
    uint32_t peak_live;
    std::vector<NDObject *> arrange;
  };
  int iter_num = 0;
  uint32_t cur_live = 0;
  std::vector<NDObject *> objects = bb.ToVector<false>();  // used in encoding
  if (bb.size() > 64) {
    // Support up to 64 nodes
    return objects;
  }
  std::unordered_map<NDObject *, uint32_t> in_degrees_bak;   // Used in schedulling
  std::unordered_map<NDObject *, uint32_t> out_degrees_bak;  // Used in deallocating
  std::vector<NDObject *> arrange_bak;                       // Record all load
  std::unordered_set<NDObject *> static_obj;
  std::vector<NDObject *> readys_bak;
  // Init in_degrees
  for (auto &obj : bb) {
    in_degrees_bak[&obj] = GetInputsNum(&obj);
    if (in_degrees_bak[&obj] == 0 && !obj.IsLoad()) {
      readys_bak.emplace_back(&obj);
    }
  }
  // Init arrange and out_degrees
  for (auto &obj : bb) {
    out_degrees_bak[&obj] = bb.GetUserNum(&obj);
    if (obj.IsLoad()) {
      cur_live++;
      arrange_bak.emplace_back(&obj);
      out_degrees_bak[&obj] += kNumUsersBig;  // Load won't be deallocated
      for (auto user : bb.GetUsers(&obj)) {
        in_degrees_bak[user]--;
        if (in_degrees_bak[user] == 0) {
          readys_bak.emplace_back(user);
        }
      }
    } else if (obj.IsStore()) {
      cur_live++;
      out_degrees_bak[obj.lhs_] += kNumUsersBig;  // Variable used by Store won't be deallocated
      static_obj.insert(obj.lhs_);
      static_obj.insert(&obj);
    }
  }
  std::unordered_map<uint64_t, DpStruct> tape;  // Used to record interediate results
  tape[Encode(readys_bak.begin(), readys_bak.end())] = DpStruct{cur_live, cur_live, arrange_bak};

  for (auto i = arrange_bak.size(); i < bb.size(); ++i) {
    std::unordered_map<uint64_t, DpStruct> new_tape;
    for (auto &[code, mem_block] : tape) {
      auto readys = Decode(objects, code);
      auto in_degrees = in_degrees_bak;
      auto out_degrees = out_degrees_bak;
      auto &arrange = mem_block.arrange;
      auto cur_live = mem_block.cur_live;
      auto peak_live = mem_block.peak_live;
      // Init in_degrees and out_degree
      for (auto i = arrange_bak.size(); i < arrange.size(); ++i) {
        auto obj = arrange[i];
        for (auto pred : GetPreds(obj)) {
          out_degrees[pred]--;
        }
        for (auto user : bb.GetUsers(obj)) {
          in_degrees[user]--;
        }
      }
      for (auto obj : readys) {
        // Allocate variable
        arrange.emplace_back(obj);
        std::unordered_set<NDObject *> new_readys(readys.begin(), readys.end());
        new_readys.erase(obj);
        // Object may be used twice by same user, so we need to really do the calculation
        for (auto user : bb.GetUsers(obj)) {
          if (--in_degrees[user] == 0) {
            new_readys.insert(user);
          }
        }
        auto new_code = Encode(new_readys.begin(), new_readys.end());
        if (new_tape.find(new_code) != new_tape.end() && new_tape[new_code].peak_live <= peak_live) {
          // skip
          arrange.pop_back();
          for (auto user : bb.GetUsers(obj)) {
            ++in_degrees[user];
          }
          continue;
        }
        iter_num++;
        // Update live, peak
        auto new_cur = cur_live;
        auto new_peak = peak_live;
        bool need_update_peak = false;
        if (static_obj.find(obj) == static_obj.end()) {
          new_cur++;
          need_update_peak = true;
        }
        auto type = obj->GetObjectType();
        auto tmp_cur = new_cur;
        // Deallocate variable
        for (auto pred : GetPreds(obj)) {
          if (--out_degrees[pred] == 0) {
            new_cur--;
            if (need_update_peak && (type == kUnary || type == kBinary || type == kBinaryS)) {
              // Inplace optimize
              need_update_peak = false;
            }
          }
        }
        if (need_update_peak) {
          new_peak = std::max(new_peak, tmp_cur);
        }
        if (new_tape.find(new_code) == new_tape.end() || new_tape[new_code].peak_live > new_peak) {
          new_tape.insert_or_assign(new_code, DpStruct{new_cur, new_peak, arrange});
        }
        // recorver to reuse arrange, out_degrees, in_degrees
        arrange.pop_back();
        for (auto pred : GetPreds(obj)) {
          ++out_degrees[pred];
        }
        for (auto user : bb.GetUsers(obj)) {
          ++in_degrees[user];
        }
      }
    }
    // replace old tape
    tape = std::move(new_tape);
  }
  return tape[0].arrange;
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

BasicBlock::BasicBlock(const std::vector<NDObject *> &objects, std::vector<NDObject *> &owner, GraphTracker *tracker)
  : objects_owner_(owner), tracker_(tracker) {
  // build linked list from objects
  list_.Build(objects, true);
  for (auto obj : objects) {
    SetHead(obj, -1);
  }
  edges_.reserve(objects.size() * 2);
  for (auto obj : objects) {
    ItePreds(obj, [this, obj](NDObject *pred) { this->AddUser(pred, obj); });
  }
}

template <bool if_update_index>
std::vector<NDObject *> BasicBlock::ToVector() {
  auto iter = begin();
  std::vector<NDObject *> res;
  res.reserve(size());
  int i = 0;
  while (iter != end()) {
    if constexpr (if_update_index) {
      iter->index_ = i++;
    }
    res.push_back(iter.get());
    ++iter;
  }
  return res;
}

void BasicBlock::Export(std::vector<NDObject *> &objects) {
  auto iter = begin();
  objects.clear();
  objects.reserve(size());
  int i = 0;
  while (iter != end()) {
    iter->index_ = i++;
    ObjectList::Prev(iter.get())->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    objects.push_back(iter.get());
    ++iter;
  }
  ObjectList::Prev(iter.get())->insn_ = nullptr;
}

BasicBlock::iterator BasicBlock::Insert(BasicBlock::iterator iter, NDObject *object) {
  if (iter.get() == object) {
    return iter;
  }
  list_.Insert(iter.get(), object);
  SetHead(object, -1);
  for (auto pred : GetPreds(object)) {
    AddUser(pred, object);
  }
  // object should be deleted by owner
  objects_owner_.push_back(object);
  return BasicBlock::iterator(object);
}

void BasicBlock::Erase(NDObject *object) {
  list_.Erase(object);
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
}

BasicBlock::iterator BasicBlock::Move(BasicBlock::iterator iter, NDObject *obj) {
  if (iter.get() == obj || iter.GetPrev().get() == obj) {
    return iterator(obj);
  }
  list_.Erase(obj);
  list_.Insert(iter.get(), obj);
  return iterator(obj);
}

void BasicBlock::UpdateInput(NDObject *obj, NDObject *old, NDObject *update) {
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
    // Now only select have more than 2 inputs
    ASSERT(obj->flags_ & OBJ_FLAG_XHS);
    auto flex = static_cast<FlexOp *>(obj);
    if (tracker_) {
      tracker_->Record(&flex->xhs_);
    }
    flex->xhs_ = update;
  }
}

void ReorderStore(BasicBlock &block) {
  for (auto iter = block.begin(); iter != block.end();) {
    if (iter->IsStore()) {
      auto obj = BasicBlock::iterator(iter->lhs_);
      auto store = iter.get();
      iter++;
      // If store is in right position, Move will do nothing
      block.Move(++obj, store);
    } else {
      iter++;
    }
  }
}

void ReorderLoad(BasicBlock &block) {
  uint16_t idx = 0;
  using ObjWithOrder = std::pair<uint16_t, NDObject *>;
  std::priority_queue<ObjWithOrder> load_order;
  std::unordered_set<NDObject *> load_set;
  // Get order of Load by usage
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    if (!iter->IsLoad()) {
      for (auto pred : GetPreds(iter.get())) {
        if (pred->IsLoad() && load_set.find(pred) == load_set.end()) {
          load_set.insert(pred);
          load_order.push({idx, pred});
        }
      }
    }
    idx++;
  }
  while (!load_order.empty()) {
    auto load = load_order.top().second;
    load_order.pop();
    block.Move(block.begin(), load);
  }
}

void InsertRemovePad(BasicBlock &block) {
  if (g_system.Arch() == kAiCore_C310) {
    return;
  }

  size_t max_depth = 1;
  auto min_type_id = kDataTypeEnd;
  for (auto &op : block) {
    auto obj_type = op.GetObjectType();
    if (obj_type == kReshape) {
      return;
    }
    max_depth = std::max(max_depth, op.nd_.size());
    min_type_id = std::min(min_type_id, op.type_id_);
  }

  TileInfo info;
  info.Reset(max_depth);
  for (auto &op : block) {
    if (auto ndd = op.Ndd(); ndd != nullptr && ndd->dims.size() != max_depth) {
      ndd->dims.resize(max_depth, 1);
    }
    op.TileCollect(info);
  }
  for (auto iter = block.begin(); iter != block.end(); iter++) {
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
  auto maxlive = MaxLive(bb);
  std::cout << "peak live: " << maxlive << std::endl;
}

void CompactPeakLiveness(BasicBlock &bb) {
  std::vector<NDObject *> backup = bb.ToVector<false>();
  auto old_peak = MaxLive(bb);
  auto new_order = ReorderObjectsHeuristic(bb);
  bb.List().Build(new_order, false);
  auto new_peak = MaxLive(bb);
  if (new_peak >= old_peak) {
    // Reorder cause a bad result, rollback
    bb.List().Build(backup, false);
  }
}

void EliminateReshape(BasicBlock &bb) {
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto op = it.get();
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
          bb.Insert(BasicBlock::iterator(op), copy);
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

std::vector<Pass> passes = {&EliminateReshape, &CompactPeakLiveness, &ReorderLoad, &ReorderStore, &InsertRemovePad};
}  // namespace dvm::pass
