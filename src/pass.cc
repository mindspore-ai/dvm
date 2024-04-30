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
#define NEXT_OBJ(a) reinterpret_cast<NDObject *>((a)->insn_)
#define PREV_OBJ(a) reinterpret_cast<NDObject *>((a)->tail_insn_)
#define ASSIGN_NEXT_OBJ(a, b) (a)->insn_ = reinterpret_cast<uint64_t *>(b)
#define ASSIGN_PREV_OBJ(a, b) (a)->tail_insn_ = reinterpret_cast<uint64_t *>(b)

namespace {
constexpr int kNumUsersBig = 100;
constexpr int kNumUsers1 = 1;
constexpr int kNumUsers2 = 2;

inline std::vector<NDObject *> GetPreds(NDObject *obj) {
  std::vector<NDObject *> res;
  if (obj->lhs_ == nullptr) {
    return res;
  }
  res.emplace_back(obj->lhs_);
  if (obj->rhs_ == nullptr) {
    return res;
  }
  res.emplace_back(obj->rhs_);
  if (obj->GetObjectType() == kSelect) {
    res.emplace_back(reinterpret_cast<SelectOp *>(obj)->cond_);
  }
  return res;
}

inline void ItePreds(NDObject *obj, std::function<void(NDObject *)> fun) {
  if (obj->lhs_ == nullptr) {
    return;
  }
  fun(obj->lhs_);
  if (obj->rhs_ == nullptr) {
    return;
  }
  fun(obj->rhs_);
  if (obj->GetObjectType() == kSelect) {
    fun(reinterpret_cast<SelectOp *>(obj)->cond_);
  }
}

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

NDObject *&GetInputRef(NDObject *obj, NDObject *input) {
  if (obj->lhs_ == input) {
    return obj->lhs_;
  }
  if (obj->rhs_ == input) {
    return obj->rhs_;
  }
  // Now only select have more than 2 inputs
  ASSERT(obj->GetObjectType() == kSelect);
  return reinterpret_cast<SelectOp *>(obj)->cond_;
}

inline std::vector<NDObject *> GetSuccs(NDObject *obj, const BasicBlock &bb) { return bb.context().GetUsers(obj); }

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
    if (obj.GetObjectType() == kLoad) {
      current_live++;
      num_users[&obj] = kNumUsersBig;
    }
    if (obj.GetObjectType() == kStore) {
      current_live++;
      num_users[obj.lhs_] = kNumUsersBig;
    }
  }

  peak = current_live;
  for (auto &obj : bb) {
    // Store doesn't introduce new variable
    if (obj.GetObjectType() == kLoad || obj.GetObjectType() == kStore) {
      continue;
    }
    if (num_users.find(&obj) == num_users.end()) {
      num_users[&obj] = GetSuccs(&obj, bb).size();
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
    if (obj.lhs_ == nullptr) {
      continue;
    }
    try_deallcate(obj.lhs_);
    if (obj.rhs_ == nullptr) {
      continue;
    }
    try_deallcate(obj.rhs_);
    // SelectOp has three inpus
    if (obj.GetObjectType() != kSelect) {
      SelectOp *select = reinterpret_cast<SelectOp *>(&obj);
      try_deallcate(select->cond_);
    }
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
    if (obj->lhs_->GetObjectType() != kLoad) {
      ++res;
    }
    if (obj->rhs_ == nullptr) {
      return res;
    }
    if (obj->rhs_->GetObjectType() != kLoad) {
      ++res;
    }
    if (obj->GetObjectType() == ObjectType::kSelect &&
        reinterpret_cast<SelectOp *>(obj)->cond_->GetObjectType() != kLoad) {
      ++res;
    }
    return res;
  };
  for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
    auto obj = iter.get();
    heights[obj] = 0;
    for (auto user : GetSuccs(iter.get(), bb)) {
      heights[obj] = std::max(heights[obj], heights[user] + 1);
    }
    if (obj->GetObjectType() == kStore || obj->GetObjectType() == kLoad) {
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
    for (auto user : GetSuccs(obj, bb)) {
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
    if (in_degrees_bak[&obj] == 0 && obj.GetObjectType() != kLoad) {
      readys_bak.emplace_back(&obj);
    }
  }
  // Init arrange and out_degrees
  for (auto &obj : bb) {
    out_degrees_bak[&obj] = GetSuccs(&obj, bb).size();
    if (obj.GetObjectType() == kLoad) {
      cur_live++;
      arrange_bak.emplace_back(&obj);
      out_degrees_bak[&obj] += kNumUsersBig;  // Load won't be deallocated
      for (auto user : GetSuccs(&obj, bb)) {
        in_degrees_bak[user]--;
        if (in_degrees_bak[user] == 0) {
          readys_bak.emplace_back(user);
        }
      }
    } else if (obj.GetObjectType() == kStore) {
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
        for (auto user : GetSuccs(obj, bb)) {
          in_degrees[user]--;
        }
      }
      for (auto obj : readys) {
        // Allocate variable
        arrange.emplace_back(obj);
        std::unordered_set<NDObject *> new_readys(readys.begin(), readys.end());
        new_readys.erase(obj);
        // Object may be used twice by same user, so we need to really do the calculation
        for (auto user : GetSuccs(obj, bb)) {
          if (--in_degrees[user] == 0) {
            new_readys.insert(user);
          }
        }
        auto new_code = Encode(new_readys.begin(), new_readys.end());
        if (new_tape.find(new_code) != new_tape.end() && new_tape[new_code].peak_live <= peak_live) {
          // skip
          arrange.pop_back();
          for (auto user : GetSuccs(obj, bb)) {
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
        for (auto user : GetSuccs(obj, bb)) {
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

void BasicBlockContext::Init(const std::vector<NDObject *> &objects) {
  // users_.resize(objects.size());
  head_.resize(objects.size(), -1);
  for (auto obj : objects) {
    ItePreds(obj, [this, obj](NDObject *pred) { this->AddUser(pred, obj); });
  }
}

void BasicBlockContext::Init(NDObjectIterator<false> begin, NDObjectIterator<false> end, size_t capcity) {
  edges_.clear();
  head_.resize(capcity, -1);
  while (begin != end) {
    auto obj = begin.get();
    ItePreds(obj, [this, obj](NDObject *pred) { this->AddUser(pred, obj); });
  };
}

void BasicBlockContext::Erase(NDObject *object) {
  head_[object->index_] = -1;
  for (auto pred : GetPreds(object)) {
    auto idx = head_[pred->index_];
    auto last = idx;
    while (idx != -1) {
      if (edges_[idx].user == object) {
        if (idx == head_[pred->index_]) {
          head_[pred->index_] = edges_[idx].next;
        } else {
          edges_[last].next = edges_[idx].next;
        }
        break;
      }
      idx = edges_[idx].next;
      last = idx;
    }
    ASSERT(idx != -1);
  }
}

BasicBlock::BasicBlock(const std::vector<NDObject *> &objects, std::vector<NDObject *> &owner)
    : sentinel_(kTypeEnd), size_(objects.size()), capacity_(objects.size()), objects_owner_(owner) {
  // build linked list from objects
  ReOrder(objects, true);
  context_.Init(objects);
}

void BasicBlock::ReOrder(const std::vector<NDObject *> &objects, bool if_update_index) {
  // build linked list from objects
  NDObject *last_object = &sentinel_;
  int index = 0;
  for (auto object : objects) {
    ASSIGN_NEXT_OBJ(last_object, object);
    ASSIGN_PREV_OBJ(object, last_object);
    last_object = object;
    if (if_update_index) {
      last_object->index_ = index;
    }
    ++index;
  }
  size_ = index;
  if (if_update_index) {
    capacity_ = index;
  }
  ASSIGN_NEXT_OBJ(last_object, &sentinel_);
  ASSIGN_PREV_OBJ(&sentinel_, last_object);
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

void BasicBlock::Clear() {
  auto iter = begin();
  while (iter != end()) {
    PREV_OBJ(iter)->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    ++iter;
  }
  PREV_OBJ(iter)->insn_ = nullptr;
}

void BasicBlock::Export(std::vector<NDObject *> &objects) {
  auto iter = begin();
  objects.clear();
  objects.reserve(size());
  int i = 0;
  while (iter != end()) {
    iter->index_ = i++;
    PREV_OBJ(iter)->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    objects.push_back(iter.get());
    ++iter;
  }
  PREV_OBJ(iter)->insn_ = nullptr;
}

BasicBlock::iterator BasicBlock::Insert(BasicBlock::iterator iter, NDObject *object) {
  if (iter.get() == object) {
    return iter;
  }
  auto prev = PREV_OBJ(iter);
  ASSIGN_PREV_OBJ(object, prev);
  ASSIGN_NEXT_OBJ(object, iter.get());
  ASSIGN_PREV_OBJ(iter, object);
  ASSIGN_NEXT_OBJ(prev, object);
  ++size_;
  object->index_ = capacity_++;
  // object should be deleted by owner
  objects_owner_.push_back(object);
  return NDObjectIterator<false>(object);
}

BasicBlock::iterator BasicBlock::Erase(BasicBlock::iterator iter) {
  ASSERT(iter != end());
  auto prev = PREV_OBJ(iter);
  auto next = NEXT_OBJ(iter);
  ASSIGN_NEXT_OBJ(prev, next);
  ASSIGN_PREV_OBJ(next, prev);
  --size_;
  return NDObjectIterator<false>(next);
}

BasicBlock::iterator BasicBlock::Move(BasicBlock::iterator iter, BasicBlock::pointer obj) {
  if (iter.get() == obj || iter.GetPrev().get() == obj) {
    return iterator(obj);
  }
  auto prev = PREV_OBJ(iter);
  ASSIGN_NEXT_OBJ(PREV_OBJ(obj), NEXT_OBJ(obj));
  ASSIGN_PREV_OBJ(NEXT_OBJ(obj), PREV_OBJ(obj));
  ASSIGN_NEXT_OBJ(prev, obj);
  ASSIGN_PREV_OBJ(iter, obj);
  ASSIGN_NEXT_OBJ(obj, iter.get());
  ASSIGN_PREV_OBJ(obj, prev);
  return iterator(obj);
}

void BasicBlock::UpdateContext() { context_.Init(begin(), end(), capacity_); }

void ReorderStore(BasicBlock &block) {
  for (auto iter = block.begin(); iter != block.end();) {
    if (iter->GetObjectType() == kStore) {
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
    if (iter->GetObjectType() != kLoad) {
      for (auto pred : GetPreds(iter.get())) {
        if (pred->GetObjectType() == kLoad && load_set.find(pred) == load_set.end()) {
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
  if (DeviceInfo::Instance().Arch() != kAiCore_C220) {
    return;
  }
  size_t max_depth = 1;
  for (auto &op : block) {
    auto obj_type = op.GetObjectType();
    if (obj_type == kReshape) {
      return;
    }
    max_depth = std::max(max_depth, op.nd_.size());
  }

  PropRange range;
  range.base = 0;
  range.depth = max_depth;
  for (auto &op : block) {
    if (op.nd_.size() != max_depth) {
      op.nd_.resize(max_depth, 1);
    }
    op.AlignProp(range);
  }
  for (auto iter = block.begin(); iter != block.end(); iter++) {
    if (iter->GetObjectType() == kStore) {
      if (iter->lhs_->obj_id_ == kElementAny || static_cast<int>(iter->nd_.size()) == range.depth) {
        continue;
      }
      uint64_t iter_size = ITEM_SIZE[iter->type_id_];
      if (iter_size == 1) {
        continue;
      }
      for (int i = 0; i < range.depth; i++) {
        iter_size *= iter->nd_[i];
      }
      if (iter_size % SIMD_BLOCK_SIZE && iter_size < SIMD_REPEAT_SIZE) {
        auto remove_pad = new RemovePadOp(iter->lhs_);
        remove_pad->nd_ = iter->lhs_->nd_;
        iter->lhs_ = remove_pad;
        block.Insert(iter, remove_pad);
      }
    }
  }
}

void PrintPeakLive(BasicBlock &bb) {
  auto maxlive = MaxLive(bb);
  printf("peak live: %lu\n", maxlive);
}

void CompactPeakLiveness(BasicBlock &bb) {
  std::vector<NDObject *> backup = bb.ToVector<false>();
  auto old_peak = MaxLive(bb);
  auto new_order = ReorderObjectsHeuristic(bb);
  bb.ReOrder(new_order);
  auto new_peak = MaxLive(bb);
  if (new_peak >= old_peak) {
    // Reorder cause a bad result, rollback
    bb.ReOrder(backup);
  }
}

namespace eliminate_reshape {
struct PropagateArgs {
  bool is_forward;
  NDObject *obj;
  NDObject *last;
  std::vector<int64_t> new_shape;
};
struct AnalysisIntermediate {
  std::vector<std::optional<std::vector<int64_t>>> need_reshape;  // idx corresbond to NDObject's index_
  std::vector<NDObject *> visited;
  std::vector<PropagateArgs> todos;

  inline void RegisterNewShape(NDObject *obj, const std::vector<int64_t> &new_shape) {
    need_reshape[obj->index_] = new_shape;
    visited.push_back(obj);
  }
};
struct ShapePacket {
  size_t start;
  size_t end;
  bool is_broadcast_axis;
};
std::vector<ShapePacket> GetShapePackets(const std::vector<int64_t> &shape_ori,
                                         const std::vector<int64_t> &shape_to_change) {
  std::vector<ShapePacket> shape_packets;
  shape_packets.reserve(3);
  ASSERT(shape_ori.size() >= 1);
  bool is_curr_broadcast_axis = false;
  ShapePacket shape_packet{0, 1, shape_to_change[0] != shape_ori[0]};
  bool &is_prev_broadcast_axis = shape_packet.is_broadcast_axis;
  for (size_t i = 1; i < shape_to_change.size(); ++i) {
    is_curr_broadcast_axis = shape_to_change[i] != shape_ori[i];
    if (is_curr_broadcast_axis != is_prev_broadcast_axis) {
      shape_packets.push_back(shape_packet);
      shape_packet.start = shape_packet.end;
      shape_packet.is_broadcast_axis = is_curr_broadcast_axis;
    }
    ++shape_packet.end;
  }
  shape_packets.push_back(shape_packet);
  return shape_packets;
}

std::vector<int64_t> TryReshape(const std::vector<int64_t> &shape_to_change, const std::vector<int64_t> &shape_ori,
                                const std::vector<int64_t> &shape_new) {
  ASSERT(shape_to_change.size() == shape_ori.size());
  auto shape_packets = GetShapePackets(shape_ori, shape_to_change);

  std::vector<int64_t> res;
  size_t j = 0;
  for (size_t i = 0; i < shape_packets.size(); ++i) {
    int64_t ori_size = 1;
    const ShapePacket &shape_packet = shape_packets[i];
    for (size_t ii = shape_packet.start; ii < shape_packet.end; ++ii) {
      ori_size *= shape_ori[ii];
    }
    size_t j_start = j;
    // Case when shape_ori has trailing 1, e.g. (8,2,1,1) -> (8,2)
    if (j_start == shape_new.size() && i == shape_packets.size() - 1 && ori_size == 1 &&
        !shape_packet.is_broadcast_axis) {
      return res;
    }

    // Find shape packet in shape_new
    int64_t new_size = 1;
    while (new_size < ori_size && j < shape_new.size()) {
      new_size *= shape_new[j];
      ++j;
    }
    // Case when shape is 1
    if (j == j_start && j < shape_new.size() && shape_new[j] == 1) {
      ++j;
    }
    if (new_size != ori_size) {
      // Can't reshape
      return {};
    }
    // Add shapes corresponding to this shape packet
    if (shape_packet.is_broadcast_axis) {
      auto new_len = j - j_start;
      if (new_len == 0) {
        return {};
      }
      auto old_len = shape_packet.end - shape_packet.start;
      if (old_len <= new_len) {
        res.insert(res.end(), shape_to_change.begin() + shape_packet.start,
                   shape_to_change.begin() + (shape_packet.start + old_len));
        for (size_t ii = old_len; ii < new_len; ++ii) {
          res.push_back(1);
        }
      } else {
        int64_t first = 1;
        for (size_t ii = 0; ii < old_len - new_len + 1; ++ii) {
          first *= shape_to_change[shape_packet.start + ii];
        }
        res.push_back(first);
        res.insert(res.end(), shape_to_change.begin() + (shape_packet.start + old_len - new_len + 1),
                   shape_to_change.begin() + old_len);
      }
    } else {
      // Case of not broadcast packet
      res.insert(res.end(), shape_new.begin() + j_start, shape_new.begin() + j);
    }
  }
  // Case when shape_new has trailing 1, e.g. (32, 4) -> (16, 2, 4, 1, 1)
  while (j < shape_new.size()) {
    if (shape_new[j++] != 1) {
      return {};
    }
    res.push_back(1);
  }
  return res;
}

bool Propagate(NDObject *obj, const std::vector<int64_t> &new_shape, NDObject *last, bool is_forward,
               const BasicBlock &bb, AnalysisIntermediate &intermediate) {
  auto &need_reshape = intermediate.need_reshape;
  auto &todos = intermediate.todos;
  if (need_reshape[obj->index_].has_value()) {
    return true;
  }
  if (obj->nd_ == new_shape) {
    return true;
  }
  std::vector<int64_t> forward_shape;
  std::vector<int64_t> backward_shape;
  auto type = obj->GetObjectType();
  switch (type) {
    case kUnary:
    case kBinary:
    case kBinaryS:
    case kSelect:
    case kCast:
      // Elementwise
      intermediate.RegisterNewShape(obj, new_shape);
      forward_shape = new_shape;
      backward_shape = new_shape;
      break;
    case kLoad:
    case kBroadcastS: {
      ASSERT(!is_forward);
      intermediate.RegisterNewShape(obj, new_shape);
      forward_shape = new_shape;
      break;
    }
    case kStore:
      ASSERT(is_forward);
      intermediate.RegisterNewShape(obj, new_shape);
      return true;
    case kReshape: {
      if (is_forward) {
        // Only change shape of input of this Reshape, no need to propagate further
        return true;
      }
      intermediate.RegisterNewShape(obj, new_shape);
      if (GetSuccs(obj, bb).size() == 1) return true;
      // Need to change all other users of this Reshape
      for (auto succ : GetSuccs(obj, bb)) {
        if (succ == last) {
          continue;
        }
        todos.push_back({true, succ, obj, new_shape});
      }
      return true;
    }
    case kBroadcastTo:
    case kReduce: {
      auto shape_change =
        is_forward ? TryReshape(obj->nd_, obj->lhs_->nd_, new_shape) : TryReshape(obj->lhs_->nd_, obj->nd_, new_shape);
      if (shape_change.empty()) {
        return false;
      }
      intermediate.RegisterNewShape(obj, is_forward ? shape_change : new_shape);
      if (is_forward) {
        forward_shape = shape_change;
      } else {
        backward_shape = shape_change;
        forward_shape = new_shape;
      }
      break;
    }
    case kElementAny: {
      if (new_shape.size() == obj->nd_.size()) {
        return true;
      }
      if (is_forward) {
        forward_shape = std::vector<int64_t>(new_shape.size(), 1);
        intermediate.RegisterNewShape(obj, forward_shape);
      } else {
        auto new_size = new_shape.size();
        auto old_size = obj->nd_.size();
        if (new_size > old_size) {
          backward_shape = obj->lhs_->nd_;
          while (old_size++ < new_size) {
            backward_shape.push_back(1);
          }
        } else {
          int64_t first = 1;
          size_t i = 0;
          while (i < old_size - new_size + 1) {
            first *= obj->lhs_->nd_[i++];
          }
          backward_shape.push_back(first);
          while (i < old_size) {
            backward_shape.push_back(obj->lhs_->nd_[i++]);
          }
          forward_shape = new_shape;
          intermediate.RegisterNewShape(obj, forward_shape);
        }
      }
      break;
    }
    default:
      return false;
  }

  if (!forward_shape.empty()) {
    for (auto succ : GetSuccs(obj, bb)) {
      todos.push_back({true, succ, obj, forward_shape});
    }
  }
  if (!backward_shape.empty()) {
    for (auto prev : GetPreds(obj)) {
      todos.push_back({false, prev, obj, backward_shape});
    }
  }
  return true;
}

// Fix start_dim_ and end_dim_
void CleanUpReduce(NDObject *obj) {
  auto &input_axis = obj->lhs_->nd_;
  auto &output_axis = obj->nd_;
  ASSERT(input_axis.size() == output_axis.size());
  size_t i = 0;
  while (input_axis[i] == output_axis[i] && i < input_axis.size()) {
    ++i;
  }
  int start = i;
  while (input_axis[i] != output_axis[i] && i < input_axis.size()) {
    ++i;
  }
  int end = i - 1;
  ASSERT(start <= end);
  auto reduce = reinterpret_cast<_ReduceOp *>(obj);
  reduce->SetRange(start, end);
}
}  // namespace eliminate_reshape

void EliminateReshape(BasicBlock &bb) {
  using namespace eliminate_reshape;

  std::unordered_set<NDObject *> reduce_to_cleanup;
  auto eliminate_reshape_impl = [&bb, &reduce_to_cleanup](bool is_forward) {
    for (auto &reshape : bb) {
      if (reshape.GetObjectType() != kReshape) {
        continue;
      }
      AnalysisIntermediate inter;
      inter.need_reshape.resize(bb.capacity());
      inter.visited.reserve(bb.capacity());
      std::vector<int64_t> &new_shape = is_forward ? reshape.lhs_->nd_ : reshape.nd_;
      inter.RegisterNewShape(&reshape, new_shape);
      bool can_eliminate = true;
      if (is_forward) {
        for (auto user : GetSuccs(&reshape, bb)) {
          if (!Propagate(user, new_shape, &reshape, true, bb, inter)) {
            can_eliminate = false;
            break;
          }
        }
      } else {
        for (auto user : GetPreds(&reshape)) {
          if (!Propagate(user, new_shape, &reshape, false, bb, inter)) {
            can_eliminate = false;
            break;
          }
        }
      }
      auto &todos = inter.todos;
      while (!todos.empty() && can_eliminate) {
        auto todo = std::move(todos.back());
        todos.pop_back();
        can_eliminate = Propagate(todo.obj, todo.new_shape, todo.last, todo.is_forward, bb, inter);
      }
      if (!can_eliminate) {
        continue;
      }
      // Reshape
      for (auto to_update : inter.visited) {
        to_update->nd_ = inter.need_reshape[to_update->index_].value();
        // Reduce need additoinal clean up
        if (to_update->GetObjectType() == kReduce) {
          reduce_to_cleanup.insert(to_update);
        }
      }
      // Delete Reshape op, and manually fix context to reduce execution time used in UpdateContext
      auto &context = bb.context();
      auto prev = reshape.lhs_;
      for (auto succ : GetSuccs(&reshape, bb)) {
        auto &input_ref = GetInputRef(succ, &reshape);
        input_ref = prev;
        context.AddUser(prev, succ);
      }
      bb.Erase(BasicBlock::iterator(&reshape));
      context.Erase(&reshape);
    }
  };

  eliminate_reshape_impl(true);
  eliminate_reshape_impl(false);
  for (auto obj : reduce_to_cleanup) {
    CleanUpReduce(obj);
  }
}

std::vector<Pass> passes = {&EliminateReshape, &CompactPeakLiveness, &ReorderLoad, &ReorderStore, &InsertRemovePad};
}  // namespace dvm::pass
