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

namespace dvm::pass {
#define NEXT_OBJ(a) reinterpret_cast<NDObject *>((a)->insn_)
#define PREV_OBJ(a) reinterpret_cast<NDObject *>((a)->tail_insn_)
#define ASSIGN_NEXT_OBJ(a, b) (a)->insn_ = reinterpret_cast<uint64_t *>(b)
#define ASSIGN_PREV_OBJ(a, b) (a)->tail_insn_ = reinterpret_cast<uint64_t *>(b)

BasicBlock::BasicBlock(const std::vector<NDObject *> &objects) : sentinel_(kTypeEnd), size_(objects.size()) {
  // build linked list from objects
  NDObject *last_object = &sentinel_;
  for (auto object : objects) {
    ASSIGN_NEXT_OBJ(last_object, object);
    ASSIGN_PREV_OBJ(object, last_object);
    last_object = object;
  }
  ASSIGN_NEXT_OBJ(last_object, &sentinel_);
  ASSIGN_PREV_OBJ(&sentinel_, last_object);
  context_.Init(objects);
}

std::vector<NDObject *> BasicBlock::ToVector() {
  auto iter = begin();
  std::vector<NDObject *> res;
  res.reserve(size());
  int i = 0;
  while (iter != end()) {
    PREV_OBJ(iter)->insn_ = nullptr;
    iter->tail_insn_ = nullptr;
    iter->index_ = i++;
    res.push_back(iter.get());
    ++iter;
  }
  PREV_OBJ(iter)->insn_ = nullptr;
  return res;
}

BasicBlock::iterator BasicBlock::Insert(BasicBlock::iterator iter, NDObject *object) {
  if (iter.get() == object) {
    return iter;
  }
  auto prev = NEXT_OBJ(iter);
  ASSIGN_PREV_OBJ(object, prev);
  ASSIGN_NEXT_OBJ(object, iter.get());
  ASSIGN_PREV_OBJ(iter, object);
  ASSIGN_NEXT_OBJ(prev, object);
  ++size_;
  return NDObjectIterator<false>(object);
}

BasicBlock::iterator BasicBlock::Erase(BasicBlock::iterator iter) {
  ASSERT(iter != end());
  auto prev = PREV_OBJ(iter);
  auto next = NEXT_OBJ(iter);
  ASSIGN_NEXT_OBJ(prev, next);
  ASSIGN_PREV_OBJ(next, prev);
  delete iter.get();
  --size_;
  return NDObjectIterator<false>(next);
}

BasicBlock::iterator BasicBlock::Move(BasicBlock::iterator iter, BasicBlock::pointer obj) {
  if (iter.get() == obj || iter.GetPrev().get() == obj) {
    return iter;
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

void BasicBlock::UpdateContext() {
  context_.Clear();
  context_.Init(begin(), end());
}

void ReorderStore(BasicBlock &block) {
  for (auto iter = block.begin(); iter != block.end();) {
    auto &users = block.context().GetUsers(iter.get());
    // Check if the variable is only used by the Store
    if (users.size() == 1 && (*users.cbegin())->GetObjectType() == kStore) {
      ++iter;
      if (iter.get() == *users.cbegin()) {
        // the Store is already in right position
        ++iter;
        continue;
      }
      block.Move(iter, *users.cbegin());
    } else {
      ++iter;
    }
  }
}
}  // namespace dvm::pass