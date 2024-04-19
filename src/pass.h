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

#ifndef _DVM_PASS_H_
#define _DVM_PASS_H_

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "ops.h"

namespace dvm::pass {
#define NEXT_OBJ(a) reinterpret_cast<NDObject *>((a)->insn_)
#define PREV_OBJ(a) reinterpret_cast<NDObject *>((a)->tail_insn_)

template <bool IsReverse>
class NDObjectIterator {
  friend NDObjectIterator<!IsReverse>;

 public:
  // these alias is used in <algorithm>
  using value_type = NDObject;
  using pointer = NDObject *;
  using reference = NDObject &;
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;

 private:
  pointer ptr_;

 public:
  NDObjectIterator(pointer ptr) : ptr_(ptr) {}

  // Get a reverse iterator to the same content
  NDObjectIterator<!IsReverse> GetReverse() const { return NDObjectIterator<!IsReverse>(ptr_); }

  NDObjectIterator &operator++() {
    ptr_ = IsReverse ? PREV_OBJ(ptr_) : NEXT_OBJ(ptr_);
    return *this;
  }

  NDObjectIterator operator++(int) {
    NDObjectIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  NDObjectIterator operator--() {
    ptr_ = IsReverse ? NEXT_OBJ(ptr_) : PREV_OBJ(ptr_);
    return *this;
  }

  NDObjectIterator operator--(int) {
    NDObjectIterator tmp = *this;
    --(*this);
    return tmp;
  }

  bool operator!=(const NDObjectIterator &other) const { return ptr_ != other.ptr_; }

  reference operator*() { return *ptr_; }

  pointer operator->() { return ptr_; }

  NDObjectIterator GetPrev() {
    NDObjectIterator iter = *this;
    return --iter;
  }

  NDObjectIterator GetNext() {
    NDObjectIterator iter = *this;
    return ++iter;
  }

  NDObject *get() { return ptr_; }
};

class BasicBlockContext {
  friend class BasicBlock;

  void Init(const std::vector<NDObject *> &objects) {
    users_.reserve(objects.size());
    for (auto object : objects) {
      users_.insert({object, std::vector<NDObject *>()});
    }
    for (auto object : objects) {
      Insert(object);
    }
  }

  void Init(NDObjectIterator<false> begin, NDObjectIterator<false> end) {
    while (begin != end) {
      users_.insert({begin.get(), std::vector<NDObject *>()});
      Insert(begin++.get());
    }
  }

  void Insert(NDObject *object) {
    if (object->lhs_ == nullptr) {
      // if lhs_ is empty, than rhs_ must be empty too
      return;
    }
    users_[object->lhs_].emplace_back(object);
    if (object->rhs_ == nullptr) {
      return;
    }
    users_[object->rhs_].emplace_back(object);
    if (object->GetObjectType() == kSelect) {
      users_[reinterpret_cast<SelectOp *>(object)->cond_].emplace_back(object);
    }
  }

  void Erase(NDObject *object) { users_.erase(object); }

  void Clear() { users_.clear(); }

 public:
  // Will fail when object not exist in the context.
  // And users may duplicate
  const std::vector<NDObject *> &GetUsers(NDObject *object) const {
    auto iter = users_.find(object);
    ASSERT(iter != users_.end());
    return iter->second;
  }

 protected:
  std::unordered_map<NDObject *, std::vector<NDObject *>> users_;
};

/// @brief Container of NDObject* in pass pipeline
class BasicBlock {
 public:
  BasicBlock(const std::vector<NDObject *> &objects, std::vector<NDObject *> &owner);

 public:
  using iterator = NDObjectIterator<false>;
  using reverse_iterator = NDObjectIterator<true>;
  using pointer = NDObject *;

  void Reinit(const std::vector<NDObject *> &objects);

  iterator begin() { return iterator(NEXT_OBJ(&sentinel_)); }

  iterator end() { return iterator(&sentinel_); }

  reverse_iterator rbegin() { return reverse_iterator(PREV_OBJ(&sentinel_)); }

  reverse_iterator rend() { return reverse_iterator(&sentinel_); }

  inline size_t size() const { return size_; }

  // Insert a object before position of iterator, ownership is move to object_owner_.
  iterator Insert(iterator iter, pointer object);

  // Remove the object at position of iterator, the NDObject won't be deleted.
  iterator Erase(iterator iter);

  // Move an object to a new position just before the iterator
  iterator Move(iterator iter, pointer object);

  void PushFront(pointer ptr) { Insert(begin(), ptr); }

  void PushBack(pointer ptr) { Insert(end(), ptr); }

  std::vector<NDObject *> ToVector();

  void Export(std::vector<NDObject *> &objects);

  // Remove use of insn_ and tail_insn_
  void Clear();

  const BasicBlockContext &context() const { return context_; }

  // Should be called after dependency graph of objects has changed
  void UpdateContext();

 protected:
  NDLoadDummy sentinel_;
  size_t size_;  // Will size be used?
  BasicBlockContext context_;
  std::vector<NDObject *> &objects_owner_;
};

#undef NEXT_OBJ
#undef PREV_OBJ

// ------ Introducing Pass--------

/// @brief Reorder Store, to reduce maximum num of live variables
/// @details Check the variable used by Store, if the Store operator is
/// the only consumer of the variable, move the Store operator
/// @example
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Binary(%0, %1)
///   %3 = Binary(%0, %1)
///   %4 = Store(%3)
///   %5 = Store(%2)
/// }
/// ---------->
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Binary(%0, %1)
///   %5 = Store(%2)
///   %3 = Binary(%0, %1)
///   %4 = Store(%3)
/// }
/// ----------
void ReorderStore(BasicBlock &block);

/// @brief Reorder Load, to bring the that were used earlier forward
/// @example
/// vkernel.graph() {
///   %0 = Load()
///   %1 = Load()
///   %2 = Unary(%1)
///   %3 = Binary(%0, %2)
///   %4 = Load()
///   %5 = Binary(%3, %4)
///   %6 = Store(%5)
/// }
/// ---------->
/// vkernel.graph() {
///   %1 = Load()
///   %0 = Load()
///   %4 = Load()
///   %2 = Unary(%1)
///   %3 = Binary(%0, %2)
///   %5 = Binary(%3, %4)
///   %6 = Store(%5)
/// }
/// ----------
void ReorderLoad(BasicBlock &block);

// For debug
void PrintPeakLive(BasicBlock &bb);

/// @brief Reorder objects, to minimum peak live variables
/// @details there are two methods now.
/// The heuristic version is quick but sometimes give suboptimal results.
/// The DP version is slow but give optimal results.
/// Currently we use the heuristic method.
void CompactPeakLiveness(BasicBlock &bb);

/// @brief Eliminate Reshape Objects if possible
/// @example
/// vkernel.graph() {
///   %0::(4,8) = Load()
///   %1::(8,4) = Load()
///   %2::(4,8) = Reshape(%1)
///   %3::(4,8) = Binary(%0, %2)
///   %4::(4,8) = Store(%3)
/// }
/// ---------->
/// vkernel.graph() {
///   %0::(8,4) = Load()
///   %1::(8,4) = Load()
///   %2::(8,4) = Binary(%0, %1)
///   %3::(8,4) = Store(%2)
/// }
/// ----------
void EliminateReshape(BasicBlock &bb);

/// @brief Optimize memory transfer operations, especially from UB (Unified Buffer) to GM (Global Memory), by
/// reorganizing non-continuous memory segments within the UB into a continuous memory layout. This significantly speeds
/// up the data transfer process to the GM.
void InsertRemovePad(BasicBlock &block);

using Pass = void (*)(BasicBlock &);
extern std::vector<Pass> passes;
}  // namespace dvm::pass

#endif  // _DVM_PASS_H_