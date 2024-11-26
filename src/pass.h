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
class BasicBlock;
class ObjectList {
 public:
  ObjectList() : sentinel_(kTypeEnd) {}
  void Build(const std::vector<NDObject *> &objects, bool reindex);

  void Insert(NDObject *pos, NDObject *obj) {
    auto prev = Prev(pos);
    SetPrev(obj, prev);
    SetNext(obj, pos);
    SetPrev(pos, obj);
    SetNext(prev, obj);
    size_++;
    obj->index_ = capacity_++;
  }

  void Erase(NDObject *obj) {
    auto prev = Prev(obj);
    auto next = Next(obj);
    SetNext(prev, next);
    SetPrev(next, prev);
    size_--;
  }

  NDObject *Begin() { return Next(&sentinel_); }
  NDObject *End() { return &sentinel_; }
  NDObject *ReverseBegin() { return Prev(&sentinel_); }
  NDObject *ReverseEnd() { return &sentinel_; }

  static NDObject *Next(NDObject *obj) { return reinterpret_cast<NDObject*>(obj->insn_); }
  static NDObject *Prev(NDObject *obj) { return reinterpret_cast<NDObject*>(obj->tail_insn_); }
  static void SetNext(NDObject *obj, NDObject *next) { obj->insn_ = reinterpret_cast<uint64_t*>(next); }
  static void SetPrev(NDObject *obj, NDObject *prev) { obj->tail_insn_ = reinterpret_cast<uint64_t*>(prev); }

  template <bool reverse>
  class Iterator {
    public:
      // these alias is used in <algorithm>
      using value_type = NDObject;
      using pointer = NDObject *;
      using reference = NDObject &;
      using iterator_category = std::bidirectional_iterator_tag;
      using difference_type = std::ptrdiff_t;

      Iterator(pointer ptr) : ptr_(ptr) {}

      Iterator &operator++() {
        ptr_ = reverse ? ObjectList::Prev(ptr_) : ObjectList::Next(ptr_);
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
      }

      Iterator operator--() {
        ptr_ = reverse ? ObjectList::Next(ptr_) : ObjectList::Prev(ptr_);
        return *this;
      }

      Iterator operator--(int) {
        Iterator tmp = *this;
        --(*this);
        return tmp;
      }

      bool operator!=(const Iterator &other) const { return ptr_ != other.ptr_; }

      reference operator*() { return *ptr_; }
      pointer operator->() { return ptr_; }

      Iterator GetPrev() {
        Iterator iter = *this;
        return --iter;
      }

      Iterator GetNext() {
        Iterator iter = *this;
        return ++iter;
      }

      NDObject *get() { return ptr_; }

    private:
      pointer ptr_;
  };

 protected:
  NDLoadDummy sentinel_;
  size_t size_;
  size_t capacity_;
  friend BasicBlock;
};

class BasicBlock {
 public:
  struct Edge {
    int64_t next;
    NDObject *user;
  };

  using iterator = ObjectList::Iterator<false>;
  using reverse_iterator = ObjectList::Iterator<true>;

  BasicBlock(const std::vector<NDObject *> &objects, std::vector<NDObject *> &owner);

  iterator begin() { return iterator(list_.Begin()); }
  iterator end() { return iterator(list_.End()); }
  reverse_iterator rbegin() { return reverse_iterator(list_.ReverseBegin()); }
  reverse_iterator rend() { return reverse_iterator(list_.ReverseEnd()); }

  inline size_t size() const { return list_.size_; }
  inline size_t capacity() const { return list_.capacity_; }

  iterator Insert(iterator iter, NDObject *object);
  void Erase(NDObject *object);
  iterator Move(iterator iter, NDObject *object);

  void PushFront(NDObject *ptr) { Insert(begin(), ptr); }
  void PushBack(NDObject *ptr) { Insert(end(), ptr); }

  template <bool if_update_index = true>
  std::vector<NDObject *> ToVector();

  void Export(std::vector<NDObject *> &objects);

  // Will fail when object is not exist in the context.
  // And users may duplicate
  std::vector<NDObject *> GetUsers(NDObject *object) const {
    std::vector<NDObject *> res;
    for (auto idx = GetHead(object); idx != -1; idx = edges_[idx].next) {
      res.push_back(edges_[idx].user);
    }
    return res;
  }

  size_t GetUserNum(NDObject *object) const {
    size_t num = 0;
    for (auto idx = GetHead(object); idx != -1; idx = edges_[idx].next) {
      num++;
    }
    return num;
  }

  bool IsMultiUsers(NDObject *object) const {
    auto idx = GetHead(object);
    return idx != -1 && edges_[idx].next != -1;
  }

  inline void AddUser(NDObject *obj, NDObject *new_user) {
    edges_.push_back({GetHead(obj), new_user});
    SetHead(obj, static_cast<int>(edges_.size() - 1));
  }

  ObjectList &List() { return list_; }

 protected:
  static int GetHead(NDObject *obj) { return obj->lead_dim_; }
  static void SetHead(NDObject *obj, int head) { obj->lead_dim_ = head; }

  ObjectList list_;
  std::vector<Edge> edges_;
  std::vector<NDObject *> &objects_owner_;
};

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

/// @brief Optimize reduceSum operation to minimize data movement and reduce the number of atomic additions
void InsertAtomicCum(BasicBlock &block);

using Pass = void (*)(BasicBlock &);
extern std::vector<Pass> passes;
}  // namespace dvm::pass

#endif  // _DVM_PASS_H_
