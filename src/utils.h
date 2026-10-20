/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef _DVM_UTILS_H_
#define _DVM_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include "system.h"

namespace dvm {

class DynamicBitset {
 public:
  explicit DynamicBitset(size_t size = 0) : size_(size) {
    if (size) words_.assign((size + kWordBits - 1) / kWordBits, 0);
  }

  void Init(size_t size) {
    size_ = size;
    words_.assign((size + kWordBits - 1) / kWordBits, 0);
  }

  void Set(size_t id) { words_[id / kWordBits] |= uint64_t{1} << (id % kWordBits); }

  void Reset(size_t id) { words_[id / kWordBits] &= ~(uint64_t{1} << (id % kWordBits)); }

  bool Test(size_t id) const { return id < size_ && (words_[id / kWordBits] & (uint64_t{1} << (id % kWordBits))) != 0; }

  void UnionWith(const DynamicBitset &other) {
    for (size_t i = 0; i < words_.size(); ++i) words_[i] |= other.words_[i];
  }

  DynamicBitset Intersection(const DynamicBitset &other) const {
    DynamicBitset result(size_);
    for (size_t i = 0; i < words_.size(); ++i) result.words_[i] = words_[i] & other.words_[i];
    return result;
  }

  bool Intersects(const DynamicBitset &other, const DynamicBitset &mask) const {
    for (size_t i = 0; i < words_.size(); ++i) {
      if (words_[i] & other.words_[i] & mask.words_[i]) return true;
    }
    return false;
  }

  template <typename Visitor>
  void ForEachDifference(const DynamicBitset &other, Visitor visit) const {
    for (size_t i = 0; i < words_.size(); ++i) {
      uint64_t remaining = words_[i] & ~other.words_[i];
      while (remaining != 0) {
        visit(i * kWordBits + static_cast<size_t>(__builtin_ctzll(remaining)));
        remaining &= remaining - 1;
      }
    }
  }

 private:
  static constexpr size_t kWordBits = 64;
  size_t size_;
  std::vector<uint64_t> words_;
};

// Incremental strict transitive closure of the contracted DAG.
class DagReachability {
 public:
  // IDs must be a topological ordering of the initial singleton graph.
  DagReachability() = default;
  explicit DagReachability(const std::vector<std::vector<size_t>> &successors);

  void Init(const std::vector<std::vector<size_t>> &successors);

  bool WouldCreateCycle(size_t a, size_t b) const;

  bool HasPath(size_t from, size_t to) const {
    ASSERT(from != to && alive_.Test(from) && alive_.Test(to));
    return descendants_[from].Test(to);
  }

  // Caller must first reject cyclic contractions with WouldCreateCycle.
  void Contract(size_t keep, size_t drop);

 private:
  DynamicBitset alive_;
  std::vector<DynamicBitset> ancestors_;
  std::vector<DynamicBitset> descendants_;
};

}  // namespace dvm

#endif  // _DVM_UTILS_H_
