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

#include "utils.h"

#include <utility>

#include "system.h"

namespace dvm {

DagReachability::DagReachability(const std::vector<std::vector<size_t>> &successors)
    : alive_(successors.size()),
      ancestors_(successors.size(), DynamicBitset(successors.size())),
      descendants_(successors.size(), DynamicBitset(successors.size())) {
  for (size_t node = 0; node < successors.size(); ++node) {
    alive_.Set(node);
    for (size_t user : successors[node]) {
      ASSERT(user > node && user < successors.size());
      ancestors_[user].UnionWith(ancestors_[node]);
      ancestors_[user].Set(node);
    }
  }
  for (size_t node = successors.size(); node > 0; --node) {
    for (size_t user : successors[node - 1]) {
      descendants_[node - 1].UnionWith(descendants_[user]);
      descendants_[node - 1].Set(user);
    }
  }
}

bool DagReachability::WouldCreateCycle(size_t a, size_t b) const {
  ASSERT(a != b && alive_.Test(a) && alive_.Test(b));
  // An external vertex on either path becomes a cycle after contraction.
  // Strict closure excludes endpoints, so a direct edge alone is legal.
  return descendants_[a].Intersects(ancestors_[b], alive_) || descendants_[b].Intersects(ancestors_[a], alive_);
}

void DagReachability::Contract(size_t keep, size_t drop) {
  ASSERT(keep != drop && alive_.Test(keep) && alive_.Test(drop));
  DynamicBitset external = alive_;
  external.Reset(keep);
  external.Reset(drop);
  auto ak = ancestors_[keep].Intersection(external);
  const auto ad = ancestors_[drop].Intersection(external);
  auto dk = descendants_[keep].Intersection(external);
  const auto dd = descendants_[drop].Intersection(external);

  // Only ancestors/descendants unique to one side acquire new paths.
  // Preserve the snapshots: the representative is not its own ancestor.
  ak.ForEachDifference(ad, [&](size_t id) { descendants_[id].UnionWith(dd); });
  ad.ForEachDifference(ak, [&](size_t id) {
    descendants_[id].UnionWith(dk);
    descendants_[id].Set(keep);
  });
  dk.ForEachDifference(dd, [&](size_t id) { ancestors_[id].UnionWith(ad); });
  dd.ForEachDifference(dk, [&](size_t id) {
    ancestors_[id].UnionWith(ak);
    ancestors_[id].Set(keep);
  });

  ak.UnionWith(ad);
  ancestors_[keep] = std::move(ak);
  dk.UnionWith(dd);
  descendants_[keep] = std::move(dk);
  // Historical dropped bits may remain in other rows. Every query and
  // propagation snapshot masks them out; their positions are never reused.
  alive_.Reset(drop);
  ancestors_[drop] = DynamicBitset();
  descendants_[drop] = DynamicBitset();
}

}  // namespace dvm
