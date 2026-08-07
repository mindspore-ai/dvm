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

#include <map>
#include <queue>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "gkernel.h"

namespace dvm {
static int sym_dump_ = -1;

namespace {
struct _GraphReloadCloner : public CloneHelper {
  IntArrayRef *GetClone(IntArrayRef *shape) override { return shape; }
  ScalarRef *GetClone(ScalarRef *scalar) override { return scalar; }
  NDObject *GetClone(NDObject *op) override { return op; }
  void SetClone(NDObject *op, NDObject *clone) override {}
};
}  // namespace

class GraphSpliter {
 public:
  GraphSpliter() = default;

  // op pattern
  static inline constexpr uint32_t kPatNone = 0;
  static inline constexpr uint32_t kPatElemwise = 1u << 1;
  static inline constexpr uint32_t kPatBroadcast = 1u << 2;
  static inline constexpr uint32_t kPatReduce = 1u << 3;
  static inline constexpr uint32_t kPatReshape = 1u << 4;
  static inline constexpr uint32_t kPatCube = 1u << 5;
  // structure pattern
  static inline constexpr uint32_t kPatReducePost = 1u << 10;

  struct Area {
    int id;
    uint32_t pattern;
    IntArrayRef *shape;
    std::vector<NDObject *> ops;
    std::map<int, std::unordered_set<NDObject *>> borders;
    uint64_t reach_mask;
    StagesKernel::Stage *stage;
  };

#define PATTERN_CALL(pat)             \
  do {                                \
    PatternMerge(&GraphSpliter::pat); \
    if (sym_dump_) DumpArea(#pat);    \
  } while (0)

  void Run(GraphKernel *kernel) {
    int64_t sym_dim_next = -10000;
    for (auto op : kernel->build_ops_) {
      op->ShapeProp(sym_dim_next);
      op->prop_id_ = -1;
    }
    InitArea(kernel->build_ops_);
    BuildBorder();
    InitReachMask();
    if (sym_dump_ < 0) {
      sym_dump_ = getenv("DVM_SPLIT_SYM_DUMP") ? 1 : 0;
    }
    if (sym_dump_) {
      for (int idx = 0; idx < static_cast<int>(kernel->build_ops_.size()); ++idx) {
        kernel->build_ops_[idx]->index_ = idx;
      }
      std::cout << "\n************ shape prop ************\n" << kernel->DumpGraph(true) << std::endl;
      DumpArea("init");
    }
    PATTERN_CALL(BroadCons);
    PATTERN_CALL(ReduceProd);
    PATTERN_CALL(ReduceCons);
    PATTERN_CALL(ReshapeProd);
    PATTERN_CALL(CubeCons);
    BuildStage(kernel);
    for (auto area : areas_) {
      delete area;
    }
    if (sym_dump_) {
      std::cout << "\n************ split ************\n" << kernel->DumpGraph(true) << std::endl;
    }
  }

  Area *BroadCons(Area *area) {
    if (area->pattern > kPatBroadcast) {
      return nullptr;
    }
    for (const auto &bt : area->borders) {
      auto b = areas_[bt.first];
      if (b->pattern <= kPatBroadcast && ShapeAffine(b->shape, area->shape) && !CheckCycle(area, b) &&
          std::find_if(b->ops.begin(), b->ops.end(), [this, area](NDObject *op) {
            return stores_.count(op) && ShapesEqual(op->shape_ref_, area->shape);
          }) == b->ops.end()) {
        return b;
      }
    }
    return nullptr;
  }

  Area *ReduceProd(Area *area) {
    if (area->pattern > kPatReduce) {
      return nullptr;
    }
    for (const auto &bt : area->borders) {
      auto b = areas_[bt.first];
      if (b->pattern <= kPatBroadcast && ShapesEqual(area->shape, b->shape) && !CheckCycle(area, b)) {
        return b;
      }
    }
    return nullptr;
  }

  Area *ReduceCons(Area *area) {
    if (area->pattern & ~(kPatBroadcast | kPatElemwise)) {
      return nullptr;
    }
    for (auto &[id, prods] : area->borders) {
      auto b = areas_[id];
      if ((b->pattern & kPatReduce) && !CheckCycle(area, b) &&
          std::find_if(prods.begin(), prods.end(), [this, area](NDObject *op) {
            return op->obj_id_ == ObjectType::kReduce && ShapesEqual(op->shape_ref_, area->shape);
          }) != prods.end()) {
        area->pattern |= kPatReducePost;
        return b;
      }
    }
    return nullptr;
  }

  Area *ReshapeProd(Area *area) {
    if (area->pattern & kPatReshape) {
      return nullptr;
    }
    for (auto op : area->ops) {
      if (op->obj_id_ != ObjectType::kReshape || op->prop_id_ == op->lhs_->prop_id_) {
        continue;
      }
      auto a = areas_[op->lhs_->prop_id_];
      if (!(a->pattern & ~(kPatElemwise | kPatBroadcast)) && !CheckCycle(area, a)) {
        return a;
      }
    }
    return nullptr;
  }

  Area *CubeCons(Area *area) {
    if (area->pattern & ~kPatElemwise) {
      return nullptr;
    }
    for (auto &[id, prods] : area->borders) {
      auto b = areas_[id];
      if ((b->pattern & kPatCube) && !CheckCycle(area, b) &&
          std::find_if(prods.begin(), prods.end(), [this, area](NDObject *op) {
            return op->obj_id_ == ObjectType::kCubeOp && ShapesEqual(op->shape_ref_, area->shape);
          }) != prods.end()) {
        return b;
      }
    }
    return nullptr;
  }

  void BuildStage(GraphKernel *graph_kernel) {
    auto flags = graph_kernel->Flags() & ~static_cast<uint32_t>(KernelFlag::kPrivate1);
    std::vector<int> active_ids;
    active_ids.reserve(areas_.size());
    std::vector<int> indegrees(areas_.size(), 0);
    std::vector<std::vector<int>> consumers(areas_.size());
    // Materialize every live area first so later store/load wiring can always find the producer stage.
    for (auto area : areas_) {
      if (area->pattern == kPatNone) {
        area->stage = nullptr;
        continue;
      }
      VKernel *kernel;
      if (area->pattern & kPatCube) {
        kernel = new MixKernel(flags);
      } else if (area->pattern & (kPatReducePost | kPatReshape)) {
        kernel = new SpecVecKernel(flags);
      } else {
        kernel = new VKernelS(flags);
      }
      area->stage = new GraphKernel::Stage(kernel);
      active_ids.push_back(area->id);
    }
    // borders records "this area depends on producer area"; invert it into producer->consumer edges.
    for (auto area : areas_) {
      if (area->pattern == kPatNone) continue;
      for (const auto &bt : area->borders) {
        auto prod_id = bt.first;
        if (areas_[prod_id]->pattern == kPatNone) continue;
        consumers[prod_id].push_back(area->id);
        indegrees[area->id]++;
      }
    }
    std::priority_queue<int> ready;
    for (auto area_id : active_ids) {
      if (indegrees[area_id] == 0) {
        ready.push(area_id);
      }
    }
    // Append stages in topological order so every cross-stage producer runs before its consumers.
    size_t ordered_num = 0;
    while (!ready.empty()) {
      auto area_id = ready.top();
      ready.pop();
      graph_kernel->AppendStage(areas_[area_id]->stage);
      ordered_num++;
      for (auto cons_id : consumers[area_id]) {
        if (--indegrees[cons_id] == 0) {
          ready.push(cons_id);
        }
      }
    }
    ASSERT(ordered_num == active_ids.size());
    // Each cross-stage value is materialized once on the producer side and then reused by all consumers.
    for (auto area : areas_) {
      if (area->pattern == kPatNone) continue;
      for (const auto &bt : area->borders) {
        for (auto op : bt.second) {
          if (auto st = stores_.find(op); st == stores_.end()) {
            auto store = new NDStore(op);
            areas_[op->prop_id_]->stage->StageStore(store);
            stores_[op] = store;
          }
        }
      }
    }
    for (auto area : areas_) {
      if (area->pattern != kPatNone) {
        auto kernel = area->stage->kernel;
        std::unordered_map<NDAccess *, NDAccess *> stage_loads;
        std::unordered_set<NDObject *> stage_ops;
        stage_loads.reserve(area->ops.size());
        stage_ops.reserve(area->ops.size());
        std::function<void(NDObject *)> append_stage_op;
        append_stage_op = [&](NDObject *op) {
          if (!op->IsSimd() || op->prop_id_ != area->id || !stage_ops.insert(op).second) {
            return;
          }
          op->ForInput([this, area, kernel, &stage_loads, &append_stage_op](NDObject *&in) {
            if (in->IsLoad()) {
              auto acc = static_cast<NDAccess *>(in);
              auto it = stage_loads.find(acc);
              if (it == stage_loads.end()) {
                // Clone graph loads per stage so child-kernel tiling stays local.
                // The original graph load also needs a reloc slot for StageBind.
                if (acc->addr_.reloc_ == nullptr) {
                  acc->addr_.Update(&acc->addr_.data);
                }
                _GraphReloadCloner cloner;
                auto clone = static_cast<NDAccess *>(in->Clone(cloner));
                clone->prop_id_ = area->id;
                area->stage->StageLoad(clone, acc);
                kernel->Append(clone);
                it = stage_loads.emplace(acc, clone).first;
              }
              in = it->second;
            } else if (in->prop_id_ != area->id) {
              ASSERT(stores_.count(in));
              auto store = stores_[in];
              NDAccess *load = nullptr;
              if (auto it = swap_loads_.find(store); it != swap_loads_.end()) {
                for (auto ld : it->second) {
                  if (ld->prop_id_ == area->id) {
                    load = ld;
                    break;
                  }
                }
              }
              if (load == nullptr || load->prop_id_ != area->id) {
                // Consumers read foreign values through an explicit stage-local load bound to the producer store.
                load = new NDLoad(nullptr, in->shape_ref_, in->type_id_);
                load->prop_id_ = area->id;
                area->stage->StageLoad(load, store);
                kernel->Append(load);
                swap_loads_[store].push_back(load);
              }
              in = load;
            } else if (in->prop_id_ == area->id) {
              append_stage_op(in);
            }
          });
          if (op->SharedNdd() && op->nd_.data != op->lhs_->nd_.data) {
            op->nd_.data = op->lhs_->nd_.data;
          }
          kernel->Append(op);
          if (auto st = stores_.find(op); st != stores_.end()) {
            kernel->Append(st->second);
          }
        };
        for (auto it = graph_kernel->build_ops_.rbegin(); it != graph_kernel->build_ops_.rend(); ++it) {
          append_stage_op(*it);
        }
      }
    }
  }

 protected:
  bool ShapesEqual(const IntArrayRef *a, const IntArrayRef *b) {
    if (a == b) return true;
    if (a == nullptr || b == nullptr || a->size != b->size) return false;
    for (size_t i = 0; i < a->size; ++i) {
      if (a->data[i] != b->data[i]) return false;
    }
    return true;
  }

  bool ShapeAffine(const IntArrayRef *small, const IntArrayRef *large) {
    if (small == large) return true;
    if (small->size > large->size) return false;
    size_t diff = large->size - small->size;
    for (size_t i = 0; i < small->size; ++i) {
      if (small->data[i] != large->data[i + diff] && small->data[i] != 1) return false;
    }
    return true;
  }

  bool CheckCycle(Area *area, Area *prod) {
    if (area->id < 64) {
      return (prod->reach_mask >> area->id) & 1;
    } else {
      std::unordered_set<Area *> visited;
      std::function<bool(Area *)> dfs;
      dfs = [this, &dfs, area, &visited](Area *a) -> bool {
        if (a == area) return true;
        if (!visited.count(a)) {
          visited.insert(a);
          for (const auto &bt : a->borders) {
            if (dfs(areas_[bt.first])) return true;
          }
        }
        return false;
      };
      return dfs(prod);
    }
  }

  void MergeArea(Area *prod, Area *cons) {
    prod->pattern |= cons->pattern;
    cons->pattern = kPatNone;
    prod->shape = cons->shape;
    prod->reach_mask |= cons->reach_mask;
    for (auto op : cons->ops) {
      op->prop_id_ = prod->id;
    }
    cons->ops.insert(cons->ops.end(), prod->ops.begin(), prod->ops.end());
    std::swap(prod->ops, cons->ops);
  }

  void InitArea(const std::vector<NDObject *> &build_ops) {
    int next_area_id = 0;
    for (auto it = build_ops.rbegin(); it != build_ops.rend(); ++it) {
      auto op = *it;
      if (!op->IsSimd()) {
        if (op->obj_id_ == ObjectType::kStore) {
          stores_[op->lhs_] = static_cast<NDAccess *>(op);
        }
        continue;
      }
      if (op->prop_id_ == -1) {
        auto area = new Area();
        bool hold_back;
        if (op->obj_id_ == ObjectType::kBroadcastTo) {
          area->pattern = kPatBroadcast;
          area->shape = op->shape_ref_;
          hold_back = true;
        } else if (op->obj_id_ == ObjectType::kReduce) {
          area->pattern = kPatReduce;
          area->shape = op->lhs_->shape_ref_;
          hold_back = false;
        } else if (op->obj_id_ == ObjectType::kReshape) {
          area->pattern = kPatReshape;
          area->shape = op->shape_ref_;
          hold_back = true;
        } else if (op->obj_id_ == ObjectType::kCubeOp) {
          area->pattern = kPatCube;
          area->shape = op->shape_ref_;
          hold_back = true;
        } else {
          area->pattern = kPatElemwise;
          area->shape = op->shape_ref_;
          hold_back = false;
        }
        area->id = next_area_id++;
        op->prop_id_ = area->id;
        area->ops.push_back(op);
        areas_.push_back(area);
        if (hold_back) {
          continue;
        }
      }
      auto area = areas_[op->prop_id_];
      op->ForInput([&area, this](NDObject *in) {
        if (!in->IsLoad() && in->obj_id_ != ObjectType::kCubeOp && in->obj_id_ != ObjectType::kReduce &&
            in->obj_id_ != ObjectType::kReshape) {
          if (in->prop_id_ == -1) {
            if (ShapesEqual(in->shape_ref_, area->shape)) {
              area->ops.push_back(in);
              in->prop_id_ = area->id;
            }
          } else if (auto a = areas_[in->prop_id_]; a != area) {
            if (ShapesEqual(a->shape, area->shape) && !CheckCycle(area, a)) {
              MergeArea(a, area);
              area = a;
            }
          }
        }
      });
    }
  }

  void BuildBorder() {
    for (auto a : areas_) {
      if (a->pattern == kPatNone) continue;
      for (auto op : a->ops) {
        op->ForInput([this, a](NDObject *in) {
          if (!in->IsLoad() && in->prop_id_ != a->id) {
            a->borders[in->prop_id_].insert(in);
          }
        });
      }
    }
  }

  void InitReachMask() {
    for (auto a : areas_) {
      a->reach_mask = 0;
      for (const auto &bt : a->borders) {
        if (bt.first < 64) {
          a->reach_mask |= 1ull << bt.first;
        }
      }
    }
    for (auto it = areas_.rbegin(); it != areas_.rend(); ++it) {
      auto a = *it;
      for (const auto &bt : a->borders) {
        if (auto b = areas_[bt.first]; b->id < 64) {
          a->reach_mask |= b->reach_mask;
        }
      }
    }
  }

  void PatternMerge(Area *(GraphSpliter::*pattern)(Area *)) {
    bool merged = true;
    while (merged) {
      merged = false;
      for (auto area : areas_) {
        if (area->pattern == kPatNone) continue;
        auto m = (this->*pattern)(area);
        if (m == nullptr) continue;
        MergeArea(m, area);
        auto &borders = m->borders;
        for (auto &[id, prods] : area->borders) {
          if (id == m->id) continue;
          if (auto it = borders.find(id); it != borders.end()) {
            it->second.merge(prods);
          } else {
            borders.emplace(id, prods);
          }
        }
        for (auto a : areas_) {
          if (a == m || a->pattern == kPatNone) continue;
          if (auto it = a->borders.find(area->id); it != a->borders.end()) {
            if (auto it2 = a->borders.find(m->id); it2 != a->borders.end()) {
              it2->second.merge(it->second);
            } else {
              a->borders.emplace(m->id, std::move(it->second));
              a->borders.erase(it);
            }
          }
        }
        merged = true;
        break;
      }
    }
  }

  void DumpArea(const char *pat) {
    std::cout << "\n************ pattern." << pat << " ************" << std::endl;
    for (size_t i = 0; i < areas_.size(); ++i) {
      auto area = areas_[i];
      if (area->pattern != kPatNone) {
        std::cout << i << ": id=" << area->id << ", pat=" << area->pattern << ", shape=" << *area->shape << ", ops=[";
        for (auto op : area->ops) {
          std::cout << op->index_;
          if (op != area->ops.back()) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
        for (auto &[id, prods] : area->borders) {
          std::cout << "   border: area=" << id << ", cross=[";
          for (auto op : prods) {
            std::cout << op->index_ << ", ";
          }
          std::cout << "]" << std::endl;
        }
      }
    }
  }

  std::vector<Area *> areas_;
  std::unordered_map<NDObject *, NDAccess *> stores_;
  std::unordered_map<NDAccess *, std::vector<NDAccess *>> swap_loads_;
};

GraphKernel::GraphKernel(uint32_t flags) : StagesKernel(flags) {}

void GraphKernel::Append(NDObject *op) { build_ops_.push_back(op); }

uint64_t GraphKernel::CodeGen() {
  if (stages_.empty()) {
    GraphSpliter spliter;
    spliter.Run(this);
    if (IsDynamic()) {
      return 0;
    }
  }
  return StagesKernel::CodeGen();
}

void GraphKernel::Clone(VKernel *base, CloneHelper &helper) {
  auto *k = static_cast<GraphKernel *>(base);
  for (auto *op : k->build_ops_) {
    build_ops_.push_back(op->CloneUpdate(helper));
  }
}

void GraphKernel::Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) {
  if (rgraph) {
    DumpRefHelper helper(oss);
    helper.DumpGraph(indent, "sym", build_ops_);
    return;
  }
  StagesKernel::Dump(oss, indent, rgraph);
}
}  // namespace dvm
