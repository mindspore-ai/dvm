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
  static inline constexpr uint32_t kPatCubePost = 1u << 11;
  static inline constexpr uint32_t kPatHorizontal = 1u << 12;

  static inline constexpr uint32_t kPatMaskOp = (1u << 10) - 1;

  struct Area {
    int id;
    uint32_t pattern;
    uint32_t epoch{0};
    int peak_live;
    uint32_t item_size;
    IntArrayRef *shape;
    std::vector<NDObject *> ops;
    std::map<int, std::vector<NDObject *>> borders;
    StagesKernel::Stage *stage{nullptr};

    int64_t GetSpace(int64_t init_size = 1) const {
      constexpr int64_t kDynDimSpace= 512;
      int64_t size = init_size;
      for (size_t i = 0; i < shape->size; ++i) {
        size *= shape->data[i] > 0 ? shape->data[i] : kDynDimSpace;
      }
      return size;
    }

    void AddBorder(int prod_id, NDObject *op) {
      auto &prods = borders[prod_id];
      if (std::find(prods.begin(), prods.end(), op) == prods.end()) {
        prods.push_back(op);
      }
    }

    void AddBorders(int prod_id, const std::vector<NDObject *> &ops) {
      for (auto op : ops) {
        AddBorder(prod_id, op);
      }
    }
  };

  struct Candidate {
    Area *a;
    Area *b;
    int64_t score;
    uint32_t pattern;
    int peak_live;
  };

  void Run(GraphKernel *kernel) {
    int64_t sym_dim_next = -10000;
    for (auto op : kernel->build_ops_) {
      op->ShapeProp(sym_dim_next);
      op->prop_id_ = -1;
    }
    InitArea(kernel->build_ops_);
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
    constexpr int kMaxIterNum = 10;
    for (int iter = 0; iter < kMaxIterNum; ++iter) {
      auto cands = CollectCandidates();
      if (cands.empty()) break;
      uint32_t epoch = fuse_epoch_;
      for (auto &c : cands) {
        if (c.a->epoch <= epoch && c.b->epoch <= epoch) {
          DoFuse(c);
        }
      }
      if (sym_dump_) {
        DumpArea("loop");
      }
    }
    BuildStage(kernel);
    for (auto area : areas_) {
      delete area;
    }
    if (sym_dump_) {
      std::cout << "\n************ split ************\n" << kernel->DumpGraph(true) << std::endl;
    }
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
    std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
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
  bool ShapeEqual(const IntArrayRef *a, const IntArrayRef *b) {
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
    std::unordered_set<Area *> visited;
    std::function<bool(Area *)> dfs;
    dfs = [this, &dfs, area, &visited](Area *a) -> bool {
      if (a == area) return true;
      if (!visited.insert(a).second) return false;
      for (const auto &bt : a->borders) {
        if (dfs(areas_[bt.first])) return true;
      }
      return false;
    };
    return dfs(prod);
  }

  void InitArea(const std::vector<NDObject *> &build_ops) {
    int next_area_id = 0;
    std::unordered_set<NDObject *> lives;
    for (auto it = build_ops.rbegin(); it != build_ops.rend(); ++it) {
      auto op = *it;
      if (!op->IsStore() && lives.count(op) == 0) {
        continue;
      }
      op->ForInput([&lives](NDObject *in) { lives.insert(in); });
    }
    for (auto op : build_ops) {
      if (!op->IsSimd()) {
        op->prop_id_ = -1;
        if (op->obj_id_ == ObjectType::kStore) {
          stores_.emplace(op->lhs_, static_cast<NDAccess *>(op));
        }
        continue;
      }
      if (lives.count(op) == 0) continue;
      auto area = new Area();
      if (op->obj_id_ == ObjectType::kBroadcastTo) {
        area->pattern = kPatBroadcast;
        area->shape = op->shape_ref_;
      } else if (op->obj_id_ == ObjectType::kReduce) {
        area->pattern = kPatReduce;
        area->shape = op->lhs_->shape_ref_;
      } else if (op->obj_id_ == ObjectType::kReshape) {
        area->pattern = kPatReshape;
        area->shape = op->shape_ref_;
      } else if (op->obj_id_ == ObjectType::kCubeOp) {
        area->pattern = kPatCube;
        area->shape = op->shape_ref_;
      } else {
        area->pattern = kPatElemwise;
        area->shape = op->shape_ref_;
        if (op->rhs_) {
          op->ForInput([this, area](NDObject *in) {
            if (area->pattern == kPatElemwise && !ShapeEqual(area->shape, in->shape_ref_)) {
              area->pattern = kPatBroadcast;
            }
          });
        }
      }
      area->id = next_area_id++;
      op->prop_id_ = area->id;
      area->item_size = ITEM_SIZE[op->type_id_];
      area->ops.push_back(op);
      area->peak_live = 1;
      op->ForInput([this, op, area](NDObject *in) {
        if (in->IsLoad()) {
          consumers_[in].push_back(op);
        } else if (in->prop_id_ != -1) {
          ASSERT(in->prop_id_ != area->id);
          area->AddBorder(in->prop_id_, in);
          consumers_[in].push_back(op);
        }
        area->peak_live++;
      });
      areas_.push_back(area);
    }
  }

  template <typename T>
  void TravelInput(NDObject *root, const T &func) {
    std::vector<NDObject *> &stack = temp_ops_;
    stack.push_back(root);
    while (!stack.empty()) {
      auto top = stack.back();
      stack.pop_back();
      if (func(top)) {
        top->ForInput([&stack](NDObject *in) { stack.push_back(in); });
      }
    }
  }

  bool ReducePostCheck(NDObject *red) {
    auto dims = static_cast<ReduceOp *>(red)->DimRef();
    if (dims->size == 0) {
      return false;
    }
    auto shape = red->lhs_->shape_ref_;
    int64_t min_dim = shape->size - 1;
    int64_t size = shape->data[min_dim];
    while (size > 0 && size > 32 * 1024 && min_dim > 0) {
      min_dim--;
      size *= shape->data[min_dim];
    }
    for (size_t i = 0; i < dims->size; ++i) {
      auto x = dims->data[i];
      if (x < 0) {
        x += shape->size;
      }
      if (x <= min_dim) return false;
    }
    return true;
  }

  std::pair<uint32_t, int64_t> VFuseCheck(Area *prod, Area *cons) {
    constexpr uint32_t kAffineMask = kPatElemwise | kPatBroadcast | kPatReduce;
    constexpr int64_t kPhaseAffine = 2;
    constexpr int64_t kPhaseCV = 1;
    constexpr int64_t kPhaseReshape = 0;
    const auto &borders = cons->borders.at(prod->id);
    uint32_t prod_op_mask = prod->pattern & kPatMaskOp;
    uint32_t cons_op_mask = cons->pattern & kPatMaskOp;
    NDObject *prod_reduce = nullptr;
    if (prod_op_mask & kPatReduce) {
      for (auto op : borders) {
        TravelInput(op, [&prod_reduce, prod](NDObject *x) {
          if (x->prop_id_ != prod->id) return true;
          if (x->obj_id_ == kReduce) {
            prod_reduce = x;
          }
          return prod_reduce != nullptr;
        });
      }
    }
    auto in_range = [](uint32_t pat, uint32_t mask) { return (pat & ~mask) == 0; };
    auto phase_score = [](int64_t phase) { return phase << 60; };
    if (in_range(prod_op_mask, kAffineMask) && in_range(cons_op_mask, kAffineMask)) {
      if (prod_reduce) {
        if (prod->GetSpace() == cons->GetSpace() && ReducePostCheck(prod_reduce)) {
          return std::make_pair(kPatReducePost, phase_score(kPhaseAffine));
        }
        return std::make_pair(kPatNone, 0);
      }
      if (ShapeEqual(cons->shape, prod->shape)) {
        return std::make_pair(kPatElemwise, phase_score(kPhaseAffine));
      }
      if (ShapeAffine(prod->shape, cons->shape)) {
        return std::make_pair(kPatBroadcast, phase_score(kPhaseAffine));
      }
      if (ShapeAffine(cons->shape, prod->shape)) {
        // TODO: broadcast store
      }
    } else if (prod_op_mask == kPatCube) {
      if (in_range(cons_op_mask, kPatElemwise | kPatBroadcast) && ShapeEqual(prod->shape, cons->shape) &&
          cons->ops.size() < 8) {
        return std::make_pair(kPatCubePost, phase_score(kPhaseCV));
      }
    } else if (in_range(prod_op_mask, kAffineMask | kPatReshape) &&
               in_range(cons_op_mask, kAffineMask | kPatReshape)) {
      if (cons_op_mask == kPatReshape) {
        return std::make_pair(kPatReshape, phase_score(kPhaseReshape) + 2);
      }
      if (prod_op_mask == kPatReshape) {
        return std::make_pair(kPatReshape, phase_score(kPhaseReshape) + 1);
      }
      if (borders.size() == 1 && (*borders.begin())->obj_id_ == kReshape) {
        if (prod_reduce) {
          if (!(cons_op_mask & kPatBroadcast) && ReducePostCheck(prod_reduce)) {
            return std::make_pair(kPatReducePost, phase_score(kPhaseReshape));
          }
          return std::make_pair(kPatNone, 0);
        }
        return std::make_pair(kPatReshape, phase_score(kPhaseReshape));
      }
    }
    return std::make_pair(kPatNone, 0);
  }

  bool HFuseCheck(Area *x, Area *y) {
    if ((x->pattern & kPatCube) || (y->pattern & kPatCube)) {
      return false;
    }
    if (!ShapeEqual(x->shape, y->shape)) {
      return false;
    }
    auto find_reduce = [](Area *a) -> ReduceOp * {
      for (auto op : a->ops) {
        if (op->obj_id_ == kReduce) {
          return static_cast<ReduceOp *>(op);
        }
      }
      return nullptr;
    };
    auto red_x = find_reduce(x);
    auto red_y = find_reduce(y);
    if (red_x != nullptr && red_y != nullptr && !ShapeEqual(red_x->DimRef(), red_y->DimRef())) {
      return false;
    }
    if (CheckCycle(x, y) || CheckCycle(y, x)) {
      return false;
    }
    return true;
  }

  int PeakLiveOps(Area *x, Area *y) {
    std::vector<NDObject *> &seq = temp_ops_;
    seq.reserve(x->ops.size() + y->ops.size());
    seq.insert(seq.end(), x->ops.begin(), x->ops.end());
    seq.insert(seq.end(), y->ops.begin(), y->ops.end());
    std::unordered_set<NDObject *> lives;
    int cur_live = 0, peak_live = 0;
    for (auto it = seq.rbegin(); it != seq.rend(); ++it) {
      auto op = *it;
      bool need_store = false;
      if (lives.count(op) == 0) {
        lives.insert(op);
        cur_live++;
        need_store = true;
      } else if (stores_.find(op) != stores_.end()) {
        need_store = true;
      }
      op->ForInput([&lives, &cur_live](NDObject *in) {
        if (lives.count(in) == 0) {
          lives.insert(in);
          cur_live++;
        }
      });
      if (cur_live > peak_live) {
        peak_live = cur_live;
      }
      if (!need_store) {
        cur_live--;
      }
    }
    seq.clear();
    return peak_live;
  }

  std::pair<int64_t, int> GetScore(Area *x, Area *y, int64_t shared) {
    int64_t x_space = x->GetSpace(x->item_size);
    int64_t y_space = y->GetSpace(y->item_size);
    int64_t score = shared * x_space;
    auto it = y->borders.find(x->id);
    if (it != y->borders.end()) {
      auto &ops = it->second;
      score += y_space * ops.size();
      for (auto op : ops) {
        if (stores_.find(op) != stores_.end()) continue;
        auto &cons = consumers_[op];
        if (std::find_if(cons.begin(), cons.end(), [y](NDObject *p) { return p->prop_id_ != y->id; }) == cons.end()) {
          score += x_space;
        }
      }
    }
    int64_t fuse_space = std::max(x_space, y_space);
    int peak_live = PeakLiveOps(x, y);
    if (static_cast<uint64_t>(peak_live * fuse_space) > g_system.CoreNum() * g_system.LocalMemSize()) {
      constexpr int64_t kLiveRatio = 3;
      int base_peak = std::max(x->peak_live, y->peak_live);
      score -= static_cast<int64_t>(peak_live - base_peak) / kLiveRatio * fuse_space;
    }
    return std::make_pair(score, peak_live);
  }

  std::vector<Candidate> CollectCandidates() {
    auto pair_key = [](Area *x, Area *y) {
      uint64_t lo = x->id;
      uint64_t hi = y->id;
      if (lo > hi) {
        std::swap(lo, hi);
      }
      return lo << 32 | hi;
    };
    std::vector<Candidate> cands;
    std::unordered_map<uint64_t, int> pair_shared;
    std::vector<Area *> distinct;
    for (auto &entry : consumers_) {
      distinct.clear();
      for (auto op : entry.second) {
        Area *x = areas_[op->prop_id_];
        if (std::find(distinct.begin(), distinct.end(), x) == distinct.end()) {
          distinct.push_back(x);
        }
      }
      if (distinct.size() > 1) {
        for (size_t i = 0; i < distinct.size(); ++i) {
          for (size_t j = i + 1; j < distinct.size(); ++j) {
            pair_shared[pair_key(distinct[i], distinct[j])]++;
          }
        }
      }
    }
    for (auto cons : areas_) {
      if (cons->pattern == kPatNone) continue;
      for (auto &entry : cons->borders) {
        auto prod = areas_[entry.first];
        if (prod->pattern == kPatNone) continue;
        auto [pattern, phase_score] = VFuseCheck(prod, cons);
        if (pattern == kPatNone || CheckCycle(cons, prod)) continue;
        int shared = 0;
        if (auto it = pair_shared.find(pair_key(prod, cons)); it != pair_shared.end()) {
          shared = it->second;
        }
        if (auto [score, peak_live] = GetScore(prod, cons, shared); score > 0) {
          cands.push_back({prod, cons, score + phase_score, pattern, peak_live});
        }
      }
    }
    for (auto &[key, count] : pair_shared) {
      auto x = areas_[static_cast<int>(key >> 32)];
      auto y = areas_[static_cast<int>(key & 0xffffffff)];
      if (x->pattern == kPatNone || y->pattern == kPatNone || !HFuseCheck(x, y)) continue;
      if (auto [score, peak_live] = GetScore(x, y, count); score > 0) {
        cands.push_back({x, y, score, kPatHorizontal, peak_live});
      }
    }
    std::sort(cands.begin(), cands.end(), [](const Candidate &l, const Candidate &r) {
      return l.score >= r.score;
    });
    return cands;
  }

  void DoFuse(const Candidate &c) {
    Area *surv = c.a;
    Area *abs = c.b;
    surv->pattern |= abs->pattern | c.pattern;
    abs->pattern = kPatNone;
    if (surv->item_size < abs->item_size) {
      surv->item_size = abs->item_size;
    }
    if (c.pattern != kPatReducePost && c.pattern != kPatCubePost) {
      surv->shape = abs->shape;
    }
    for (auto op : abs->ops) {
      op->prop_id_ = surv->id;
    }
    surv->ops.insert(surv->ops.end(), abs->ops.begin(), abs->ops.end());
    surv->epoch = abs->epoch = ++fuse_epoch_;
    surv->peak_live = c.peak_live;
    for (auto &[id, prods] : abs->borders) {
      if (id != surv->id) {
        surv->AddBorders(id, prods);
      }
    }
    surv->borders.erase(abs->id);
    for (auto cons : areas_) {
      if (cons->pattern == kPatNone) continue;
      if (auto it = cons->borders.find(abs->id); it != cons->borders.end()) {
        cons->AddBorders(surv->id, it->second);
        cons->borders.erase(it);
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
          std::cout << "   inport: id=" << id << ", ops=[";
          size_t num = prods.size();
          for (auto op : prods) {
            std::cout << op->index_;
            if (--num) {
              std::cout << ", ";
            }
          }
          std::cout << "]" << std::endl;
        }
      }
    }
  }

  std::vector<Area *> areas_;
  std::unordered_map<NDObject *, NDAccess *> stores_;
  std::unordered_map<NDAccess *, std::vector<NDAccess *>> swap_loads_;
  std::unordered_map<NDObject *, std::vector<NDObject *>> consumers_;
  std::vector<NDObject *> temp_ops_;
  uint32_t fuse_epoch_{0};
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
