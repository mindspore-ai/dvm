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

#ifndef _DVM_TUNING_H_
#define _DVM_TUNING_H_
#include <map>
#include <mutex>
#include <condition_variable>
#include "dvm.h"
#include "ops.h"

namespace dvm {
struct TuningInfo {
  TuningInfo() = default;
  TuningInfo(int64_t m0_, int64_t n0_, int64_t k0_, uint32_t swizzle_, uint32_t core_loop_, uint32_t block_dim_)
   : m0(m0_), n0(n0_), k0(k0_), swizzle(swizzle_), core_loop(core_loop_), block_dim(block_dim_) {}
  int64_t m0{0};
  int64_t n0{0};
  int64_t k0{0};
  uint32_t swizzle{0};
  uint32_t core_loop{0};
  uint32_t block_dim{0};
};

enum TunerType { kOnlineTuner = 0, kLazyTuner, kUnknownTuner };
class CubeTuner {
 public:
  using Key = std::pair<uint64_t, uint64_t>;

  CubeTuner(TunerType type) : type_(type) {}
  virtual ~CubeTuner();
  virtual void GenTile(CubeOp *op, vCubeOp *code) = 0;
  TunerType Type() const { return type_; }

  static std::map<Key, TuningInfo> &CacheTable() {
    static std::map<Key, TuningInfo> table;
    return table;
  }

  Key GenKey(CubeOp *op, vCubeOp *code) {
    uint64_t key_batch = (uint64_t)code->batch_a0 << 48 | (uint64_t)code->batch_a1 << 32 | code->batch_b0 << 16 | code->batch_b1;
    uint64_t key_shape = op->m_real_ << 44 | op->n_real_ << 24 | op->k_real_ << 2;
    if (op->type_id_ == dvm::kFloat32) key_shape |= 4ul;
    if (op->trans_a_) key_shape |= 2ul;
    if (op->trans_b_) key_shape |= 1ul;
    return std::make_pair(key_batch, key_shape);
  }

 protected:
  TunerType type_;
};

class OnlineCubeTuner : public CubeTuner {
 public:
  OnlineCubeTuner() : CubeTuner(kOnlineTuner) {}
  void GenTile(CubeOp *op, vCubeOp *code) override;

 protected:
  struct TuneData {
    float best_time{1e6};
    TuningInfo best_para;
    Kernel kernel;
  };
  void TileV3(TuneData &td, CubeOp *mm, vCubeOp *op);
  void Tuning(TuneData &td, const TuningInfo &parameter);
};

class LazyCubeTuner : public CubeTuner {
 public:
  LazyCubeTuner() : CubeTuner(kLazyTuner) {}
  ~LazyCubeTuner() override;

  void GenTile(CubeOp *op, vCubeOp *code) override;
  int Launch(CubeOp *op, Code &code, void *stream);

 protected:
  enum TuningStage { kTileTuning = 0, kSwizzleTuning };
  struct Context {
    ~Context() {
      for (auto info : tile_space) {
        if (info) delete info;
      }
      for (auto info : swizzle_space) {
        if (info) delete info;
      }
    }
    std::vector<TuningInfo *> tile_space;
    std::vector<TuningInfo *> swizzle_space;
    float best_time;
    int best_idx{-1};
    int next_idx{0};
    int gen_cnt{0};
    int run_cnt{0};
    std::mutex mutex_;
    std::condition_variable cond_var_;
    TuningStage tuning_stage{kTileTuning};
  };

  void BuildTileSpace(CubeOp *op, vCubeOp *code, std::vector<TuningInfo *> &space);
  void BuildSwizzleSpace(vCubeOp *code, TuningInfo *best_tile, std::vector<TuningInfo *> &space);
  std::map<Key, Context *> context_;
};
}  // namespace dvm
#endif  // _DVM_TUNING_H_
