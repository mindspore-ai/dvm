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

#include <cstdlib>
#include <iostream>
#include "acl/acl_rt.h"
#include "tuning.h"
#include "msprof.h"
#include "xkernel.h"
#include "ops.h"

namespace dvm {
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t MAX_BIAS_SIZE = 1024;

class ManualCubeTuner : public CubeTuner {
 public:
  explicit ManualCubeTuner(const TuningInfo &info) : CubeTuner(kUnknownTuner), info_(info) {}
  void GenTile(CubeOp *op, vCubeOp *code) override {
    op->m0_ = code->m0 = info_.m0;
    op->n0_ = code->n0 = info_.n0;
    op->k0_ = code->k0 = info_.k0;
    code->swizzle = info_.swizzle;
    op->core_loop_ = info_.core_loop;
    op->block_dim_ = info_.block_dim;
  }

 private:
  const TuningInfo &info_;
};

CubeTuner::~CubeTuner() {}

void OnlineCubeTuner::GenTile(CubeOp *op, vCubeOp *code) {
  auto &tuning_table = CubeTuner::CacheTable();
  TuningInfo &best_tuning = tuning_table[GenKey(op, code)];
  if (best_tuning.swizzle == 0) {
    void *dev_M_, *dev_N_, *dev_O_;
    ERROR_CHECK(aclrtMalloc(&dev_M_, op->lhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ERROR_CHECK(aclrtMalloc(&dev_N_, op->rhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ERROR_CHECK(aclrtMalloc(&dev_O_, op->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    TuneData td;
    td.kernel.Reset(KernelType::kCube, 0);
    auto m_input = td.kernel.Load(dev_M_, op->lhs_->shape_ref_, DataType::kFloat16);
    auto n_input = td.kernel.Load(dev_N_, op->rhs_->shape_ref_, DataType::kFloat16);
    auto matmul = new CubeOp(m_input, n_input, op->trans_a_, op->trans_b_);
    if (op->type_id_ == dvm::kFloat32) matmul->SetOutFp32(false);
    td.kernel.GetImpl()->Append(matmul);
    matmul->SetRealShape(op->nd_[1], op->nd_[0], op->k_real_, 0, 0);  // in batch fold m_real != op->nd_[1]
    (void)td.kernel.Store(dev_O_, matmul);
    TileV3(td, op, code);
    best_tuning = td.best_para;
    aclrtFree(dev_M_);
    aclrtFree(dev_N_);
    aclrtFree(dev_O_);
  }
  code->m0 = op->m0_ = best_tuning.m0;
  code->n0 = op->n0_ = best_tuning.n0;
  code->k0 = op->k0_ = best_tuning.k0;
  code->swizzle = best_tuning.swizzle;
  op->core_loop_ = best_tuning.core_loop;
  op->block_dim_ = best_tuning.block_dim;
}

void OnlineCubeTuner::TileV3(TuneData &td, CubeOp *mm, vCubeOp *op) {
  auto l0c_max = g_system.L0CSize() / FP32_SIZE;
  auto bias_size = mm->bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_max = (g_system.L1Size() / 2 - bias_size) / ITEM_SIZE[mm->lhs_->type_id_];
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  uint32_t round_m = RoundUp<uint32_t>(mm->m_align_, BLOCK_SIZE);
  uint32_t round_n = RoundUp<uint32_t>(mm->n_align_, BLOCK_SIZE);
  uint32_t round_k = RoundUp<uint32_t>(mm->k_align_, BLOCK_SIZE);
  uint32_t block_dim = 0;
  auto tile_select = [&](uint32_t x, uint32_t y) {
    // 1. get m0, n0, k0
    uint32_t m0, n0, k0;
    if (!mm->trans_a_) {
      k0 = x;
      n0 = y;
      if (k0 > round_k || n0 > round_n) return;
      uint64_t mx = std::min(l0c_max / n0, (l1_max - k0 * n0) / k0);
      m0 = RoundDown<uint32_t>(mx, mx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      ASSERT((k0 * n0 < l1_max) && (m0 > 0));
      if (m0 > round_m) m0 = round_m;
    } else if (!mm->trans_b_) {  // trans_a && !trans_b_
      m0 = x;
      n0 = y;
      if (m0 > round_m || n0 > round_n) return;
      uint64_t kx = l1_max / (m0 + n0);
      k0 = RoundDown<uint32_t>(kx, kx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      if (m0 * n0 > l0c_max || k0 == 0) return;
      if (k0 > round_k) k0 = round_k;
    } else {  // trans_a && trans_b_
      k0 = x;
      m0 = y;
      if (k0 > round_k || m0 > round_m) return;
      uint64_t nx = std::min(l0c_max / m0, (l1_max - k0 * m0) / k0);
      n0 = RoundDown<uint32_t>(nx, nx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      ASSERT((k0 * m0 < l1_max) && (n0 > 0));
      if (n0 > round_n) n0 = round_n;
    }
    // 2. get core_loop, block_dim
    uint32_t m_loop = CeilDiv(op->m_real, m0);
    uint32_t n_loop = CeilDiv(op->n_real, n0);
    uint32_t core_loop = m_loop * n_loop * mm->batch_c0_ * mm->batch_c1_;
    block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    for (uint32_t cnt = std::min(block_dim, m_loop); cnt >= 1; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_nZ, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min(block_dim, n_loop); cnt >= 1; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_zN, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min({n_loop, m_loop, block_dim}); cnt >= V_CUBE_DIAGONAL_MIN_DIM; --cnt) {
      auto swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_N, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
      swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_Z, cnt);
      Tuning(td, {m0, n0, k0, swizzle, core_loop, block_dim});
    }
  };
  uint32_t align_max = 512 / ITEM_SIZE[mm->lhs_->type_id_];
  for (uint32_t x = align_max; x >= BLOCK_SIZE; x >>= 1) {
    for (uint32_t y = align_max; y >= x; y >>= 1) {
      tile_select(x, y);
      if (x != y) {
        tile_select(y, x);
      }
      if (block_dim == core_num) {
        return;
      }
    }
  }
}

void OnlineCubeTuner::Tuning(TuneData &td, const TuningInfo &parameter) {
  ManualCubeTuner tuner(parameter);
  static_cast<MixKernel *>(td.kernel.GetImpl())->SetTuner(&tuner);
  td.kernel.CodeGen();
  RepeatProfiler profiler;
  profiler.Reset();
  uint32_t test_num = 10;
  for (uint32_t i = 0; i < test_num; i++) {
    profiler.RecordStart(nullptr);
    ERROR_CHECK(td.kernel.Launch(nullptr, 0, nullptr, nullptr));
    profiler.RecordEnd(nullptr);
  }
  auto mean_time = (profiler.total_us_ - profiler.min_us_ - profiler.max_us_) / (test_num - 2);
  if (mean_time < td.best_time) {
    td.best_time = mean_time;
    td.best_para = parameter;
  }
}

LazyCubeTuner::~LazyCubeTuner() {
  for (auto it = context_.begin(); it != context_.end(); ++it) {
    delete it->second;
  }
}

void LazyCubeTuner::GenTile(CubeOp *op, vCubeOp *code) {
  auto &tuning_table = CubeTuner::CacheTable();
  auto key = GenKey(op, code);
  TuningInfo *info;
  uint32_t space_idx = (uint32_t)-1;
  auto it = tuning_table.find(key);
  if (it != tuning_table.end()) {
    info = &it->second;
  } else {
    Context *ctx;
    auto ctx_it = context_.find(key);
    if (ctx_it == context_.end()) {
      ctx = new Context();
      BuildTileSpace(op, code, ctx->tile_space);
      context_[key] = ctx;
    } else {
      ctx = ctx_it->second;
    }
    auto &current_space = ctx->tuning_stage == kTileTuning ? ctx->tile_space : ctx->swizzle_space;
    int space_size = current_space.size();
    constexpr int repeat = 2;
    if (ctx->gen_cnt == space_size * repeat) {
      if (ctx->gen_cnt != ctx->run_cnt) {
        // means some tiling configure in the current generated tiling space have not been launched,
        // wait launch completed
        std::unique_lock<std::mutex> lock(ctx->mutex_);
        ctx->cond_var_.wait(lock, [ctx] { return ctx->gen_cnt == ctx->run_cnt; });
      }
      int cur_idx = ctx->best_idx >= 0 ? (ctx->best_idx % space_size) : 0;
      if (ctx->tuning_stage == kSwizzleTuning) {  // tuning finished
        auto it = tuning_table.emplace(key, *ctx->swizzle_space[cur_idx]);
        info = &it.first->second;
        context_.erase(key);
        delete ctx;
      } else {  // tile_space finished, switch to swizzle_space
        BuildSwizzleSpace(code, ctx->tile_space[cur_idx], ctx->swizzle_space);
        ctx->tuning_stage = kSwizzleTuning;
        ctx->run_cnt = 0;
        ctx->best_idx = -1;
        ctx->next_idx = 1;
        ctx->gen_cnt = 1;
        space_idx = 0;
        info = ctx->swizzle_space[space_idx];
      }
    } else {
      space_idx = ctx->next_idx;
      info = current_space[space_idx];
      ctx->next_idx = (space_idx + 1) % space_size;
      ctx->gen_cnt++;
    }
  }
  code->unique_id = space_idx;  // reuse
  code->m0 = op->m0_ = info->m0;
  code->n0 = op->n0_ = info->n0;
  code->k0 = op->k0_ = info->k0;
  code->swizzle = info->swizzle;
  op->core_loop_ = info->core_loop;
  op->block_dim_ = info->block_dim;
}

int LazyCubeTuner::Launch(CubeOp *op, Code &code, void *stream) {
  auto cube_code = reinterpret_cast<vCubeOp *>(code.data_ + code.HeadSize());
  uint32_t space_idx = cube_code->unique_id;
  if (space_idx == (uint32_t)-1) {
    return code.Launch(nullptr, stream);
  }
  TimeProfiler profiler;
  profiler.RecordStart(stream);
  auto err = code.Launch(nullptr, stream);
  float time = profiler.RecordEnd(stream);
  auto ctx = context_[GenKey(op, cube_code)];
  if (!err && (ctx->best_idx < 0 || time < ctx->best_time)) {
    ctx->best_idx = ctx->run_cnt;
    ctx->best_time = time;
  }
  ctx->run_cnt++;
  if (ctx->run_cnt == ctx->gen_cnt) {
    std::unique_lock<std::mutex> lock(ctx->mutex_);
    ctx->cond_var_.notify_all();
  }
  return 0;
}

void LazyCubeTuner::BuildTileSpace(CubeOp *op, vCubeOp *code, std::vector<TuningInfo *> &space) {
  auto l0c_max = g_system.L0CSize() / FP32_SIZE;
  auto bias_size = op->bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_max = (g_system.L1Size() / 2 - bias_size) / ITEM_SIZE[op->lhs_->type_id_];
  auto core_num = g_system.CoreNum(CoreType::kAIC);
  uint32_t round_m = RoundUp<uint32_t>(op->m_align_, BLOCK_SIZE);
  uint32_t round_n = RoundUp<uint32_t>(op->n_align_, BLOCK_SIZE);
  uint32_t round_k = RoundUp<uint32_t>(op->k_align_, BLOCK_SIZE);
  uint32_t block_dim = 0;
  auto tile_select = [&](uint32_t m0, uint32_t n0) {
    if (m0 > round_m || n0 > round_n) return;
    uint32_t kx = l1_max / (m0 + n0);
    uint32_t k0 = RoundDown<uint32_t>(kx, kx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
    if (k0 == 0) return;
    if (k0 > round_k) k0 = round_k;
    // 2. get core_loop, block_dim
    uint32_t m_loop = CeilDiv(code->m_real, m0);
    uint32_t n_loop = CeilDiv(code->n_real, n0);
    uint32_t core_loop = m_loop * n_loop * op->batch_c0_ * op->batch_c1_;
    block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    uint32_t cnt = 7;
    space.push_back(
      new TuningInfo(m0, n0, k0, vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_nZ, cnt), core_loop, block_dim));
    space.push_back(
      new TuningInfo(m0, n0, k0, vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_zN, cnt), core_loop, block_dim));
  };
  uint32_t align_max = 512 / ITEM_SIZE[op->lhs_->type_id_];
  for (uint32_t x = align_max; x >= BLOCK_SIZE; x >>= 1) {
    for (uint32_t y = align_max; y >= x; y >>= 1) {
      if (x * y > l0c_max) continue;
      tile_select(x, y);
      if (x != y) {
        tile_select(y, x);
      }
      if (block_dim == core_num) {
        return;
      }
    }
  }
}

void LazyCubeTuner::BuildSwizzleSpace(vCubeOp *code, TuningInfo *best_tile, std::vector<TuningInfo *> &space) {
  uint32_t m_loop = CeilDiv(code->m_real, (uint32_t)best_tile->m0);
  uint32_t n_loop = CeilDiv(code->n_real, (uint32_t)best_tile->n0);
  for (uint32_t cnt = std::min(best_tile->block_dim, m_loop); cnt >= 1; --cnt) {
    space.push_back(
      new TuningInfo(best_tile->m0, best_tile->n0, best_tile->k0, cnt, best_tile->core_loop, best_tile->block_dim));
  }
  for (uint32_t cnt = std::min(best_tile->block_dim, n_loop); cnt >= 1; --cnt) {
    space.push_back(new TuningInfo(best_tile->m0, best_tile->n0, best_tile->k0, 1u << 16 | cnt, best_tile->core_loop,
                                   best_tile->block_dim));
  }
}
}  // namespace dvm
