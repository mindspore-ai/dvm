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

#include <cstdlib>
#include <iostream>
#include <cstring>
#ifndef VK_SIM_MODEL
#include "acl/acl_rt.h"
#endif
#include "tuning.h"
#include "xkernel.h"
#include "ops.h"

namespace dvm {
#define ASCEND_CALL(func)                                                                               \
  do {                                                                                                  \
    auto err = (func);                                                                                  \
    if (err != 0) {                                                                                     \
      std::cerr << "Ascend error in function " << #func << " : " << static_cast<int>(err) << std::endl; \
      exit(0);                                                                                          \
    }                                                                                                   \
  } while (0)

constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t MAX_BIAS_SIZE = 1024;

class ManualCubeTuner : public CubeTuner {
 public:
  ManualCubeTuner(const TuningInfo &info) : CubeTuner(kUnknownTuner), info_(info) {}
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
#ifndef VK_SIM_MODEL
  auto &tuning_table = CubeTuner::CacheTable();
  TuningInfo &best_tuning = tuning_table[GenKey(op, code)];
  if (best_tuning.swizzle == 0) {
    void *dev_M_, *dev_N_, *dev_O_;
    ASCEND_CALL(aclrtMalloc(&dev_M_, op->lhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ASCEND_CALL(aclrtMalloc(&dev_N_, op->rhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ASCEND_CALL(aclrtMalloc(&dev_O_, op->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    kernel_ = new Kernel();
    kernel_->Reset(kStaticMix);
    auto m_input = kernel_->Load(dev_M_, op->lhs_->shape_ref_, DType::kFloat16);
    auto n_input = kernel_->Load(dev_N_, op->rhs_->shape_ref_, DType::kFloat16);
    auto matmul = new CubeOp(m_input, n_input, op->trans_a_, op->trans_b_);
    if (op->type_id_ == dvm::kFloat32) matmul->SetOutFp32(false);
    kernel_->GetImpl()->Append(matmul);
    matmul->SetRealShape(op->m_real_, op->n_real_, op->k_real_, 0, 0);
    (void)kernel_->Store(dev_O_, matmul);
    TileV3(op, code);
    best_tuning = best_tuning_;
    aclrtFree(dev_M_);
    aclrtFree(dev_N_);
    aclrtFree(dev_O_);
    delete kernel_;
  }
  code->m0 = op->m0_ = best_tuning.m0;
  code->n0 = op->n0_ = best_tuning.n0;
  code->k0 = op->k0_ = best_tuning.k0;
  code->swizzle = best_tuning.swizzle;
  op->core_loop_ = best_tuning.core_loop;
  op->block_dim_ = best_tuning.block_dim;
#endif
}

void OnlineCubeTuner::TileV3(CubeOp *mm, vCubeOp *op) {
  auto l0c_max = System::Instance().L0CSize() / FP32_SIZE;
  auto bias_size = mm->bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_max = (System::Instance().L1Size() / 2 - bias_size) / ITEM_SIZE[mm->lhs_->type_id_];
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
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
    uint32_t core_loop = m_loop * n_loop * std::max(op->batch_a0, op->batch_b0) * std::max(op->batch_a1, op->batch_b1);
    block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    for (uint32_t cnt = std::min(block_dim, m_loop); cnt >= 1; --cnt) {
      auto swizzle = cnt;
      Tuning({m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min(block_dim, n_loop); cnt >= 1; --cnt) {
      auto swizzle = 1u << 16 | cnt;
      Tuning({m0, n0, k0, swizzle, core_loop, block_dim});
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

void OnlineCubeTuner::Tuning(const TuningInfo &parameter) {
#ifndef VK_SIM_MODEL
  ManualCubeTuner tuner(parameter);
  static_cast<MixKernel *>(kernel_->GetImpl())->SetTuner(&tuner);
  kernel_->CodeGen();
  float min_us = 1e6;
  float max_us = 0.0f;
  float total_us = 0.0f;
  aclrtEvent start, end;
  ASCEND_CALL(aclrtCreateEvent(&start));
  ASCEND_CALL(aclrtCreateEvent(&end));
  uint32_t test_num = 10;
  for (uint32_t i = 0; i < test_num; i++) {
    ASCEND_CALL(aclrtRecordEvent(start, nullptr));
    ASCEND_CALL(kernel_->Launch(nullptr, nullptr));
    ASCEND_CALL(aclrtRecordEvent(end, nullptr));
    ASCEND_CALL(aclrtSynchronizeStream(nullptr));
    float time_us = 0.0f;
    ASCEND_CALL(aclrtEventElapsedTime(&time_us, start, end));
    time_us *= 1000.0;
    if (time_us < min_us) {
      min_us = time_us;
    }
    if (time_us > max_us) {
      max_us = time_us;
    }
    total_us += time_us;
  }
  ASCEND_CALL(aclrtDestroyEvent(start));
  ASCEND_CALL(aclrtDestroyEvent(end));
  auto mean_time = (total_us - min_us - max_us) / (test_num - 2);
  if (mean_time < best_time_) {
    best_time_ = mean_time;
    best_tuning_ = parameter;
  }
#endif
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
    int cur_idx = ctx->best_idx >= 0 ? ctx->best_idx : 0;
    if (ctx->gen_cnt == space_size * repeat) {
      if (ctx->gen_cnt != ctx->run_cnt) {
        info = current_space[cur_idx];
      } else if (ctx->tuning_stage == kSwizzleTuning) {
        auto it = tuning_table.emplace(key, *ctx->swizzle_space[cur_idx]);
        info = &it.first->second;
        context_.erase(key);
        delete ctx;
      } else { // ctx->tuning_stage == kSwizzleTuning
        BuildSwizzleSpace(code, ctx->tile_space[cur_idx], ctx->swizzle_space);
        ctx->tuning_stage = kSwizzleTuning;
        ctx->run_cnt = 0;
        ctx->best_idx = 0;
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
  code->unique_id = space_idx; // reuse
  code->m0 = op->m0_ = info->m0;
  code->n0 = op->n0_ = info->n0;
  code->k0 = op->k0_ = info->k0;
  code->swizzle = info->swizzle;
  op->core_loop_ = info->core_loop;
  op->block_dim_ = info->block_dim;
}

int LazyCubeTuner::Launch(CubeOp *op, Code &code, void *stream) {
#ifndef VK_SIM_MODEL
  auto cube_code = reinterpret_cast<vCubeOp *>(code.data_ + code.HeadSize());
  uint32_t space_idx = cube_code->unique_id;
  if (space_idx == (uint32_t)-1) {
    return code.Launch(nullptr, stream);
  }
  aclrtEvent start, end;
  auto err1 = aclrtCreateEvent(&start);
  auto err2 = aclrtCreateEvent(&end);
  if (err1 || err2) return -1;
  auto err3 = aclrtRecordEvent(start, stream);
  auto err4 = code.Launch(nullptr, stream);
  auto err5 = aclrtRecordEvent(end, stream);
  auto err6 = aclrtSynchronizeStream(stream);
  if (err3 || err4 || err5 || err6) return -1;
  float time = 0.0f;
  auto err7 = aclrtEventElapsedTime(&time, start, end);
  (void)aclrtDestroyEvent(start);
  (void)aclrtDestroyEvent(end);
  auto ctx = context_[GenKey(op, cube_code)];
  if (!err7 && (ctx->best_idx < 0 || time < ctx->best_time)) {
    ctx->best_idx = space_idx;
    ctx->best_time = time;
  }
  ctx->run_cnt++;
#endif
  return 0;
}

void LazyCubeTuner::BuildTileSpace(CubeOp *op, vCubeOp *code, std::vector<TuningInfo *> &space) {
  auto l0c_max = System::Instance().L0CSize() / FP32_SIZE;
  auto bias_size = op->bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_max = (System::Instance().L1Size() / 2 - bias_size) / ITEM_SIZE[op->lhs_->type_id_];
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
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
    uint32_t core_loop =
      m_loop * n_loop * std::max(code->batch_a0, code->batch_b0) * std::max(code->batch_a1, code->batch_b1);
    block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    uint32_t cnt = 7;
    space.push_back(new TuningInfo(m0, n0, k0, cnt, core_loop, block_dim));
    space.push_back(new TuningInfo(m0, n0, k0, 1u << 16 | cnt, core_loop, block_dim));
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
