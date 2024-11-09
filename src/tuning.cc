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
#include "acl/acl_rt.h"
#include "tuning.h"
#include "kernel.h"
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

inline __attribute__((always_inline)) uint32_t RoundUp(uint32_t num, uint32_t rnd) {
  if (rnd == 0) {
    return 0;
  }
  return (num + rnd - 1) / rnd * rnd;
}

inline __attribute__((always_inline)) uint32_t RoundDown(uint32_t num, uint32_t rnd) {
  if (rnd == 0) {
    return 0;
  }
  return num / rnd * rnd;
}
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;

void ManualMatMul::SetTiling(const TuningInfo &info) {
  m0_ = info.m0;
  n0_ = info.n0;
  k0_ = info.k0;
  swizzle_ = info.swizzle;
  core_loop_ = info.core_loop;
  block_dim_ = info.block_dim;
}

void TunedMatMul::GenTiling(vCubeOp *op) {
  auto &tuning_table = TunedMatMul::GetTuningTable();
  uint64_t key_batch = (uint64_t)op->batch_a0 << 48 | (uint64_t)op->batch_a1 << 32 | op->batch_b0 << 16 |op->batch_b1;
  uint64_t key_shape = m_real_ << 44 | n_real_ << 24 | k_real_ << 2;
  if (type_id_ == dvm::kFloat32) key_shape |= 4ul;
  if (trans_a_) key_shape |= 2ul;
  if (trans_b_) key_shape |= 1ul;
  TuningInfo &best_tuning = tuning_table[{key_batch, key_shape}];
  if (best_tuning.swizzle == 0) {
    void *dev_M_, *dev_N_, *dev_O_;
    ASCEND_CALL(aclrtMalloc(&dev_M_, lhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ASCEND_CALL(aclrtMalloc(&dev_N_, rhs_->Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ASCEND_CALL(aclrtMalloc(&dev_O_, Size() + 512, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    kernel_ = new Kernel();
    kernel_->Reset(kStaticMix);
    auto m_input = kernel_->Load(dev_M_, lhs_->shape_ref_, DType::kFloat16);
    auto n_input = kernel_->Load(dev_N_, rhs_->shape_ref_, DType::kFloat16);
    matmul_ = new ManualMatMul(m_input, n_input, trans_a_, trans_b_);
    if (type_id_ == dvm::kFloat32) matmul_->SetOutFp32(false);
    kernel_->GetImpl()->Append(matmul_);
    matmul_->SetRealShape(m_real_, n_real_, k_real_, 0, 0);
    (void)kernel_->Store(dev_O_, matmul_);
    TileV3(op);
    best_tuning = best_tuning_;
    aclrtFree(dev_M_);
    aclrtFree(dev_N_);
    aclrtFree(dev_O_);
    delete kernel_;
  }
  op->m0 = m0_ = best_tuning.m0;
  op->n0 = n0_ = best_tuning.n0;
  op->k0 = k0_ = best_tuning.k0;
  op->swizzle = best_tuning.swizzle;
  core_loop_ = best_tuning.core_loop;
  block_dim_ = best_tuning.block_dim;
}

void TunedMatMul::TileV3(vCubeOp *op) {
  auto l0c_max = System::Instance().L0CSize() / FP32_SIZE;
  auto l1_max = System::Instance().L1Size() / 2 / ITEM_SIZE[lhs_->type_id_];
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
  uint32_t round_m = RoundUp(m_align_, BLOCK_SIZE);
  uint32_t round_n = RoundUp(n_align_, BLOCK_SIZE);
  uint32_t round_k = RoundUp(k_align_, BLOCK_SIZE);
  auto tile_select = [&](uint32_t x, uint32_t y) {
    // 1. get m0, n0, k0
    uint32_t m0, n0, k0;
    if (!trans_a_) {
      k0 = x;
      n0 = y;
      if (k0 > round_k || n0 > round_n) return;
      uint64_t mx = std::min(l0c_max / n0, (l1_max - k0 * n0) / k0);
      m0 = RoundDown(mx, mx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      ASSERT((k0 * n0 < l1_max) && (m0 > 0));
      if (m0 > round_m) m0 = round_m;
    } else if (!trans_b_) {  // trans_a && !trans_b_
      m0 = x;
      n0 = y;
      if (m0 > round_m || n0 > round_n) return;
      uint64_t kx = l1_max / (m0 + n0);
      k0 = RoundDown(kx, kx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      if (m0 * n0 > l0c_max || k0 == 0) return;
      if (k0 > round_k) k0 = round_k;
    } else {  // trans_a && trans_b_
      k0 = x;
      m0 = y;
      if (k0 > round_k || m0 > round_m) return;
      uint64_t nx = std::min(l0c_max / m0, (l1_max - k0 * m0) / k0);
      n0 = RoundDown(nx, nx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      ASSERT((k0 * m0 < l1_max) && (n0 > 0));
      if (n0 > round_n) n0 = round_n;
    }
    // 2. get core_loop, block_dim
    uint32_t m_loop = CeilDiv(op->m_real, m0);
    uint32_t n_loop = CeilDiv(op->n_real, n0);
    uint32_t core_loop = m_loop * n_loop * std::max(op->batch_a0, op->batch_b0) * std::max(op->batch_a1, op->batch_b1);
    uint32_t block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    for (uint32_t cnt = std::min(block_dim, m_loop); cnt >= 1; --cnt) {
      auto swizzle = cnt;
      Tuning({m0, n0, k0, swizzle, core_loop, block_dim});
    }
    for (uint32_t cnt = std::min(block_dim, n_loop); cnt >= 1; --cnt) {
      auto swizzle = 1u << 16 | cnt;
      Tuning({m0, n0, k0, swizzle, core_loop, block_dim});
    }
    block_dim_ = block_dim;
  };
  uint32_t align_max = 512 / ITEM_SIZE[lhs_->type_id_];
  for (uint32_t x = align_max; x >= BLOCK_SIZE; x >>= 1) {
    for (uint32_t y = align_max; y >= x; y >>= 1) {
      tile_select(x, y);
      if (x != y) {
        tile_select(y, x);
      }
      if (block_dim_ == core_num) {
        return;
      }
    }
  }
}

void TunedMatMul::Tuning(const TuningInfo &parameter) {
  matmul_->SetTiling(parameter);
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
}
}  // namespace dvm