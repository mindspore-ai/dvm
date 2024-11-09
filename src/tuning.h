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
#include <vector>
#include <map>
#include "dvm.h"
#include "ops.h"

namespace dvm {
struct TuningInfo {
  int64_t m0{0};
  int64_t n0{0};
  int64_t k0{0};
  uint32_t swizzle{0};
  uint32_t core_loop{0};
  uint32_t block_dim{0};
};

class ManualMatMul : public CubeOp {
 public:
  using CubeOp::CubeOp;
  void SetTiling(const TuningInfo &info);
  void GenTiling(vCubeOp *op) override {
    op->m0 = m0_;
    op->n0 = n0_;
    op->k0 = k0_;
    op->swizzle = swizzle_;
  };
  uint32_t swizzle_{0};
};

class TunedMatMul : public CubeOp {
 public:
  using CubeOp::CubeOp;
  void GenTiling(vCubeOp *op) override;
  void TileV3(vCubeOp *op);
  void Tuning(const TuningInfo &parameter);
  static std::map<std::pair<uint64_t, uint64_t>, TuningInfo> &GetTuningTable() {
    static std::map<std::pair<uint64_t, uint64_t>, TuningInfo> table;
    return table;
  }

 private:
  float best_time_{1e6};
  Kernel *kernel_{nullptr};
  ManualMatMul *matmul_{nullptr};
  TuningInfo best_tuning_;
};
}  // namespace dvm
#endif  // _DVM_TUNING_H_