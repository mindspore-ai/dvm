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

#ifndef _DVM_OPS_M_H_
#define _DVM_OPS_M_H_

#include <map>
#include <mutex>
#include <condition_variable>
#include "ops.h"
#include "ops_m.h"

namespace dvm {

class CubeTuner;
class CubeOp : public NDObject {
 public:
  static constexpr uint32_t BLOCK_SIZE = 16;
  static constexpr uint32_t CUBE_BLOCK_SIZE = 256;
  static constexpr uint32_t AXES_ALIGN_SIZE = 512;
  static constexpr uint32_t CONST_512 = 512;
  static constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 7;
  static constexpr uint32_t DEFAULT_DIAGONAL_SWIZZLE_COUNT = 8;
  static constexpr int64_t MAX_SPLIT_K = 20480;
  static constexpr int64_t MIN_SPLIT_K = 4096;
  static constexpr int64_t ALIGN_256 = 256;
  static constexpr int64_t ALIGN_128 = 128;
  static constexpr int64_t ALIGN_32 = 32;
  struct Tactics {
    bool enable_splitk;
    bool enable_pad;
    bool enable_bias_cast;

    int64_t k_stride;
    int64_t lhs_pad_size;
    int64_t rhs_pad_size;
  };

  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b);
  CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias);

  uint64_t Emit(VectorKernel &k) override { return 0; }
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void NormalizeCube();
  virtual void InferTactics(Tactics &t) const;
  virtual void CodeGen(vCubeOp *code, CubeTuner *tuner);
  virtual void NormalizeOutput();
  virtual void GenTiling(vCubeOp *code);

  uint64_t PostFusionWorkSpace() const {
    uint64_t pingpong_size = m0_ * n0_ * ITEM_SIZE[type_id_];
    return core_loop_ < block_dim_ * 2 ? pingpong_size * core_loop_ : pingpong_size * block_dim_ * 2;
  }
  void SetRealShape(int64_t m, int64_t n, int64_t k, size_t offset_a, size_t offset_b) {
    set_real_ = true;
    m_real_ = m;
    n_real_ = n;
    k_real_ = k;
    offset_a_ = offset_a;
    offset_b_ = offset_b;
  }
  void SetOutFp32(bool atomic_add) {
    atomic_add_ = atomic_add;
    type_id_ = kFloat32;
  }
  void TryBatchFold() { batch_fold_ = !trans_a_ && lhs_->nd_.size() > 2 && rhs_->nd_.size() == 2; }
  uint64_t BaseSize() const { return m0_ * n0_ * ITEM_SIZE[type_id_]; }

  void Recover() {
    if (batch_fold_) {
      ASSERT(lhs_->nd_.size() > 2);
      uint64_t batch_size = lhs_->nd_[2];
      if (lhs_->nd_.size() > 3) batch_size *= lhs_->nd_[3];
      m_real_ /= batch_size;
      m_align_ /= batch_size;
    }
  }
  void Clear() {
    atomic_add_ = false;
    batch_fold_ = false;
    set_real_ = false;
    type_id_ = lhs_->type_id_;
    offset_a_ = offset_b_ = 0;
  }

  NDAccess *output_{nullptr};
  uint64_t block_dim_{0};
  uint64_t core_loop_{0};
  int64_t m_align_{0};
  int64_t n_align_{0};
  int64_t k_align_{0};
  int64_t ka_align_{0};
  int64_t kb_align_{0};
  int64_t m_real_{0};
  int64_t n_real_{0};
  int64_t k_real_{0};
  int64_t m0_{0};
  int64_t n0_{0};
  int64_t k0_{0};
  bool trans_a_{false};
  bool trans_b_{false};
  bool pingpong_store_{false};
  bool atomic_add_{false};
  bool batch_fold_{false};
  bool set_real_{false};
  NDObject *bias_{nullptr};
  uint32_t batch_c0_{0};
  uint32_t batch_c1_{0};
  NDSpaceData ndd_;

 protected:
  float CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0);
  void Tile(vCubeOp *code);
  void TileV1(vCubeOp *op);
  void TileV2(vCubeOp *op, uint32_t swizzle_type);

  size_t offset_a_{0};
  size_t offset_b_{0};
  ShapeWithRef shape_;
};

class GmmOp : public CubeOp {
 public:
  GmmOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias, NDObject *group_list,
        GmmSplitType group_type, GmmListType group_list_type);

  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void InferTactics(Tactics &t) const override;
  void NormalizeOutput() override;
  void CodeGen(vCubeOp *code, CubeTuner *tuner) override;
  void GenTiling(vCubeOp *code) override;

  NDObject *group_list_;
  GmmSplitType group_type_;
  GmmListType group_list_type_;
};

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

  explicit CubeTuner(TunerType type) : type_(type) {}
  virtual ~CubeTuner();
  virtual void GenTile(CubeOp *op, vCubeOp *code) = 0;
  TunerType Type() const { return type_; }

  static std::map<Key, TuningInfo> &CacheTable() {
    static std::map<Key, TuningInfo> table;
    return table;
  }

  Key GenKey(CubeOp *op, vCubeOp *code) {
    uint64_t key_batch = (uint64_t)op->batch_c0_ << 32 | (uint64_t)op->batch_c1_ << V_CUBE_BCAST_C1_OFFSET | (uint64_t)code->batch_cast;
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
#endif  // _DVM_OPS_M_H_
