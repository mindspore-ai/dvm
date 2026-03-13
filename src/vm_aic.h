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
#define _CCE_KERNEL_
#include "isa.h"

#define V_GMM_BLOCK_OFF_OFFSET 32
#define V_GMM_BLOCK_OFF_BIT 32
#define V_GMM_LAST_OFFSET 8
#define V_GMM_LAST_BIT 24
#define V_GMM_G_SIZE_OFFSET 0
#define V_GMM_G_SIZE_BITS 8

template <typename M, typename Derived>
class TileVisitorBase {
 public:
  static constexpr uint64_t INVALID_TILE_POS = uint64_t(-1);

  __force_inline__ __aicore__ TileVisitorBase(__gm__ vCubeOp *__restrict__ op, uint64_t blockidx, uint64_t blocknum,
                                              uint64_t tile_num, M &m) {
    swizzle_ = op->swizzle;
    gm_msg_ = reinterpret_cast<__gm__ vMixGroupMsg *>(op->gm_pos + sizeof(vMixGroupMsg) * blockidx);
    pos_last_ = pos_ = INVALID_TILE_POS;
    vblock_ = blockidx | blocknum << 48;
    if (m.flags & V_CUBE_FLAG_GROUPED_LIST) {
      group_info_ = op->group_list_size << V_GMM_G_SIZE_OFFSET;
      gm_group_list_ = reinterpret_cast<__gm__ int64_t *>(op->gm_group_list);
      tile_loop_ = blockidx;
      UpdateGmm<true>(m);
    } else {
      group_info_ = 0;
      tile_loop_ = tile_num << 32 | blockidx;
    }
    UpdateNext(m);
  }

  template <bool init>
  __force_inline__ __aicore__ void UpdateGmm(M &m) {
    if constexpr (!init) {
      if (m.flags & V_CUBE_FLAG_GROUP_K) {
        m.gm_a += (m.flags & V_CUBE_FLAG_TRANS_A) ? m.k_real * m.m_align : m.k_real;
        m.gm_b += (m.flags & V_CUBE_FLAG_TRANS_B) ? m.k_real : m.k_real * m.n_align;
        m.gm_bias += m.n_real * m.BiasTypeSize();
      } else {
        m.gm_a += (m.flags & V_CUBE_FLAG_TRANS_A) ? m.m_real : m.m_real * m.ka_align;
        m.gm_b += m.kb_align * m.n_align;
        m.gm_bias += m.n_real * m.BiasTypeSize();
      }
      if (!(m.flags & V_CUBE_FLAG_PINGPONG_STORE)) {
        m.gm_c += m.m_real * m.n_real * (m.flags & V_CUBE_FLAG_OUT_FP32 ? sizeof(float) : sizeof(uint16_t));
      }
      group_info_ += (m.m_loop * m.n_loop) << V_GMM_BLOCK_OFF_OFFSET;
    }
    int64_t stride = *gm_group_list_ - LastList();
    if (!(m.flags & V_CUBE_FLAG_GMM_SIZE_MODE)) {
      group_info_ += stride << V_GMM_LAST_OFFSET;
    }
    group_info_--;
    if (m.flags & V_CUBE_FLAG_GROUP_K) {
      m.k_real = static_cast<uint64_t>(stride);
      m.k_loop = CeilDiv(m.k_real, m.k0);
      m.shuffle_k = vBlockIdx() % m.k_loop;
      m.k_actual = (m.shuffle_k == m.k_loop - 1) ? (m.k_real - m.shuffle_k * m.k0) : m.k0;
    } else {
      m.m_real = static_cast<uint64_t>(stride);
      m.m_loop = CeilDiv(m.m_real, m.m0);
    }
    gm_group_list_++;
    pos_last_ = pos_ = INVALID_TILE_POS;
    tile_loop_ += (m.m_loop * m.n_loop) << 32;
  }

  __force_inline__ __aicore__ void PipelineSync(const M &m) {}
  __force_inline__ __aicore__ void VectorSync(M &m) {}
  __force_inline__ __aicore__ bool Forward(M &m) {
    if ((m.flags & V_CUBE_FLAG_GROUP_SET) && pos_ != INVALID_TILE_POS) {
      static_cast<Derived *>(this)->VectorSync(m);
    }
    if (unlikely(m.flags & V_CUBE_FLAG_PIPELINE)) {
      static_cast<Derived *>(this)->PipelineSync(m);
    }
    while (GroupListSize() > 0 && !HasNext()) {
      UpdateGmm<false>(m);
      UpdateNext(m);
    }
    if (!HasNext()) {
      return false;
    }
    pos_last_ = pos_;
    pos_ = pos_next_;
    tile_loop_ += vBlockNum();
    UpdateNext(m);
    return true;
  }

  __force_inline__ __aicore__ void UpdateNext(const M &m) {
    uint64_t tile_idx = vGetBitRange(tile_loop_, 0, 32);
    uint64_t tile_num = vGetBitRange(tile_loop_, 32, 32);
    if (tile_idx < tile_num) {
      tile_idx -= BlockIdxOffset();
      uint64_t midx, nidx, cidx;
      vCubeOp::IndexCompute(tile_idx, swizzle_, m.m_loop, m.n_loop, midx, nidx, cidx);
      pos_next_ = cidx << V_MM_POS_C_OFFSET | midx << V_MM_POS_M_OFFSET | nidx << V_MM_POS_N_OFFSET;
    } else {
      pos_next_ = INVALID_TILE_POS;
    }
  }

  __force_inline__ __aicore__ bool HasNext() const { return pos_next_ != INVALID_TILE_POS; }
  __force_inline__ __aicore__ bool HasLast() const { return pos_last_ != INVALID_TILE_POS; }
  __force_inline__ __aicore__ uint64_t CurC() const { return vGetBitRange(pos_, V_MM_POS_C_OFFSET, V_MM_POS_C_BITS); }
  __force_inline__ __aicore__ uint64_t CurM() const { return vGetBitRange(pos_, V_MM_POS_M_OFFSET, V_MM_POS_M_BITS); }
  __force_inline__ __aicore__ uint64_t CurN() const { return vGetBitRange(pos_, V_MM_POS_N_OFFSET, V_MM_POS_N_BITS); }
  __force_inline__ __aicore__ uint64_t NextC() const {
    return vGetBitRange(pos_next_, V_MM_POS_C_OFFSET, V_MM_POS_C_BITS);
  }
  __force_inline__ __aicore__ uint64_t NextM() const {
    return vGetBitRange(pos_next_, V_MM_POS_M_OFFSET, V_MM_POS_M_BITS);
  }
  __force_inline__ __aicore__ uint64_t NextN() const {
    return vGetBitRange(pos_next_, V_MM_POS_N_OFFSET, V_MM_POS_N_BITS);
  }
  __force_inline__ __aicore__ uint64_t LastC() const {
    return vGetBitRange(pos_last_, V_MM_POS_C_OFFSET, V_MM_POS_C_BITS);
  }
  __force_inline__ __aicore__ uint64_t LastM() const {
    return vGetBitRange(pos_last_, V_MM_POS_M_OFFSET, V_MM_POS_M_BITS);
  }
  __force_inline__ __aicore__ uint64_t LastN() const {
    return vGetBitRange(pos_last_, V_MM_POS_N_OFFSET, V_MM_POS_N_BITS);
  }
  __force_inline__ __aicore__ uint64_t GroupListSize() const {
    return vGetBitRange(group_info_, V_GMM_G_SIZE_OFFSET, V_GMM_G_SIZE_BITS);
  }
  __force_inline__ __aicore__ uint64_t BlockIdxOffset() const {
    return vGetBitRange(group_info_, V_GMM_BLOCK_OFF_OFFSET, V_GMM_BLOCK_OFF_BIT);
  }
  __force_inline__ __aicore__ uint64_t LastList() const {
    return vGetBitRange(group_info_, V_GMM_LAST_OFFSET, V_GMM_LAST_BIT);
  }
  __force_inline__ __aicore__ uint64_t vBlockIdx() const { return vblock_ & 0xfffful; }
  __force_inline__ __aicore__ uint64_t vBlockNum() const { return vblock_ >> 48; }

  uint64_t pos_;
  uint64_t pos_next_;
  uint64_t pos_last_;
  uint64_t swizzle_;
  uint64_t tile_loop_;
  __gm__ vMixGroupMsg *gm_msg_;
  __gm__ int64_t *__restrict__ gm_group_list_{nullptr};
  uint64_t group_info_;
  uint64_t vblock_;
};