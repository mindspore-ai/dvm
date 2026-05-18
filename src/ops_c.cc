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

#include "kernel.h"
#include "comm.h"
#include "ops_m.h"
#include "ops_c.h"

namespace dvm {
static void ReserveCommEvent(VectorKernel &k) {
  if (k.forward_event_num_ > 7) {
    k.forward_event_num_ = 7;
  }
  if (k.backward_event_num_ > 6) {
    k.backward_event_num_ = 6;
  }
}

void NDMultiLoad::Normalize(std::vector<NDObject *> &run_ops) {
  auto dims = shape_ref_->size;
  ndd_.dims.resize(dims);
  gap_ = ITEM_SIZE[type_id_];
  for (size_t i = 0; i < shape_ref_->size; i++) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
    gap_ *= ndd_.dims[i];
  }
  gap_ = gap_ / comm_->GetRankSize();
  ndd_.dims[shape_ref_->size - 1] /= comm_->GetRankSize();
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
}

uint64_t NDMultiLoad::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  int64_t lead_align = ndd_.lead_stride();
  uint64_t src_tile_stride_ = ndd_.stride_back() / lead_align * ndd_.lead_dim();
  vMultiLoad op;
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  op.from = addr_.gm;
  op.multi_size = rank_size;
  op.xbuf_size = xbuf_size_;
  op.gap = gap_;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
  op.body_iter = ndd_.stride_back() / lead_align;
  op.tail_iter = tail_dim_ <= ndd_.lead_idx() ? op.body_iter : op.body_iter / ndd_[tail_dim_] * tail_size_;
  op.iter_size = ndd_.lead_dim() * ITEM_SIZE[type_id_];
  op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
  op.round_rank = 0;
  op.peer_mem = comm_->GetPeerMemPtr(rank_id);
  op.flag_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_FLAG_OFFSET;
  op.rank_id = rank_id;
  op.tile_stride2 = ndd_.stride_back() * ITEM_SIZE[type_id_];

  addr_.Update(insn_ + vMultiLoad::RELOC_OFFSET);
  k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(insn_ + vMultiLoad::UNIQUEID_OFFSET));
  return vMultiLoad::Encode(insn_, vAccInsnID::V_MULTI_LOAD, op, nullptr);
}

NDObject *NDMultiLoad::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  return new NDMultiLoad(static_cast<uint8_t *>(addr_.gm), shape_ref, type_id_, comm_);
}

void NDMultiLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "MultiLoad"; }

std::atomic<uint32_t> CommIdWrap::unique_id_ = 0;
int CommIdWrap::LaunchWrap(void *workspace, void *stream) {
  uint32_t cur_id = ++unique_id_;
  for (auto id : ids_) {
    *id = cur_id;
  }
  return next_->LaunchWrap(workspace, stream);
}

ReduceScatterOp::ReduceScatterOp(NDObject *input, const Communicator *comm)
    : CommOp(input, comm, ObjectType::kReduceScatter) {
  add_id_ = GetBinaryInsnID(kAdd, type_id_);
  shape_ref_ = &shape_;
  multi_load_ = input->obj_id_ == kMultiLoad;
  if (multi_load_) store_lhs_ = false;
}

ReduceScatterOp::~ReduceScatterOp() {}

void ReduceScatterOp::Normalize(std::vector<NDObject *> &run_ops) {
  const IntArrayRef *input_shape_ref = lhs_->shape_ref_;
  size_t start_idx = input_shape_ref->data[0] != comm_->GetRankSize() ? 1 : 0;
  if (multi_load_) {
    start_idx = 0;
  }
  shape_.Resize(input_shape_ref->size + start_idx);
  for (size_t i = 0; i < input_shape_ref->size; ++i) {
    shape_[i + start_idx] = input_shape_ref->data[i];
  }
  if (start_idx == 1) {
    shape_[0] = comm_->GetRankSize();
    shape_[1] /= shape_[0];
  }
  ndd_.dims = lhs_->nd_.dims();
  if (multi_load_) {
    shape_[0] /= comm_->GetRankSize();
  } else {
    ASSERT(ndd_[ndd_.size() - 1] == comm_->GetRankSize());
    ndd_.dims[ndd_.size() - 1] = 1;
    shape_[0] = 1;
  }

  // init CommOp related member
  // suppose only 1 multiload is allowd, reserve xbuf for multiload here
  xbuf_reserve_ = multi_load_ ? 2 : 3;
  // Also Reserve for MultiLoad
  code_reserve_ = sizeof(uint64_t) * (7 * (comm_->GetRankSize() - 1) + 4 + 3);
}

void ReduceScatterOp::FoldProp(NDObject *op, PropRange &range) {
  auto self = static_cast<ReduceScatterOp *>(op);
  if (self->multi_load_) {
    return;
  }
  auto &lhs_nd = self->lhs_->nd_;
  auto &ndd = self->ndd_;
  BroadReduceFoldProp<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, range);
}

void ReduceScatterOp::TileCollect(NDObject *op, TileInfo &info) {
  auto self = static_cast<ReduceScatterOp *>(op);
  if (self->multi_load_) {
    return;
  }
  auto &lhs_nd = self->lhs_->nd_;
  auto &ndd = self->ndd_;
  BroadReduceTileCollect<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, info);
}

NDObject *ReduceScatterOp::Clone(CloneHelper &h) { return new ReduceScatterOp(h.GetClone(lhs_), comm_); }

void ReduceScatterOp::Dump(bool verbose, std::ostringstream &oss) { oss << "ReduceScatter"; }

void ReduceScatterOp::Tile(const TileParam &tp) {
  if (CollectRoundTile(ndd_.dims, tp, round_tile_) && tp.tail > 0) {
    ASSERT(tail_dim_ == -1);  // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

uint64_t ReduceScatterOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  if (multi_load_) {
    return MultiLoadEmit(k);
  }
  uint64_t rounds[2];
  ASSERT(!round_tile_.empty());
  BuildDimRounds(round_tile_, rounds);

  // TODO: support matmul post fusion ReduceScatter
  auto store_id = mix_ ? vAccInsnID::V_PEER_STORE_MIX : vAccInsnID::V_PEER_STORE;
  auto load_id = mix_ ? vAccInsnID::V_PEER_LOAD_MIX : vAccInsnID::V_PEER_LOAD;
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t forward_event = g_system.EventNum() - 1;
  uint64_t backward_event = g_system.EventNum() - 1;
  uint64_t backward_event2 = backward_event - 1;
  uint64_t code_size = 0;
  uint64_t *current_insn{nullptr};

  // nop
  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[2];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.lenburst = GetBlocks(tile_stride);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event2 << V_HEAD_B_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  vPeerDMA p_store;
  p_store.comm_type = CommType::kCommReduceScatter;
  p_store.peer_mem = comm_->GetPeerMemPtr(rank_id);
  p_store.flag_mem = p_store.peer_mem + PEERMEM_FLAG_OFFSET;
  p_store.xn = store_xbuf_;
  p_store.tile_stride = tile_stride_size;
  p_store.lenburst = GetBlocks(tile_stride);
  p_store.tail_lenburst = p_store.lenburst;
  p_store.round_rank = 0;
  current_insn = insn_ + code_size;
  code_size += vPeerDMA::Encode(current_insn, store_id, p_store, nullptr);
  k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event2 << V_M_HEAD_SET_EVENT_OFFSET;

  bool is_begin = true;
  bool is_ping = true;
  uint64_t rhs = 0;
  for (int i = 1; i < rank_size; ++i) {
    rhs = xbufs_[is_ping ? 0 : 1];
    // PeerLoad
    vPeerDMA p_load;
    p_load.comm_type = CommType::kCommReduceScatter;
    p_load.peer_mem = comm_->GetPeerMemPtr(i + rank_id);
    p_load.flag_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_FLAG_OFFSET;
    p_load.xn = rhs;
    p_load.tile_stride = tile_stride_size;
    p_load.lenburst = GetBlocks(tile_stride);
    p_load.tail_lenburst = tail_dim_ < 0 ? p_load.lenburst : GetBlocks(tile_stride / ndd_[tail_dim_] * tail_size_);
    p_load.round_rank = 2;
    p_load.rank_id = rank_id;
    p_load.event_id = backward_event2;
    if (!is_begin && rank_size - i > 1) {
      p_load.set_flag = true;
    }
    if (i > 2) {
      p_load.wait_flag = true;
    }
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, load_id, p_load, rounds);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

    vBinary add;
    add.xd = xbuf_;
    add.xn = is_begin ? lhs_->xbuf_ : xbuf_;
    add.xm = rhs;
    add.count = ndd_.stride_back();
    current_insn = insn_ + code_size;
    code_size += vBinary::Encode(current_insn, add_id_, add);
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
    is_begin = false;
    is_ping = !is_ping;
  }

  tail_insn_ = current_insn;
  return code_size;
}

uint64_t ReduceScatterOp::MultiLoadEmit(VectorKernel &k) {
  auto load_id = vAccInsnID::V_PEER_LOAD;
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t forward_event = g_system.EventNum() - 1;
  uint64_t backward_event = g_system.EventNum() - 1;
  uint64_t backward_event2 = backward_event - 1;
  int code_size = 0;
  uint64_t *current_insn{nullptr};

  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  bool is_begin = true;
  bool is_ping = true;
  uint64_t rhs = 0;
  for (int i = 1; i < rank_size; ++i) {
    rhs = xbufs_[is_ping ? 0 : 1];
    vPeerDMA p_load;
    p_load.comm_type = CommType::kCommReduceScatter;
    p_load.peer_mem = comm_->GetPeerMemPtr(rank_id + rank_size - i) + (i - 1) * tile_stride_size;
    p_load.flag_mem = comm_->GetPeerMemPtr(rank_id + rank_size - i) + PEERMEM_FLAG_OFFSET;
    p_load.xn = rhs;
    p_load.tile_stride = tile_stride_size * (rank_size - 1);
    p_load.lenburst = GetBlocks(tile_stride);
    p_load.tail_lenburst = tail_dim_ < 0 ? p_load.lenburst : GetBlocks(tile_stride / ndd_[tail_dim_] * tail_size_);
    p_load.rank_id = rank_id;
    p_load.event_id = backward_event2;
    if (!is_begin && rank_size - i > 1) {
      p_load.set_flag = true;
    }
    if (i > 2) {
      p_load.wait_flag = true;
    }
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, load_id, p_load, nullptr);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

    vBinary add;
    add.xd = xbuf_;
    add.xn = is_begin ? lhs_->xbuf_ : xbuf_;
    add.xm = rhs;
    add.count = ndd_.stride_back();
    current_insn = insn_ + code_size;
    code_size += vBinary::Encode(current_insn, add_id_, add);
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
    is_begin = false;
    is_ping = !is_ping;
  }

  tail_insn_ = current_insn;
  return code_size;
}

void ReduceScatterOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ReduceScatterOp *>(op);
  auto &shape = self->shape_;
  auto input_shape = op->lhs_->shape_ref_;
  auto rank_size = self->comm_->GetRankSize();
  shape.Resize(input_shape->size);
  for (size_t i = 0; i < input_shape->size; ++i) {
    if (i == 0) {
      auto dim = input_shape->data[i];
      if (dim < 0) {
        shape[i] = dim;
      } else {
        ASSERT(dim % rank_size == 0);
        shape[i] = dim / rank_size;
      }
    } else {
      shape[i] = input_shape->data[i];
    }
  }
}

AllReduceOpBase::AllReduceOpBase(int op_type, NDObject *input, const Communicator *comm)
    : CommOp(input, comm, ObjectType::kAllReduce) {
  if (op_type == ReduceType::kSum) {
    insn_id_ = GetBinaryInsnID(kAdd, type_id_ == kBFloat16 ? kFloat32 : type_id_);
  } else if (op_type == ReduceType::kMax) {
    insn_id_ = GetBinaryInsnID(kMaximum, type_id_ == kBFloat16 ? kFloat32 : type_id_);
  } else {
    ASSERT(0);
  }
  max_type_ = type_id_ == kBFloat16 ? kFloat32 : type_id_;
}

void AllReduceOpBase::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    // ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

void AllReduceOpBase::Dump(bool verbose, std::ostringstream &oss) { oss << "AllReduce"; }

void AllReduceOpBase::Normalize(std::vector<NDObject *> &run_ops) {
  ndd_.dims = lhs_->nd_.dims();
  if (ndd_.dims.prod() > 128 * static_cast<uint32_t>(comm_->GetRankSize())) {
    use_twoshot_ = true;
  }
  // init CommOp related member
  xbuf_reserve_ = cube_op_ == nullptr ? 3 : 2;
  // Rely on vBinary, vCopy, vNop, vPeerDMA(vPingpongPeerLoad)
  code_reserve_ = use_twoshot_ ? (sizeof(uint64_t) * ((2 * vPeerDMA::ROUND_OFFSET + 2) * (comm_->GetRankSize() - 1) +
                                                      2 * vPeerDMA::ROUND_OFFSET + 6))
                               : sizeof(uint64_t) * ((vPeerDMA::ROUND_OFFSET + 2) * comm_->GetRankSize() + 1);
}

template <bool is_bf16>
void AllReduceOp<is_bf16>::Normalize(std::vector<NDObject *> &run_ops) {
  AllReduceOpBase::Normalize(run_ops);
  if constexpr (is_bf16) {
    // cast to and from f32 in Emit()
    xbuf_reserve_ += 1;
    code_reserve_ += sizeof(uint64_t) * (comm_->GetRankSize() + 1) * 2;
  }
}

template <bool is_bf16>
uint64_t AllReduceOp<is_bf16>::MatmulEmit(VectorKernel &k) {
  uint64_t *current_insn = insn_;
  uint64_t code_size = 0;
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t lead_align = ndd_.lead_stride();
  uint64_t tile_stride = ndd_.stride_back() / lead_align * ndd_.lead_dim();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  uint64_t forward_event = g_system.EventNum() - 1;
  uint64_t backward_event = g_system.EventNum() - 1;
  uint64_t backward_event2 = backward_event - 1;

  // nop
  code_size += vNop::Encode(insn_);

  if (use_twoshot_) {
    // Prepare variables
    // Use the fact that lead_align is divisible by simd_width
    uint64_t nburst = ndd_.stride_back() / lead_align;
    uint64_t per_rank_nburst = nburst / rank_size;
    uint64_t last_rank_nburst = nburst - per_rank_nburst * (rank_size - 1);
    uint64_t this_rank_nburst = rank_id == rank_size - 1 ? last_rank_nburst : per_rank_nburst;
    uint64_t lenburst = ndd_.lead_dim() * ITEM_SIZE[type_id_];
    uint64_t pad_size = lead_align * ITEM_SIZE[type_id_] - lenburst;
    uint64_t per_rank_offset = per_rank_nburst * lead_align * ITEM_SIZE[type_id_];
    uint64_t this_rank_offset = this_rank_nburst * lead_align * ITEM_SIZE[type_id_];
    uint64_t per_rank_load_offset = per_rank_nburst * lenburst;
    ASSERT(per_rank_offset % 32 == 0);
    // Should be 32Byte aligned in UB
    uint64_t this_rank_count = this_rank_nburst * lead_align;

    // Used by statge 2
    uint64_t per_rank_lenburst = per_rank_offset / 32;
    uint64_t last_rank_lenburst = last_rank_nburst * lead_align * ITEM_SIZE[type_id_] / 32;
    uint64_t this_rank_lenburst = this_rank_nburst * lead_align * ITEM_SIZE[type_id_] / 32;

    // Twoshot Stage 1
    bool is_begin = true;
    uint64_t binary_dst = xbufs_[1] + per_rank_offset * rank_id;
    uint64_t rhs = xbufs_[0];
    uint64_t rhs2 = 0;
    uint64_t lhs = lhs_->xbuf_ + per_rank_offset * rank_id;
    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      // avoid address after cast
      ASSERT(xbuf_size_ % (2 * SIMD_BLOCK_SIZE) == 0);
      binary_dst = xbufs_[1] + xbuf_size_ / 2;
      vUnary op;
      op.xd = binary_dst;
      auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
      op.xn = lhs;
      op.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = binary_dst;
    }

    for (int i = 1; i < rank_size; ++i) {
      rhs2 = rhs;
      vPingPongPeerLoad ppp_load;
      vPingPongLoad &pp_load = ppp_load.base;
      pp_load.from = comm_->GetPeerMemPtr(rank_id + i);
      pp_load.xn = rhs;
      pp_load.tile_stride = tile_stride_size;
      pp_load.body_iter = this_rank_nburst;
      pp_load.tail_iter = this_rank_nburst;
      pp_load.iter_size = lenburst;
      pp_load.pad_size = pad_size;
      pp_load.pingpong = 0;
      pp_load.pingpong_stride = cube_op_->m0_ * cube_op_->n0_ * ITEM_SIZE[type_id_];
      pp_load.round_rank = 0;
      ppp_load.peer_mem_offset = rank_id * per_rank_load_offset;
      current_insn = insn_ + code_size;
      code_size += vPingPongPeerLoad::Encode(current_insn, vAccInsnID::V_PINGPONG_PEER_LOAD, ppp_load, nullptr);
      k.GetVisitor<MixVisitCoder>()->AddReloc(current_insn, vPingPongPeerLoad::SHARD_OFFSET);
      k.comm_op_->id_wrap_.ids_.emplace_back(
        reinterpret_cast<uint32_t *>(current_insn + vPingPongPeerLoad::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
      if (is_begin && rank_size > 2) {
        *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | backward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
      }

      if constexpr (is_bf16) {
        rhs2 = xbufs_[2];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
        op.xn = rhs;
        op.count = this_rank_count;
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary binary_op;  // this binary op can be add or max
      binary_op.xd = binary_dst;
      binary_op.xn = is_begin ? lhs : binary_dst;
      binary_op.xm = rhs2;
      binary_op.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, insn_id_, binary_op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      if (type_id_ != kBFloat16) {
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }
      is_begin = false;
      rhs += this_rank_offset;
    }
    // Last Add backsync first PingPongPeerLoad
    if (rank_size > 2) {
      *current_insn |= 0x1ul << V_HEAD_BACK_SET_OFFSET | backward_event << V_HEAD_B_SET_EVENT_OFFSET;
    }

    // f32 will be casted back to bf16
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbufs_[1] + per_rank_offset * rank_id;
      auto cast_id = GetCastInsnID(kFloat32, kBFloat16);
      op.xn = binary_dst;
      op.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      binary_dst = xbufs_[1] + per_rank_offset * rank_id;
    }

    // TwoShot stage 2
    *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event << V_HEAD_B_WAIT_EVENT_OFFSET;
    vPeerDMA p_store;
    p_store.peer_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_OFFSET;
    p_store.flag_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_FLAG_OFFSET;
    p_store.xn = binary_dst;  // store result of Allreduce
    p_store.tile_stride = tile_stride_size;
    p_store.lenburst = this_rank_lenburst;
    p_store.tail_lenburst = p_store.lenburst;
    p_store.round_rank = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE_MIX, p_store, nullptr);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event << V_M_HEAD_SET_EVENT_OFFSET;

    for (int i = 1; i < rank_size; ++i) {
      uint64_t dst = xbufs_[1] + ((i + rank_id) % rank_size) * per_rank_offset;
      bool is_last = (i + rank_id == rank_size - 1);
      // PeerLoad
      vPeerDMA p_load;
      p_load.flag_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_TWOSHOT_FLAG_OFFSET;
      p_load.xn = dst;
      p_load.tile_stride = tile_stride_size;
      p_load.peer_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_TWOSHOT_OFFSET;
      p_load.lenburst = is_last ? last_rank_lenburst : per_rank_lenburst;
      p_load.tail_lenburst = p_load.lenburst;
      p_load.round_rank = 0;  // TODO: consider broadcast
      current_insn = insn_ + code_size;
      code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD_MIX, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    }
    // last peerload sync copy
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
    vCopy cp;
    cp.xn = xbufs_[1];
    cp.xd = xbuf_;
    cp.lenburst = GetBlocks(tile_stride);
    current_insn = insn_ + code_size;
    code_size += vCopy::Encode(current_insn, V_COPY, cp);
    *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

    current_insn = insn_ + code_size;
    code_size += vNop::Encode(current_insn);
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  } else {
    // OneShot
    uint64_t lhs = lhs_->xbuf_;

    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbuf_;
      auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
      op.xn = lhs_->xbuf_;
      op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = xbuf_;
    }

    bool is_begin = true;
    uint64_t binary_dst = xbuf_;
    uint64_t rhs = 0;
    uint64_t rhs2 = 0;
    bool is_ping = true;
    for (int i = 1; i < rank_size; ++i) {
      rhs = xbufs_[is_ping ? 0 : 1];
      rhs2 = rhs;
      vPingPongPeerLoad ppp_load;
      vPingPongLoad &pp_load = ppp_load.base;
      pp_load.from = comm_->GetPeerMemPtr(rank_id + i);
      pp_load.xn = rhs;
      pp_load.tile_stride = tile_stride_size;
      pp_load.body_iter = ndd_.stride_back() / lead_align;
      pp_load.tail_iter =
        tail_dim_ <= ndd_.lead_idx() ? pp_load.body_iter : pp_load.body_iter / ndd_[tail_dim_] * tail_size_;
      pp_load.iter_size = ndd_.lead_dim() * ITEM_SIZE[type_id_];
      pp_load.pad_size = lead_align * ITEM_SIZE[type_id_] - pp_load.iter_size;
      pp_load.pingpong = 0;
      pp_load.pingpong_stride = cube_op_->m0_ * cube_op_->n0_ * ITEM_SIZE[type_id_];
      pp_load.round_rank = 0;
      ppp_load.peer_mem_offset = 0;
      ppp_load.event_id = backward_event2;
      if (!is_begin && rank_size - i > 1) {
        ppp_load.set_flag = true;
      }
      if (i > 2) {
        ppp_load.wait_flag = true;
      }
      current_insn = insn_ + code_size;
      code_size += vPingPongPeerLoad::Encode(current_insn, vAccInsnID::V_PINGPONG_PEER_LOAD, ppp_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(
        reinterpret_cast<uint32_t *>(current_insn + vPingPongPeerLoad::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
      if (is_begin && rank_size > 2) {
        *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | backward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
      }

      // bf16 would be cast to f32 to be added
      if constexpr (is_bf16) {
        rhs2 = xbufs_[2];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
        op.xn = rhs;
        op.count = ndd_.stride_back();
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary binary_op;  // this binary op can be add or max
      binary_op.xd = binary_dst;
      binary_op.xn = is_begin ? lhs : binary_dst;
      binary_op.xm = rhs2;
      binary_op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, insn_id_, binary_op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      if (type_id_ != kBFloat16) {
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }
      is_begin = false;
      is_ping = !is_ping;
    }
    // add backward sync pingpongpeerload
    if (rank_size > 2) {
      *current_insn |= 0x1ul << V_HEAD_BACK_SET_OFFSET | backward_event << V_HEAD_B_SET_EVENT_OFFSET;
    }
    // f32 will be casted back to bf16
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbuf_;
      auto cast_id = GetCastInsnID(kFloat32, kBFloat16);
      op.xn = binary_dst;
      op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
  }
  tail_insn_ = current_insn;
  return code_size;
}

template <bool is_bf16>
uint64_t AllReduceOp<is_bf16>::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  if (cube_op_ != nullptr) {
    return MatmulEmit(k);
  }
  auto store_id = mix_ ? vAccInsnID::V_PEER_STORE_MIX : vAccInsnID::V_PEER_STORE;
  auto load_id = mix_ ? vAccInsnID::V_PEER_LOAD_MIX : vAccInsnID::V_PEER_LOAD;
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t forward_event = g_system.EventNum() - 1;
  uint64_t backward_event = g_system.EventNum() - 1;
  uint64_t backward_event2 = backward_event - 1;
  int code_size = 0;
  uint64_t *current_insn{nullptr};

  // nop
  code_size += vNop::Encode(insn_);

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[2];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.lenburst = GetBlocks(tile_stride);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event2 << V_HEAD_B_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  current_insn = insn_ + code_size;
  vPeerDMA p_store;
  p_store.peer_mem = comm_->GetPeerMemPtr(rank_id);
  p_store.flag_mem = p_store.peer_mem + PEERMEM_FLAG_OFFSET;
  p_store.xn = store_xbuf_;
  p_store.tile_stride = tile_stride_size;
  p_store.lenburst = GetBlocks(tile_stride);
  p_store.tail_lenburst = p_store.lenburst;
  p_store.round_rank = 0;
  code_size += vPeerDMA::Encode(current_insn, store_id, p_store, nullptr);
  k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event2 << V_M_HEAD_SET_EVENT_OFFSET;

  if (use_twoshot_) {
    uint64_t num_in_block = SIMD_BLOCK_SIZE / ITEM_SIZE[type_id_];
    ASSERT(ndd_.stride_back() % num_in_block == 0);
    uint64_t repeat_full = ndd_.stride_back() / num_in_block;
    // make sure block(32Byte) aligned
    uint64_t per_rank_block = (repeat_full / rank_size);
    uint64_t last_rank_block = repeat_full - per_rank_block * (rank_size - 1);
    uint64_t this_rank_block = rank_id == rank_size - 1 ? last_rank_block : per_rank_block;
    uint64_t per_rank_offset = per_rank_block * SIMD_BLOCK_SIZE;
    uint64_t this_rank_offset = this_rank_block * SIMD_BLOCK_SIZE;
    ASSERT(per_rank_offset % 32 == 0);  // Should be 32Byte aligned in UB

    // TwoShot stage 1:
    // Copy data to peermem
    // Reduce the data this rank is responsible for
    bool is_begin = true;
    uint64_t rhs = xbufs_[0];
    uint64_t rhs2 = 0;
    auto lhs = lhs_->xbuf_ + per_rank_offset * rank_id;
    uint64_t binary_dst = xbufs_[1] + per_rank_offset * rank_id;  // used as destnation of add

    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      // avoid address after cast
      ASSERT(xbuf_size_ % (2 * SIMD_BLOCK_SIZE) == 0);
      binary_dst = xbufs_[1] + xbuf_size_ / 2;
      vUnary op;
      op.xd = binary_dst;
      auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
      op.xn = lhs;
      op.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = binary_dst;
    }

    for (int i = 1; i < rank_size; ++i) {
      rhs2 = rhs;
      // PeerLoad
      vPeerDMA p_load;
      p_load.flag_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_FLAG_OFFSET;
      p_load.xn = rhs;
      p_load.tile_stride = tile_stride_size;
      p_load.peer_mem = comm_->GetPeerMemPtr(i + rank_id) + rank_id * per_rank_offset;
      p_load.lenburst = this_rank_block;
      // TODO: consider tail, If use tail will faster?(less mte2 in tail tile)
      p_load.tail_lenburst = p_load.lenburst;
      p_load.round_rank = 0;  // TODO: consider broadcast
      current_insn = insn_ + code_size;
      code_size += vPeerDMA::Encode(current_insn, load_id, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

      // bf16 would be cast to f32 to be added
      if constexpr (is_bf16) {
        rhs2 = xbufs_[3];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
        op.xn = rhs;
        op.count = this_rank_block * num_in_block;
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary binary_op;
      binary_op.xd = binary_dst;
      binary_op.xn = is_begin ? lhs : binary_dst;
      binary_op.xm = rhs2;
      binary_op.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, insn_id_, binary_op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      if (type_id_ != kBFloat16) {
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }
      is_begin = false;
      rhs += this_rank_offset;
    }

    // f32 will be casted back to bf16
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbufs_[1] + per_rank_offset * rank_id;
      auto cast_id = GetCastInsnID(kFloat32, kBFloat16);
      op.xn = binary_dst;
      op.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      binary_dst = xbufs_[1] + per_rank_offset * rank_id;
    }

    // TwoShot stage 2
    // Copy reduced data to peermem
    // load other reduced data

    // Store should wait last Add
    *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event << V_HEAD_B_WAIT_EVENT_OFFSET;
    vPeerDMA p_store2;
    p_store2.peer_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_OFFSET;
    p_store2.flag_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_FLAG_OFFSET;
    p_store2.xn = binary_dst;  // store result of Allreduce
    p_store2.tile_stride = tile_stride_size;
    p_store2.lenburst = this_rank_block;
    p_store2.tail_lenburst = p_store2.lenburst;
    p_store2.round_rank = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, store_id, p_store2, nullptr);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event << V_M_HEAD_SET_EVENT_OFFSET;

    for (int i = 1; i < rank_size; ++i) {
      uint64_t dst = xbufs_[1] + ((i + rank_id) % rank_size) * per_rank_offset;
      bool is_last = (i + rank_id == rank_size - 1);
      // PeerLoad
      vPeerDMA p_load;
      p_load.flag_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_TWOSHOT_FLAG_OFFSET;
      p_load.xn = dst;
      p_load.tile_stride = tile_stride_size;
      p_load.peer_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_TWOSHOT_OFFSET;
      p_load.lenburst = is_last ? last_rank_block : per_rank_block;
      p_load.tail_lenburst = p_load.lenburst;
      p_load.round_rank = 0;
      current_insn = insn_ + code_size;
      code_size += vPeerDMA::Encode(current_insn, load_id, p_load, nullptr);
      if (i == 1) {
        *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | backward_event2 << V_M_HEAD_WAIT_EVENT_OFFSET;
      }
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    }

    // last peerload sync copy
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
    vCopy cp;
    cp.xn = xbufs_[1];
    cp.xd = xbuf_;
    cp.lenburst = GetBlocks(tile_stride);
    current_insn = insn_ + code_size;
    code_size += vCopy::Encode(current_insn, V_COPY, cp);
    *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BACK_SET_OFFSET | backward_event2 << V_HEAD_B_SET_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

    current_insn = insn_ + code_size;
    code_size += vNop::Encode(current_insn);
    *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  } else {
    // OneShot
    auto lhs = lhs_->xbuf_;
    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbuf_;
      auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
      op.xn = lhs_->xbuf_;
      op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = xbuf_;
    }

    bool is_begin = true;
    bool is_ping = true;
    uint64_t rhs = 0;
    uint64_t rhs2 = 0;
    for (int i = 1; i < rank_size; ++i) {
      rhs = xbufs_[is_ping ? 0 : 1];
      rhs2 = rhs;
      // PeerLoad
      vPeerDMA p_load;
      p_load.peer_mem = comm_->GetPeerMemPtr(i + rank_id);
      p_load.flag_mem = comm_->GetPeerMemPtr(i + rank_id) + PEERMEM_FLAG_OFFSET;
      p_load.xn = rhs;
      p_load.tile_stride = tile_stride * ITEM_SIZE[type_id_];
      p_load.lenburst = GetBlocks(tile_stride);
      p_load.tail_lenburst = tail_dim_ < 0 ? p_load.lenburst : GetBlocks(tile_stride / ndd_[tail_dim_] * tail_size_);
      p_load.round_rank = 0;
      p_load.event_id = backward_event2;
      if (!is_begin && rank_size - i > 1) {
        p_load.set_flag = true;
      }
      if (i > 2) {
        p_load.wait_flag = true;
      }
      current_insn = insn_ + code_size;
      code_size += vPeerDMA::Encode(current_insn, load_id, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

      // bf16 would be cast to f32 to be added
      if constexpr (is_bf16) {
        rhs2 = xbufs_[3];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = GetCastInsnID(kBFloat16, kFloat32);
        op.xn = rhs;
        op.count = ndd_.stride_back();
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary binary_op;
      binary_op.xd = xbuf_;
      binary_op.xn = is_begin ? lhs : xbuf_;
      binary_op.xm = rhs2;
      binary_op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, insn_id_, binary_op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      if (type_id_ != kBFloat16) {
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }
      is_begin = false;
      is_ping = !is_ping;
    }

    // f32 will be casted back to bf16
    if constexpr (is_bf16) {
      vUnary op;
      op.xd = xbuf_;
      auto cast_id = GetCastInsnID(kFloat32, kBFloat16);
      op.xn = xbuf_;
      op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
  }
  tail_insn_ = current_insn;
  return code_size;
}

template <bool is_bf16>
NDObject *AllReduceOp<is_bf16>::Clone(CloneHelper &h) {
  auto op = new AllReduceOp<is_bf16>(ReduceType::kSum, h.GetClone(lhs_), comm_);
  op->insn_id_ = insn_id_;
  return op;
}

template class AllReduceOp<false>;
template class AllReduceOp<true>;

AllGatherOp::AllGatherOp(NDObject *input, const Communicator *comm) : CommOp(input, comm, ObjectType::kAllGather) {
  shape_ref_ = &shape_;
}

// input shape: [a, b], AllGather nd: [b, a, r]，AllGather shape: [a * r, b]
void AllGatherOp::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < size; i++) {
    shape_[i] = lhs_->shape_ref_->data[i];
  }
  ndd_.dims.resize(size + 1);
  for (size_t i = 0; i < size; i++) {
    ndd_.dims[i] = shape_[size - i - 1];
  }
  shape_[0] *= comm_->GetRankSize();
  ndd_.dims[size] = comm_->GetRankSize();

  xbuf_reserve_ = 2;
  code_reserve_ = sizeof(uint64_t) * (5 * (comm_->GetRankSize() + 1) + 6);
  round_tile_.resize(0);
}

void AllGatherOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    // ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

void AllGatherOp::FoldProp(NDObject *op, PropRange &range) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<AllGatherOp *>(op)->ndd_;
  BroadReduceFoldProp<PropRange::BROADCAST>(lhs_nd.data->dims, ndd.dims, range);
}

void AllGatherOp::TileCollect(NDObject *op, TileInfo &info) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<AllGatherOp *>(op)->ndd_;
  BroadReduceTileCollect<PropRange::BROADCAST>(lhs_nd.data->dims, ndd.dims, info);
}

uint64_t AllGatherOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t code_size = 0;
  uint64_t *current_insn{nullptr};
  uint64_t forward_event = g_system.EventNum() - 1;
  uint64_t backward_event = g_system.EventNum() - 1;

  // TODO: If we do not consider prologue fusion of AllGather, then we can find load in this way.
  // But if we consider prologue fusion, this is not a general way to get round_tile_ of load.
  auto load = static_cast<NDLoad *>(this->lhs_);
  round_tile_ = load->round_tile_;
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }

  // nop
  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[0];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.lenburst = GetBlocks(tile_stride);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event << V_HEAD_B_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  vPeerDMA p_store;
  p_store.comm_type = CommType::kCommAllGather;
  p_store.peer_mem = comm_->GetPeerMemPtr(rank_id);
  p_store.flag_mem = p_store.peer_mem + PEERMEM_FLAG_OFFSET;
  p_store.xn = store_xbuf_;
  p_store.tile_stride = tile_stride_size;
  p_store.lenburst = GetBlocks(tile_stride);
  p_store.tail_lenburst = p_store.lenburst;
  p_store.round_rank = round_tile_.size();
  p_store.rank_id = rank_id;
  current_insn = insn_ + code_size;
  code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE, p_store, rounds);
  k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event << V_M_HEAD_SET_EVENT_OFFSET;

  for (int i = 0; i < rank_size; ++i) {
    auto ub_addr = xbufs_[1];
    // PeerLoad
    vPeerDMA p_load;
    p_load.comm_type = CommType::kCommAllGather;
    p_load.peer_mem = comm_->GetPeerMemPtr(i);
    p_load.flag_mem = comm_->GetPeerMemPtr(i) + PEERMEM_FLAG_OFFSET;
    p_load.rank_id = i;
    p_load.xn = ub_addr;
    p_load.tile_stride = tile_stride * ITEM_SIZE[type_id_];
    p_load.lenburst = GetBlocks(tile_stride);
    p_load.tail_lenburst = tail_dim_ < 0 ? p_load.lenburst : GetBlocks(tile_stride / ndd_[tail_dim_] * tail_size_);
    p_load.round_rank = round_tile_.size();
    p_load.event_id = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD, p_load, rounds);
    if (i == 0) {
      *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | backward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
    }
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  }

  // last peerload sync copy
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
  cp.xn = xbufs_[1];
  cp.xd = xbuf_;
  cp.lenburst = GetBlocks(tile_stride);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BACK_SET_OFFSET | backward_event << V_HEAD_B_SET_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  current_insn = insn_ + code_size;
  code_size += vNop::Encode(current_insn);
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  tail_insn_ = current_insn;
  return code_size;
}

NDObject *AllGatherOp::Clone(CloneHelper &h) { return new AllGatherOp(h.GetClone(lhs_), comm_); }

void AllGatherOp::Dump(bool verbose, std::ostringstream &oss) { oss << "AllGather"; }

void AllGatherOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<AllGatherOp *>(op);
  auto &shape = self->shape_;
  auto input_shape = op->lhs_->shape_ref_;
  auto rank_size = self->comm_->GetRankSize();
  shape.Resize(input_shape->size);
  for (size_t i = 0; i < input_shape->size; ++i) {
    if (i == 0) {
      auto dim = input_shape->data[i];
      if (dim < 0) {
        shape[i] = dim;
      } else {
        shape[i] = dim * rank_size;
      }
    } else {
      shape[i] = input_shape->data[i];
    }
  }
}

AllGatherV2Op::AllGatherV2Op(NDObject *input, const Communicator *comm)
    : CommOp(input, comm, ObjectType::kAllGatherV2) {
  shape_ref_ = &shape_;
}

// input shape: [a, b], AllGatherV2 nd: [b, a]，AllGather shape: [a * r, b]
void AllGatherV2Op::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < size; i++) {
    shape_[i] = lhs_->shape_ref_->data[i];
  }
  shape_[0] *= comm_->GetRankSize();

  ndd_.dims = lhs_->nd_.dims();
  xbuf_reserve_ = comm_->GetRankSize();
  code_reserve_ = 5 * sizeof(uint64_t) * (comm_->GetRankSize() + 1);
}

uint64_t AllGatherV2Op::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t code_size = 0;
  uint64_t *current_insn{nullptr};
  auto forward_event = g_system.EventNum() - 1;
  auto backward_event = g_system.EventNum() - 1;

  // nop
  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[0];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.lenburst = GetBlocks(tile_stride);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event << V_HEAD_B_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  vPeerDMA p_store;
  p_store.peer_mem = comm_->GetPeerMemPtr(rank_id);
  p_store.flag_mem = p_store.peer_mem + PEERMEM_FLAG_OFFSET;
  p_store.xn = store_xbuf_;
  p_store.tile_stride = tile_stride_size;
  p_store.lenburst = GetBlocks(tile_stride);
  p_store.tail_lenburst = p_store.lenburst;
  p_store.round_rank = 0;
  current_insn = insn_ + code_size;
  code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE, p_store, nullptr);
  k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | forward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | backward_event << V_M_HEAD_SET_EVENT_OFFSET;

  for (int i = 0; i < rank_size; ++i) {
    if (i == rank_id) {
      continue;
    }
    auto ub_addr = xbufs_[i];
    // PeerLoad
    vPeerDMA p_load;
    p_load.peer_mem = comm_->GetPeerMemPtr(i);
    p_load.flag_mem = comm_->GetPeerMemPtr(i) + PEERMEM_FLAG_OFFSET;
    p_load.xn = ub_addr;
    p_load.tile_stride = tile_stride * ITEM_SIZE[type_id_];
    p_load.lenburst = GetBlocks(tile_stride);
    p_load.tail_lenburst = tail_dim_ < 0 ? p_load.lenburst : GetBlocks(tile_stride / ndd_[tail_dim_] * tail_size_);
    p_load.round_rank = 0;
    p_load.event_id = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD, p_load, nullptr);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    if (i == rank_size - 1 || (rank_id == rank_size - 1 && i == rank_size - 2)) {
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
    }
  }

  cp.xn = lhs_->xbuf_;
  cp.xd = xbufs_[rank_id];
  cp.lenburst = CeilDiv(tile_stride_size, 32UL);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
  tail_insn_ = current_insn;
  return code_size;
}

NDObject *AllGatherV2Op::Clone(CloneHelper &h) { return new AllGatherV2Op(h.GetClone(lhs_), comm_); }

void AllGatherV2Op::Dump(bool verbose, std::ostringstream &oss) { oss << "AllGatherV2"; }
}  // namespace dvm
