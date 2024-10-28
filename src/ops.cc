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
#include <set>
#include <string>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <mutex>
#include <float.h>
#include "ops.h"
#include "kernel.h"

namespace dvm {
namespace {
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 7;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_32 = 32;

struct InsnIdTable {
  const char *name;
  vSimdInsnID ids[kTypeEnd];
};

static const InsnIdTable unary_id_list[kUnaryOpEnd] = {
  // must keep consistent order with UnaryOpType
  {"Sqrt",       {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE}},
  {"Abs",        {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_NONE}},
  {"Log",        {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE}},
  {"Exp",        {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE}},
  {"Reciprocal", {V_NONE, V_REC_FP16, V_NONE, V_REC, V_NONE}},
  {"IsFinite",   {V_NONE, V_ISFINITE_FP16, V_NONE, V_ISFINITE, V_NONE}},
  {"LogicalNot", {V_NOT_BOOL, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"Round",      {V_NONE, V_ROUND_FP16, V_NONE, V_ROUND, V_NONE}},
  {"Floor",      {V_NONE, V_FLOOR_FP16, V_NONE, V_FLOOR, V_NONE}},
  {"Ceil",       {V_NONE, V_CEIL_FP16, V_NONE, V_CEIL, V_NONE}},
  {"Trunc",      {V_NONE, V_TRUNC_FP16, V_NONE, V_TRUNC, V_NONE}}
};

static const InsnIdTable binary_id_list[] = {
  // must keep consistent order with BinaryOpType
  {"Equal",      {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"NotEqual",   {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"Greater",    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"GreaterEqual",{V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"Less",       {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"LessEqual",  {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"Add",        {V_NONE, V_ADD_FP16, V_NONE, V_ADD, V_ADD_INT32}},
  {"Sub",        {V_NONE, V_SUB_FP16, V_NONE, V_SUB, V_SUB_INT32}},
  {"Mul",        {V_NONE, V_MUL_FP16, V_NONE, V_MUL, V_MUL_INT32}},
  {"Div",        {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE}},
  {"Pow",        {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},  // power: individual implement
  {"Maximum",    {V_NONE, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32}},
  {"Minimum",    {V_NONE, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32}},
  {"LogicalAnd", {V_NONE, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32}},
  {"LogicalOr",  {V_NONE, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32}}
};

static const InsnIdTable binarys_id_list[] = {
  // must keep consistent order with BinarySOpType
  {"Equal",       {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"NotEqual",    {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Greater",     {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"GreaterEqual",{V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Less",        {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"LessEqual",   {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Add",         {V_NONE, V_ADDS_FP16, V_NONE, V_ADDS, V_ADDS_INT32}},
  {"Mul",         {V_NONE, V_MULS_FP16, V_NONE, V_MULS, V_MULS_INT32}},
  {"Maximum",     {V_NONE, V_MAXS_FP16, V_NONE, V_MAXS, V_MAXS_INT32}},
  {"Minimum",     {V_NONE, V_MINS_FP16, V_NONE, V_MINS, V_MINS_INT32}}
};

static const vSimdInsnID cast_id_list[][kTypeEnd] = {
  {V_NONE, V_CAST_BOOL_TO_FP16, V_NONE, V_NONE, V_NONE},                             // V_BOOL
  {V_CAST_FP16_TO_BOOL, V_NONE, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE, V_CAST_BF16_TO_FP32, V_CAST_BF16_TO_INT32},               // V_BFLOAT16
  {V_NONE, V_CAST_FP32_TO_FP16, V_CAST_FP32_TO_BF16, V_NONE, V_CAST_FP32_TO_INT32},  // V_FLOAT32
  {V_NONE, V_CAST_INT32_TO_FP16, V_NONE, V_CAST_INT32_TO_FP32, V_NONE},              // V_INT32
};

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

inline uint64_t DMAConfig(uint64_t sid, uint64_t nBurst, uint64_t lenBurst,
                          uint64_t srcStride, uint64_t dstStride) {
  return dstStride << 48 | srcStride << 32 | lenBurst << 16 | nBurst << 4 | sid;
}

int EmitCopy(bcodeptr_t insn, uint64_t xd, uint64_t xn, uint64_t bytes) {
  vCopy op;
  op.xd = xd;
  op.xn = xn;
  op.config = DMAConfig(0, 1, (bytes + 31) >> 5, 0, 0);
  return vCopy::Encode(insn, V_COPY, op);
}

NDObject* GetBroadcastOp(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
  dvm::_BroadcastOp *broadcast_op = nullptr;
  if (stuff_idx < stuff_ops.size()) {
    broadcast_op = static_cast<dvm::_BroadcastOp *>(stuff_ops[stuff_idx]);
    broadcast_op->lhs_ = obj;
  } else {
    broadcast_op = new dvm::_BroadcastOp(obj);
    stuff_ops.push_back(broadcast_op);
  }
  broadcast_op->nd_ = dst_shape;
  stuff_idx++;
  return broadcast_op;
}

NDObject* InsertBroadcastOpsInBetween(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
  // output shape is not dst_shape, but the last inbetween shape, which just need only one broadcast op to reach the dst_shape
  bool broadcast_flag = false;
  auto temp_shape = obj->nd_; // TODO: inplace optimize
  const auto &src_shape = obj->nd_;
  dvm::NDObject *output_obj = obj;
  for (size_t i = 0; i < src_shape.size(); i++) {
    if (src_shape[i] != dst_shape[i]) {
      broadcast_flag = true;
      temp_shape[i] = dst_shape[i];
    } else if (broadcast_flag) {
      if (temp_shape == dst_shape) {
        break;
      }
      broadcast_flag = false;
      output_obj = GetBroadcastOp(output_obj, temp_shape, stuff_ops, stuff_idx);
    }
  }
  return output_obj;
}

NDObject* InsertImplicitBroadcast(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
  // output shape is dst_shape
  auto new_input = InsertBroadcastOpsInBetween(obj, dst_shape, stuff_ops, stuff_idx);
  auto last_broadcast_op = GetBroadcastOp(new_input, dst_shape, stuff_ops, stuff_idx);
  return last_broadcast_op;
}

uint32_t EmitClearPad(uint64_t *pc, NDObject *op, uint64_t simd_width) {
  vClearPad clr_op;
  auto lead_dim = op->lead_dim_;
  clr_op.xd = op->xbuf_;
  clr_op.iter_size = op->nd_[lead_dim];
  clr_op.iter_stride = op->strides_[lead_dim];
  clr_op.iter_num = op->strides_.back() / op->strides_[lead_dim];
  clr_op.simd_width = simd_width;
  return vClearPad::Encode(pc, V_CLR_PAD, clr_op);
}

void BuildDimRounds(const DimArray &round_tile, uint64_t rounds[]) {
  switch (round_tile.size()) {
    case 1: {
      auto r1 = round_tile[0];
      rounds[0] = r1;
      break;
    }
    case 2: {
      auto r1 = round_tile[0] * round_tile[1];
      auto r2 = round_tile[1];
      rounds[0] = r2 << 32 | r1;
      break;
    }
    case 3: {
      auto r1 = round_tile[0] * round_tile[1] * round_tile[2];
      auto r2 = round_tile[1];
      auto r3 = round_tile[2];
      rounds[0] = r2 << 32 | r1;
      rounds[1] = r3;
      break;
    }
    case 4: {
      auto r1 = round_tile[0] * round_tile[1] * round_tile[2] * round_tile[3];
      auto r2 = round_tile[1];
      auto r3 = round_tile[2] * round_tile[3];
      auto r4 = round_tile[3];
      rounds[0] = r2 << 32 | r1;
      rounds[1] = r4 << 32 | r3;
      break;
    }
    default:
      EXCEPTION_IF(true, "multi-broadcast rank exceed max limit(4)");
      break;
  }
}
}  // namespace

MemPool<512, 8192> NDObject::mem_pool_;

int64_t NDObject::Size() {
  return std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) * ITEM_SIZE[type_id_];
}

void NDObject::Tile(const TileParam &tp) {
  bool pointwise = nd_[tp.start] > 1;
  for (int i = tp.start + 1; i <= tp.end; ++i) {
    pointwise = pointwise || nd_[i] > 1;
    nd_[i] = 1;
  }
  if (pointwise) { // broadcast source or reduce des: all 1, donot need to tile
    nd_[tp.start] = tp.tile;
  }
}

void NDObject::UpdateStride(uint64_t simd_width) {
  strides_.resize(nd_.size());
  size_t i = 0;
  for (; i < nd_.size() - 1; ++i) {
    if (nd_[i] > 1) {
      break;
    }
    strides_[i] = 1;
  }
  lead_dim_ = i;
  int64_t align = simd_width;
  strides_[i] = CeilDiv(nd_[i], align) * align;
  for (++i; i < nd_.size(); ++i) {
    strides_[i] = nd_[i] * strides_[i - 1];
  }
}

void NDObject::Dump(std::ostringstream &oss) {
  oss << "NDObject";
}

int NDLoadDummy::Emit(VectorKernel &k) {
  *insn_ = vMakeHead(V_LOAD_DUMMY, 0, 1, V_PIPE_LOAD);
  return 1;
}

void NDLoadDummy::Dump(std::ostringstream &oss) {
  oss << "LoadDummy";
}

void NDLoad::Tile(const TileParam &tp) {
  if (tp.num > 1) {
    bool is_broadcast = true;
    for (int i = tp.start; i <= tp.end; ++i) {
      if (nd_[i] != 1) {
        is_broadcast = false;
        break;
      }
    }
    if (is_broadcast) {
      if (round_tile_.size() % 2 == 0) {
        round_tile_.push_back(tp.num);
      } else {
        round_tile_.back() *= tp.num;
      }
    } else {
      if (!round_tile_.empty()) {
        if (round_tile_.size() % 2 == 0) {
          round_tile_.back() *= tp.num;
        } else {
          round_tile_.push_back(tp.num);
        }
      }
      if (tp.tail > 0) {
        ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
        tail_dim_ = tp.start;
        tail_size_ = tp.tail;
      }
    }
  }
  NDObject::Tile(tp);
}

int NDLoad::Emit(VectorKernel &k) {
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }
  int64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lead_align == nd_[lead_dim_] || lead_align == strides_.back()) {
    vDMA op;
    op.gm = gm_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(src_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(src_tile_stride_ / nd_[tail_dim_] * tail_size_);
    op.round_rank = round_tile_.size();
    reloc_addr_ = insn_ + vDMA::RELOC_OFFSET;
    return vDMA::Encode(insn_, vLoadInsnID::V_LOAD, vPipe::V_PIPE_LOAD, op, rounds);
  } else { // align
    vLoad op;
    op.from = gm_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.body_iter = strides_.back() / lead_align;
    op.tail_iter = tail_dim_ <= lead_dim_ ? op.body_iter : op.body_iter / nd_[tail_dim_] * tail_size_;
    op.iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
    op.round_rank = round_tile_.size();
    reloc_addr_ = insn_ + vLoad::RELOC_OFFSET;
    return vLoad::Encode(insn_, vLoadInsnID::V_LOAD_2, op, rounds);
  }
}

void NDLoad::Normalize(std::vector<NDObject*> &run_ops) {
  auto dims = shape_ref_->size;
  nd_.resize(dims);
  for (size_t i = 0; i < shape_ref_->size; i++) {
    nd_[i] = shape_ref_->data[dims - i - 1];
  }
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
}

void NDLoad::Dump(std::ostringstream &oss) {
  oss << "Load";
}

void NDPadStore::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < pad_shape_->size; i++) {
    shape_[size - 1 - i] = lhs_->shape_ref_->data[size - 1 - i] + pad_shape_->data[pad_shape_->size - 1 - i];
  }
  for (size_t i = pad_shape_->size; i < size; i++) {
    shape_[size - 1 - i] = lhs_->shape_ref_->data[size - 1 - i];
  }
  nd_ = lhs_->nd_;
}

void NDPadStore::AlignProp(PropRange &range) {
  range.depth = 1;
}

void NDPadStore::FoldProp(PropRange &range) {
  range.depth = nd_.size() - 1;
}

int NDPadStore::Emit(VectorKernel &k) {
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  vSliceSL op;
  auto size = shape_ref_->size;
  op.gm = gm_;
  op.xn = lhs_->xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_[lead_dim_];
  op.src_m = shape_ref_->data[size - 2];
  op.src_n = shape_ref_->data[size - 1];

  op.slice_m = lhs_->shape_ref_->data[size - 2];
  op.slice_n = lhs_->shape_ref_->data[size - 1];
  op.slice_k = 1;
  for (size_t i = 0; i + 2 < size; i++) {
    op.slice_k *= shape_ref_->data[i];
  }
  op.type_size = ITEM_SIZE[type_id_];
  op.offset = 0;
  op.one_flag = 0;
  if (op.slice_n == 1 && lead_dim_ != 0) {
    op.pad_size = 0;
    op.one_flag = 1;
  }
  reloc_addr_ = insn_ + vSliceSL::RELOC_OFFSET;
  return vSliceSL::Encode(insn_, vStoreInsnID::V_SLICE_STORE, V_PIPE_STORE, op);
}

void NDPadStore::Dump(std::ostringstream &oss) {
  oss << "PadLoad";
}

void NDSLoad::AlignProp(PropRange &range) {
  range.depth = 1;
}

void NDSStore::AlignProp(PropRange &range) {
  range.depth = 1;
}

void NDSLoad::FoldProp(PropRange &range) {
  range.depth = std::min(static_cast<int>(nd_.size() - 1), range.depth);
}

void NDSStore::FoldProp(PropRange &range) {
  range.depth = std::min(static_cast<int>(nd_.size() - 1), range.depth);
}

void NDSLoad::Tile(const TileParam &tp) {
  if (tp.group_tile) {
    NDObject::Tile(tp);
  } else {
    NDLoad::Tile(tp);
  }
}

void NDSStore::Tile(const TileParam &tp) {
  NDObject::Tile(tp);
}

int NDSStore::Emit(VectorKernel &k) { // TODO: broadcast
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  vSStore op;
  op.gm = gm_;
  op.xn = lhs_->xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_[lead_dim_];
  op.slice_m = cube_op_->m0_;
  op.slice_n = cube_op_->n0_;
  op.src_n = cube_op_->n_real_;
  op.tail_m = cube_op_->m_real_ % cube_op_->m0_;
  op.tail_n = cube_op_->n_real_ % cube_op_->n0_;
  op.type_size = ITEM_SIZE[type_id_];
  reloc_addr_ = insn_ + vSStore::RELOC_OFFSET;
  return vSStore::Encode(insn_, vStoreInsnID::V_SSTORE, op);;
}

int NDSLoad::Emit(VectorKernel &k) {
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (cube_op_->output_ == this && cube_op_->pingpong_store_) {
    uint64_t rounds[2];
    if (!round_tile_.empty()) {
      BuildDimRounds(round_tile_, rounds);
    }
    vPingPongLoad op;
    op.from = gm_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.body_iter = strides_.back() / lead_align;
    op.tail_iter = tail_dim_ <= lead_dim_ ? op.body_iter : op.body_iter / nd_[tail_dim_] * tail_size_;
    op.iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
    op.pingpong = 0;
    op.pingpong_stride = cube_op_->m0_ * cube_op_->n0_ * ITEM_SIZE[type_id_];
    op.round_rank = round_tile_.size();
    reloc_addr_ = insn_ + vPingPongLoad::RELOC_OFFSET;
    return vPingPongLoad::Encode(insn_, vLoadInsnID::V_PINGPONG_LOAD, op, rounds);
  } else {
    vSLoad op;
    op.gm = gm_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_;
    op.pad_size = lead_align - nd_[lead_dim_];
    op.slice_m = cube_op_->m0_;
    op.slice_n = cube_op_->n0_;
    size_t shape_size = shape_ref_->size;
    op.src_n = cube_op_->n_real_;
    op.tail_m = cube_op_->m_real_ % cube_op_->m0_;
    op.tail_n = cube_op_->n_real_ % cube_op_->n0_;
    auto broadcast_m = shape_size < 2 || shape_ref_->data[shape_size - 2] == 1;
    auto broadcast_n = shape_size < 1 || shape_ref_->data[shape_size - 1] == 1;
    op.flags = broadcast_m << 1 | broadcast_n;
    op.type_size = ITEM_SIZE[type_id_];
    reloc_addr_ = insn_ + vSLoad::RELOC_OFFSET;
    return vSLoad::Encode(insn_, vLoadInsnID::V_SLOAD, op);
  }
}

void NDSLoad::Dump(std::ostringstream &oss) {
  oss << "SLoad";
}

void NDSStore::Dump(std::ostringstream &oss) {
  oss << "SStore";
}

void NDSliceLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(src_ref_->size <= 3);
  NDLoad::Normalize(run_ops);
}

void NDSliceLoad::AlignProp(PropRange &range) {
  range.depth = 1;
}

void NDSliceLoad::FoldProp(PropRange &range) {
  range.depth = nd_.size() - 1;
}

int64_t NDSliceLoad::CalcOffset() {
  uint64_t src_offset = 0;
  std::vector<int64_t> start(src_ref_->size);
  for (size_t i = 0; i < src_ref_->size; i++) {
    start[i] = start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i];
  }
  if (src_ref_->size == 1) {
    src_offset = start[0];
  } else if (src_ref_->size == 2) {
    src_offset = start[0] * src_ref_->data[1] + start[1];
  } else {
    src_offset = start[0] * src_ref_->data[1] * src_ref_->data[2] + start[1] * src_ref_->data[2] + start[2];
  }
  src_offset *= ITEM_SIZE[type_id_];
  return src_offset;
}

int NDSliceLoad::Emit(VectorKernel &k) {
  auto reloc_offset = CalcOffset();
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  vSliceSL op;
  auto size = size_ref_->size;
  op.gm = gm_;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_[lead_dim_];
  op.slice_k = 1;
  if (nd_.size() == 1) {
    op.slice_m = 1;
    op.slice_n = size_ref_->data[0];
    op.src_m = 1;
    op.src_n = src_ref_->data[0];
  } else {
    op.slice_m = size_ref_->data[size - 2];
    op.slice_n = size_ref_->data[size - 1];
    op.src_m = src_ref_->data[size - 2];
    op.src_n = src_ref_->data[size - 1];
    for (size_t i = 0; i + 2 < size; i++) {
      op.slice_k *= size_ref_->data[i];
    }
  }
  op.type_size = ITEM_SIZE[type_id_];
  op.offset = reloc_offset;
  op.one_flag = 0;
  reloc_addr_ = insn_ + vSliceSL::RELOC_OFFSET;
  return vSliceSL::Encode(insn_, vLoadInsnID::V_SLICE_LOAD, V_PIPE_LOAD, op);
}

void NDSliceLoad::Dump(std::ostringstream &oss) {
  oss << "SliceLoad";
}

void NDStridedSliceLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(std::all_of(step_ref_->data, step_ref_->data + step_ref_->size, [](int64_t i) { return i == 1; }));
  shape_.Resize(src_ref_->size);
  for (size_t i = 0; i < src_ref_->size; i++) {
    int64_t end = end_ref_->data[i] < 0 ? end_ref_->data[i] + src_ref_->data[i] : end_ref_->data[i];
    int64_t start = start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i];
    shape_[i] = end - start;
  }
  size_ref_ = shape_ref_;
  NDSliceLoad::Normalize(run_ops);
}

void NDStridedSliceLoad::Dump(std::ostringstream &oss) {
  oss << "StridedSliceLoad";
}

void NDStore::Normalize(std::vector<NDObject*> &run_ops) {
  nd_ = lhs_->nd_;
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
}

void NDStore::Tile(const TileParam &tp) {
  if (tp.num > 1) {
    bool is_broadcast = true;
    for (int i = tp.start; i <= tp.end; ++i) {
      if (nd_[i] != 1) {
        is_broadcast = false;
        break;
      }
    }
    if (is_broadcast) {
      if (round_tile_.size() % 2 == 0) {
        round_tile_.push_back(tp.num);
      } else {
        round_tile_.back() *= tp.num;
      }
    } else {
      if (!round_tile_.empty()) {
        if (round_tile_.size() % 2 == 0) {
          round_tile_.back() *= tp.num;
        } else {
          round_tile_.push_back(tp.num);
        }
      }
      if (tp.tail > 0) {
        ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
        tail_dim_ = tp.start;
        tail_size_ = tp.tail;
      }
    }
  }
  NDObject::Tile(tp);
}

int NDStore::Emit(VectorKernel &k) {
  uint64_t lead_align = LeadAlign();
  ASSERT(lead_align == static_cast<uint64_t>(lhs_->LeadAlign()));
  uint64_t dst_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lhs_->obj_id_ == kElementAny) {
    vStoreStatus op;
    op.xn = lhs_->xbuf_;
    op.to = reinterpret_cast<uint64_t>(gm_);
    reloc_addr_ = insn_ + vStoreStatus::RELOC_OFFSET;
    return vStoreStatus::Encode(insn_, V_STORE_STATUS, op);
  }
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
    if (lhs_->obj_id_ == kReduce) {
      auto red_op = lhs_->Cast<ReduceOp*>();
      auto build_atomic_store = [this, lead_align, dst_tile_stride_, red_op](vStoreAtomic &op) {
        op.to = reinterpret_cast<uint64_t>(gm_);
        op.xn = lhs_->xbuf_;
        op.cum_flag = (lhs_->RealObjType() == kAtomicCum && (round_tile_.size() & 1));
        op.iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
        op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        op.iter_num = strides_.back() / lead_align;
        if (tail_dim_ < 0 || red_op->InRange(tail_dim_)) {
          op.iter_tail = op.iter_num;
        } else {
          ASSERT(op.iter_num > 1); // inner reduce is divided. outer reduce is not lead
          op.iter_tail = op.iter_num / nd_[tail_dim_] * tail_size_;
        }
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.round_rank = round_tile_.size();
      };
      int code_size;
      Code &code = k.code_;
      if (System::Instance().deterministic_) {
        vStoreAtomicDeterm op;
        build_atomic_store(op.base);
        op.core_tile_num = (k.tile_num_ + code.block_dim_ - 1) / code.block_dim_;
        op.tail_tile_num = k.tile_num_ % op.core_tile_num ? k.tile_num_ % op.core_tile_num + 1 : 0;
        op.stride_num = Size() / op.base.tile_stride;
        reloc_addr_ = insn_ + vStoreAtomicDeterm::RELOC_OFFSET;
        code_size = vStoreAtomicDeterm::Encode(insn_, V_STORE_ATOMIC_DETERM, code.block_dim_, op, rounds);
      } else {
        vStoreAtomic op;
        build_atomic_store(op);
        reloc_addr_ = insn_ + vStoreAtomic::RELOC_OFFSET;
        code_size = vStoreAtomic::Encode(insn_, V_STORE_ATOMIC, op, rounds);
      }
      red_op->GenClearKernel(this);
      code.sub_codes_.push_back(&(red_op->clear_kernel_->code_));
      code.reloc_reuse_.emplace_back(red_op->clear_store_->reloc_addr_, reloc_addr_);
      return code_size;
    }
  }
  if (lead_align == static_cast<uint64_t>(lhs_->nd_[lhs_->lead_dim_])) {
    vDMA op;
    op.gm = gm_;
    op.xn = lhs_->xbuf_;
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(dst_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(dst_tile_stride_/ nd_[tail_dim_] * tail_size_);
    op.round_rank = round_tile_.size();
    reloc_addr_ = insn_ + vDMA::RELOC_OFFSET;
    return vDMA::Encode(insn_, vStoreInsnID::V_STORE, vPipe::V_PIPE_STORE, op, rounds);
  } else {
    vStore op;
    uint64_t iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    uint64_t pad_size = lead_align * ITEM_SIZE[type_id_] - iter_size;
    uint64_t body_iter = strides_.back() / lead_align;
    uint64_t lead_tiling, tail_iter;
    if (lhs_->RealObjType() == ObjectType::kRemovePad) {
      if (tail_dim_ < 0) {
        iter_size *= body_iter;
        tail_iter = iter_size;
      } else if (body_iter == 1) {
        tail_iter = tail_size_ * ITEM_SIZE[type_id_];
      } else {
        tail_iter = body_iter / nd_[tail_dim_] * tail_size_ * iter_size;
        iter_size *= body_iter;
      }
      lead_tiling = 1;
      body_iter = 1;
      pad_size = 0;
    } else if (body_iter == 1) {
      lead_tiling = 1;
      tail_iter = tail_dim_ < 0 ? iter_size : tail_size_ * ITEM_SIZE[type_id_];
    } else {
      lead_tiling = 0;
      tail_iter = tail_dim_ < 0 ? body_iter : body_iter / nd_[tail_dim_] * tail_size_;
    }
    op.xn = lhs_->xbuf_;
    op.to = reinterpret_cast<uint64_t>(gm_);
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.iter_num = body_iter;
    op.iter_tail = tail_iter;
    op.iter_size = iter_size;
    op.pad_size = pad_size;
    op.round_rank = round_tile_.size();
    op.lead_tiling = lead_tiling;
    reloc_addr_ = insn_ + vStore::RELOC_OFFSET;
    return vStore::Encode(insn_, vStoreInsnID::V_STORE_2, op, rounds);
  }
}

void NDStore::Dump(std::ostringstream &oss) {
  oss << "Store";
}

WrapOp::WrapOp(NDObject *inner, ObjectType wrap_id)
  : FlexOp(inner->lhs_, inner->rhs_, inner->type_id_, inner->obj_id_), inner_(inner), wrap_id_(wrap_id) {
  shape_ref_ = inner->shape_ref_;
  if (inner->flags_ & OBJ_FLAG_XHS) {
    SetXhs(static_cast<FlexOp*>(inner)->xhs_);
  }
  ws_num_ = 1; // for inner_xbuf_
  if (inner->flags_ & OBJ_FLAG_WORKSPACE) {
    ws_num_ += static_cast<FlexOp*>(inner)->ws_num_;
  }
  ASSERT(!(flags_ & OBJ_FLAG_WRAP));
  flags_ |= OBJ_FLAG_WRAP;
}

void WrapOp::Normalize(std::vector<NDObject*> &run_ops) {
  inner_->Normalize(run_ops);
  nd_ = inner_->nd_;
  lhs_ = inner_->lhs_;
  rhs_ = inner_->rhs_;
  if (inner_->flags_ & OBJ_FLAG_XHS) {
    xhs_ = static_cast<FlexOp*>(inner_)->xhs_;
  }
}
void WrapOp::Tile(const TileParam &tp) {
  inner_->Tile(tp);
  nd_ = inner_->nd_;
}
void WrapOp::AlignProp(PropRange &range) { inner_->AlignProp(range); }
void WrapOp::FoldProp(PropRange &range) { inner_->FoldProp(range); }

int CopyOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, strides_.back() * ITEM_SIZE[type_id_]);
}

void CopyOp::Dump(std::ostringstream &oss) {
  oss << "Copy";
}

void ReshapeOp::Normalize(std::vector<NDObject*> &run_ops) {
  // update nd_/shape_
  auto dims = dst_shape_ref_->size;
  nd_.resize(dims);
  shape_.Resize(dims);
  int64_t sz = 1;
  size_t update_axis = dims;
  for (size_t i = 0; i < dims; ++i) {
    auto sh = dst_shape_ref_->data[i];
    shape_[i] = sh;
    auto nd_i = dims - i - 1;
    if (sh == -1) {
      update_axis = nd_i;
    } else {
      sz *= sh;
      nd_[nd_i] = sh;
    }
  }
  if (update_axis != dims) {
    int64_t input_sz = 1;
    auto &nd = lhs_->nd_;
    for (size_t i = 0; i < nd.size(); ++i) {
      input_sz *= nd[i];
    }
    auto v = input_sz / sz;
    nd_[update_axis] = v;
    shape_[dims - 1 - update_axis] = v;
  }
}

int ReshapeOp::Emit(VectorKernel &k) {
  if (lhs_->nd_[lhs_->lead_dim_] == nd_[lead_dim_]) {
    return CopyOp::Emit(k);
  }
  vReshape op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xd_lead = nd_[lead_dim_];
  op.xn_lead = lhs_->nd_[lhs_->lead_dim_];
  op.xd_pad = strides_[lead_dim_] - op.xd_lead;
  op.xn_pad = lhs_->strides_[lhs_->lead_dim_] - op.xn_lead;
  op.dup_size = strides_.back() / strides_[lead_dim_];
  return vReshape::Encode(insn_, (type_id_ == kFloat32 || type_id_ == kInt32) ? V_RESHAPE_B32 : V_RESHAPE_B32, op);
}

void ReshapeOp::Dump(std::ostringstream &oss) {
  oss << "Reshape";
}

UnaryOp::UnaryOp(int op_type, NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kUnary), op_type_(op_type) {
 shape_ref_ = input->shape_ref_;
}

int UnaryOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  ASSERT(size_t(op_type_) < sizeof(unary_id_list) / sizeof(InsnIdTable));
  auto id = unary_id_list[op_type_].ids[type_id_];
  return vUnary::Encode(insn_, id, op);
}

void UnaryOp::Dump(std::ostringstream &oss) {
  oss << unary_id_list[op_type_].name;
}

int UnaryOp::QueryId(const std::string &op_name) {
  for (int i = 0; i < static_cast<int>(sizeof(unary_id_list) / sizeof(InsnIdTable)); ++i) {
    if (op_name == unary_id_list[i].name) return i;
  }
  return -1;
}

int IsFinite16Op::Emit(VectorKernel &k) {
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = wss_[0];
  op.repeat = strides_.back() / k.simd_width_;
  return vBinary::Encode(insn_, V_ISFINITE_FP16, op);
}

void IsFinite16Op::Dump(std::ostringstream &oss) {
  oss << "IsFinite";
}

int AtomicCumOp::Emit(VectorKernel &k) {
  if (!(round_tile_->size() & 1)) {
    int size = InnerEmit(k, insn_, xbuf_);
    tail_insn_ = inner_->tail_insn_;
    return size;
  }
  int size = InnerEmit(k, insn_, inner_xbuf_);
  vAtomicCum op;
  uint64_t rounds[2];
  BuildDimRounds(*round_tile_, rounds);
  op.xd = xbuf_;
  op.xn = inner_xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  op.round_rank = round_tile_->size();
  tail_insn_ = insn_ + size;
  size += vAtomicCum::Encode(tail_insn_, V_ATOMICCUM, op, rounds);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  return size;
}

void AtomicCumOp::Dump(std::ostringstream &oss) {
  oss << "AtomicCum.";
  inner_->Dump(oss);
}

int RemovePadOp::Emit(VectorKernel &k) {
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_REMOVEPAD_U16, V_REMOVEPAD_U16, V_REMOVEPAD, V_REMOVEPAD};
  if (nd_[lead_dim_] == strides_[lead_dim_] || strides_.back() == strides_[lead_dim_]) {
    int size = InnerEmit(k, insn_, xbuf_);
    tail_insn_ = inner_->tail_insn_;
    return size;
  }
  int size = InnerEmit(k, insn_, inner_xbuf_);
  vRemovePad op;
  op.xd = xbuf_;
  op.xn = inner_xbuf_;
  op.repeat = strides_.back() / strides_[lead_dim_];
  op.iter_num = nd_[lead_dim_];
  op.rs = GetBlocks(strides_[lead_dim_]);
  tail_insn_ = insn_ + size;
  ASSERT(id_list[type_id_] != V_NONE);
  size += vRemovePad::Encode(tail_insn_, id_list[type_id_], op);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  return size;
}

void RemovePadOp::Dump(std::ostringstream &oss) {
  oss << "RemovePad.";
  inner_->Dump(oss);
}

void ElementAnyOp::Normalize(std::vector<NDObject *> &run_ops) {
  tail_dim_ = -1;
  tail_size_ = 0;
  nd_.resize(lhs_->nd_.size(), 1);
}

void ElementAnyOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int ElementAnyOp::Emit(VectorKernel &k) {
  uint32_t insn_num = 1;
  uint32_t size = 0;
  if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
    size = EmitClearPad(insn_, lhs_, k.simd_width_);
    tail_insn_ = insn_ + size;
    insn_num++;
  }
  vElementAny op;
  op.xn = lhs_->xbuf_ ; 
  op.xd = xbuf_;
  op.rs = GetBlocks(k.simd_width_);
  op.iter_size = lhs_->strides_.back();
  op.tail_size = tail_dim_ < 0
                    ? lhs_->strides_.back()
                    : lhs_->strides_.back() / lhs_->nd_[tail_dim_] * tail_size_;

  op.repeat = lhs_->strides_.back() / k.simd_width_;
  size += vElementAny::Encode(tail_insn_, V_ELEMENT_ANY, op);

  if (insn_num > 1) {
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return size;
}

void ElementAnyOp::Dump(std::ostringstream &oss) {
  oss << "ElementAny";
}

int CastOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  auto id = cast_id_list[lhs_->type_id_][type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

void CastOp::Dump(std::ostringstream &oss) {
  oss << "Cast";
}

template <typename T>
BinaryScalarOp<T>::BinaryScalarOp(int op_type, NDObject *input, T scalar)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kBinaryS), op_type_(op_type), scalar_(scalar) {
  shape_ref_ = input->shape_ref_;
}

template <typename T>
int BinaryScalarOp<T>::Emit(VectorKernel &k) {
  vBinaryS<T> op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  op.scalar = scalar_;
  ASSERT(size_t(op_type_) < sizeof(binarys_id_list) / sizeof(InsnIdTable));
  auto id = binarys_id_list[op_type_].ids[type_id_];
  return vBinaryS<T>::Encode(insn_, id, op);
}

template <typename T>
void BinaryScalarOp<T>::Dump(std::ostringstream &oss) {
  oss << binarys_id_list[op_type_].name << "<" << scalar_ << ">";
}

template class BinaryScalarOp<float>;
template class BinaryScalarOp<int32_t>;

CompareScalarOp::CompareScalarOp(int op_type, NDObject *input, float scalar)
    : FlexOp(input, nullptr, input->type_id_, ObjectType::kCompareS), scalar_(scalar) {
  ws_num_ = 1;
  cmp_op_ = op_type;
  shape_ref_ = input->shape_ref_;
}

int CompareScalarOp::Emit(VectorKernel &k) {
  vCompareS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  op.scalar = scalar_;
  op.ws = wss_[0];
  op.type = cmp_op_;
  return vCompareS::Encode(insn_, type_id_ == kFloat32 ? V_CMPS : V_CMPS_FP16, op);
}

void CompareScalarOp::Dump(std::ostringstream &oss) {
  oss << "CompareS<" << cmp_op_ << ", " << scalar_ << ">";
}

_BinaryNormalizer::~_BinaryNormalizer() {
  for (auto op : lhs_stuff_ops_) {
    delete op;
  }
  for (auto op : rhs_stuff_ops_) {
    delete op;
  }
}

void _BinaryNormalizer::Normalize(NDObject *self, std::vector<NDObject*> &run_ops) {
  // recover original input
  if (!lhs_stuff_ops_.empty()) {
    self->lhs_ = lhs_stuff_ops_[0]->lhs_;
  }
  if (!rhs_stuff_ops_.empty()) {
    self->rhs_ = rhs_stuff_ops_[0]->lhs_;
  }
  // update shape_ref_
  auto lhs_data = self->lhs_->shape_ref_->data;
  auto rhs_data = self->rhs_->shape_ref_->data;
  auto lhs_sz = self->lhs_->shape_ref_->size;
  auto rhs_sz = self->rhs_->shape_ref_->size;
  if (lhs_sz > rhs_sz) {
    shape_.Resize(lhs_sz);
    auto diff = lhs_sz - rhs_sz;
    for (size_t i = 0; i < diff; ++i) {
      shape_[i] = lhs_data[i];
    }
    for (size_t i = diff; i < lhs_sz; ++i) {
      shape_[i] = lhs_data[i] == 1 ? rhs_data[i - diff]: lhs_data[i];
    }
  } else {
    shape_.Resize(rhs_sz);
    auto diff = rhs_sz - lhs_sz;
    for (size_t i = 0; i < diff; ++i) {
      shape_[i] = rhs_data[i];
    }
    for (size_t i = diff; i < rhs_sz; ++i) {
      shape_[i] = rhs_data[i] == 1 ? lhs_data[i - diff]: rhs_data[i];
    }
  }
  // update nd_
  const auto &lhs_nd = self->lhs_->nd_;
  const auto &rhs_nd = self->rhs_->nd_;
  auto &nd = self->nd_;
  auto lhs_dim = lhs_nd.size();
  auto rhs_dim = rhs_nd.size();
  auto res_dim = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
  nd.resize(res_dim, 1);
  bool lhs_need_broadcast = false;
  bool rhs_need_broadcast = false;
  for (size_t i = 0; i < res_dim; ++i) {
    auto lhs_axis = i < lhs_dim ? lhs_nd[i] : 1;
    auto rhs_axis = i < rhs_dim ? rhs_nd[i] : 1;
    if (lhs_axis > rhs_axis) {
      nd[i] = lhs_axis;
      rhs_need_broadcast = true;
    } else if (lhs_axis < rhs_axis) {
      nd[i] = rhs_axis;
      lhs_need_broadcast = true;
    } else {
      nd[i] = lhs_axis;
    }
  }
  if (self->flags_ & OBJ_FLAG_EAGER) {
    if (lhs_need_broadcast) {
      size_t stuff_idx = run_ops.size();
      self->lhs_ = InsertImplicitBroadcast(self->lhs_, nd, run_ops, stuff_idx);
    }
    if (rhs_need_broadcast) {
      size_t stuff_idx = run_ops.size();
      self->rhs_ = InsertImplicitBroadcast(self->rhs_, nd, run_ops, stuff_idx);
    }
    return;
  }
  if (lhs_need_broadcast) {
    size_t stuff_idx = 0;
    self->lhs_ = InsertImplicitBroadcast(self->lhs_, nd, lhs_stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(lhs_stuff_ops_[i]);
    }
  }
  if (rhs_need_broadcast) {
    size_t stuff_idx = 0;
    self->rhs_ = InsertImplicitBroadcast(self->rhs_, nd, rhs_stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(rhs_stuff_ops_[i]);
    }
  }
}

BinaryOp::BinaryOp(int op_type, NDObject *lhs, NDObject *rhs) : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kBinary), op_type_(op_type) {
  shape_ref_ = &norm_.shape_;
}

int BinaryOp::Emit(VectorKernel &k) {
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  ASSERT(size_t(op_type_) < sizeof(binary_id_list) / sizeof(InsnIdTable));
  auto id = binary_id_list[op_type_].ids[type_id_];
  return vBinary::Encode(insn_, id, op);
}

void BinaryOp::Dump(std::ostringstream &oss) {
  oss << binary_id_list[op_type_].name;
}

int BinaryOp::QueryId(const std::string &op_name) {
  for (int i = 0; i < static_cast<int>(sizeof(binary_id_list) / sizeof(InsnIdTable)); ++i) {
    if (op_name == binary_id_list[i].name) return i;
  }
  return -1;
}

CompareOp::CompareOp(int op_type, NDObject *lhs, NDObject *rhs) : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kCompare) {
  ws_num_ = 1;
  cmp_op_ = op_type;
  shape_ref_ = &norm_.shape_;
}

int CompareOp::Emit(VectorKernel &k) {
  vCompare op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.type = cmp_op_;
  op.ws = wss_[0];
  op.repeat = strides_.back() / k.simd_width_;
  return vCompare::Encode(insn_, type_id_ == kFloat32 ? V_CMP : V_CMP_FP16, op);
}

void CompareOp::Dump(std::ostringstream &oss) {
  oss << "Compare<" << cmp_op_ << ">";
}

int PowerOp::Emit(VectorKernel &k) {
  ASSERT(type_id_ == dvm::kFloat32);
  vBinaryWS op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  op.ws0 = wss_[0];
  op.ws1 = wss_[1];
  return vBinaryWS::Encode(insn_, V_POW, op);
}

void PowerOp::Dump(std::ostringstream &oss) {
  oss << "Power";
}

void SelectOp::Normalize(std::vector<NDObject *> &run_ops) {
  // recover original input
  NDObject **input[] = {&lhs_, &rhs_, &xhs_};
  for (size_t i = 0; i < 3; i++) {
    if (!stuff_ops_[i].empty()) {
      *input[i] = stuff_ops_[i][0]->lhs_;
    }
  }
  auto max_size = std::max({lhs_->shape_ref_->size, rhs_->shape_ref_->size, xhs_->shape_ref_->size});
  shape_.Resize(max_size);
  for (size_t i = 0; i < max_size; ++i) {
    int64_t len[3];
    for (size_t j = 0; j < 3; j++) {
      auto obj = *input[j];
      len[j] = (obj->shape_ref_->size < i + 1) ? 1 : obj->shape_ref_->data[obj->shape_ref_->size - 1 - i];
    }
    shape_[max_size - 1 - i] = std::max({len[0], len[1], len[2]});
  }

  // update nd_
  auto &lhs_nd = lhs_->nd_;
  auto &rhs_nd = rhs_->nd_;
  auto &xhs_nd = xhs_->nd_;
  bool lhs_bc = false;
  bool rhs_bc = false;
  bool xhs_bc = false;
  auto max_dims = std::max({lhs_nd.size(), rhs_nd.size(), xhs_nd.size()});
  nd_.resize(max_dims);
  for (size_t i = 0; i < max_dims; ++i) {
    auto lhs_dim = i < lhs_nd.size() ? lhs_nd[i] : 1;
    auto rhs_dim = i < rhs_nd.size() ? rhs_nd[i] : 1;
    auto xhs_dim = i < xhs_nd.size() ? xhs_nd[i] : 1;
    nd_[i] = std::max({lhs_dim, rhs_dim, xhs_dim});
    if (nd_[i] != lhs_dim) lhs_bc = true;
    if (nd_[i] != rhs_dim) rhs_bc = true;
    if (nd_[i] != xhs_dim) xhs_bc = true;
  }
  auto insert_broadcast = [&input, &run_ops, this](size_t idx) {
    size_t stuff_idx = 0;
    *input[idx] = InsertImplicitBroadcast(*input[idx], nd_, stuff_ops_[idx], stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(stuff_ops_[idx][i]);
    }
  };
  if (lhs_bc) insert_broadcast(0);
  if (rhs_bc) insert_broadcast(1);
  if (xhs_bc) insert_broadcast(2);
}

SelectOp::~SelectOp() {
  for (size_t j = 0; j < 3; ++j) {
    for (auto op : stuff_ops_[j]) {
      delete op;
    }
  }
}

int SelectOp::Emit(VectorKernel &k) {
  vSelect op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_SEL_FP16, V_NONE, V_SEL, V_SEL_INT32};
  op.repeat = strides_.back() / k.simd_width_;
  op.xm = rhs_->xbuf_;
  op.cond =  xhs_->xbuf_;
  op.ws = wss_[0];
  ASSERT(id_list[type_id_] != V_NONE);
  return vSelect::Encode(insn_, id_list[type_id_], op);
}

void SelectOp::Dump(std::ostringstream &oss) {
  oss << "Select";
}

void _BroadcastOp::FoldProp(PropRange &range) {
  int state = 0; // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

void _BroadcastOp::AlignProp(PropRange &range) {
  int state = 0; // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

int _BroadcastOp::Emit(VectorKernel &k) {
  int start_dim = -1;
  int end_dim = -1;
  for (size_t i = lead_dim_; i < nd_.size(); ++i) {
    if (start_dim == -1) {
      if (nd_[i] != lhs_->nd_[i])  {
        end_dim = i;
        start_dim = i;
      }
    } else if (nd_[i] != lhs_->nd_[i] || nd_[i] == 1) {
      end_dim = i;
    } else {
      break;
    }
  }
  int64_t offset; 
  if (start_dim == 0 || start_dim == lead_dim_) {
    offset = EmitBroadcastX(insn_, end_dim, k.simd_width_);
  } else {
    offset = EmitBroadcastY(insn_, start_dim, end_dim, k.simd_width_);
  }
  return offset;
}

int64_t _BroadcastOp::EmitBroadcastX(uint64_t *p, int end_dim, int64_t simd_width) {
  vBroadcastX op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_[end_dim] / simd_width;
  int64_t rank_size = static_cast<int64_t>(strides_.size());
  op.lead_num = end_dim + 1 < rank_size ? nd_[end_dim + 1] : 1;
  op.iter_num = end_dim + 2 <  rank_size ? strides_.back() / strides_[end_dim + 1] : 1;
  op.lead_pad = lhs_->strides_[lhs_->lead_dim_] - lhs_->nd_[lhs_->lead_dim_];
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_BROADCAST_X_B16, V_NONE, V_BROADCAST_X_B32, V_BROADCAST_X_B32};
  ASSERT(id_list[type_id_] != V_NONE);
  return vBroadcastX::Encode(p, id_list[type_id_], op);
}

int64_t _BroadcastOp::EmitBroadcastY(uint64_t *p, int start_dim, int end_dim, int64_t simd_width) {
  vBroadcastY op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  if (start_dim > 0) {
    op.iter_num = strides_.back() / strides_[end_dim];
    op.dup_num = strides_[end_dim]  / strides_[start_dim - 1];
    op.dup_stride = strides_[start_dim - 1] * ITEM_SIZE[type_id_] / SIMD_BLOCK_SIZE;
  } else {
    op.iter_num = 1;
    op.dup_num = 1;
    op.dup_stride = strides_.back() * ITEM_SIZE[type_id_] / SIMD_BLOCK_SIZE;
  }
  return vBroadcastY::Encode(p, V_BROADCAST_Y, op);
}

void _BroadcastOp::Dump(std::ostringstream &oss) {
  oss << "Broadcast";
}

BroadcastOp::~BroadcastOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
}

void BroadcastOp::Normalize(std::vector<NDObject*> &run_ops) {
  // recover original input
  if (!stuff_ops_.empty()) {
    lhs_ = stuff_ops_[0]->lhs_;
  }
  // update nd_ from shape_ref_
  auto dims = dst_shape_ref_->size;
  nd_.resize(dims);
  shape_.Resize(dims);
  auto offset = dims - lhs_->shape_ref_->size;  // dst_shape dims >= x_shape dims
  for (size_t i = 0; i < dims; ++i) {
    shape_[i] = dst_shape_ref_->data[i];
    if (shape_[i] == -1) {
      // e.g. x_shape (4, 1), dst_shape (2, -1, 1) --> dst_shape (2, 4, 1)
      shape_[i] = lhs_->shape_ref_->data[i - offset];
    }
    nd_[dims - 1 - i] = shape_[i];
  }
  if (flags_ & OBJ_FLAG_EAGER) {
    size_t stuff_idx = run_ops.size();
    lhs_ = InsertBroadcastOpsInBetween(lhs_, nd_, run_ops, stuff_idx);
  } else {
    size_t stuff_idx = 0;
    lhs_ = InsertBroadcastOpsInBetween(lhs_, nd_, stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(stuff_ops_[i]);
    }
  }
}

template <typename T>
void BroadcastScalarOp<T>::Normalize(std::vector<NDObject*> &run_ops) {
  // update nd_ from shape_ref_
  auto dims = shape_ref_->size;
  nd_.resize(dims);
  for (size_t i = 0; i < dims; ++i) {
    nd_[i] = shape_ref_->data[dims - i - 1];
  }
}

template <typename T>
int BroadcastScalarOp<T>::Emit(VectorKernel &k) {
  vBroadcastS<T> op;
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_BROADCAST_S_FP16, V_NONE, V_BROADCAST_S, V_BROADCAST_S_INT32};
  op.scalar = scalar_;
  op.xd = xbuf_;
  op.repeat = strides_.back() / k.simd_width_;
  ASSERT(id_list[type_id_] != V_NONE);
  return vBroadcastS<T>::Encode(insn_, id_list[type_id_], op);
}

template <typename T>
void BroadcastScalarOp<T>::Dump(std::ostringstream &oss) {
  oss << "BroadcastS<" << scalar_ << ">";
}

template class BroadcastScalarOp<float>;
template class BroadcastScalarOp<int32_t>;

void _ReduceOp::FoldProp(PropRange &range) {
  int state = 0; // -1 - reduce ; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      break;
    }
    if (state == 0) {
      if (nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
}

void _ReduceOp::AlignProp(PropRange &range) {
  int state = 0; // -1 - reduce; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      break;
    }
    if (state == 0) {
      if (nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
}

void _ReduceOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int _ReduceOp::Emit(VectorKernel &k) {
  ASSERT(red_op_ == ReduceOp::SUM);
  if (nd_[lhs_->lead_dim_] == lhs_->nd_[lhs_->lead_dim_] && strides_.back() == lhs_->strides_.back()) {
    return EmitCopy(insn_, xbuf_, lhs_->xbuf_, strides_.back() * ITEM_SIZE[type_id_]);
  } else if (start_dim_ <= lhs_->lead_dim_) { // reduce x
    uint32_t size = 0;
    uint32_t insn_num = 1;
    if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
      size = EmitClearPad(insn_, lhs_, k.simd_width_);
      tail_insn_ = insn_ + size;
      insn_num++;
    }
    vReduceX op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.red_size = lhs_->strides_[end_dim_];
    if (tail_dim_ >= 0 && tail_dim_ <= end_dim_) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_size = lhs_->strides_.back() / lhs_->strides_[end_dim_];
    if (nd_[lead_dim_] > 1) {
      op.dup_block = nd_[lead_dim_];
      op.dup_pad = strides_[lead_dim_] - nd_[lead_dim_];
    } else {
      op.dup_block = op.dup_size;
      op.dup_pad = 0;
    }
    size += vReduceX::Encode(tail_insn_, V_RSUM_X, op);
    if (insn_num > 1) {
      *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
    return size;
  } else {
    vReduceY op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.iter_size = strides_[start_dim_];
    op.red_size = lhs_->strides_[end_dim_] / op.iter_size;
    ASSERT(op.red_size > 1);
    if (InRange(tail_dim_)) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = strides_.back() / strides_[end_dim_];
    return vReduceY::Encode(insn_, V_RSUM_Y, op);
  }
}

void _ReduceOp::Dump(std::ostringstream &oss) {
  oss << "Reduce";
}

ReduceOp::~ReduceOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
  if (clear_kernel_ != nullptr) {
    delete clear_kernel_;
  }
}

void ReduceOp::Normalize(std::vector<NDObject*> &run_ops) {
  DimArray dims;
  DimArray shape_dims;
  ASSERT(dims_ref_ != nullptr);
  NDObject *input = stuff_ops_.empty() ? lhs_ : stuff_ops_[0]->lhs_;
  auto input_shape_ref = input->shape_ref_;
  //update dims
  auto lhs_dim = input_shape_ref->size;
  if (dims_ref_->size == 0) {
    shape_dims.resize(lhs_dim);
    dims.resize(lhs_dim);
    for (int64_t i = 0; i < static_cast<int64_t>(lhs_dim); ++i) {
      shape_dims[i] = i;
      dims[i] = i;
    }
  } else {
    std::set<int64_t> dims_set;
    for (size_t i = 0; i < dims_ref_->size; i++) {
      auto dim = dims_ref_->data[i];
      if (dim < 0) {
        dim += lhs_dim;
      }
      dims_set.insert(dim);
    }
    size_t dim_size = 0;
    for (auto it = dims_set.begin(); it != dims_set.end(); ++it) {
      shape_dims[dim_size++] = *it;
    }
    int64_t back_idx = shape_dims[dim_size - 1] + 1;
    int64_t back_end = input->nd_.size() - 1;
    while (back_idx <= back_end && input->nd_[back_end - back_idx] == 1) { // align fold may flip dims
      shape_dims[dim_size++] = back_idx++;
    }
    shape_dims.resize(dim_size);
    dims.resize(dim_size);
    for (size_t i = 0; i < dim_size; ++i) {
      dims[i] = lhs_dim - shape_dims[dim_size - i - 1] - 1;
    }
  }
  // update shape_ref_
  int shape_size = 0;
  int dim_idx = 0;
  for (int i = 0; i < static_cast<int>(input_shape_ref->size); ++i) {
    if (i != shape_dims[dim_idx]) {
      shape_[shape_size++] = input_shape_ref->data[i];
    } else {
      dim_idx++;
      if (keepdims_) {
        shape_[shape_size++] = 1;
      }
    }
  }
  shape_.Resize(shape_size);
  size_t stuff_idx = 0;
  int red_start = -1, red_end = -1, red_ext = -1, lead_dim = -1;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto d = dims[i];
    if (input->nd_[d] == 1) continue;
    if (d != red_ext) {
      if (lead_dim  == -1) {
        for (lead_dim = 0; lead_dim < d && input->nd_[lead_dim] == 1; lead_dim++);
      }
      if (red_start >= 0) {
        _ReduceOp *obj;
        if (flags_ & OBJ_FLAG_EAGER) {
          obj = new _ReduceOp(input, red_op_);
        } else {
          if (stuff_idx == stuff_ops_.size()) {
            stuff_ops_.push_back(new _ReduceOp(input, red_op_));
          }
          obj = stuff_ops_[stuff_idx];
        }
        stuff_idx++;
        obj->nd_ = input->nd_;
        for (int j = red_start; j <= red_end; ++j) {
          obj->nd_[j] = 1;
        }
        // align tile may revert to 0. let lead reduce to 0
        obj->SetRange(red_start == lead_dim ? 0 : red_start, red_end);
        run_ops.push_back(obj);
        input = obj;
      }
      red_start = d;
    }
    red_end = d;
    for (red_ext = d + 1; red_ext < static_cast<int>(input->nd_.size()) && input->nd_[red_ext] == 1; red_ext++);
  }
  nd_ = input->nd_;
  if (red_start != -1) {
    lhs_ = input;
    for (int j = red_start; j <= red_end; ++j) {
      nd_[j] = 1;
    }
    SetRange(red_start == lead_dim ? 0 : red_start, red_end);
  } else {
    ASSERT(stuff_idx == 0);
    start_dim_ = end_dim_ = -1;
  }
}

void ReduceOp::GenClearKernel(NDAccess *store) {
 if (clear_kernel_ == nullptr) {
    clear_kernel_ = new VKernelD();
    auto dummy_load = new NDLoadDummy(type_id_);
    clear_kernel_->Append(dummy_load);
    clear_shape_.data = &clear_shape_data_;
    clear_shape_.size = 1;
    auto broadcast_scalar_op = new BroadcastScalarOp<float>(0.0, &clear_shape_, type_id_, dummy_load);
    clear_kernel_->Append(broadcast_scalar_op);
    clear_store_ = new NDStore(store->gm_, broadcast_scalar_op);
    clear_kernel_->Append(clear_store_);
  }
  clear_shape_data_ = std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{});
  if (System::Instance().deterministic_) {
    clear_shape_data_ += 32 / sizeof(float);
  }
  clear_kernel_->CodeGen();
}

void AffinePropOp::FoldProp(PropRange &range) {
  enum _State { kStateUndeterm = 0, kStateElemwise, kStateBroadcast };
  _State state = kStateUndeterm;
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    auto d1 = x_->nd_[i];
    auto d2 = y_->nd_[i];
    if (state == kStateUndeterm) {
      if (d1 != d2) {
        state = kStateBroadcast;
      } else if (d1 != 1) {
        state = kStateElemwise;
      }
    } else if (state == kStateElemwise) {
      if (d1 != d2) {
        break;
      }
    } else if (d1 == d2 && d1 > 1) {
      break;
    }
    new_depth++;
  }
  if (state == kStateBroadcast && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

void AffinePropOp::AlignProp(PropRange &range) {
  enum _State { kStateUndeterm = 0, kStateElemwise, kStateBroadcast };
  _State state = kStateUndeterm;
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    auto d1 = x_->nd_[i];
    auto d2 = y_->nd_[i];
    if (state == kStateUndeterm) {
      if (d1 != d2) {
        state = kStateBroadcast;
      } else if (d1 != 1) {
        state = kStateElemwise;
      }
    } else if (state == kStateElemwise) {
      if (d1 != d2) {
        break;
      }
    } else if (d1 == d2 && d1 > 1) {
      break;
    }
    new_depth++;
  }
  if (state == kStateBroadcast && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

int AffinePropOp::Emit(VectorKernel &k) {
  ASSERT(0);
  return 0;
}

void AffinePropOp::Dump(std::ostringstream &oss) {
  oss << "AffineProp";
}

void CubeOp::ComputeBroadcastShape(NDObject *lhs, NDObject *rhs) {
  int n = std::max(lhs->nd_.size(), rhs->nd_.size());
  nd_.resize(n);
  nd_[0] = n_real_;
  nd_[1] = m_real_;
  for (int i = 2; i < n; ++i) {
    auto dim1 = i < static_cast<int>(lhs->nd_.size()) ? lhs_->nd_[i] : 1;
    auto dim2 = i < static_cast<int>(rhs->nd_.size()) ? rhs_->nd_[i] : 1;
    // TODO: nd_[2] = dim1 >= dim2 ? dim1 : dim2;
    if (dim1 == dim2) {
      nd_[i] = dim1;
    } else if (dim1 == 1) {
      nd_[i] = dim2;
    } else if (dim2 == 1) {
      nd_[i] = dim1;
    } else {
      // should not reach here, because this case can not be broadcasted.
      ASSERT(0);
    }
  }
  shape_.resize(n);
  for (size_t i = 0; i < nd_.size(); ++i) {
    shape_[i] = nd_[nd_.size() - 1 - i];
  }
}

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b)
    : NDObject(lhs, rhs, lhs->type_id_, kCubeOp), trans_a_(trans_a), trans_b_(trans_b){};

CubeOp::~CubeOp() {
  if (lhs_->IsLoad()) {
    delete lhs_;
  }
  if (rhs_->IsLoad()) {
    delete rhs_;
  }
  if (output_->IsStore()) {
    delete output_;
  }
}

void CubeOp::InitPadShape() {
  auto GetPad = [](int64_t pad_size) -> std::vector<int64_t> {
    if (pad_size % ALIGN_128 == 0 || (pad_size <= ALIGN_256 && pad_size % ALIGN_32 == 0)) {
      return {};
    }
    return {ALIGN_256 - pad_size % ALIGN_256};
  };
  pad_a_ = GetPad(trans_a_ ? m_align_ : k_align_);
  pad_b_ = GetPad(trans_b_ ? k_align_ : n_align_);
}

void CubeOp::NormalizeCube() {
  m_align_ = trans_a_ ? lhs_->nd_[0] : lhs_->nd_[1];
  // Only pad the rows, which may result in matrices A and B where some K matrices are padded and some are not.
  // Therefore, we take the maximum among them.
  k_align_ = std::max(trans_a_ ? lhs_->nd_[1] : lhs_->nd_[0], trans_b_ ? rhs_->nd_[0] : rhs_->nd_[1]);
  n_align_ = trans_b_ ? rhs_->nd_[1] : rhs_->nd_[0];
  m_real_ = m_align_;
  k_real_ = k_align_;
  n_real_ = n_align_;
  NormalizeOutput();
}

void CubeOp::NormalizeOutput() {
  if (lhs_->nd_.size() == 2 && rhs_->nd_.size() == 2) {
    nd_.resize(2);
    nd_[0] = n_real_;
    nd_[1] = m_real_;
    shape_ = {m_real_, n_real_};
  } else {
    ComputeBroadcastShape(lhs_, rhs_);
  }
  shape_ref_data_ = shape_;
  shape_ref_ = &shape_ref_data_;
}

float CubeOp::CostFunc(vCubeOp *op, uint32_t m0, uint32_t n0) {
  float a_coef = 5.0f;
  float b_coef = 5.0f;
  float bw_coef = 1.0f;
  auto m_loop = CeilDiv(op->m_real, m0);
  auto n_loop = CeilDiv(op->n_real, n0);
  if (m_loop == 0 || n_loop == 0) {
    return FLT_MAX;
  }
  auto core_need = m_loop * n_loop;
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
  auto l2_num = System::Instance().L2Size() / ITEM_SIZE[lhs_->type_id_];
  uint32_t block_dim = core_need < core_num ? core_need : core_num;
  uint32_t m_once = block_dim < n_loop ? m0 : block_dim / n_loop * m0;

  uint32_t n_once = block_dim < n_loop ? core_num * n0 : op->n_real;
  if (m_once * op->k_real > l2_num) {
      a_coef = bw_coef;
  }
  if (n_once * op->k_real > l2_num) {
      b_coef = bw_coef;
  }
  // calibrate bandwidth
  a_coef = a_coef * block_dim / core_num;
  b_coef = b_coef * block_dim / core_num;
  return static_cast<float>(m_real_) * static_cast<float>(n_loop) / a_coef +
         static_cast<float>(n_real_) * static_cast<float>(m_loop) / b_coef;
}

void CubeOp::Tile(vCubeOp *op) {
  auto pri_flag = m_align_ < n_align_ ? false : true;
  auto m_round = RoundUp(static_cast<uint32_t>(m_align_), BLOCK_SIZE);
  auto n_round = RoundUp(static_cast<uint32_t>(n_align_), BLOCK_SIZE);
  auto pri_axis = pri_flag ? m_round : n_round;
  auto axis = pri_flag ? n_round : m_round;
  auto axis_max = AXES_ALIGN_SIZE / ITEM_SIZE[lhs_->type_id_];
  auto pri_axis0_max = pri_axis < axis_max ? pri_axis : axis_max;
  auto axis0_max = axis < axis_max ? axis : axis_max;
  auto l0c_num = System::Instance().L0CSize() / FP32_SIZE;
  uint32_t pri_axis0_init = BLOCK_SIZE;
  uint32_t axis0_init = BLOCK_SIZE;
  // The maximum value can be returned by cost function.
  // cost function: 1.0f / (a_coef * n0) + 1.0f / (b_coef * m0)
  // a_coef and b_coef are not less than 1 / core_num, which is 0.04
  // m0 and n0 are not less than BLOCK_SIZE, which is 16
  float min_cost = FLT_MAX;
  // m0, n0
  for (uint32_t pri_axis0 = pri_axis0_init; pri_axis0 <= pri_axis0_max; pri_axis0 *= 2) {
    for (uint32_t axis0 = axis0_init; axis0 <= axis0_max; axis0 *= 2) {
      if (pri_axis0 * axis0 > l0c_num) {
        break;
      }
      auto m0 = pri_flag ? pri_axis0 : axis0;
      auto n0 = pri_flag ? axis0 : pri_axis0;
      auto cost = CostFunc(op, m0, n0);
      if (cost < min_cost) {
        min_cost = cost;
        op->m0 = m0;
        op->n0 = n0;
      }
    }
  }
  // k0
  uint32_t cubeBlockSize = CUBE_BLOCK_SIZE;
  uint32_t kBlockSize = BLOCK_SIZE;
  auto l1_ping_pong_num = System::Instance().L1Size() / 2 / ITEM_SIZE[lhs_->type_id_];
  auto k0_max = l1_ping_pong_num / (op->m0 + op->n0);
  op->k0 = k0_max < cubeBlockSize ? RoundDown(k0_max, kBlockSize) : RoundDown(k0_max, cubeBlockSize);
  if (op->k0 > CONST_512) {
    op->k0 = RoundDown(op->k0, CONST_512);
  }
  if (op->k0 > op->k_real) {
    op->k0 = op->k_real;
    if (op->k0 % BLOCK_SIZE) {
      op->k0 += BLOCK_SIZE - op->k0 % BLOCK_SIZE;
    }
  }
  m0_ = op->m0;
  n0_ = op->n0;
  k0_ = op->k0;
}

void CubeOp::Dump(std::ostringstream &oss) {
  oss << "MatMul";
}

void CubeOp::GetSwizzleConfig(vCubeOp *op) {
  uint32_t swizzle_cnt = DEFAULT_SWIZZLE_COUNT;
  uint32_t swizzle_dir = 0;
  if (System::Instance().SocName() == kAscend910B4) {
    float mincost = op->m_align + op->n_align;
    for (size_t i = 1; i <= block_dim_; i++) {
      uint32_t c = (block_dim_ + i - 1) / i;
      float cost;
      if (i * op->n0 + op->m_align < op->m0 * i + op->n_align) {  // zN
        uint32_t mem_a_zN = c * op->m0;
        uint32_t mem_b_zN = i * op->n0;
        cost = mem_a_zN + mem_b_zN;
        if (cost <= mincost) {
          swizzle_dir = 1;
          mincost = cost;
          swizzle_cnt = i;
        }
      } else {  // nZ
        uint32_t mem_a_nZ = c * op->n0;
        uint32_t mem_b_nZ = i * op->m0;
        cost = mem_a_nZ + mem_b_nZ;
        if (cost < mincost) {
          swizzle_dir = 0;
          mincost = cost;
          swizzle_cnt = i;
        }
      }
    }
  } else {
    if (op->m_real > op->n_real) {
      swizzle_dir = 0;
      uint32_t m_loop = CeilDiv(op->m_real, op->m0);
      swizzle_cnt = std::min(swizzle_cnt, m_loop);
    } else {
      swizzle_dir = 1;
      uint32_t n_loop = CeilDiv(op->n_real, op->n0);
      swizzle_cnt = std::min(swizzle_cnt, n_loop);
    }
  }
  op->swizzle = swizzle_dir << 16 | swizzle_cnt;
}

static uint32_t GetSwizzle(uint64_t major, uint64_t minor, uint64_t major_loop, uint64_t minor_loop, uint64_t k_real,
                           bool major_align, bool minor_align, uint32_t block_dim, float &mincost) {
  constexpr float L2_BW = 5.0f;
  const uint64_t CACHE_LINE = 512 / ITEM_SIZE[kFloat16];
  uint32_t core_num = System::Instance().CoreNum(CoreType::kCube);
  uint64_t cache_limit = (System::Instance().L2Size() / ITEM_SIZE[kFloat16] - 256 * 128 * 8 * core_num) / k_real;
  uint64_t swizzle_cnt = 0;
  uint64_t minsize = major * major_loop + minor * minor_loop;
  for (uint64_t cnt = std::min(static_cast<uint64_t>(block_dim), major_loop); cnt >= 1; --cnt) {
    float major_hit, minor_hit;
    uint64_t width = std::min(block_dim * 2 / cnt, minor_loop);
    uint64_t major_need = major * cnt * 2;
    uint64_t minor_need = minor * width;
    if (major_align) major_need = RoundUp(major_need, CACHE_LINE);
    if (minor_align) minor_need = RoundUp(minor_need, CACHE_LINE);
    if (minor * minor_loop + major_need * 2 < cache_limit) {
      uint64_t size = major * cnt + minor * width;
      if (size >= minsize && mincost < 3.125f) continue;
      minsize = size;
      major_hit = static_cast<float>(minor_loop - 1) / minor_loop;
      minor_hit = static_cast<float>(major_loop - 1) / major_loop;
    } else if (major_need + minor_need < cache_limit) {
      major_hit = static_cast<float>(minor_loop - 1) / minor_loop;
      minor_hit = static_cast<float>(major_loop - ((major_loop - 1) / cnt + 1)) / major_loop;
    } else {
      major_hit = static_cast<float>(width - 1) / width;
      minor_hit = static_cast<float>(cnt - 1) / cnt;
      if (major_align && major < CACHE_LINE) {
        uint64_t size = major * cnt;
        major_hit = major_hit * size / RoundUp(size, CACHE_LINE);
      }
      if (minor_align && minor < CACHE_LINE) {
        uint64_t size = minor * width;
        minor_hit = minor_hit * size / RoundUp(size, CACHE_LINE);
      }
      float k_hit = static_cast<float>(cache_limit) / (major_need + minor_need);
      major_hit *= k_hit;
      minor_hit *= k_hit;
    }
    float major_coef = L2_BW / (major_hit + (1.0f - major_hit) * L2_BW);
    float minor_coef = L2_BW / (minor_hit + (1.0f - minor_hit) * L2_BW);
    if (block_dim < core_num) {
      major_coef = major_coef * block_dim / core_num;
      minor_coef = minor_coef * block_dim / core_num;
    }
    float cost = 1.0f / (major_coef * static_cast<float>(minor)) + 1.0f / (minor_coef * static_cast<float>(major));
    //std::cout << "swizzle=(" << cnt << ", " << width << "), hit=(" << major_hit << ", " << minor_hit << "), coef=(" << major_coef << ", " << minor_coef << "), cost=" << cost << std::endl;
    if (cost < mincost) {
      mincost = cost;
      swizzle_cnt = cnt;
    }
  }
  return swizzle_cnt;
}

void CubeOp::TileV2(vCubeOp *op) {
  auto l0c_max = System::Instance().L0CSize() / FP32_SIZE;
  auto l1_max = System::Instance().L1Size() / 2 / ITEM_SIZE[lhs_->type_id_];
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
  float mincost = 3.125f;
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
    } else if (!trans_b_) { // trans_a && !trans_b_
      m0 = x;
      n0 = y;
      if (m0 > round_m || n0 > round_n) return;
      uint64_t kx = l1_max / (m0 + n0);
      k0 = RoundDown(kx, kx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      if (m0 * n0 > l0c_max || k0 == 0) return;
      if (k0 > round_k) k0 = round_k;
    } else { // trans_a && trans_b_
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
    bool swizzle_zN = m_align_ < n_align_;
    //std::cout << "param: m0=" << m0 << ", n0=" << n0 << ", k0=" << k0 << ", core_loop=" << core_loop << ", block_dim=" << block_dim << ", swizzle_zN=" << swizzle_zN << std::endl;
    uint32_t swizzle = swizzle_zN ? GetSwizzle(n0, m0, n_loop, m_loop, k_real_, !trans_b_, trans_a_, block_dim, mincost)
                      : GetSwizzle(m0, n0, m_loop, n_loop, k_real_, trans_a_, !trans_b_, block_dim, mincost);
    if (swizzle) {
      op->m0 = m0_ = m0;
      op->n0 = n0_ = n0;
      op->k0 = k0_ = k0;
      op->swizzle = swizzle_zN ? 1u << 16 | swizzle : swizzle;
      block_dim_ = block_dim;
      core_loop_ = core_loop;
    }
  };
  block_dim_ = 0;
  uint32_t align_max = 512 / ITEM_SIZE[lhs_->type_id_];
  for (uint32_t x = align_max; x >= BLOCK_SIZE; x >>= 1) {
    for (uint32_t y = align_max; y >= x; y >>= 1) {
      tile_select(x, y);
      if (x != y) {
        tile_select(y, x);
      }
      if (block_dim_ > 0) {
        return;
      }
    }
  }
}

void CubeOp::GenTiling(vCubeOp *op) {
  static int tiling_ver = -1;
  if (tiling_ver == -1) {
    const char *ver = getenv("DVM_MATMUL_TILING");
    tiling_ver = ver != nullptr ? std::stoi(ver) : 0;
  }
  if (tiling_ver == 2) {
    TileV2(op);
  } else {
    Tile(op);
    auto m_loop = CeilDiv(op->m_real, op->m0);
    auto n_loop = CeilDiv(op->n_real, op->n0);
    core_loop_ = m_loop * n_loop * std::max(op->batch_a0, op->batch_b0) * std::max(op->batch_a1, op->batch_b1);
    auto core_num = System::Instance().CoreNum(CoreType::kCube);
    block_dim_ = core_loop_ < core_num ? core_loop_ : core_num;
    GetSwizzleConfig(op);
  }
}

void CubeOp::CodeGen(vCubeOp *op) {
  op->m_align = m_align_;
  op->n_align = n_align_;
  op->k_align = k_align_;
  op->m_real = m_real_;
  op->n_real = n_real_;
  op->k_real = k_real_;
  op->a_size = lhs_->nd_[0] * lhs_->nd_[1];
  op->b_size = rhs_->nd_[0] * rhs_->nd_[1];
  op->offset_a = offset_a_;
  op->offset_b = offset_b_;
  if (lhs_->IsLoad()) {
    auto a = static_cast<NDAccess*>(lhs_);
    op->gm_a = reinterpret_cast<uint64_t>(a->gm_);
    op->batch_a1 = a->nd_.size() > 2 ? static_cast<uint32_t>(a->nd_[2]) : 1;
    op->batch_a0 = a->nd_.size() > 3 ? static_cast<uint32_t>(a->nd_[3]) : 1;
  } else {
    ASSERT(0); // TODO: pre fusion
  }
  if (rhs_->IsLoad()) {
    auto b = static_cast<NDAccess*>(rhs_);
    op->gm_b = reinterpret_cast<uint64_t>(b->gm_);
    op->batch_b1 = b->nd_.size() > 2 ? static_cast<uint32_t>(b->nd_[2]) : 1;
    op->batch_b0 = b->nd_.size() > 3 ? static_cast<uint32_t>(b->nd_[3]) : 1;
  } else {
    ASSERT(0); // TODO: pre fusion
  }
  auto c = static_cast<NDAccess*>(output_);
  op->gm_c = reinterpret_cast<uint64_t>(c->gm_);
  op->flags = trans_a_ ? V_CUBE_FLAG_TRANS_A : 0;
  if (trans_b_)  op->flags |= V_CUBE_FLAG_TRANS_B;
  if (type_id_ == dvm::kFloat32) op->flags |= V_CUBE_FLAG_OUT_FP32;
  if (atomic_add_) op->flags |= V_CUBE_FLAG_ATOMIC_ADD;
  auto dtype = lhs_->type_id_;
  ASSERT(dtype == dvm::kFloat16 || dtype == dvm::kBFloat16);
  op->dtype = dtype == dvm::kFloat16 ? vCubeOp::FP16 : vCubeOp::BF16;
  GenTiling(op);
  //std::cout << "result tiling: m0=" << op->m0 << ", n0=" << op->n0 << ", k0=" << op->k0 << ", swizzle=(" << (op->swizzle >> 16) << ", " << (op->swizzle & 0xfffful) << ")" << std::endl;
}
} // namespace dvm
