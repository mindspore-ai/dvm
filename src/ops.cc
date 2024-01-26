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

#include <string>
#include <cstring>
#include <vector>
#include "ops.h"

namespace dvm {
namespace {
inline uint64_t DMAConfig(uint64_t sid, uint64_t nBurst, uint64_t lenBurst,
                          uint64_t srcStride, uint64_t dstStride) {
  return dstStride << 48 | srcStride << 32 | lenBurst << 16 | nBurst << 4 | sid;
}

inline uint64_t MakeHead(uint64_t pipe, uint64_t len, uint64_t ext, uint64_t id) {
  ASSERT(len % 8 == 0);
  uint64_t head = ext << V_HEAD_EXT_OFFSET | (len / 8) << V_HEAD_SIZE_OFFSET |
                  id << V_HEAD_ID_OFFSET;
  if (pipe == V_PIPE_SIMD) {
    head |= 1ul << V_HEAD_IS_SIMD_OFFSET;
  }
  return head;
}

NDObject* GetBroadcastOp(NDObject *obj, const std::vector<int64_t> &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
  dvm::_BroadcastOp *broadcast_op = nullptr;
  if (stuff_idx < stuff_ops.size()) {
    broadcast_op = static_cast<dvm::_BroadcastOp *>(stuff_ops[stuff_idx]);
    broadcast_op->lhs_ = obj;
    broadcast_op->nd_ = dst_shape;
  } else {
    broadcast_op = new dvm::_BroadcastOp(obj, dst_shape);
    stuff_ops.push_back(broadcast_op);
  }
  stuff_idx++;
  return broadcast_op;
}

NDObject* InsertBroadcastOpsInBetween(NDObject *obj, const std::vector<int64_t> &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
  // output shape is not dst_shape, but the last inbetween shape, which just need only one broadcast op to reach the dst_shape
  bool broadcast_flag = false;
  auto temp_shape = obj->nd_;
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

NDObject* InsertImplicitBroadcast(NDObject *obj, const std::vector<int64_t> &dst_shape, std::vector<NDObject*> &stuff_ops, size_t &stuff_idx) {
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
}  // namespace

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
  uint64_t align = simd_width;
  strides_[i] = (nd_[i] + align - 1) / align * align;
  for (++i; i < nd_.size(); ++i) {
    strides_[i] = nd_[i] * strides_[i - 1];
  }
}

std::pair<uint32_t, uint32_t> NDLoadDummy::Emit(const Code &code) {
  *insn = MakeHead(V_PIPE_LOAD, sizeof(uint64_t), 0, V_LOAD_DUMMY);
  return std::make_pair(sizeof(uint64_t), 1);
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
      if (round_ == 0) {
        factor_ = tp.num;
        round_ = 1;
      } else {
        ASSERT(round_ == 1); // restrict: only continuous broadcast
        factor_ *= tp.num;
      }
    } else if (factor_ > 1) {
      round_ *= tp.num;
      factor_ *= tp.num;
    }
    if (!is_broadcast && tp.tail > 0) {
      ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
      tail_dim_ = tp.start;
      tail_size_ = tp.tail;
    }
  }
  NDObject::Tile(tp);
}

std::pair<uint32_t, uint32_t> NDLoad::Emit(const Code &code) {
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lead_align == static_cast<uint64_t>(nd_[lead_dim_]) || lead_dim_ + 1 == static_cast<int>(nd_.size())) {
    vDMA op;
    op.gm = src_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(src_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(src_tile_stride_ / nd_[tail_dim_] * tail_size_);
    if (factor_) {
      op.round = round_;
      op.factor = factor_;
      op.has_round = 1;
    } else {
      op.round = op.factor = op.has_round = 0;
    }
    auto size = vDMA::Encode(insn, vMemInsnID::V_LOAD, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  } else { // align
    vLoad op;
    op.from = src_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.body_iter = strides_.back() / lead_align;
    op.tail_iter = tail_dim_ <= lead_dim_ ? op.body_iter : op.body_iter / nd_[tail_dim_] * tail_size_;
    op.iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
    if (factor_) {
      op.round = round_;
      op.factor = factor_;
      op.has_round = 1;
    } else {
      op.round = op.factor = op.has_round = 0;
    }
    auto size = vLoad::Encode(insn, vMemInsnID::V_LOAD_2, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  }
}

void NDLoad::Reloc(void *src, bool update_insn) {
  src_ = static_cast<uint8_t *>(src);
  if (!update_insn) {
    return;
  }
  uint64_t id = (*insn >> V_HEAD_ID_OFFSET) & V_HEAD_ID_MASK;
  if (id == V_LOAD) {
    vDMA::Reloc(insn, src_);
  } else {
    vLoad::Reloc(insn, src_);
  }
}

void NDStore::Reloc(void *dst, bool update_insn) {
  dst_ = static_cast<uint8_t *>(dst);
  if (!update_insn) {
    return;
  }
  uint64_t id = (*insn >> V_HEAD_ID_OFFSET) & V_HEAD_ID_MASK;
  if (id == V_STORE) {
    vDMA::Reloc(insn, dst_);
  } else if(id == V_STORE_STATUS) {
    vStoreStatus *op = reinterpret_cast<vStoreStatus *>(insn);
    op->to = dst_;
  } else {
    vStore *op = reinterpret_cast<vStore *>(insn);
    op->to = dst_;
  }
}

void NDStore::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    ASSERT(tail_dim_ == -1); // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

std::pair<uint32_t, uint32_t> NDStore::Emit(const Code &code) {
  uint64_t lead_align = LeadAlign();
  ASSERT(lead_align == static_cast<uint64_t>(lhs_->LeadAlign()));
  uint64_t dst_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lhs_->obj_id_ == kElementAny) {
    vStoreStatus *op = reinterpret_cast<vStoreStatus *>(insn);
    op->to = dst_;
    op->head = MakeHead(V_PIPE_STORE, sizeof(vStoreStatus), lhs_->xbuf_, V_STORE_STATUS);
    return std::make_pair(sizeof(vStoreStatus), 1);
  }
  if (lhs_->obj_id_ == kReduce) {
    auto red_op = static_cast<ReduceOp*>(lhs_);
    if (red_op->factor_ > 1) {
      vStoreAtomic op;
      op.to = reinterpret_cast<uint64_t>(dst_);
      op.xn = lhs_->xbuf_;
      op.iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
      op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
      op.iter_num = strides_.back() / lead_align;
      if (tail_dim_ < 0 || red_op->InRange(tail_dim_)) {
        op.iter_tail = op.iter_num;
      } else {
        op.iter_tail = op.iter_num / nd_[tail_dim_] * tail_size_;
      }
      op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
      op.round = red_op->round_;
      op.factor = red_op->factor_;
      ASSERT(type_id_ == kFloat32 || DeviceInfo::Instance().Arch() == kAiCore_C220);
      op.type = type_id_ == kFloat32 ? 0 : 1;
      auto size = vStoreAtomic::Encode(insn, V_STORE_ATOMIC, op);
      atomic_ = true;
      return std::make_pair(size*sizeof(uint64_t), 1);
    }
  }
  atomic_ = false;
  if (lead_align == static_cast<uint64_t>(lhs_->nd_[lhs_->lead_dim_])) {
    vDMA op;
    op.gm = dst_;
    op.xn = lhs_->xbuf_;
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(dst_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(dst_tile_stride_/ nd_[tail_dim_] * tail_size_);
    op.round = op.factor = op.has_round = 0;
    auto size = vDMA::Encode(insn, vMemInsnID::V_STORE, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  } else {
    vStore *op = reinterpret_cast<vStore*>(insn);
    uint64_t ext = (dst_tile_stride_ * ITEM_SIZE[type_id_]) << V_X_BITS | lhs_->xbuf_;
    op->head = MakeHead(V_PIPE_STORE, sizeof(vStore), ext, vMemInsnID::V_STORE_2);
    op->to = dst_;
    uint64_t iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    uint64_t pad_size = lead_align * ITEM_SIZE[type_id_] - iter_size;
    uint64_t body_iter = strides_.back() / lead_align;
    uint64_t lead_tiling, tail_iter;
    if (body_iter == 1) {
      lead_tiling = 1;
      tail_iter = tail_dim_ < 0 ? iter_size : tail_size_ * ITEM_SIZE[type_id_];
    } else {
      lead_tiling = 0;
      tail_iter = tail_dim_ < 0 ? body_iter : body_iter / nd_[tail_dim_] * tail_size_;
    }
    op->config = lead_tiling << 62 | pad_size << 54 | iter_size << 36 | tail_iter << 18 | body_iter;
    return std::make_pair(sizeof(vStore), 1);
  }
}

std::pair<uint32_t, uint32_t> CopyOp::Emit(const Code &code) {
  vCopy *op = reinterpret_cast<vCopy*>(insn);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  op->head = MakeHead(V_PIPE_SIMD, sizeof(vCopy), ext, V_COPY);
  uint64_t lenburst = GetBlocks(strides_.back());
  op->config = DMAConfig(0, 1, lenburst, 0, 0);
  return std::make_pair(sizeof(vCopy), 1);
}

void ReshapeOp::Normalize(std::vector<NDObject*> &run_ops) {
  // update nd_ from shape_ref_
  auto dims = shape_ref_->size;
  nd_.resize(dims);
  int64_t sz = 1;
  size_t update_axis = dims;
  for (size_t i = 0; i < dims; ++i) {
    auto sh = shape_ref_->data[i];
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
    for (auto sh : lhs_->nd_) {
      input_sz *= sh;
    }
    nd_[update_axis] = input_sz / sz;
  }
}

UnaryOp::UnaryOp(int op_type, NDObject *input)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kUnary) {
  static const vOpInsnID id_list[][kTypeEnd] = {  // must keep consistent order with UnaryOpType
    {V_NONE, V_SQRT_FP16, V_SQRT, V_NONE},
    {V_NONE, V_RSQRT_FP16, V_RSQRT, V_NONE},
    {V_NONE, V_ABS_FP16, V_ABS, V_NONE},
    {V_NONE, V_LOG_FP16, V_LOG, V_NONE},
    {V_NONE, V_EXP_FP16, V_EXP, V_NONE},
    {V_NONE, V_REC_FP16, V_REC, V_NONE},
    {V_NONE, V_ISFINITE_FP16, V_ISFINITE, V_NONE},
    {V_NOT_INT8, V_NOT_FP16, V_NOT, V_NONE}};
  id_ = id_list[op_type][type_id_];
  ASSERT(id_ != V_NONE);
  shape_ref_ = input->shape_ref_;
}

std::pair<uint32_t, uint32_t> UnaryOp::Emit(const Code &code) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / code.simd_width;
  auto size = vUnary::Encode(insn, id_, op);
  return std::make_pair(size * sizeof(uint64_t), 1);
}

std::pair<uint32_t, uint32_t> ElementAnyOp::Emit(const Code &code) {
  uint32_t insn_num = 1;
  uint32_t size = 0;
  if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
    size = EmitClearPad(insn, lhs_, code.simd_width);
    tail_insn = insn + size;
    insn_num++;
  }
  vElementAny op;
  op.xn = lhs_->xbuf_ ; 
  op.xd = xbuf_;
  op.rs = GetBlocks(code.simd_width);
  op.burst_len = GetBlocks(lhs_->strides_.back());
  op.repeat = lhs_->strides_.back() / code.simd_width;
  size += vElementAny::Encode(tail_insn, type_id_ == kFloat32 ? V_ELEMENT_ANY : V_ELEMENT_ANY_FP16, op);

  if (insn_num > 1) {
    *(tail_insn) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return std::make_pair(size * sizeof(uint64_t), insn_num);
}

std::pair<uint32_t, uint32_t> _CastOp::Emit(const Code &code) {
  static const vOpInsnID id_list[][kTypeEnd] = {
    {V_NONE, V_CAST_INT8_TO_FP16, V_NONE, V_NONE},
    {V_CAST_FP16_TO_INT8, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32},
    {V_NONE, V_CAST_FP32_TO_FP16, V_NONE, V_CAST_FP32_TO_INT32},
    {V_NONE, V_CAST_INT32_TO_FP16, V_CAST_INT32_TO_FP32, V_NONE}};
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / code.simd_width;
  auto size = vUnary::Encode(insn, id_list[lhs_->type_id_][type_id_], op);
  return std::make_pair(size * sizeof(uint64_t), 1);
}

static const int g_cast_staff_type[kTypeEnd][kTypeEnd] = {
  {-1,        -1,    kFloat16, kFloat16},  // V_INT8
  {-1,        -1,    -1,        -1},       // V_FLOAT16
  {kFloat16,  -1,    -1,        -1},       // V_FLOAT32
  {kFloat16,  -1,    -1,        -1},       // V_INT32
};

CastOp::CastOp(NDObject *input, DType type_id) : _CastOp(input, type_id) {
  shape_ref_ = input->shape_ref_;
  auto stuff_type = g_cast_staff_type[input->type_id_][type_id_];
  if (stuff_type != -1) {
    stuff_op_ = new _CastOp(input, static_cast<DType>(stuff_type));
    lhs_ = stuff_op_;
  }
}

CastOp::~CastOp() {
  if (stuff_op_) {
    delete stuff_op_;
  }
}

void CastOp::Normalize(std::vector<NDObject*> &run_ops) {
  auto input = Input();
  if (stuff_op_ != nullptr)  {
    stuff_op_->nd_ = input->nd_;
    stuff_op_->index_ = run_ops.size();
    run_ops.push_back(stuff_op_);
  }
  nd_ = input->nd_;
}

BinaryScalarOp::BinaryScalarOp(int op_type, NDObject *input, float scalar)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kBinaryS), scalar_(scalar) {
  static const vOpInsnID id_list[][kTypeEnd] = {  // must keep consistent order with BinarySOpType
    {V_NONE, V_ADDS_FP16, V_ADDS, V_NONE},
    {V_NONE, V_MULS_FP16, V_MULS, V_NONE},
    {V_NONE, V_MAXS_FP16, V_MAXS, V_NONE},
    {V_NONE, V_MINS_FP16, V_MINS, V_NONE}};
  id_ = id_list[op_type][type_id_];
  ASSERT(id_ != V_NONE);
  shape_ref_ = input->shape_ref_;
}

std::pair<uint32_t, uint32_t> BinaryScalarOp::Emit(const Code &code) {
  vBinaryS *op = reinterpret_cast<vBinaryS*>(insn);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  op->head = MakeHead(V_PIPE_SIMD, sizeof(vBinaryS), ext, id_);
  uint64_t rs = GetBlocks(code.simd_width);
  int64_t repeat = strides_.back() / code.simd_width;
  op->data = rs << 18 | repeat;
  op->scalar = scalar_;
  return std::make_pair(sizeof(vBinaryS), 1);
}

BinaryOp::BinaryOp(int op_type, NDObject *lhs, NDObject *rhs)
    : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kBinary) {
  static const vOpInsnID id_list[][kTypeEnd] = {  // must keep consistent order with BinaryOpType
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_CMP, V_NONE},
    {V_NONE, V_ADD_FP16, V_ADD, V_NONE},
    {V_NONE, V_SUB_FP16, V_SUB, V_NONE},
    {V_NONE, V_MUL_FP16, V_MUL, V_NONE},
    {V_NONE, V_DIV_FP16, V_DIV, V_NONE},
    {V_NONE, V_POW_FP16, V_POW, V_NONE},
    {V_NONE, V_MAX_FP16, V_MAX, V_NONE},
    {V_NONE, V_MIN_FP16, V_MIN, V_NONE},
    {V_AND_INT8, V_AND_FP16, V_AND, V_NONE},
    {V_OR_INT8, V_OR_FP16, V_OR, V_NONE}};
  id_ = id_list[op_type][type_id_];
  ASSERT(id_ != V_NONE);
  // compare op in BinaryOpType must keep consistent order with vCompareType
  cmp_op_ = op_type < V_CMP_ALL ? op_type : -1;
  shape_ref_ = new ShapeRef();
}

BinaryOp::~BinaryOp() {
  for (auto op : lhs_stuff_ops_) {
    delete op;
  }
  for (auto op : rhs_stuff_ops_) {
    delete op;
  }
  delete shape_ref_;
}

void BinaryOp::Normalize(std::vector<NDObject*> &run_ops) {
  // recover original input
  if (!lhs_stuff_ops_.empty()) {
    lhs_ = lhs_stuff_ops_[0]->lhs_;
  }
  if (!rhs_stuff_ops_.empty()) {
    rhs_ = rhs_stuff_ops_[0]->lhs_;
  }
  // update shape_ref_
  auto lhs_data = lhs_->shape_ref_->data;
  auto rhs_data = rhs_->shape_ref_->data;
  auto lhs_sz = lhs_->shape_ref_->size;
  auto rhs_sz = rhs_->shape_ref_->size;
  if (lhs_sz > rhs_sz) {
    shape_.resize(lhs_sz);
    auto diff = lhs_sz - rhs_sz;
    for (size_t i = 0; i < diff; ++i) {
      shape_[i] = lhs_data[i];
    }
    for (size_t i = diff; i < lhs_sz; ++i) {
      shape_[i] = lhs_data[i] == 1 ? rhs_data[i - diff]: lhs_data[i];
    }
  } else {
    shape_.resize(rhs_sz);
    auto diff = rhs_sz - lhs_sz;
    for (size_t i = 0; i < diff; ++i) {
      shape_[i] = rhs_data[i];
    }
    for (size_t i = diff; i < rhs_sz; ++i) {
      shape_[i] = rhs_data[i] == 1 ? lhs_data[i - diff]: rhs_data[i];
    }
  }
  *shape_ref_ = shape_;
  // update nd_
  const auto &lhs_nd = lhs_->nd_;
  const auto &rhs_nd = rhs_->nd_;
  auto lhs_dim = lhs_nd.size();
  auto rhs_dim = rhs_nd.size();
  auto res_dim = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
  nd_.resize(res_dim, 1);
  bool lhs_need_broadcast = false;
  bool rhs_need_broadcast = false;
  for (size_t i = 0; i < res_dim; ++i) {
    auto lhs_axis = i < lhs_dim ? lhs_nd[i] : 1;
    auto rhs_axis = i < rhs_dim ? rhs_nd[i] : 1;
    if (lhs_axis > rhs_axis) {
      nd_[i] = lhs_axis;
      rhs_need_broadcast = true;
    } else if (lhs_axis < rhs_axis) {
      nd_[i] = rhs_axis;
      lhs_need_broadcast = true;
    } else {
      nd_[i] = lhs_axis;
    }
  }
  if (lhs_need_broadcast) {
    size_t stuff_idx = 0;
    lhs_ = InsertImplicitBroadcast(lhs_, nd_, lhs_stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      lhs_stuff_ops_[i]->index_ = run_ops.size();
      run_ops.push_back(lhs_stuff_ops_[i]);
    }
  }
  if (rhs_need_broadcast) {
    size_t stuff_idx = 0;
    rhs_ = InsertImplicitBroadcast(rhs_, nd_, rhs_stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      rhs_stuff_ops_[i]->index_ = run_ops.size();
      run_ops.push_back(rhs_stuff_ops_[i]);
    }
  }
}

std::pair<uint32_t, uint32_t> BinaryOp::Emit(const Code &code) {
  if (id_ == V_CMP || id_ == V_CMP_FP16) { // TODO: use child class of BinaryOp
    vCompare op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.xm = rhs_->xbuf_;
    op.type = cmp_op_;
    op.repeat = strides_.back() / code.simd_width;
    auto size = vCompare::Encode(insn, id_, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  } else {
    vBinary op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.xm = rhs_->xbuf_;
    op.repeat = strides_.back() / code.simd_width;
    auto size = vBinary::Encode(insn, id_, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  }
}

SelectOp::~SelectOp() {
  if (cond_stuff_ != nullptr) {
    delete cond_stuff_;
  }
}

void SelectOp::Normalize(std::vector<NDObject*> &run_ops) {
  auto cond = cond_stuff_ == nullptr ? cond_ : cond_stuff_->Input();
  if (cond->type_id_ == kInt8) {
    if (cond_stuff_ == nullptr) {
      cond_stuff_  = new CastOp(cond, lhs_->type_id_);
    }
    cond_stuff_->Normalize(run_ops);
    cond_stuff_->index_ = run_ops.size();
    run_ops.push_back(cond_stuff_);
    cond_ = cond_stuff_;
  }
  nd_ = lhs_->nd_;
}

std::pair<uint32_t, uint32_t> SelectOp::Emit(const Code &code) {
  vSelect *op = reinterpret_cast<vSelect *>(insn);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  op->head = MakeHead(V_PIPE_SIMD, sizeof(vSelect), ext, type_id_ == kFloat32? V_SEL : V_SEL_FP16);
  uint64_t stride = GetBlocks(code.simd_width);
  int64_t repeat = strides_.back() / code.simd_width;
  op->data = stride << 60 | repeat << 36 | rhs_->xbuf_ << 18 | cond_->xbuf_;
  return std::make_pair(sizeof(vSelect), 1);
}

int _BroadcastOp::FoldPropY(int base, int depth) {
  int state = 0; // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = base; i != base - depth; --i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      return new_depth;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  return new_depth;
}

int _BroadcastOp::FoldPropX(int depth) {
  int state = 0; // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = 0; i < depth; ++i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      return new_depth;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  return new_depth;
}

std::pair<uint32_t, uint32_t> _BroadcastOp::Emit(const Code &code) {
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
    offset = EmitBroadcastX(insn, end_dim, code.simd_width);
  } else {
    offset = EmitBroadcastY(insn, start_dim, end_dim, code.simd_width);
  }
  return std::make_pair(offset, 1);
}

int64_t _BroadcastOp::EmitBroadcastX(uint64_t *p, int end_dim, int64_t simd_width) {
  vBroadcastX op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_[end_dim] / simd_width;
  int64_t rank_size = static_cast<int64_t>(strides_.size());
  op.lead_num = end_dim + 1 < rank_size ? nd_[end_dim + 1] : 1;
  op.iter_num = end_dim + 2 <  rank_size ? strides_.back() / strides_[end_dim + 1] : 1;
  auto size = vBroadcastX::Encode(p, type_id_ == kFloat32 ? V_BROADCAST_X:V_BROADCAST_X_FP16, op);
  return size * sizeof(uint64_t); 
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
  auto size = vBroadcastY::Encode(p, V_BROADCAST_Y, op);
  return size * sizeof(uint64_t); 
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
  auto dims = shape_ref_->size;
  nd_.resize(dims);
  for (size_t i = 0; i < dims; ++i) {
    nd_[i] = shape_ref_->data[dims - i - 1];
  }
  size_t stuff_idx = 0;
  lhs_ = InsertBroadcastOpsInBetween(lhs_, nd_, stuff_ops_, stuff_idx);
  for (size_t i = 0; i < stuff_idx; ++i) {
    stuff_ops_[i]->index_ = run_ops.size();
    run_ops.push_back(stuff_ops_[i]);
  }
}

std::pair<uint32_t, uint32_t> BroadcastScalarOp::Emit(const Code &code) {
  vBroadcastS *op = reinterpret_cast<vBroadcastS *>(insn);
  op->head = MakeHead(V_PIPE_SIMD, sizeof(vBroadcastS), xbuf_, type_id_ == kFloat32 ? V_BROADCAST_S : V_BROADCAST_S_FP16);
  op->scalar = this->scalar_;
  uint64_t stride = GetBlocks(code.simd_width);
  int64_t repeat = strides_.back() / code.simd_width;
  op->data = stride << 18 | repeat;
  return std::make_pair(sizeof(vBroadcastS), 1);
}

int _ReduceOp::FoldPropY(int base, int depth) {
  int state = 0; // -1 - reduce ; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = base; i != base - depth; --i) {
    if ((state == -1 && nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      return new_depth;
    }
    if (state == 0) {
      if (nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  return new_depth;
}

int _ReduceOp::FoldPropX(int depth) {
  int state = 0; // -1 - reduce; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = 0; i < depth; ++i) {
    if ((state == -1 && nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != nd_[i])) {
      return new_depth;
    }
    if (state == 0) {
      if (nd_[i] > 1) state = 1;
      else if (lhs_->nd_[i] != nd_[i]) state = -1;
    }
    new_depth++;
  }
  return new_depth;
}

void _ReduceOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

std::pair<uint32_t, uint32_t> _ReduceOp::Emit(const Code &code) {
  ASSERT(red_op_ == ReduceOp::SUM);
  if (start_dim_ <= lhs_->lead_dim_) { // reduce x
    uint32_t size = 0;
    uint32_t insn_num = 1;
    if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
      size = EmitClearPad(insn, lhs_, code.simd_width);
      tail_insn = insn + size;
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
    size += vReduceX::Encode(tail_insn, type_id_ == kFloat32 ? V_RSUM_X : V_RSUM_X_FP16, op);
    if (insn_num > 1) {
      *(tail_insn) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
    return std::make_pair(size*sizeof(uint64_t), insn_num);
  } else {
    vReduceY op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.iter_size = strides_[start_dim_];
    op.red_size = lhs_->strides_[end_dim_] / op.iter_size;
    if (InRange(tail_dim_)) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = strides_.back() / strides_[end_dim_];
    uint32_t size = vReduceY::Encode(insn, type_id_ == kFloat32 ? V_RSUM_Y : V_RSUM_Y_FP16, op);
    return std::make_pair(size * sizeof(uint64_t), 1);
  }
}

ReduceOp::~ReduceOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
  delete shape_ref_;
}

void ReduceOp::Normalize(std::vector<NDObject*> &run_ops) {
  // update shape_ref_
  auto input_shape_ref = stuff_ops_.empty() ? lhs_->shape_ref_ : stuff_ops_[0]->lhs_->shape_ref_;
  shape_.clear();
  shape_.reserve(input_shape_ref->size);
  int dim_idx = 0;
  for (int i = 0; i < static_cast<int>(input_shape_ref->size); ++i) {
    if (i != shape_dims_[dim_idx]) {
      shape_.push_back(input_shape_ref->data[i]);
    } else {
      dim_idx++;
      if (keepdims_) {
        shape_.push_back(1);
      }
    }
  }
  *shape_ref_ = shape_;
  size_t stuff_idx = 0;
  int red_start = dims_.front();
  int red_ext = dims_.front() + 1;
  NDObject *input = lhs_;
  nd_ = input->nd_;
  bool real_reduce = nd_[red_start] > 1;
  nd_[red_start] = 1;
  while (red_ext < static_cast<int>(nd_.size()) && nd_[red_ext] == 1) red_ext++;
  for (size_t i = 1; i < dims_.size(); ++i) {
    auto d = dims_[i];
    if (d < red_ext) continue;
    if (d > red_ext && real_reduce) {
      if (stuff_idx == stuff_ops_.size()) {
        stuff_ops_.push_back(new _ReduceOp(input, red_op_));
      }
      auto obj = stuff_ops_[stuff_idx++];
      input = obj;
      std::swap(obj->nd_, nd_);
      nd_ = obj->nd_;
      obj->index_ = run_ops.size();
      obj->SetRange(red_start, red_ext-1);
      red_start = d;
      run_ops.push_back(obj);
      real_reduce = false;
    }
    real_reduce = real_reduce || nd_[d] > 1;
    nd_[d] = 1;
    red_ext = d + 1;
    while (red_ext < static_cast<int>(nd_.size()) && nd_[red_ext] == 1) red_ext++;
  }
  lhs_ = input;
  SetRange(red_start, red_ext-1);
}

void ReduceOp::Tile(const TileParam &tp) {
  if (!(tp.end < start_dim_ || tp.start > end_dim_)) {
    if (round_ == 0) {
      factor_ = tp.num;
      round_ = 1;
    } else {
      ASSERT(round_ == 1); // restrict: only continuous reduce
      factor_ *= tp.num;
    }
  } else if (factor_ > 1) {
    round_ *= tp.num;
    factor_ *= tp.num;
  }
  _ReduceOp::Tile(tp);
}

std::pair<uint32_t, uint32_t> ReduceOp::Emit(const Code &code) {
  auto ret = _ReduceOp::Emit(code);
  if (factor_ > 0 && nd_[lead_dim_] != strides_[lead_dim_]) {
    tail_insn = insn + ret.first / sizeof(uint64_t);
    auto size = EmitClearPad(tail_insn, this, code.simd_width);
    *(tail_insn) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    ret.first += size * sizeof(uint64_t);
    ret.second++;
  }
  return ret;
}
} // namespace dvm
