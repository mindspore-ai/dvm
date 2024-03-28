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
#include <set>
#include <algorithm>
#include "ops.h"
#include "kernel.h"

namespace dvm {
namespace {
inline uint64_t DMAConfig(uint64_t sid, uint64_t nBurst, uint64_t lenBurst,
                          uint64_t srcStride, uint64_t dstStride) {
  return dstStride << 48 | srcStride << 32 | lenBurst << 16 | nBurst << 4 | sid;
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
  int64_t align = simd_width;
  strides_[i] = CeilDiv(nd_[i], align) * align;
  for (++i; i < nd_.size(); ++i) {
    strides_[i] = nd_[i] * strides_[i - 1];
  }
}

int NDLoadDummy::Emit(Code &code) {
  *insn_ = vMakeHead(V_LOAD_DUMMY, 0, 1, V_PIPE_LOAD);
  return 1;
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

int NDLoad::Emit(Code &code) {
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    switch (round_tile_.size()) {
      case 1: {
        auto r1 = round_tile_[0];
        rounds[0] = 0xfffffffful << 32 | r1;
        break;
      }
      case 2: {
        auto r1 = round_tile_[0] * round_tile_[1];
        auto r2 = round_tile_[1];
        rounds[0] = r2 << 32 | r1;
        break;
      }
      case 3: {
        auto r1 = round_tile_[0] * round_tile_[1] * round_tile_[2];
        auto r2 = round_tile_[1];
        auto r3 = round_tile_[2];
        rounds[0] = r2 << 32 | r1;
        rounds[1] = 0xfffffffful << 32 | r3;
        break;
      }
      case 4: {
        auto r1 = round_tile_[0] * round_tile_[1] * round_tile_[2] * round_tile_[3];
        auto r2 = round_tile_[1];
        auto r3 = round_tile_[2] * round_tile_[3];
        auto r4 = round_tile_[3];
        rounds[0] = r2 << 32 | r1;
        rounds[1] = r4 << 32 | r3;
        break;
      }
      default:
        EXCEPTION_IF(true, "multi-broadcast rank exceed max limit(4)");
        break;
    }
  }
  int64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lead_align == nd_[lead_dim_] || lead_align == strides_.back()) {
    vDMA op;
    op.gm = src_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(src_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(src_tile_stride_ / nd_[tail_dim_] * tail_size_);
    op.round_rank = round_tile_.size();
    reloc_addr_ = insn_ + vDMA::RELOC_OFFSET;
    return vDMA::Encode(insn_, vLoadInsnID::V_LOAD, vPipe::V_PIPE_LOAD, op, rounds);
  } else { // align
    vLoad op;
    op.from = src_;
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
  round_tile_.clear();
}

void NDSliceLoad::AlignProp(PropRange &range) {
  range.depth = 1;
}

void NDSliceLoad::FoldProp(PropRange &range) {
  range.depth = size_ref_->size - 1;
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

int NDSliceLoad::Emit(Code &code) {
  reloc_offset_ = CalcOffset();
  if (nd_.size() == 1) {
    src_ += reloc_offset_;
    return NDLoad::Emit(code);
  }
  uint64_t lead_align = LeadAlign();
  uint64_t src_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  vSliceLoad op;
  op.gm = src_ + reloc_offset_;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_[lead_dim_];
  if (nd_.size() == 2) {
    op.slice_k = 1;
    op.slice_n = size_ref_->data[0];
    op.slice_m = size_ref_->data[1];
    op.src_n = src_ref_->data[0];
    op.src_m = src_ref_->data[1];
  } else {
    op.slice_k = size_ref_->data[0];
    op.slice_n = size_ref_->data[1];
    op.slice_m = size_ref_->data[2];
    op.src_n = src_ref_->data[1];
    op.src_m = src_ref_->data[2];
  }
  op.type_size = ITEM_SIZE[type_id_];
  reloc_addr_ = insn_ + vSliceLoad::RELOC_OFFSET;
  return vSliceLoad::Encode(insn_, vLoadInsnID::V_SLICE_LOAD, op);
}

void NDStridedSliceLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(std::all_of(step_ref_->data, step_ref_->data + step_ref_->size, [](int64_t i) { return i == 1; }));
  shape_.resize(src_ref_->size);
  for (size_t i = 0; i < src_ref_->size; i++) {
    int64_t end = end_ref_->data[i] < 0 ? end_ref_->data[i] + src_ref_->data[i] : end_ref_->data[i];
    int64_t start = start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i];
    shape_[i] = end - start;
  }
  *shape_ref_ = shape_;
  size_ref_ = shape_ref_;
  NDSliceLoad::Normalize(run_ops);
}

NDStore::~NDStore() {
  if (clear_kernel_ != nullptr) {
    delete clear_kernel_;
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

int NDStore::Emit(Code &code) {
  uint64_t lead_align = LeadAlign();
  ASSERT(lead_align == static_cast<uint64_t>(lhs_->LeadAlign()));
  uint64_t dst_tile_stride_ = strides_.back() / lead_align * nd_[lead_dim_];
  if (lhs_->obj_id_ == kElementAny) {
    vStoreStatus *op = reinterpret_cast<vStoreStatus *>(insn_);
    op->to = dst_;
    op->head = vMakeHead(V_STORE_STATUS, lhs_->xbuf_, sizeof(vStoreStatus) / sizeof(uint64_t), V_PIPE_STORE);
    reloc_addr_ = insn_ + vStoreStatus::RELOC_OFFSET;
    return sizeof(vStoreStatus) / sizeof(uint64_t);
  } else if (lhs_->obj_id_ == kReduce || (lhs_->obj_id_ == kRemovePad && lhs_->lhs_->obj_id_ == kReduce)) {
    auto reduce_op = lhs_->obj_id_ == kRemovePad ? lhs_->lhs_ : lhs_;
    auto red_op = static_cast<ReduceOp *>(reduce_op);
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
      if (lhs_->obj_id_ == kRemovePad) {
        op.pad_size = 0;
      }
      if (clear_kernel_ == nullptr) {
        clear_kernel_ = new VKernelD();
        auto dummy_load = new NDLoadDummy(type_id_);
        clear_kernel_->Append(dummy_load);
        auto broadcast_scalar_op = new BroadcastScalarOp<float>(0.0, shape_ref_, type_id_, dummy_load);
        clear_kernel_->Append(broadcast_scalar_op);
        clear_store_ = new NDStore(dst_, broadcast_scalar_op);
        clear_kernel_->Append(clear_store_);
      }
      clear_kernel_->CodeGen();
      code.atomic_clean_.push_back(clear_kernel_->GetCode());
      reloc_addr_ = insn_ + vStoreAtomic::RELOC_OFFSET;
      return vStoreAtomic::Encode(insn_, V_STORE_ATOMIC, op);
    }
  }
  if (lead_align == static_cast<uint64_t>(lhs_->nd_[lhs_->lead_dim_])) {
    vDMA op;
    op.gm = dst_;
    op.xn = lhs_->xbuf_;
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.lenburst = GetBlocks(dst_tile_stride_);
    op.tail_lenburst = tail_dim_ < 0 ? op.lenburst : GetBlocks(dst_tile_stride_/ nd_[tail_dim_] * tail_size_);
    op.round_rank = 0;
    reloc_addr_ = insn_ + vDMA::RELOC_OFFSET;
    return vDMA::Encode(insn_, vStoreInsnID::V_STORE, vPipe::V_PIPE_STORE, op, nullptr);
  } else {
    vStore *op = reinterpret_cast<vStore*>(insn_);
    uint64_t ext = (dst_tile_stride_ * ITEM_SIZE[type_id_]) << V_X_BITS | lhs_->xbuf_;
    op->head = vMakeHead(vStoreInsnID::V_STORE_2, ext, sizeof(vStore) / sizeof(uint64_t), V_PIPE_STORE);
    op->to = dst_;
    uint64_t iter_size = nd_[lead_dim_] * ITEM_SIZE[type_id_];
    uint64_t pad_size = lead_align * ITEM_SIZE[type_id_] - iter_size;
    uint64_t body_iter = strides_.back() / lead_align;
    uint64_t lead_tiling, tail_iter;
    if (lhs_->obj_id_ == ObjectType::kRemovePad) {
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
    op->config = lead_tiling << 62 | pad_size << 54 | iter_size << 36 | tail_iter << 18 | body_iter;
    reloc_addr_ = insn_ + vStore::RELOC_OFFSET;
    return sizeof(vStore) / sizeof(uint64_t);
  }
}

int CopyOp::Emit(Code &code) {
  vCopy *op = reinterpret_cast<vCopy*>(insn_);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  op->head = vMakeHead(V_COPY, ext, sizeof(vCopy) / sizeof(uint64_t), V_PIPE_SIMD);
  uint64_t lenburst = GetBlocks(strides_.back());
  op->config = DMAConfig(0, 1, lenburst, 0, 0);
  return sizeof(vCopy) / sizeof(uint64_t);
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

int ReshapeOp::Emit(Code &code) {
  EXCEPTION_IF(lhs_->nd_[lhs_->lead_dim_] != nd_[lead_dim_],"Reshape with diffrent leading pad is not support");
  return CopyOp::Emit(code);
}

UnaryOp::UnaryOp(int op_type, NDObject *input) : NDObject(input, nullptr, input->type_id_, ObjectType::kUnary) {
  static const vSimdInsnID id_list[][kTypeEnd] = {
    // must keep consistent order with UnaryOpType
    {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE},
    {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_NONE},
    {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE},
    {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE},
    {V_NONE, V_REC_FP16, V_NONE, V_REC, V_NONE},
    {V_NONE, V_ISFINITE_FP16, V_NONE, V_ISFINITE, V_NONE},
    {V_NOT_INT8, V_NONE, V_NONE, V_NONE, V_NONE}};
  id_ = id_list[op_type][type_id_];
  ASSERT(id_ != V_NONE);
  shape_ref_ = input->shape_ref_;
}

int UnaryOp::Emit(Code &code) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / code.simd_width_;
  return vUnary::Encode(insn_, id_, op);
}

RemovePadOp::RemovePadOp(NDObject *input) : CopyOp(input) {
  ASSERT(ITEM_SIZE[type_id_] != 1);
  obj_id_ = ObjectType::kRemovePad;
}

int RemovePadOp::Emit(Code &code) {
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_REMOVEPAD_U16, V_REMOVEPAD_U16, V_REMOVEPAD, V_REMOVEPAD};
  if (nd_[lead_dim_] == strides_[lead_dim_] || strides_.back() == strides_[lead_dim_]) {
    return CopyOp::Emit(code);
  }
  vRemovePad op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / strides_[lead_dim_];
  op.iter_num = nd_[lead_dim_];
  op.rs = GetBlocks(strides_[lead_dim_]);
  return vRemovePad::Encode(insn_, id_list[type_id_], op);
}

void ElementAnyOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int ElementAnyOp::Emit(Code &code) {
  uint32_t insn_num = 1;
  uint32_t size = 0;
  if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
    size = EmitClearPad(insn_, lhs_, code.simd_width_);
    tail_insn_ = insn_ + size;
    insn_num++;
  }
  vElementAny op;
  op.xn = lhs_->xbuf_ ; 
  op.xd = xbuf_;
  op.rs = GetBlocks(code.simd_width_);
  op.iter_size = lhs_->strides_.back();
  op.tail_size = tail_dim_ < 0
                    ? lhs_->strides_.back()
                    : lhs_->strides_.back() / lhs_->nd_[tail_dim_] * tail_size_;

  op.repeat = lhs_->strides_.back() / code.simd_width_;
  size += vElementAny::Encode(tail_insn_, type_id_ == kFloat32 ? V_ELEMENT_ANY : V_ELEMENT_ANY_FP16, op);

  if (insn_num > 1) {
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return size;
}

int CastOp::Emit(Code &code) {
  static const vSimdInsnID id_list[][kTypeEnd] = {
    {V_NONE, V_CAST_INT8_TO_FP16, V_NONE, V_NONE, V_NONE},                             // V_INT8
    {V_CAST_FP16_TO_INT8, V_NONE, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32},  // V_FLOAT16
    {V_NONE, V_NONE, V_NONE, V_CAST_BF16_TO_FP32, V_CAST_BF16_TO_INT32},               // V_BFLOAT16
    {V_NONE, V_CAST_FP32_TO_FP16, V_CAST_FP32_TO_BF16, V_NONE, V_CAST_FP32_TO_INT32},  // V_FLOAT32
    {V_NONE, V_CAST_INT32_TO_FP16, V_NONE, V_CAST_INT32_TO_FP32, V_NONE},              // V_INT32
  };
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = strides_.back() / code.simd_width_;
  return vUnary::Encode(insn_, id_list[lhs_->type_id_][type_id_], op);
}

template <typename T>
BinaryScalarOp<T>::BinaryScalarOp(int op_type, NDObject *input, T scalar)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kBinaryS), scalar_(scalar) {
  static const vSimdInsnID id_list[][kTypeEnd] = {  // must keep consistent order with BinarySOpType
    {V_NONE, V_ADDS_FP16, V_NONE, V_ADDS, V_ADDS_INT32},
    {V_NONE, V_MULS_FP16, V_NONE, V_MULS, V_MULS_INT32},
    {V_NONE, V_MAXS_FP16, V_NONE, V_MAXS, V_MAXS_INT32},
    {V_NONE, V_MINS_FP16, V_NONE, V_MINS, V_MINS_INT32}};
  id_ = id_list[op_type][type_id_];
  ASSERT(id_ != V_NONE);
  shape_ref_ = input->shape_ref_;
}

template <typename T>
int BinaryScalarOp<T>::Emit(Code &code) {
  vBinaryS<T> *op = reinterpret_cast<vBinaryS<T> *>(insn_);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  op->head = vMakeHead(id_, ext, sizeof(vBinaryS<T>) / sizeof(uint64_t), V_PIPE_SIMD);
  uint64_t rs = GetBlocks(code.simd_width_);
  int64_t repeat = strides_.back() / code.simd_width_;
  op->data = rs << 18 | repeat;
  op->scalar = scalar_;
  return sizeof(vBinaryS<T>) / sizeof(uint64_t);
}

template class BinaryScalarOp<float>;
template class BinaryScalarOp<int32_t>;

BinaryOp::BinaryOp(int op_type, NDObject *lhs, NDObject *rhs) : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kBinary) {
  static const vSimdInsnID id_list[][kTypeEnd] = {
    // must keep consistent order with BinaryOpType
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE},
    {V_NONE, V_ADD_FP16, V_NONE, V_ADD, V_ADD_INT32},
    {V_NONE, V_SUB_FP16, V_NONE, V_SUB, V_SUB_INT32},
    {V_NONE, V_MUL_FP16, V_NONE, V_MUL, V_MUL_INT32},
    {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE},
    {V_NONE, V_POW_FP16, V_NONE, V_POW, V_NONE},
    {V_NONE, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32},
    {V_NONE, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32},
    {V_AND_INT8, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32},
    {V_OR_INT8, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32}};
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
      run_ops.push_back(lhs_stuff_ops_[i]);
    }
  }
  if (rhs_need_broadcast) {
    size_t stuff_idx = 0;
    rhs_ = InsertImplicitBroadcast(rhs_, nd_, rhs_stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(rhs_stuff_ops_[i]);
    }
  }
}

int BinaryOp::Emit(Code &code) {
  if (id_ == V_CMP || id_ == V_CMP_FP16) { // TODO: use child class of BinaryOp
    vCompare op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.xm = rhs_->xbuf_;
    op.type = cmp_op_;
    op.repeat = strides_.back() / code.simd_width_;
    return vCompare::Encode(insn_, id_, op);
  } else {
    vBinary op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.xm = rhs_->xbuf_;
    op.repeat = strides_.back() / code.simd_width_;
    return vBinary::Encode(insn_, id_, op);
  }
}

int SelectOp::Emit(Code &code) {
  vSelect *op = reinterpret_cast<vSelect *>(insn_);
  uint64_t ext = xbuf_ << V_X_BITS | lhs_->xbuf_;
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_SEL_FP16, V_NONE, V_SEL, V_SEL_INT32};
  op->head = vMakeHead(id_list[type_id_], ext, sizeof(vSelect) / sizeof(uint64_t), V_PIPE_SIMD);
  uint64_t stride = GetBlocks(code.simd_width_);
  int64_t repeat = strides_.back() / code.simd_width_;
  op->data = stride << 60 | repeat << 36 | rhs_->xbuf_ << 18 | cond_->xbuf_;
  return sizeof(vSelect) / sizeof(uint64_t);
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

int _BroadcastOp::Emit(Code &code) {
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
    offset = EmitBroadcastX(insn_, end_dim, code.simd_width_);
  } else {
    offset = EmitBroadcastY(insn_, start_dim, end_dim, code.simd_width_);
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
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_BROADCAST_X_FP16, V_NONE, V_BROADCAST_X, V_BROADCAST_X_INT32};
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
    run_ops.push_back(stuff_ops_[i]);
  }
}

template <typename T>
int BroadcastScalarOp<T>::Emit(Code &code) {
  vBroadcastS<T> *op = reinterpret_cast<vBroadcastS<T> *>(insn_);
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_BROADCAST_S_FP16, V_NONE, V_BROADCAST_S, V_BROADCAST_S_INT32};
  op->head = vMakeHead(id_list[type_id_], xbuf_, sizeof(vBroadcastS<T>) / sizeof(uint64_t), V_PIPE_SIMD);
  op->scalar = scalar_;
  uint64_t stride = GetBlocks(code.simd_width_);
  int64_t repeat = strides_.back() / code.simd_width_;
  op->data = stride << 18 | repeat;
  return sizeof(vBroadcastS<T>) / sizeof(uint64_t);
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

int _ReduceOp::Emit(Code &code) {
  ASSERT(red_op_ == ReduceOp::SUM);
  if (start_dim_ <= lhs_->lead_dim_) { // reduce x
    uint32_t size = 0;
    uint32_t insn_num = 1;
    if (lhs_->nd_[lhs_->lead_dim_] != lhs_->strides_[lhs_->lead_dim_]) {
      size = EmitClearPad(insn_, lhs_, code.simd_width_);
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
    if (InRange(tail_dim_)) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = strides_.back() / strides_[end_dim_];
    return vReduceY::Encode(insn_, V_RSUM_Y, op);
  }
}

ReduceOp::~ReduceOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
  delete shape_ref_;
}

void ReduceOp::Normalize(std::vector<NDObject*> &run_ops) {
  ASSERT(dims_ref_ != nullptr);
  factor_ = 0;
  round_ = 0;
  NDObject *input = stuff_ops_.empty() ? lhs_ : stuff_ops_[0]->lhs_;
  auto input_shape_ref = input->shape_ref_;
  //update dims
  shape_dims_.resize(dims_ref_->size);
  for (size_t i = 0; i < dims_ref_->size; i++) {
    shape_dims_[i] = dims_ref_->data[i];
  }
  auto lhs_dim = input_shape_ref->size;
  if (shape_dims_.empty()) {
    shape_dims_.resize(lhs_dim);
    dims_.resize(lhs_dim);
    for (int64_t i = 0; i < static_cast<int64_t>(lhs_dim); ++i) {
      shape_dims_[i] = i;
      dims_[i] = i;
    }
  } else {
    std::set<int64_t> dims_set;
    std::for_each(shape_dims_.begin(), shape_dims_.end(), [&dims_set, lhs_dim](int64_t &n) {
      if (n < 0) {
        n += lhs_dim;
      }
      dims_set.insert(n);
    });
    shape_dims_.assign(dims_set.begin(), dims_set.end());
    int back_idx = shape_dims_.back() + 1;
    int back_end = input->nd_.size() - 1;
    while (back_idx <= back_end && input->nd_[back_end - back_idx] == 1) { // align fold may flip dims
      shape_dims_.push_back(back_idx++);
    }
    auto size = shape_dims_.size();
    dims_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      dims_[i] = lhs_dim - shape_dims_[size - i - 1] - 1;
    }
  }
  // update shape_ref_
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
      auto end = std::min(end_dim_, tp.end);
      for (auto i = std::max(start_dim_, tp.start); i <= end; ++i) {
        if (lhs_->nd_[i] > 1) {
          factor_ = tp.num;
          round_ = 1;
          break;
        }
      }
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

int ReduceOp::Emit(Code &code) {
  auto num = _ReduceOp::Emit(code);
  if (DeviceInfo::Instance().Arch() != kAiCore_C220 && factor_ > 0 &&
      nd_[lead_dim_] != strides_[lead_dim_]) {
    tail_insn_ = insn_ + num;
    auto size = EmitClearPad(tail_insn_, this, code.simd_width_);
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    num += size;
  }
  return num;
}
} // namespace dvm
