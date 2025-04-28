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
#include "comm.h"
#include "tuning.h"

namespace dvm {
namespace {
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 7;
constexpr uint32_t DEFAULT_DIAGONAL_SWIZZLE_COUNT = 8;
constexpr uint32_t MAX_BIAS_SIZE = 1024;
constexpr int64_t MAX_SPLIT_K = 20480;
constexpr int64_t MIN_SPLIT_K = 4096;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_32 = 32;
const size_t MAX_SLICE_DIM = 3;

struct InsnIdTable {
  const char *name;
  vSimdInsnID ids[kTypeEnd];
};

static const InsnIdTable unary_id_list[kUnaryOpEnd] = {
  // must keep consistent order with UnaryOpType
  {"Sqrt", {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE}},
  {"Abs", {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_NONE}},
  {"Log", {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE}},
  {"Exp", {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE}},
  {"Reciprocal", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"IsFinite", {V_NONE, V_ISFINITE_FP16, V_NONE, V_ISFINITE, V_NONE}},
  {"LogicalNot", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"Round", {V_NONE, V_NONE, V_NONE, V_ROUND, V_NONE}},
  {"Floor", {V_NONE, V_NONE, V_NONE, V_FLOOR, V_NONE}},
  {"Ceil", {V_NONE, V_NONE, V_NONE, V_CEIL, V_NONE}},
  {"Trunc", {V_NONE, V_NONE, V_NONE, V_TRUNC, V_NONE}}};

static const InsnIdTable binary_id_list[] = {
  // must keep consistent order with BinaryOpType
  {"Equal", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_CMP_INT32}},
  {"NotEqual", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_CMP_INT32}},
  {"Greater", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"GreaterEqual", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"Less", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"LessEqual", {V_NONE, V_CMP_FP16, V_NONE, V_CMP, V_NONE}},
  {"Add", {V_NONE, V_ADD_FP16, V_NONE, V_ADD, V_ADD_INT32}},
  {"Sub", {V_NONE, V_SUB_FP16, V_NONE, V_SUB, V_SUB_INT32}},
  {"Mul", {V_NONE, V_MUL_FP16, V_NONE, V_MUL, V_MUL_INT32}},
  {"Div", {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE}},
  {"Pow", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},  // power: individual implement
  {"Maximum", {V_NONE, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32}},
  {"Minimum", {V_NONE, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32}},
  {"LogicalAnd", {V_NONE, V_MIN_FP16, V_NONE, V_MIN, V_MIN_INT32}},
  {"LogicalOr", {V_NONE, V_MAX_FP16, V_NONE, V_MAX, V_MAX_INT32}}};

static const InsnIdTable binarys_id_list[] = {
  // must keep consistent order with BinarySOpType
  {"Equal", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"NotEqual", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Greater", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"GreaterEqual", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Less", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"LessEqual", {V_NONE, V_CMPS_FP16, V_NONE, V_CMPS, V_NONE}},
  {"Add", {V_NONE, V_ADDS_FP16, V_NONE, V_ADDS, V_ADDS_INT32}},
  {"Mul", {V_NONE, V_MULS_FP16, V_NONE, V_MULS, V_MULS_INT32}},
  {"Div", {V_NONE, V_SDIV_FP16, V_NONE, V_SDIV, V_NONE}},
  {"Maximum", {V_NONE, V_MAXS_FP16, V_NONE, V_MAXS, V_MAXS_INT32}},
  {"Minimum", {V_NONE, V_MINS_FP16, V_NONE, V_MINS, V_MINS_INT32}}};

static const vSimdInsnID cast_id_list[][kTypeEnd] = {
  {V_NONE, V_CAST_BOOL_TO_FP16, V_NONE, V_NONE, V_NONE},                             // V_BOOL
  {V_CAST_FP16_TO_BOOL, V_NONE, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE, V_CAST_BF16_TO_FP32, V_CAST_BF16_TO_INT32},               // V_BFLOAT16
  {V_NONE, V_CAST_FP32_TO_FP16, V_CAST_FP32_TO_BF16, V_NONE, V_CAST_FP32_TO_INT32},  // V_FLOAT32
  {V_NONE, V_CAST_INT32_TO_FP16, V_NONE, V_CAST_INT32_TO_FP32, V_NONE},              // V_INT32
};

inline uint64_t DMAConfig(uint64_t sid, uint64_t nBurst, uint64_t lenBurst, uint64_t srcStride, uint64_t dstStride) {
  return dstStride << 48 | srcStride << 32 | lenBurst << 16 | nBurst << 4 | sid;
}

template <typename T>
inline uint32_t EncodeScalar(T scalar) {
  union Scalar {
    T val;
    uint32_t encode;
  } data;
  data.val = scalar;
  return data.encode;
}

template <typename T>
inline uint32_t EncodeScalar(T scalar, DType type) {
  switch (type) {
    case kFloat16:
      return static_cast<Float16>(static_cast<float>(scalar)).int_value();
    case kBFloat16:
      return static_cast<BFloat16>(static_cast<float>(scalar)).int_value();
    case kFloat32:
      return EncodeScalar(static_cast<float>(scalar));
    case kInt32:
      return EncodeScalar(static_cast<int32_t>(scalar));
    default:
      return 0;
  }
}

int EmitCopy(bcodeptr_t insn, uint64_t xd, uint64_t xn, uint64_t bytes) {
  vCopy op;
  op.xd = xd;
  op.xn = xn;
  op.config = DMAConfig(0, 1, (bytes + 31) >> 5, 0, 0);
  return vCopy::Encode(insn, V_COPY, op);
}

NDObject *GetBroadcastOp(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject *> &stuff_ops,
                         size_t &stuff_idx) {
  dvm::_BroadcastOp *broadcast_op = nullptr;
  if (stuff_idx < stuff_ops.size()) {
    broadcast_op = static_cast<dvm::_BroadcastOp *>(stuff_ops[stuff_idx]);
    broadcast_op->lhs_ = obj;
  } else {
    broadcast_op = new dvm::_BroadcastOp(obj);
    stuff_ops.push_back(broadcast_op);
  }
  broadcast_op->ndd_.dims = dst_shape;
  stuff_idx++;
  return broadcast_op;
}

NDObject *InsertBroadcastOpsInBetween(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject *> &stuff_ops,
                                      size_t &stuff_idx) {
  // output shape is not dst_shape, but the last inbetween shape, which just need only one broadcast op to reach the
  // dst_shape
  bool broadcast_flag = false;
  auto temp_shape = obj->nd_.dims();  // TODO: inplace optimize
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

NDObject *InsertImplicitBroadcast(NDObject *obj, const DimArray &dst_shape, std::vector<NDObject *> &stuff_ops,
                                  size_t &stuff_idx) {
  // output shape is dst_shape
  auto new_input = InsertBroadcastOpsInBetween(obj, dst_shape, stuff_ops, stuff_idx);
  auto last_broadcast_op = GetBroadcastOp(new_input, dst_shape, stuff_ops, stuff_idx);
  return last_broadcast_op;
}

int64_t SelectSimdWidth(int64_t iter_size, DType type_id) {
  auto block_width = ITEM_SIMD_WIDTH_MAX[type_id] >> 3;
  ASSERT(iter_size % block_width == 0);
  auto iter_block = iter_size / block_width;
  if (iter_block <= 8) return iter_size;
  auto simd_block = 8;
  auto repeat = iter_block / simd_block;
  while (repeat * simd_block != iter_block) {
    repeat = iter_block / (--simd_block);
  }
  return simd_block * block_width;
}

uint32_t EmitClearPad(uint64_t *pc, NDObject *op, uint64_t iter_tail) {
  vClearPad clr_op;
  clr_op.xd = op->xbuf_;
  clr_op.iter_size = op->nd_.lead_dim();
  clr_op.iter_stride = op->nd_.lead_stride();
  clr_op.iter_num = op->nd_.stride_back() / clr_op.iter_stride;
  clr_op.simd_width = SelectSimdWidth(clr_op.iter_stride, op->type_id_);
  clr_op.iter_tail = iter_tail;
  return vClearPad::Encode(pc, V_CLR_PAD, clr_op);
}

void UpdateRoundTile(bool pointwise, int64_t num, DimArray &round_tile) {
  if (!pointwise) {
    if (round_tile.size() % 2 == 0) {
      round_tile.push_back(num);
    } else {
      round_tile.back() *= num;
    }
  } else {
    if (!round_tile.empty()) {
      if (round_tile.size() % 2 == 0) {
        round_tile.back() *= num;
      } else {
        round_tile.push_back(num);
      }
    }
  }
}

bool CollectRoundTile(uint32_t elem_mask, const TileParam &tp, DimArray &round_tile) {
  bool pointwise = false;
  if (tp.num > 1) {
    pointwise = (elem_mask >> tp.start) & ((2u << (tp.end - tp.start)) - 1);
    UpdateRoundTile(pointwise, tp.num, round_tile);
  }
  return pointwise;
}

bool CollectRoundTile(const DimArray &nd, const TileParam &tp, DimArray &round_tile) {
  bool pointwise = false;
  if (tp.num > 1) {
    for (int i = tp.start; i <= tp.end; ++i) {
      if (nd[i] != 1) {
        pointwise = true;
        break;
      }
    }
    UpdateRoundTile(pointwise, tp.num, round_tile);
  }
  return pointwise;
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

std::ostream &operator<<(std::ostream &oss, const ShapeRef &shape) {
  oss << "[";
  for (size_t i = 0; i < shape.size; i++) {
    if (i) {
      oss << ",";
    }
    oss << shape.data[i];
  }
  oss << "]";
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const Float16 &scalar) {
  oss << static_cast<float>(scalar);
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const BFloat16 &scalar) {
  oss << static_cast<float>(scalar);
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const DimArray &nd) {
  oss << "[";
  if (nd.size() > 0) {
    for (size_t i = 0; i < nd.size() - 1; ++i) {
      oss << nd[i] << ",";
    }
    oss << nd.back();
  }
  oss << "]";
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const NDSpace &nd) {
  oss << nd.dims();
  return oss;
}

MemPool<512, 8192> NDObject::mem_pool_;

const NDObjectAttr NDObject::attrs_[ObjectType::kObjectBulk] = {
  {kGenLoad, true, false},    // LoadDummy
  {kGenLoad, true, false},    // MultiLoad
  {kGenLoad, true, false},    // Load
  {kGenStore, false, true},   // PadStore
  {kGenStore, true, true},    // Store
  {kGenComm, true, false},    // ReduceScatter
  {kGenComm, true, false},    // AllGather
  {kGenComm, true, false},    // AllGatherV2
  {kGenComm, true, false},    // AllReduce
  {kGenSimd1, true, false},   // Reshape
  {kGenSimd1, true, true},    // Copy
  {kGenSimd1, true, true},    // Unary
  {kGenSimd2, true, true},    // Binary
  {kGenSimd1, true, true},    // Cast
  {kGenSimd1, true, true},    // BinaryS
  {kGenSimd1, false, false},  // BroadcastTo
  {kGenSimd0, true, false},   // BroadcastS
  {kGenFlex, false, false},   // Reduce
  {kGenSimd3, true, true},    // Select
  {kGenSimd1, false, false},  // ElemAny
  {kGenSimd1, true, true},    // RemovePad
  {kGenFlex, true, true},     // Power
  {kGenFlex, true, true},     // Compare
  {kGenFlex, true, true},     // CompareS
};

class AtomicCleanWrap : public CodeWrap {
 public:
  AtomicCleanWrap(NDAccess *store) {
    clear_shape_.data = &clear_shape_data_;
    clear_shape_.size = 1;
    auto type = store->type_id_;
    auto dummy_load = new NDLoadDummy(type);
    kernel_.Append(dummy_load);
    auto op = new BroadcastScalarOp<float>(0.0, &clear_shape_, type, dummy_load);
    kernel_.Append(op);
    store_ = new NDStore(store->addr_.gm, op);
    kernel_.Append(store_);
  }

  void CodeGen(Code &code, NDAccess *store) {
    clear_shape_data_ = std::accumulate(store->shape_ref_->data, store->shape_ref_->data + store->shape_ref_->size, 1LL,
                                        std::multiplies{});
    if (System::Instance().deterministic_) {
      clear_shape_data_ += 32 / sizeof(float);
    }
    kernel_.CodeGen();
    code.BindOpFast(store_->addr_, store->addr_);
  }

  int LaunchWrap(void *workspace, void *stream) {
    kernel_.code_.Launch(nullptr, stream);
    return next_->LaunchWrap(workspace, stream);
  }

  bool DasWrap(std::ostringstream &oss) {
    kernel_.code_.DisAssemble(oss);
    auto ret = next_->DasWrap(oss);
    oss << std::endl;
    return ret;
  }

 private:
  VKernelD kernel_;
  NDStore *store_;
  ShapeRef clear_shape_;
  int64_t clear_shape_data_;
};

int64_t NDObject::Size() {
  return std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) *
         ITEM_SIZE[type_id_];
}

void NDObject::Tile(const TileParam &tp) {
  if (auto ndd = Ndd(); ndd != nullptr) {
    auto &dims = ndd->dims;
    bool pointwise = dims[tp.start] > 1;
    for (int i = tp.start + 1; i <= tp.end; ++i) {
      pointwise = pointwise || dims[i] > 1;
      dims[i] = 1;
    }
    if (pointwise) {  // broadcast source or reduce des: all 1, donot need to tile
      dims[tp.start] = tp.tile;
    }
  }
}

void NDObject::Shard(const ShardParam &sp) {
  if (auto ndd = Ndd(); ndd != nullptr) {
    auto &dims = ndd->dims;
    for (size_t i = sp.base; i < dims.size(); ++i) {
      dims[i] = i < static_cast<size_t>(sp.base + ShardParam::PARTIAL_SIZE) && dims[i] != 1 ? sp.tile[i - sp.base] : 1;
    }
  }
}

void NDObject::Dump(bool verbose, std::ostringstream &oss) { oss << "NDObject"; }

int NDLoadDummy::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  *insn_ = vMakeHead(V_LOAD_DUMMY, 0, 1, V_PIPE_LOAD);
  return 1;
}

void NDLoadDummy::Dump(bool verbose, std::ostringstream &oss) { oss << "LoadDummy"; }

void NDLoad::Shard(const ShardParam &sp) {
  for (int i = ndd_.size() - 1; i > sp.base + 1; --i) {
    if (auto factor = sp.dom->operator[](i); factor > 1) {
      UpdateRoundTile(ndd_[i] > 1, factor, round_tile_);
    }
  }
  flags_ |= OBJ_FLAG_LOAD_SHARD_ROUND;
  if (ndd_[sp.base + 1] == 1 && sp.tile[sp.base + 1] > 1) {
    flags_ |= OBJ_FLAG_LOAD_SHARD_BCAST1;
  }
  if (ndd_[sp.base] == 1 && sp.tile[sp.base] > 1) {
    flags_ |= OBJ_FLAG_LOAD_SHARD_BCAST0;
  }
  NDObject::Shard(sp);
}

void NDLoad::Tile(const TileParam &tp) {
  if (!(flags_ & OBJ_FLAG_LOAD_SHARD_ROUND) && CollectRoundTile(ndd_.dims, tp, round_tile_) && tp.tail > 0) {
    ASSERT(tail_dim_ == -1);  // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int NDLoad::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }
  int64_t lead_align = ndd_.lead_stride();
  int64_t lead_dim = ndd_.lead_dim();
  uint64_t src_tile_stride_ = ndd_.stride_back() / lead_align * lead_dim;
  if (auto shard = k.root_dom_.shard_) {
    constexpr int SHARD_M = 1;
    constexpr int SHARD_N = 0;
    if (flags_ & OBJ_FLAG_LOAD_PINGPONG) {
      vPingPongLoad op;
      op.from = addr_.gm;
      op.xn = xbuf_;
      op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
      op.body_iter = ndd_.stride_back() / lead_align;
      op.tail_iter = tail_dim_ <= ndd_.lead_idx() ? op.body_iter : op.body_iter / ndd_[tail_dim_] * tail_size_;
      op.iter_size = lead_dim * ITEM_SIZE[type_id_];
      op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
      op.pingpong = 0;
      op.pingpong_stride = shard->tile[SHARD_N] * shard->tile[SHARD_M] * ITEM_SIZE[type_id_];
      op.round_rank = round_tile_.size();
      addr_.Update(insn_ + vPingPongLoad::RELOC_OFFSET);
      return vPingPongLoad::Encode(insn_, vAccInsnID::V_PINGPONG_LOAD, op, rounds);
    } else {
      vSLoad op;
      op.gm = addr_.gm;
      op.xn = xbuf_;
      op.tile_stride = src_tile_stride_;
      op.pad_size = lead_align - lead_dim;
      op.broadcast_m = flags_ & OBJ_FLAG_LOAD_SHARD_BCAST1;
      op.broadcast_n = flags_ & OBJ_FLAG_LOAD_SHARD_BCAST0;
      op.round_rank = round_tile_.size();
      op.type_size = ITEM_SIZE[type_id_];
      addr_.Update(insn_ + vSLoad::RELOC_OFFSET);
      k.GetVisitor<MixVisitCoder>()->AddReloc(insn_, vSLoad::SHARD_OFFSET);
      return vSLoad::Encode(insn_, vAccInsnID::V_SLOAD, op, rounds);
    }
  }  // end shard_
  vLoad op;
  op.from = addr_.gm;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_ * ITEM_SIZE[type_id_];
  op.body_iter = ndd_.stride_back() / lead_align;
  op.iter_size = lead_dim * ITEM_SIZE[type_id_];
  if (op.body_iter == 1) {
    op.tail_iter = tail_dim_ < 0 ? op.iter_size : tail_size_ * ITEM_SIZE[type_id_];
  } else {
    op.tail_iter = tail_dim_ < 0 ? op.body_iter : op.body_iter / ndd_[tail_dim_] * tail_size_;
  }
  op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
  op.round_rank = round_tile_.size();
  addr_.Update(insn_ + vLoad::RELOC_OFFSET);
  return vLoad::Encode(insn_, vAccInsnID::V_LOAD, op, rounds);
}

void NDLoad::Normalize(std::vector<NDObject *> &run_ops) {
  auto dims = shape_ref_->size;
  ndd_.dims.resize(dims);
  for (size_t i = 0; i < shape_ref_->size; i++) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
  }
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
}

void NDLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "Load"; }

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

int NDMultiLoad::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
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

void NDMultiLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "MultiLoad"; }

void NDPadStore::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < size; i++) {
    shape_[i] = lhs_->shape_ref_->data[i];
  }
  shape_[size - 1] += pad_size_;
  nd_ = lhs_->nd_;
}

void NDPadStore::AlignProp(PropRange &range) { range.depth = 1; }

void NDPadStore::FoldProp(PropRange &range) { range.depth = nd_.size() - 1; }

int NDPadStore::Emit(VectorKernel &k) {
  uint64_t lead_align = nd_.lead_stride();
  uint64_t src_tile_stride_ = nd_.stride_back() / lead_align * nd_.lead_dim();
  vSliceSL op;
  auto size = shape_ref_->size;
  op.gm = addr_.gm;
  op.xn = lhs_->xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_.lead_dim();
  op.slice_k = op.slice_m = 1;
  op.src_m = 1;
  op.src_n = shape_ref_->data[size - 1];
  for (size_t i = 0; i + 1 < size; i++) {
    op.src_m *= shape_ref_->data[i];
  }
  op.slice_n = lhs_->shape_ref_->data[size - 1];
  op.type_size = ITEM_SIZE[type_id_];
  op.offset = 0;
  op.one_flag = 0;
  if (op.slice_n == 1 && nd_.lead_idx() != 0) {
    op.pad_size = 0;
    op.one_flag = 1;
  }
  op.round_rank = 0;
  addr_.Update(insn_ + vSliceSL::RELOC_OFFSET);
  return vSliceSL::Encode(insn_, vAccInsnID::V_SLICE_STORE, V_PIPE_STORE, op, nullptr);
}

void NDPadStore::Dump(bool verbose, std::ostringstream &oss) { oss << "PadStore"; }

void NDSliceLoad::Normalize(std::vector<NDObject *> &run_ops) { NDLoad::Normalize(run_ops); }

void NDSliceLoad::AlignProp(PropRange &range) { range.depth = 1; }

void NDSliceLoad::FoldProp(PropRange &range) { range.depth = 1; }

int64_t NDSliceLoad::CalcOffset() {
  uint64_t src_offset = 0;
  std::vector<int64_t> start(src_ref_->size);
  for (size_t i = 0; i < src_ref_->size; i++) {
    start[i] = start_ref_ == nullptr
                 ? 0
                 : (start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i]);
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
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }
  auto reloc_offset = CalcOffset();
  uint64_t lead_align = ndd_.lead_stride();
  uint64_t src_tile_stride_ = ndd_.stride_back() / lead_align * ndd_.lead_dim();
  vSliceSL op;
  auto size = size_ref_->size;
  op.gm = addr_.gm;
  op.xn = xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - ndd_.lead_dim();
  op.slice_k = 1;
  if (ndd_.size() == 1) {
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
  op.round_rank = round_tile_.size();
  op.type_size = ITEM_SIZE[type_id_];
  op.offset = reloc_offset;
  op.one_flag = 0;
  addr_.Update(insn_ + vSliceSL::RELOC_OFFSET);
  return vSliceSL::Encode(insn_, vAccInsnID::V_SLICE_LOAD, V_PIPE_LOAD, op, rounds);
}

void NDSliceLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "SliceLoad"; }

void NDStridedSliceLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(std::all_of(step_ref_->data, step_ref_->data + step_ref_->size, [](int64_t i) { return i == 1; }));
  shape_.Resize(src_ref_->size);
  for (size_t i = 0; i < src_ref_->size; i++) {
    int64_t end = end_ref_->data[i] < 0 ? end_ref_->data[i] + src_ref_->data[i] : end_ref_->data[i];
    int64_t start = start_ref_ == nullptr
                      ? 0
                      : (start_ref_->data[i] < 0 ? start_ref_->data[i] + src_ref_->data[i] : start_ref_->data[i]);
    shape_[i] = std::min(end, src_ref_->data[i]) - start;
  }
  size_ref_ = shape_ref_;
  NDSliceLoad::Normalize(run_ops);
}

void NDStridedSliceLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "StridedSliceLoad"; }

void NDStore::Normalize(std::vector<NDObject *> &run_ops) {
  nd_ = lhs_->nd_;
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
  UpdateDimMask();
}

void NDStore::Shard(const ShardParam &sp) {
  for (int i = nd_.size() - 1; i > sp.base + 1; --i) {
    if (auto factor = sp.dom->operator[](i); factor > 1) {
      UpdateRoundTile(elem_dim_mask_ & (1u << i), factor, round_tile_);
    }
  }
  flags_ |= OBJ_FLAG_STORE_SHARD_ROUND;
  if ((elem_dim_mask_ & 1u << (sp.base + 1)) == 0 && sp.tile[sp.base + 1] > 1) {
    flags_ |= OBJ_FLAG_STORE_SHARD_BCAST1;
  }
  if ((elem_dim_mask_ & 1u << (sp.base)) == 0 && sp.tile[sp.base] > 1) {
    flags_ |= OBJ_FLAG_STORE_SHARD_BCAST0;
  }
  NDObject::Shard(sp);
}

void NDStore::Tile(const TileParam &tp) {
  if (!(flags_ & OBJ_FLAG_STORE_SHARD_ROUND) && CollectRoundTile(elem_dim_mask_, tp, round_tile_) && tp.tail > 0) {
    ASSERT(tail_dim_ == -1);  // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
}

int NDStore::Emit(VectorKernel &k) {
  int64_t lead_align = nd_.lead_stride();
  int64_t lead_dim = nd_.lead_dim();
  ASSERT(lead_align == lhs_->nd_.lead_stride());
  uint64_t dst_tile_stride_ = nd_.stride_back() / lead_align * lead_dim;
  if (lhs_->obj_id_ == kElementAny) {
    vStoreCond op;
    op.xn = lhs_->xbuf_;
    op.to = addr_.data;
    op.tile_stride = 0;
    op.dtype_shift = 2;
    op.pad_size = 0;
    op.iter_size = 0;
    op.round_rank = 0;
    op.cond_offset = insn_ - lhs_->tail_insn_ - vElementAny::STORE_COND_OFFSET;
    addr_.Update(insn_ + vStoreCond::RELOC_OFFSET);
    return vStoreCond::Encode(insn_, vAccInsnID::V_STORE_COND, op, nullptr);
  }
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
    if (k.comm_op_ && k.comm_op_->GetObjectType() == kReduceScatter) {
      // ReduceScatter round store accroding to rank_id
      vStoreRS op;
      op.to = addr_.data;
      op.xn = lhs_->xbuf_;
      op.iter_size = lead_dim * ITEM_SIZE[type_id_];
      op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
      op.iter_num = nd_.stride_back() / lead_align;
      op.rank_id = k.comm_op_->comm_->GetRankId();
      if (tail_dim_ < 0) {
        op.iter_tail = op.iter_num;
      } else {
        ASSERT(op.iter_num > 1);
        op.iter_tail = op.iter_num / nd_[tail_dim_] * tail_size_;
      }
      op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
      op.round_rank = round_tile_.size();
      addr_.Update(insn_ + vStoreRS::RELOC_OFFSET);
      return vStoreRS::Encode(insn_, V_STORE_RS, op, rounds);
    }
    if (lhs_->obj_id_ == kReduce) {
      auto red_op = static_cast<ReduceOp *>(lhs_);
      if (k.GetVisitor<RedVisitCoder>()) {
        ASSERT(tail_dim_ < 0);
        vStoreCond op;
        op.xn = lhs_->xbuf_;
        op.to = addr_.data;
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.dtype_shift = ITEM_SIZE[type_id_] == 4 ? 2 : 1;
        if (lead_align == lead_dim || nd_.stride_back() == lead_align) {
          op.pad_size = 0;
          op.iter_size = 0;
        } else {
          op.iter_size = lead_dim * ITEM_SIZE[type_id_];
          op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        }
        op.cond_offset = insn_ - red_op->tail_insn_ - vReduceJoin::STORE_COND_OFFSET;
        op.round_rank = round_tile_.size();
        addr_.Update(insn_ + vStoreCond::RELOC_OFFSET);
        return vStoreCond::Encode(insn_, vAccInsnID::V_STORE_COND, op, rounds);
      } else {
        int code_size;
        Code &code = k.code_;
        vStoreAtomic op;
        op.to = addr_.data;
        op.xn = lhs_->xbuf_;
        op.cum_flag = (round_tile_.size() & 1);
        op.iter_size = lead_dim * ITEM_SIZE[type_id_];
        op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        op.iter_num = nd_.stride_back() / lead_align;
        if (tail_dim_ < 0 || red_op->InRange(tail_dim_)) {
          op.iter_tail = op.iter_num;
        } else {
          ASSERT(op.iter_num > 1);  // inner reduce is divided. outer reduce is not lead
          op.iter_tail = op.iter_num / nd_[tail_dim_] * tail_size_;
        }
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.round_rank = round_tile_.size();
        addr_.Update(insn_ + vStoreAtomic::RELOC_OFFSET);
        code_size = vStoreAtomic::Encode(insn_, V_STORE_ATOMIC, op, rounds);
        auto clean_wrap = red_op->clean_wrap_;
        if (clean_wrap == nullptr) {
          red_op->clean_wrap_ = clean_wrap = new AtomicCleanWrap(this);
        }
        clean_wrap->CodeGen(code, this);
        code.InsertWrap(clean_wrap);
        return code_size;
      }
    }
  }
  if (k.comm_op_ && k.comm_op_->GetObjectType() == kAllGatherV2) {
    vStoreAG op;
    op.to = addr_.data;
    op.xn = k.comm_op_->xbufs_[0];
    op.iter_size = lead_dim * ITEM_SIZE[type_id_];
    op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
    op.iter_num = nd_.stride_back() / lead_align;
    if (op.iter_num == 1) {
      op.iter_tail = tail_dim_ < 0 ? op.iter_size : tail_size_ * ITEM_SIZE[type_id_];
    } else {
      op.iter_tail = tail_dim_ < 0 ? op.iter_num : op.iter_num / nd_[tail_dim_] * tail_size_;
    }
    // op.iter_tail = tail_dim_ < 0 ? op.iter_num : op.iter_num / nd_[tail_dim_] * tail_size_;
    op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
    op.rank_id = k.comm_op_->comm_->GetRankId();
    op.rank_size = k.comm_op_->comm_->GetRankSize();
    op.xbuf_size =
      reinterpret_cast<uint64_t>(k.comm_op_->xbufs_[1]) - reinterpret_cast<uint64_t>(k.comm_op_->xbufs_[0]);
    op.shard_stride = std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) *
                      ITEM_SIZE[type_id_] / op.rank_size;
    addr_.Update(insn_ + vStoreAG::RELOC_OFFSET);
    return vStoreAG::Encode(insn_, V_STORE_AG, op);
  }
  if (k.root_dom_.shard_) {
    vSStore op;
    op.gm = addr_.gm;
    op.xn = lhs_->xbuf_;
    op.tile_stride = dst_tile_stride_;
    op.pad_size = lead_align - lead_dim;
    op.broadcast_m = flags_ & OBJ_FLAG_STORE_SHARD_BCAST1;
    op.broadcast_n = flags_ & OBJ_FLAG_STORE_SHARD_BCAST0;
    op.round_rank = round_tile_.size();
    op.type_size = ITEM_SIZE[type_id_];
    addr_.Update(insn_ + vSStore::RELOC_OFFSET);
    k.GetVisitor<MixVisitCoder>()->AddReloc(insn_, vSStore::SHARD_OFFSET);
    return vSStore::Encode(insn_, vAccInsnID::V_SSTORE, op, rounds);
  }
  vStore op;
  uint64_t iter_size = lead_dim * ITEM_SIZE[type_id_];
  uint64_t pad_size = lead_align * ITEM_SIZE[type_id_] - iter_size;
  uint64_t body_iter = nd_.stride_back() / lead_align;
  uint64_t tail_iter;
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
    body_iter = 1;
    pad_size = 0;
  } else if (body_iter == 1) {
    tail_iter = tail_dim_ < 0 ? iter_size : tail_size_ * ITEM_SIZE[type_id_];
  } else {
    tail_iter = tail_dim_ < 0 ? body_iter : body_iter / nd_[tail_dim_] * tail_size_;
  }
  op.xn = lhs_->xbuf_;
  op.to = addr_.data;
  op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
  op.iter_num = body_iter;
  op.iter_tail = tail_iter;
  op.iter_size = iter_size;
  op.pad_size = pad_size;
  op.round_rank = round_tile_.size();
  addr_.Update(insn_ + vStore::RELOC_OFFSET);
  return vStore::Encode(insn_, vAccInsnID::V_STORE, op, rounds);
}

void NDStore::Dump(bool verbose, std::ostringstream &oss) { oss << "Store"; }

int CopyOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, nd_.stride_back() * ITEM_SIZE[type_id_]);
}

void CopyOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Copy"; }

void ReshapeOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update nd_/shape_
  auto dims = dst_shape_ref_->size;
  ndd_.dims.resize(dims);
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
      ndd_.dims[nd_i] = sh;
    }
  }
  if (update_axis != dims) {
    int64_t input_sz = 1;
    auto &nd = lhs_->nd_;
    for (size_t i = 0; i < nd.size(); ++i) {
      input_sz *= nd[i];
    }
    auto v = input_sz / sz;
    ndd_.dims[update_axis] = v;
    shape_[dims - 1 - update_axis] = v;
  }
}

int ReshapeOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  if (lhs_->nd_.lead_dim() == ndd_.lead_dim()) {
    return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
  }
  vReshape op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xd_lead = ndd_.lead_dim();
  op.xn_lead = lhs_->nd_.lead_dim();
  op.xd_pad = ndd_.lead_stride() - op.xd_lead;
  op.xn_pad = lhs_->nd_.lead_stride() - op.xn_lead;
  op.dup_size = ndd_.stride_back() / ndd_.lead_stride();
  return vReshape::Encode(insn_, (type_id_ == kFloat32 || type_id_ == kInt32) ? V_RESHAPE_B32 : V_RESHAPE_B32, op);
}

void ReshapeOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Reshape"; }

UnaryOp::UnaryOp(int op_type, NDObject *input)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kUnary), op_type_(op_type) {
  shape_ref_ = input->shape_ref_;
}

int UnaryOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  ASSERT(size_t(op_type_) < sizeof(unary_id_list) / sizeof(InsnIdTable));
  auto id = unary_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

void UnaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << unary_id_list[op_type_].name; }

int UnaryOp::QueryId(const std::string &op_name) {
  for (int i = 0; i < static_cast<int>(sizeof(unary_id_list) / sizeof(InsnIdTable)); ++i) {
    if (op_name == unary_id_list[i].name) return i;
  }
  return -1;
}

int RemovePadOp::Emit(VectorKernel &k) {
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_REMOVEPAD_U16, V_REMOVEPAD_U16, V_REMOVEPAD, V_REMOVEPAD};
  if (nd_.lead_dim() == nd_.lead_stride() || nd_.stride_back() == nd_.lead_stride()) {
    return CopyOp::Emit(k);
  }
  vRemovePad op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.repeat = nd_.stride_back() / nd_.lead_stride();
  op.iter_num = nd_.lead_dim();
  op.rs = GetBlocks(nd_.lead_stride());
  return vRemovePad::Encode(insn_, id_list[type_id_], op);
}

void RemovePadOp::Dump(bool verbose, std::ostringstream &oss) { oss << "RemovePad"; }

void ElementAnyOp::Normalize(std::vector<NDObject *> &run_ops) {
  tail_dim_ = -1;
  tail_size_ = 0;
  ndd_.dims.resize(lhs_->nd_.size(), 1);
}

void ElementAnyOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int ElementAnyOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  uint64_t clr_tail = 0;
  vElementAny op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.simd_width = SelectSimdWidth(lhs_->nd_.stride_back(), type_id_);
  op.repeat = lhs_->nd_.stride_back() / op.simd_width;
  if (tail_dim_ < 0) {
    op.repeat_tail = op.repeat;
  } else if (tail_dim_ >= 0 && tail_dim_ <= ndd_.lead_idx()) {
    uint64_t tail_size = static_cast<uint64_t>(tail_size_);
    op.repeat_tail = CeilDiv(tail_size, op.simd_width);
    if (op.repeat_tail * op.simd_width != tail_size) {
      clr_tail = tail_size;
    }
  } else {
    op.repeat_tail = op.repeat / lhs_->nd_[tail_dim_] * tail_size_;
  }
  uint32_t insn_num = 1;
  uint32_t size = 0;
  if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail) {
    size = EmitClearPad(insn_, lhs_, clr_tail);
    tail_insn_ = insn_ + size;
    insn_num++;
  }
  size += vElementAny::Encode(tail_insn_, V_ELEMENT_ANY, op);

  if (insn_num > 1) {
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return size;
}

void ElementAnyOp::Dump(bool verbose, std::ostringstream &oss) { oss << "ElementAny"; }

int CastOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  auto id = cast_id_list[lhs_->type_id_][type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

void CastOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Cast"; }

template <typename T>
BinaryScalarOp<T>::BinaryScalarOp(int op_type, NDObject *input, T scalar)
    : NDObject(input, nullptr, input->type_id_, ObjectType::kBinaryS), op_type_(op_type), scalar_(scalar) {
  shape_ref_ = input->shape_ref_;
}

template <typename T>
int BinaryScalarOp<T>::Emit(VectorKernel &k) {
  vBinaryS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = EncodeScalar(scalar_, type_id_);
  ASSERT(size_t(op_type_) < sizeof(binarys_id_list) / sizeof(InsnIdTable));
  auto id = binarys_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vBinaryS::Encode(insn_, id, op);
}

template <typename T>
void BinaryScalarOp<T>::Dump(bool verbose, std::ostringstream &oss) {
  oss << binarys_id_list[op_type_].name;
  if (verbose) {
    oss << "<" << scalar_ << ">";
  }
}

template class BinaryScalarOp<float>;
template class BinaryScalarOp<int32_t>;
template class BinaryScalarOp<Float16>;
template class BinaryScalarOp<BFloat16>;

template <typename T>
CompareScalarOp<T>::CompareScalarOp(int op_type, NDObject *input, T scalar)
    : FlexOp(input, nullptr, input->type_id_, ObjectType::kCompareS), scalar_(scalar) {
  ws_num_ = 1;
  flags_ |= OBJ_FLAG_FLEX_INPL_WS;
  cmp_op_ = op_type;
  shape_ref_ = input->shape_ref_;
}

template <typename T>
int CompareScalarOp<T>::Emit(VectorKernel &k) {
  vCompareS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = EncodeScalar(scalar_, type_id_);
  op.ws = wss_[0];
  op.type = cmp_op_;
  return vCompareS::Encode(insn_, type_id_ == kFloat32 ? V_CMPS : V_CMPS_FP16, op);
}

template <typename T>
void CompareScalarOp<T>::Dump(bool verbose, std::ostringstream &oss) {
  oss << "CompareS";
  if (verbose) {
    oss << "<" << cmp_op_ << ", " << scalar_ << ">";
  }
}

template class CompareScalarOp<float>;
template class CompareScalarOp<int32_t>;
template class CompareScalarOp<Float16>;
template class CompareScalarOp<BFloat16>;

_BinaryNormalizer::~_BinaryNormalizer() {
  for (auto op : lhs_stuff_ops_) {
    delete op;
  }
  for (auto op : rhs_stuff_ops_) {
    delete op;
  }
}

void _BinaryNormalizer::Normalize(NDObject *self, std::vector<NDObject *> &run_ops) {
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
      shape_[i] = lhs_data[i] == 1 ? rhs_data[i - diff] : lhs_data[i];
    }
  } else {
    shape_.Resize(rhs_sz);
    auto diff = rhs_sz - lhs_sz;
    for (size_t i = 0; i < diff; ++i) {
      shape_[i] = rhs_data[i];
    }
    for (size_t i = diff; i < rhs_sz; ++i) {
      shape_[i] = rhs_data[i] == 1 ? lhs_data[i - diff] : rhs_data[i];
    }
  }
  // update nd_
  const auto &lhs_nd = self->lhs_->nd_;
  const auto &rhs_nd = self->rhs_->nd_;
  DimArray nd;
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
  } else {
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
  self->nd_ = self->lhs_->nd_;
}

BinaryOp::BinaryOp(int op_type, NDObject *lhs, NDObject *rhs)
    : NDObject(lhs, rhs, lhs->type_id_, ObjectType::kBinary), op_type_(op_type) {
  shape_ref_ = &norm_.shape_;
}

int BinaryOp::Emit(VectorKernel &k) {
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back();
  ASSERT(size_t(op_type_) < sizeof(binary_id_list) / sizeof(InsnIdTable));
  auto id = binary_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vBinary::Encode(insn_, id, op);
}

void BinaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << binary_id_list[op_type_].name; }

int BinaryOp::QueryId(const std::string &op_name) {
  for (int i = 0; i < static_cast<int>(sizeof(binary_id_list) / sizeof(InsnIdTable)); ++i) {
    if (op_name == binary_id_list[i].name) return i;
  }
  return -1;
}

CompareOp::CompareOp(int op_type, NDObject *lhs, NDObject *rhs)
    : FlexOp(lhs, rhs, lhs->type_id_, ObjectType::kCompare) {
  ws_num_ = 1;
  flags_ |= OBJ_FLAG_FLEX_INPL_WS;
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
  op.count = nd_.stride_back();
  auto id = binary_id_list[cmp_op_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vCompare::Encode(insn_, id, op);
}

void CompareOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Compare";
  if (verbose) {
    oss << "<" << cmp_op_ << ">";
  }
}

int PowerOp::Emit(VectorKernel &k) {
  ASSERT(type_id_ == dvm::kFloat32);
  vBinaryWS op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back();
  op.ws0 = wss_[0];
  op.ws1 = wss_[1];
  return vBinaryWS::Encode(insn_, V_POW, op);
}

void PowerOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Power"; }

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
  DimArray nd;
  nd.resize(max_dims);
  for (size_t i = 0; i < max_dims; ++i) {
    auto lhs_dim = i < lhs_nd.size() ? lhs_nd[i] : 1;
    auto rhs_dim = i < rhs_nd.size() ? rhs_nd[i] : 1;
    auto xhs_dim = i < xhs_nd.size() ? xhs_nd[i] : 1;
    nd[i] = std::max({lhs_dim, rhs_dim, xhs_dim});
    if (nd[i] != lhs_dim) lhs_bc = true;
    if (nd[i] != rhs_dim) rhs_bc = true;
    if (nd[i] != xhs_dim) xhs_bc = true;
  }
  auto insert_broadcast = [&input, &run_ops, this, &nd](size_t idx) {
    size_t stuff_idx = 0;
    *input[idx] = InsertImplicitBroadcast(*input[idx], nd, stuff_ops_[idx], stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(stuff_ops_[idx][i]);
    }
  };
  if (lhs_bc) insert_broadcast(0);
  if (rhs_bc) insert_broadcast(1);
  if (xhs_bc) insert_broadcast(2);
  nd_ = lhs_->nd_;
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
  op.count = nd_.stride_back();
  op.xm = rhs_->xbuf_;
  op.cond = xhs_->xbuf_;
  op.ws = wss_[0];
  ASSERT(id_list[type_id_] != V_NONE);
  return vSelect::Encode(insn_, id_list[type_id_], op);
}

void SelectOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Select"; }

void _BroadcastOp::FoldProp(PropRange &range) {
  int state = 0;  // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

void _BroadcastOp::AlignProp(PropRange &range) {
  int state = 0;  // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

int _BroadcastOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  int start_dim = -1;
  int end_dim = -1;
  for (size_t i = ndd_.lead_idx(); i < ndd_.size(); ++i) {
    if (start_dim == -1) {
      if (ndd_[i] != lhs_->nd_[i]) {
        end_dim = i;
        start_dim = i;
      }
    } else if (ndd_[i] != lhs_->nd_[i] || ndd_[i] == 1) {
      end_dim = i;
    } else {
      break;
    }
  }
  int64_t offset;
  if (start_dim == 0 || start_dim == ndd_.lead_idx()) {
    offset = EmitBroadcastX(insn_, end_dim);
  } else {
    offset = EmitBroadcastY(insn_, start_dim, end_dim);
  }
  return offset;
}

int64_t _BroadcastOp::EmitBroadcastX(uint64_t *p, int end_dim) {
  vBroadcastX op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = ndd_.stride(end_dim);
  int64_t rank_size = static_cast<int64_t>(ndd_.size());
  op.lead_num = end_dim + 1 < rank_size ? ndd_[end_dim + 1] : 1;
  op.iter_num = end_dim + 2 < rank_size ? ndd_.stride_back() / ndd_.stride(end_dim + 1) : 1;
  op.lead_pad = lhs_->nd_.lead_stride() - lhs_->nd_.lead_dim();
  const static vSimdInsnID id_list[kTypeEnd] = {V_NONE, V_BROADCAST_X_B16, V_NONE, V_BROADCAST_X_B32,
                                                V_BROADCAST_X_B32};
  ASSERT(id_list[type_id_] != V_NONE);
  return vBroadcastX::Encode(p, id_list[type_id_], op);
}

int64_t _BroadcastOp::EmitBroadcastY(uint64_t *p, int start_dim, int end_dim) {
  vBroadcastY op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  if (start_dim > 0) {
    op.iter_num = ndd_.stride_back() / ndd_.stride(end_dim);
    op.dup_num = ndd_.stride(end_dim) / ndd_.stride(start_dim - 1);
    op.dup_stride = ndd_.stride(start_dim - 1) * ITEM_SIZE[type_id_] / SIMD_BLOCK_SIZE;
  } else {
    op.iter_num = 1;
    op.dup_num = 1;
    op.dup_stride = ndd_.stride_back() * ITEM_SIZE[type_id_] / SIMD_BLOCK_SIZE;
  }
  return vBroadcastY::Encode(p, V_BROADCAST_Y, op);
}

void _BroadcastOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Broadcast"; }

BroadcastOp::~BroadcastOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
}

void BroadcastOp::Normalize(std::vector<NDObject *> &run_ops) {
  // recover original input
  if (!stuff_ops_.empty()) {
    lhs_ = stuff_ops_[0]->lhs_;
  }
  // update nd_ from shape_ref_
  auto dims = dst_shape_ref_->size;
  ndd_.dims.resize(dims);
  shape_.Resize(dims);
  auto offset = dims - lhs_->shape_ref_->size;  // dst_shape dims >= x_shape dims
  for (size_t i = 0; i < dims; ++i) {
    shape_[i] = dst_shape_ref_->data[i];
    if (shape_[i] == -1) {
      // e.g. x_shape (4, 1), dst_shape (2, -1, 1) --> dst_shape (2, 4, 1)
      shape_[i] = lhs_->shape_ref_->data[i - offset];
    }
    ndd_.dims[dims - 1 - i] = shape_[i];
  }
  if (flags_ & OBJ_FLAG_EAGER) {
    size_t stuff_idx = run_ops.size();
    lhs_ = InsertBroadcastOpsInBetween(lhs_, ndd_.dims, run_ops, stuff_idx);
  } else {
    size_t stuff_idx = 0;
    lhs_ = InsertBroadcastOpsInBetween(lhs_, ndd_.dims, stuff_ops_, stuff_idx);
    for (size_t i = 0; i < stuff_idx; ++i) {
      run_ops.push_back(stuff_ops_[i]);
    }
  }
}

template <typename T>
void BroadcastScalarOp<T>::Normalize(std::vector<NDObject *> &run_ops) {
  // update nd_ from shape_ref_
  auto dims = shape_ref_->size;
  ndd_.dims.resize(dims);
  for (size_t i = 0; i < dims; ++i) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
  }
}

template <typename T>
int BroadcastScalarOp<T>::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  vBroadcastS op;
  op.scalar = EncodeScalar(scalar_, type_id_);
  op.xd = xbuf_;
  op.count = ndd_.stride_back();
  return vBroadcastS::Encode(insn_, ITEM_SIZE[type_id_] == sizeof(uint32_t) ? V_BROADCAST_S : V_BROADCAST_S_B16, op);
}

template <typename T>
void BroadcastScalarOp<T>::Dump(bool verbose, std::ostringstream &oss) {
  oss << "BroadcastS";
  if (verbose) {
    oss << "<" << scalar_ << ">";
  }
}

template class BroadcastScalarOp<float>;
template class BroadcastScalarOp<int32_t>;
template class BroadcastScalarOp<Float16>;
template class BroadcastScalarOp<BFloat16>;

void _ReduceOp::FoldProp(PropRange &range) {
  int state = 0;  // -1 - reduce ; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && ndd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (ndd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
}

void _ReduceOp::AlignProp(PropRange &range) {
  int state = 0;  // -1 - reduce; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && ndd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (ndd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
  range.simd_dim = 1;
}

void _ReduceOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int _ReduceOp::Emit(VectorKernel &k) {
  if (ndd_.strides.empty()) {
    ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  }
  ASSERT(red_op_ == ReduceOp::SUM);
  if (ndd_.lead_dim() == lhs_->nd_.lead_dim() && ndd_.stride_back() == lhs_->nd_.stride_back()) {
    return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
  } else if (start_dim_ <= lhs_->nd_.lead_idx()) {  // reduce x
    ASSERT(start_dim_ >= 0 && end_dim_ >= 0);
    vReduceX op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.red_size = lhs_->nd_.stride(end_dim_);
    op.simd_width = SelectSimdWidth(op.red_size, type_id_);
    uint64_t clr_tail = 0;
    if (tail_dim_ >= 0 && tail_dim_ <= end_dim_) {
      auto tail_size = static_cast<uint64_t>(tail_size_);
      op.red_tail = RoundUp(tail_size, op.simd_width);
      if (op.red_tail != tail_size) {
        clr_tail = tail_size;
      }
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_size = lhs_->nd_.stride_back() / lhs_->nd_.stride(end_dim_);
    if (auto lead_dim = ndd_.lead_dim(); lead_dim > 1) {
      op.dup_block = lead_dim;
      op.dup_pad = ndd_.lead_stride() - lead_dim;
    } else {
      op.dup_block = op.dup_size;
      op.dup_pad = 0;
    }
    uint32_t size = 0;
    uint32_t insn_num = 1;
    if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail > 0) {
      size = EmitClearPad(insn_, lhs_, clr_tail);
      tail_insn_ = insn_ + size;
      insn_num++;
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
    op.iter_size = ndd_.stride(start_dim_);
    op.simd_width = SelectSimdWidth(op.iter_size, type_id_);
    op.red_size = lhs_->nd_.stride(end_dim_) / op.iter_size;
    ASSERT(op.red_size > 1);
    if (InRange(tail_dim_)) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = ndd_.stride_back() / ndd_.stride(end_dim_);
    return vReduceY::Encode(insn_, V_RSUM_Y, op);
  }
}

void _ReduceOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Reduce"; }

void ReduceOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Reduce";
  if (verbose) {
    oss << "<";
    oss << *dims_ref_ << ", ";
    oss << std::boolalpha << keepdims_;
    oss << ">";
  }
}

ReduceOp::~ReduceOp() {
  for (auto op : stuff_ops_) {
    delete op;
  }
  if (clean_wrap_ != nullptr) {
    delete clean_wrap_;
  }
}

void ReduceOp::Normalize(std::vector<NDObject *> &run_ops) {
  DimArray dims;
  DimArray shape_dims;
  ASSERT(dims_ref_ != nullptr);
  NDObject *input = stuff_ops_.empty() ? lhs_ : stuff_ops_[0]->lhs_;
  auto input_shape_ref = input->shape_ref_;
  // update dims
  auto lhs_dim = input_shape_ref->size;
  if (dims_ref_->size == 0) {
    shape_dims.resize(lhs_dim);
    dims.resize(lhs_dim);
    for (int64_t i = 0; i < static_cast<int64_t>(lhs_dim); ++i) {
      shape_dims[i] = i;
      dims[i] = i;
    }
    shape_.Resize(0);
    if (keepdims_) {
      for (size_t i = 0; i < lhs_dim; ++i) {
        shape_[i] = 1;
      }
      shape_.Resize(lhs_dim);
    }
  } else {
    size_t dim_size = dims_ref_->size;
    for (size_t i = 0; i < dim_size; i++) {
      auto dim = dims_ref_->data[i];
      if (dim < 0) {
        dim += lhs_dim;
      }
      if (i == 0 || dim > shape_dims[i - 1]) {
        shape_dims[i] = dim;
      } else {
        size_t i_pos = i;
        for (; i_pos > 0 && shape_dims[i_pos - 1] > dim; --i_pos) {
          shape_dims[i_pos] = shape_dims[i_pos - 1];
        }
        shape_dims[i_pos] = dim;
      }
    }
    // update shape_ref_
    size_t shape_size = 0;
    size_t dim_idx = 0;
    for (size_t i = 0; i < lhs_dim; ++i) {
      if (dim_idx >= dim_size || static_cast<int64_t>(i) != shape_dims[dim_idx]) {
        shape_[shape_size++] = input_shape_ref->data[i];
      } else {
        dim_idx++;
        if (keepdims_) {
          shape_[shape_size++] = 1;
        }
      }
    }
    shape_.Resize(shape_size);
    // update dims
    int64_t back_idx = shape_dims[dim_size - 1] + 1;
    int64_t back_end = input->nd_.size() - 1;
    while (back_idx <= back_end && input->nd_[back_end - back_idx] == 1) {  // align fold may flip dims
      shape_dims[dim_size++] = back_idx++;
    }
    shape_dims.resize(dim_size);
    dims.resize(dim_size);
    for (size_t i = 0; i < dim_size; ++i) {
      dims[i] = lhs_dim - shape_dims[dim_size - i - 1] - 1;
    }
  }
  size_t stuff_idx = 0;
  int red_start = -1, red_end = -1, red_ext = -1, lead_dim = -1;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto d = dims[i];
    if (input->nd_[d] == 1) continue;
    if (d != red_ext) {
      if (lead_dim == -1) {
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
        obj->ndd_.dims = input->nd_.dims();
        ndd_.strides.resize(0);
        for (int j = red_start; j <= red_end; ++j) {
          obj->ndd_.dims[j] = 1;
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
  ndd_.dims = input->nd_.dims();
  if (red_start != -1) {
    lhs_ = input;
    for (int j = red_start; j <= red_end; ++j) {
      ndd_.dims[j] = 1;
    }
    SetRange(red_start == lead_dim ? 0 : red_start, red_end);
  } else {
    ASSERT(stuff_idx == 0);
    start_dim_ = end_dim_ = -1;
  }
  visit_.Clear();
  round_tile_.resize(0);
}

void ReduceOp::Tile(const TileParam &tp) {
  CollectRoundTile(ndd_.dims, tp, round_tile_);
  _ReduceOp::Tile(tp);
}

int ReduceOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  if (ws_num_ == 2) {
    return EmitDeterm(k);
  }
  if (!(round_tile_.size() & 1)) {
    return _ReduceOp::Emit(k);
  }
  auto out_xbuf = xbuf_;
  xbuf_ = wss_[0];
  int size = _ReduceOp::Emit(k);
  vAtomicCum op;
  uint64_t rounds[2];
  BuildDimRounds(round_tile_, rounds);
  op.xd = out_xbuf;
  op.xn = xbuf_;
  xbuf_ = out_xbuf;
  op.count = ndd_.stride_back();
  op.round_rank = round_tile_.size();
  tail_insn_ = insn_ + size;
  size += vAtomicCum::Encode(tail_insn_, V_ATOMICCUM, op, rounds);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  return size;
}

static bool GenTileVisit(VectorKernel &k, const DimArray &round_tile, RedVisitCoder &coder) {
  auto round_depth = round_tile.size();
  if (round_depth == 0) {
    return false;
  }
  uint64_t core_limit = k.code_.block_dim_;
  if (round_depth == 1) {
    auto v = reinterpret_cast<vVisitRed1 *>(k.code_.data_ + k.code_.mem_size_ - sizeof(vVisitRed1));
    uint64_t r1 = round_tile[0];
    v->head = 0;
    v->r1 = r1;
    v->e = (k.tile_num_ / r1) << 32;
    coder.code_ = reinterpret_cast<uint64_t *>(v);
    coder.code_size_ = sizeof(vVisitRed1);
    coder.visit_id_ = V_VISIT_RED_1;
    coder.block_num_ = std::min(core_limit, k.tile_num_);
  } else if (round_depth == 2) {
    auto v = reinterpret_cast<vVisitRed2 *>(k.code_.data_ + k.code_.mem_size_ - sizeof(vVisitRed2));
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    v->head = 0;
    v->e = (k.tile_num_ / r1) << 32;
    v->e1_r1 = e1 << 32 | r1;
    coder.code_ = reinterpret_cast<uint64_t *>(v);
    coder.code_size_ = sizeof(vVisitRed2);
    coder.visit_id_ = V_VISIT_RED_2;
    coder.block_num_ = std::min(core_limit, k.tile_num_);
  } else if (round_depth == 3) {
    auto v = reinterpret_cast<vVisitRed3 *>(k.code_.data_ + k.code_.mem_size_ - sizeof(vVisitRed3));
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r2 = e1 << 32 | r2;
    v->r1 = r1;
    coder.code_ = reinterpret_cast<uint64_t *>(v);
    coder.code_size_ = sizeof(vVisitRed3);
    coder.visit_id_ = V_VISIT_RED_3;
    coder.block_num_ = std::min(core_limit, k.tile_num_ / r2);
  } else {
    ASSERT(round_depth == 4);
    auto v = reinterpret_cast<vVisitRed4 *>(k.code_.data_ + k.code_.mem_size_ - sizeof(vVisitRed4));
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    uint64_t e2 = round_tile[3];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r1 = e1 << 32 | r1;
    v->e2_r2 = e2 << 32 | r2;
    coder.code_ = reinterpret_cast<uint64_t *>(v);
    coder.code_size_ = sizeof(vVisitRed4);
    coder.visit_id_ = V_VISIT_RED_4;
    coder.block_num_ = std::min(core_limit, k.tile_num_ / r2);
  }
  return true;
}

int ReduceOp::EmitDeterm(VectorKernel &k) {
  if (!GenTileVisit(k, round_tile_, visit_)) {
    return _ReduceOp::Emit(k);
  }
  if (!k.GetVisitor<RedVisitCoder>()) {
    k.AddVisitor(&visit_);
    visit_.ws_size_ = ndd_.stride_back() * sizeof(float) * visit_.block_num_;
  }
  auto out_xbuf = xbuf_;
  xbuf_ = wss_[0];
  int size = _ReduceOp::Emit(k);
  vReduceJoin op;
  op.xd = xbuf_ = out_xbuf;
  op.xn = wss_[0];
  op.xs = wss_[1];
  if (ndd_.lead_stride() == ndd_.lead_dim()) {
    op.iter_num = 1;
    op.iter_stride = ndd_.stride_back();
  } else if (ndd_.lead_stride() == ndd_.stride_back()) {
    op.iter_num = 1;
    op.iter_stride = ndd_.lead_dim();
  } else {
    op.iter_stride = ndd_.lead_stride();
    op.iter_num = ndd_.stride_back() / op.iter_stride;
  }
  op.ws = 0;
  tail_insn_ = insn_ + size;
  size += vReduceJoin::Encode(tail_insn_, op);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  k.GetVisitor<RedVisitCoder>()->AddReloc(tail_insn_, 0);
  ws_reloc_.ws = 0;
  ws_reloc_.Update(tail_insn_ + vReduceJoin::RELOC_OFFSET);
  k.code_.BindWorkspace(ws_reloc_, 0);  // TODO: mutli workspace
  if (k.forward_event_num_ > 6) {
    k.forward_event_num_ = 6;
  }
  if (k.backward_event_num_ > 6) {
    k.backward_event_num_ = 6;
  }
  return size;
}

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b)
    : NDObject(lhs, rhs, lhs->type_id_, kCubeOp), trans_a_(trans_a), trans_b_(trans_b) {
  shape_ref_ = &shape_;
  nd_.data = &ndd_;
}

CubeOp::CubeOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias)
    : CubeOp(lhs, rhs, trans_a, trans_b) {
  bias_ = bias;
}

void CubeOp::InferCubeConfig() {
  auto GetPad = [this](int64_t pad_size, int64_t &pad) {
    if (pad_size % ALIGN_128 == 0 || (pad_size <= ALIGN_256 && pad_size % ALIGN_32 == 0)) {
      return;
    }
    pad = ALIGN_256 - pad_size % ALIGN_256;
    tactics_.enable_pad = true;
  };
  GetPad(trans_a_ ? m_align_ : k_align_, tactics_.lhs_pad_size);
  GetPad(trans_b_ ? k_align_ : n_align_, tactics_.rhs_pad_size);

  int64_t k_stride = System::Instance().L2Size() / (m_real_ + n_real_) / 2;
  if ((k_stride << 1) < k_real_ && k_real_ > MAX_SPLIT_K) {
    tactics_.enable_splitk = true;
    tactics_.k_stride = std::min(k_stride / ALIGN_256 * ALIGN_256, MAX_SPLIT_K);
    tactics_.k_stride = std::max(tactics_.k_stride, MIN_SPLIT_K);
  }

  if (bias_ && bias_->type_id_ == kBFloat16) {
    tactics_.enable_bias_cast = true;
  }
}

void CubeOp::NormalizeCube() {
  m_align_ = trans_a_ ? lhs_->nd_[0] : lhs_->nd_[1];
  // Only pad the rows, which may result in matrices A and B where some K matrices are padded and some are not.
  // Therefore, we take the maximum among them.
  ka_align_ = trans_a_ ? lhs_->nd_[1] : lhs_->nd_[0];
  kb_align_ = trans_b_ ? rhs_->nd_[0] : rhs_->nd_[1];
  k_align_ = std::max(ka_align_, kb_align_);
  n_align_ = trans_b_ ? rhs_->nd_[1] : rhs_->nd_[0];
  m_real_ = m_align_;
  k_real_ = k_align_;
  n_real_ = n_align_;
  NormalizeOutput();
}

void CubeOp::NormalizeOutput() {
  size_t n = std::max(lhs_->nd_.size(), rhs_->nd_.size());
  ndd_.dims.resize(n);
  ndd_.dims[0] = n_real_;
  ndd_.dims[1] = m_real_;
  for (size_t i = 2; i < n; ++i) {
    auto dim1 = i < lhs_->nd_.size() ? lhs_->nd_[i] : 1;
    auto dim2 = i < rhs_->nd_.size() ? rhs_->nd_[i] : 1;
    // TODO: nd_[2] = dim1 >= dim2 ? dim1 : dim2;
    if (dim1 != 1 && dim2 != 1 && dim2 != dim1) {
      ASSERT(0);
    }
    ndd_.dims[i] = std::max(dim1, dim2);
  }
  shape_.Resize(n);
  for (size_t i = 0; i < ndd_.size(); ++i) {
    shape_[i] = nd_[ndd_.size() - 1 - i];
  }
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
  auto bias_size = bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_ping_pong_num = (System::Instance().L1Size() / 2 - bias_size) / ITEM_SIZE[lhs_->type_id_];
  auto k0_max = l1_ping_pong_num / (op->m0 + op->n0);
  op->k0 =
    k0_max < cubeBlockSize ? RoundDown<uint32_t>(k0_max, kBlockSize) : RoundDown<uint32_t>(k0_max, cubeBlockSize);
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

void CubeOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "MatMul";
  if (verbose) {
    oss << "<" << trans_a_ << ", " << trans_b_ << ">";
  }
}

void CubeOp::GetSwizzleConfig(vCubeOp *op) {
  uint32_t swizzle_cnt = DEFAULT_SWIZZLE_COUNT;
  uint32_t visit_type = 0;
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
          visit_type = V_CUBE_SWIZ_VISIT_zN;
          mincost = cost;
          swizzle_cnt = i;
        }
      } else {  // nZ
        uint32_t mem_a_nZ = c * op->n0;
        uint32_t mem_b_nZ = i * op->m0;
        cost = mem_a_nZ + mem_b_nZ;
        if (cost < mincost) {
          visit_type = V_CUBE_SWIZ_VISIT_nZ;
          mincost = cost;
          swizzle_cnt = i;
        }
      }
    }
  } else {
    if (op->m_real > op->n_real) {
      visit_type = V_CUBE_SWIZ_VISIT_nZ;
      uint32_t m_loop = CeilDiv(op->m_real, op->m0);
      swizzle_cnt = std::min(swizzle_cnt, m_loop);
    } else {
      visit_type = V_CUBE_SWIZ_VISIT_zN;
      uint32_t n_loop = CeilDiv(op->n_real, op->n0);
      swizzle_cnt = std::min(swizzle_cnt, n_loop);
    }
  }
  op->swizzle = vCubeOp::SwizzleEncode(visit_type, swizzle_cnt);
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
    // std::cout << "swizzle=(" << cnt << ", " << width << "), hit=(" << major_hit << ", " << minor_hit << "), coef=("
    // << major_coef << ", " << minor_coef << "), cost=" << cost << std::endl;
    if (cost < mincost) {
      mincost = cost;
      swizzle_cnt = cnt;
    }
  }
  return swizzle_cnt;
}

void CubeOp::TileV2(vCubeOp *op) {
  auto l0c_max = System::Instance().L0CSize() / FP32_SIZE;
  auto bias_size = bias_ ? MAX_BIAS_SIZE : 0;
  auto l1_max = (System::Instance().L1Size() / 2 - bias_size) / ITEM_SIZE[lhs_->type_id_];
  auto core_num = System::Instance().CoreNum(CoreType::kCube);
  float mincost = 3.125f;
  uint32_t round_m = RoundUp<uint32_t>(m_align_, BLOCK_SIZE);
  uint32_t round_n = RoundUp<uint32_t>(n_align_, BLOCK_SIZE);
  uint32_t round_k = RoundUp<uint32_t>(k_align_, BLOCK_SIZE);
  auto tile_select = [&](uint32_t x, uint32_t y) {
    // 1. get m0, n0, k0
    uint32_t m0, n0, k0;
    if (!trans_a_) {
      k0 = x;
      n0 = y;
      if (k0 > round_k || n0 > round_n) return;
      uint64_t mx = std::min(l0c_max / n0, (l1_max - k0 * n0) / k0);
      m0 = RoundDown<uint32_t>(mx, mx > CUBE_BLOCK_SIZE ? CUBE_BLOCK_SIZE : BLOCK_SIZE);
      ASSERT((k0 * n0 < l1_max) && (m0 > 0));
      if (m0 > round_m) m0 = round_m;
    } else if (!trans_b_) {  // trans_a && !trans_b_
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
    uint32_t core_loop = m_loop * n_loop * batch_c0_ * batch_c1_;
    uint32_t block_dim = core_loop < core_num ? core_loop : core_num;
    // 3. select swizzle
    uint32_t swizzle_type = m_align_ < n_align_ ? V_CUBE_SWIZ_VISIT_zN : V_CUBE_SWIZ_VISIT_nZ;
    // std::cout << "param: m0=" << m0 << ", n0=" << n0 << ", k0=" << k0 << ", core_loop=" << core_loop << ",
    // block_dim=" << block_dim << ", swizzle_zN=" << swizzle_zN << std::endl;
    uint32_t swizzle_cnt = swizzle_type == V_CUBE_SWIZ_VISIT_zN
                             ? GetSwizzle(n0, m0, n_loop, m_loop, k_real_, !trans_b_, trans_a_, block_dim, mincost)
                             : GetSwizzle(m0, n0, m_loop, n_loop, k_real_, trans_a_, !trans_b_, block_dim, mincost);
    if (swizzle_cnt) {
      op->m0 = m0_ = m0;
      op->n0 = n0_ = n0;
      op->k0 = k0_ = k0;
      op->swizzle = vCubeOp::SwizzleEncode(swizzle_type, swizzle_cnt);
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
  if (tiling_ver != 1) {
    TileV2(op);
  } else {
    Tile(op);
    auto m_loop = CeilDiv(op->m_real, op->m0);
    auto n_loop = CeilDiv(op->n_real, op->n0);
    core_loop_ = m_loop * n_loop * batch_c0_ * batch_c1_;
    auto core_num = System::Instance().CoreNum(CoreType::kCube);
    block_dim_ = core_loop_ < core_num ? core_loop_ : core_num;
    GetSwizzleConfig(op);
  }
}

void CubeOp::CodeGen(vCubeOp *op, CubeTuner *tuner) {
  op->m_align = m_align_;
  op->n_align = n_align_;
  op->k_align = k_align_;
  op->ka_align = ka_align_;
  op->kb_align = kb_align_;
  op->m_real = m_real_;
  op->n_real = n_real_;
  op->k_real = k_real_;
  op->offset_a = offset_a_;
  op->offset_b = offset_b_;
  auto a = static_cast<NDAccess *>(lhs_);
  op->gm_a = a->addr_.data;
  uint32_t batch_a1 = a->nd_.size() > 2 ? static_cast<uint32_t>(a->nd_[2]) : 1;
  uint32_t batch_a0 = a->nd_.size() > 3 ? static_cast<uint32_t>(a->nd_[3]) : 1;
  auto b = static_cast<NDAccess *>(rhs_);
  op->gm_b = b->addr_.data;
  uint32_t batch_b1 = b->nd_.size() > 2 ? static_cast<uint32_t>(b->nd_[2]) : 1;
  uint32_t batch_b0 = b->nd_.size() > 3 ? static_cast<uint32_t>(b->nd_[3]) : 1;
  if (batch_fold_) {
    auto batch_fold = batch_a1 * batch_a0;
    batch_a1 = 1;
    batch_a0 = 1;
    if (m_real_ == ndd_[1]) {
      op->m_align = m_align_ *= batch_fold;
      op->m_real = m_real_ *= batch_fold;
    }
  }
  batch_c0_ = std::max(batch_a0, batch_b0);
  batch_c1_ = std::max(batch_a1, batch_b1);
  op->batch_cast = 0;
  if (batch_c0_ != batch_a0) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_A0;
  if (batch_c0_ != batch_b0) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_B0;
  if (batch_c1_ != batch_a1) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_A1;
  if (batch_c1_ != batch_b1) op->batch_cast |= V_CUBE_BCAST_FLAG_BCAST_B1;
  if (op->batch_cast) op->batch_cast |= batch_c1_ << V_CUBE_BCAST_C1_OFFSET;
  auto c = static_cast<NDAccess *>(output_);
  op->gm_c = c->addr_.data;
  op->flags = trans_a_ ? V_CUBE_FLAG_TRANS_A : 0;
  if (trans_b_) op->flags |= V_CUBE_FLAG_TRANS_B;
  if (type_id_ == dvm::kFloat32) op->flags |= V_CUBE_FLAG_OUT_FP32;
  if (atomic_add_) op->flags |= V_CUBE_FLAG_ATOMIC_ADD;
  if (bias_) {
    ASSERT(bias_->type_id_ == kFloat32 || bias_->type_id_ == kFloat16);
    op->flags |= (V_CUBE_FLAG_BIAS_FP16) * (bias_->type_id_ == kFloat16);
    op->flags |= V_CUBE_FLAG_WITH_BIAS;
    op->gm_bias = static_cast<NDAccess *>(bias_)->addr_.data;
  }
  ASSERT(lhs_->type_id_ == dvm::kFloat16 || lhs_->type_id_ == dvm::kBFloat16);
  uint64_t dtype = lhs_->type_id_ == dvm::kFloat16 ? vCubeOp::FP16 : vCubeOp::BF16;
  op->flags |= dtype << V_CUBE_FLAG_DTYPE_OFFSET;
  if (tuner) {
    tuner->GenTile(this, op);
  } else {
    GenTiling(op);
  }
  op->m_loop = CeilDiv(op->m_real, op->m0);
  op->n_loop = CeilDiv(op->n_real, op->n0);
  op->k_loop = CeilDiv(op->k_real, op->k0);
  op->group_num = core_loop_;
}

GmmOp::GmmOp(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias, NDObject *group_list,
             GroupType group_type)
    : CubeOp(lhs, rhs, trans_a, trans_b, bias), group_list_(group_list), group_type_(group_type) {
  obj_id_ = kGmmOp;
}

void GmmOp::NormalizeOutput() {
  if (group_type_ == kSplit_M) {
    ASSERT(rhs_->shape_ref_->size == 3);
    ASSERT(group_list_->shape_ref_->data[0] == rhs_->shape_ref_->data[0]);
    ASSERT(bias_ == nullptr || bias_->shape_ref_->data[0] == rhs_->shape_ref_->data[0]);
    ndd_.dims.resize(2);
    ndd_.dims[0] = n_real_;
    ndd_.dims[1] = m_real_;
  }
  if (group_type_ == kSplit_K) {
    ndd_.dims.resize(3);
    ndd_.dims[0] = n_real_;
    ndd_.dims[1] = m_real_;
    ndd_.dims[2] = group_list_->shape_ref_->data[0];
  }
  shape_.Resize(ndd_.dims.size());
  for (size_t i = 0; i < ndd_.size(); ++i) {
    shape_[i] = ndd_[ndd_.size() - 1 - i];
  }
}

void GmmOp::GenTiling(vCubeOp *op) {
  Tile(op);
  op->swizzle = vCubeOp::SwizzleEncode(V_CUBE_SWIZ_VISIT_DIAGONAL_Z, DEFAULT_DIAGONAL_SWIZZLE_COUNT);
}

void GmmOp::CodeGen(vCubeOp *op, CubeTuner *tuner) {
  CubeOp::CodeGen(op, nullptr);
  batch_c0_ = 1;
  batch_c1_ = 1;
  op->batch_cast = 0;
  op->flags |= V_CUBE_FLAG_GROUPED_LIST;
  if (group_type_ == kSplit_K) {
    op->flags |= V_CUBE_FLAG_GROUP_K;
  }
  op->gm_group_list = static_cast<NDAccess *>(group_list_)->addr_.data;
  op->group_list_size = group_list_->shape_ref_->data[0];

  block_dim_ = System::Instance().CoreNum(CoreType::kCube);
  if (group_type_ == kSplit_K) {
    core_loop_ = op->m_loop * op->n_loop * op->group_list_size;
  }
  if (group_type_ == kSplit_M) {
    core_loop_ = block_dim_;
  }
  op->group_num = core_loop_;
}

void GmmOp::InferCubeConfig() {
  CubeOp::InferCubeConfig();
  if (group_type_ == kSplit_K) {
    tactics_.enable_splitk = false;
  }
}

static void ReserveCommEvent(VectorKernel &k) {
  if (k.forward_event_num_ > 7) {
    k.forward_event_num_ = 7;
  }
  if (k.backward_event_num_ > 6) {
    k.backward_event_num_ = 6;
  }
}

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
  add_id_ = binary_id_list[kAdd].ids[type_id_];
  shape_ref_ = &shape_;
  multi_load_ = input->obj_id_ == kMultiLoad;
  if (multi_load_) store_lhs_ = false;
}

ReduceScatterOp::~ReduceScatterOp() {
  if (reshape_op_) {
    delete reshape_op_;
  }
}

void ReduceScatterOp::Normalize(std::vector<NDObject *> &run_ops) {
  if (reshape_op_) {
    // recover original lhs_
    lhs_ = reshape_op_->lhs_;
    delete reshape_op_;
    reshape_op_ = nullptr;
  }
  const ShapeRef *input_shape_ref = lhs_->shape_ref_;
  size_t start_idx = lhs_->nd_[lhs_->nd_.size() - 1] != comm_->GetRankSize() ? 1 : 0;
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
    reshape_shape_ = shape_;
    // eg: 4p, [256, 256] -> [4, 64, 256]
    reshape_op_ = new ReshapeOp(lhs_, &reshape_shape_);
    reshape_op_->Normalize(run_ops);
    run_ops.push_back(reshape_op_);
    lhs_ = reshape_op_;
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

void ReduceScatterOp::FoldProp(PropRange &range) {
  if (multi_load_) {
    CommOp::FoldProp(range);
    return;
  }
  int state = 0;  // -1 - reduce ; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (ndd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
}

void ReduceScatterOp::AlignProp(PropRange &range) {
  if (multi_load_) {
    CommOp::AlignProp(range);
    return;
  }
  int state = 0;  // -1 - reduce; 1 - elemwise, 0 - undetemite
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && ndd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (ndd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::REDUCE) {
    range.affine = PropRange::REDUCE;
  }
  range.depth = new_depth;
}

void ReduceScatterOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ReduceScatter";
  if (lhs_ == reshape_op_) {
    oss << "(WithReshape)";
  }
}

void ReduceScatterOp::Tile(const TileParam &tp) {
  if (CollectRoundTile(ndd_.dims, tp, round_tile_) && tp.tail > 0) {
    ASSERT(tail_dim_ == -1);  // restrict: only one unalign tile
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

int ReduceScatterOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
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
  uint64_t forward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event2 = backward_event - 1;
  int code_size = 0;
  uint64_t *current_insn{nullptr};

  // nop
  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[2];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
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
  code_size += vPeerDMA::Encode(current_insn, store_id, vPipe::V_PIPE_STORE, p_store, nullptr);
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
    code_size += vPeerDMA::Encode(current_insn, load_id, vPipe::V_PIPE_LOAD, p_load, rounds);
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

int ReduceScatterOp::MultiLoadEmit(VectorKernel &k) {
  auto load_id = vAccInsnID::V_PEER_LOAD;
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t forward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event = System::Instance().EventNum() - 1;
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
    code_size += vPeerDMA::Encode(current_insn, load_id, vPipe::V_PIPE_LOAD, p_load, nullptr);
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

AllReduceOpBase::AllReduceOpBase(NDObject *input, const Communicator *comm)
    : CommOp(input, comm, ObjectType::kAllReduce) {
  add_id_ = binary_id_list[kAdd].ids[type_id_ == kBFloat16 ? kFloat32 : type_id_];
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
int AllReduceOp<is_bf16>::MatmulEmit(VectorKernel &k) {
  uint64_t *current_insn = insn_;
  uint64_t code_size = 0;
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  uint64_t lead_align = ndd_.lead_stride();
  uint64_t tile_stride = ndd_.stride_back() / lead_align * ndd_.lead_dim();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  uint64_t forward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event = System::Instance().EventNum() - 1;
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
    uint64_t add_dst = xbufs_[1] + per_rank_offset * rank_id;
    uint64_t rhs = xbufs_[0];
    uint64_t rhs2 = 0;
    uint64_t lhs = lhs_->xbuf_ + per_rank_offset * rank_id;
    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      // avoid address after cast
      ASSERT(xbuf_size_ % (2 * SIMD_BLOCK_SIZE) == 0);
      add_dst = xbufs_[1] + xbuf_size_ / 2;
      vUnary op;
      op.xd = add_dst;
      auto cast_id = cast_id_list[kBFloat16][kFloat32];
      op.xn = lhs;
      op.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = add_dst;
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
        auto cast_id = cast_id_list[kBFloat16][kFloat32];
        op.xn = rhs;
        op.count = this_rank_count;
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary add;
      add.xd = add_dst;
      add.xn = is_begin ? lhs : add_dst;
      add.xm = rhs2;
      add.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, add_id_, add);
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
      auto cast_id = cast_id_list[kFloat32][kBFloat16];
      op.xn = add_dst;
      op.count = this_rank_count;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      add_dst = xbufs_[1] + per_rank_offset * rank_id;
    }

    // TwoShot stage 2
    *current_insn |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | forward_event << V_HEAD_SET_EVENT_OFFSET;
    *current_insn |= 0x1ul << V_HEAD_BACK_WAIT_OFFSET | backward_event << V_HEAD_B_WAIT_EVENT_OFFSET;
    vPeerDMA p_store;
    p_store.peer_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_OFFSET;
    p_store.flag_mem = comm_->GetPeerMemPtr(rank_id) + PEERMEM_TWOSHOT_FLAG_OFFSET;
    p_store.xn = add_dst;  // store result of Allreduce
    p_store.tile_stride = tile_stride_size;
    p_store.lenburst = this_rank_lenburst;
    p_store.tail_lenburst = p_store.lenburst;
    p_store.round_rank = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE_MIX, vPipe::V_PIPE_STORE, p_store, nullptr);
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
      code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD_MIX, vPipe::V_PIPE_LOAD, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    }
    // last peerload sync copy
    *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
    vCopy cp;
    cp.xn = xbufs_[1];
    cp.xd = xbuf_;
    cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
    current_insn = insn_ + code_size;
    // code_size += vNop::Encode(current_insn);
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
      auto cast_id = cast_id_list[kBFloat16][kFloat32];
      op.xn = lhs_->xbuf_;
      op.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = xbuf_;
    }

    bool is_begin = true;
    uint64_t add_dst = xbuf_;
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
        auto cast_id = cast_id_list[kBFloat16][kFloat32];
        op.xn = rhs;
        op.count = ndd_.stride_back();
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary add;
      add.xd = add_dst;
      add.xn = is_begin ? lhs : add_dst;
      add.xm = rhs2;
      add.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, add_id_, add);
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
      auto cast_id = cast_id_list[kFloat32][kBFloat16];
      op.xn = add_dst;
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
int AllReduceOp<is_bf16>::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
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
  uint64_t forward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event = System::Instance().EventNum() - 1;
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
  cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
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
  code_size += vPeerDMA::Encode(current_insn, store_id, vPipe::V_PIPE_STORE, p_store, nullptr);
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
    uint64_t add_dst = xbufs_[1] + per_rank_offset * rank_id;  // used as destnation of add

    // bf16 would be cast to f32 to be added
    if constexpr (is_bf16) {
      // avoid address after cast
      ASSERT(xbuf_size_ % (2 * SIMD_BLOCK_SIZE) == 0);
      add_dst = xbufs_[1] + xbuf_size_ / 2;
      vUnary op;
      op.xd = add_dst;
      auto cast_id = cast_id_list[kBFloat16][kFloat32];
      op.xn = lhs;
      op.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      lhs = add_dst;
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
      code_size += vPeerDMA::Encode(current_insn, load_id, vPipe::V_PIPE_LOAD, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

      // bf16 would be cast to f32 to be added
      if constexpr (is_bf16) {
        rhs2 = xbufs_[3];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = cast_id_list[kBFloat16][kFloat32];
        op.xn = rhs;
        op.count = this_rank_block * num_in_block;
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary add;
      add.xd = add_dst;
      add.xn = is_begin ? lhs : add_dst;
      add.xm = rhs2;
      add.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, add_id_, add);
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
      auto cast_id = cast_id_list[kFloat32][kBFloat16];
      op.xn = add_dst;
      op.count = this_rank_block * num_in_block;
      current_insn = insn_ + code_size;
      code_size += vUnary::Encode(current_insn, cast_id, op);
      *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
      add_dst = xbufs_[1] + per_rank_offset * rank_id;
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
    p_store2.xn = add_dst;  // store result of Allreduce
    p_store2.tile_stride = tile_stride_size;
    p_store2.lenburst = this_rank_block;
    p_store2.tail_lenburst = p_store2.lenburst;
    p_store2.round_rank = 0;
    current_insn = insn_ + code_size;
    code_size += vPeerDMA::Encode(current_insn, store_id, vPipe::V_PIPE_STORE, p_store2, nullptr);
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
      code_size += vPeerDMA::Encode(current_insn, load_id, vPipe::V_PIPE_LOAD, p_load, nullptr);
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
    cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
    current_insn = insn_ + code_size;
    // code_size += vNop::Encode(current_insn);
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
      auto cast_id = cast_id_list[kBFloat16][kFloat32];
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
      code_size += vPeerDMA::Encode(current_insn, load_id, vPipe::V_PIPE_LOAD, p_load, nullptr);
      k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;

      // bf16 would be cast to f32 to be added
      if constexpr (is_bf16) {
        rhs2 = xbufs_[3];
        vUnary op;
        op.xd = rhs2;
        auto cast_id = cast_id_list[kBFloat16][kFloat32];
        op.xn = rhs;
        op.count = ndd_.stride_back();
        current_insn = insn_ + code_size;
        code_size += vUnary::Encode(current_insn, cast_id, op);
        *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
      }

      vBinary add;
      add.xd = xbuf_;
      add.xn = is_begin ? lhs : xbuf_;
      add.xm = rhs2;
      add.count = ndd_.stride_back();
      current_insn = insn_ + code_size;
      code_size += vBinary::Encode(current_insn, add_id_, add);
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
      auto cast_id = cast_id_list[kFloat32][kBFloat16];
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

void AllGatherOp::FoldProp(PropRange &range) {
  int state = 0;  // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = range.base; i != range.base - range.depth; --i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

void AllGatherOp::AlignProp(PropRange &range) {
  int state = 0;  // -1 - broadcast; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  for (int i = 0; i < range.depth; ++i) {
    if ((state == -1 && lhs_->nd_[i] > 1) || (state == 1 && lhs_->nd_[i] != ndd_[i])) {
      break;
    }
    if (state == 0) {
      if (lhs_->nd_[i] > 1)
        state = 1;
      else if (lhs_->nd_[i] != ndd_[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < PropRange::BROADCAST) {
    range.affine = PropRange::BROADCAST;
  }
  range.depth = new_depth;
}

int AllGatherOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  int code_size = 0;
  uint64_t *current_insn{nullptr};
  uint64_t forward_event = System::Instance().EventNum() - 1;
  uint64_t backward_event = System::Instance().EventNum() - 1;

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
  cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
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
  code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE, vPipe::V_PIPE_STORE, p_store, rounds);
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
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD, vPipe::V_PIPE_LOAD, p_load, rounds);
    if (i == 0) {
      *current_insn |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | backward_event << V_M_HEAD_WAIT_EVENT_OFFSET;
    }
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
  }

  // last peerload sync copy
  *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
  cp.xn = xbufs_[1];
  cp.xd = xbuf_;
  cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
  current_insn = insn_ + code_size;
  // code_size += vNop::Encode(current_insn);
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

void AllGatherOp::Dump(bool verbose, std::ostringstream &oss) { oss << "AllGather"; }

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

int AllGatherV2Op::Emit(VectorKernel &k) {
  ndd_.UpdateStride(ndd_.dims, k.LeadAlign());
  ReserveCommEvent(k);
  k.code_.InsertWrap(&id_wrap_);
  uint64_t tile_stride = ndd_.stride_back();
  uint64_t tile_stride_size = tile_stride * ITEM_SIZE[type_id_];
  auto rank_size = comm_->GetRankSize();
  auto rank_id = comm_->GetRankId();
  int code_size = 0;
  uint64_t *current_insn{nullptr};
  auto forward_event = System::Instance().EventNum() - 1;
  auto backward_event = System::Instance().EventNum() - 1;

  // nop
  code_size += vNop::Encode(insn_);
  *insn_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;

  // copy
  vCopy cp;
  auto store_xbuf_ = xbufs_[0];
  cp.xn = lhs_->xbuf_;
  cp.xd = store_xbuf_;
  cp.config = DMAConfig(0, 1, GetBlocks(tile_stride), 0, 0);
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
  code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_STORE, vPipe::V_PIPE_STORE, p_store, nullptr);
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
    code_size += vPeerDMA::Encode(current_insn, vAccInsnID::V_PEER_LOAD, vPipe::V_PIPE_LOAD, p_load, nullptr);
    k.comm_op_->id_wrap_.ids_.emplace_back(reinterpret_cast<uint32_t *>(current_insn + vPeerDMA::UNIQUEID_OFFSET));
    if (i == rank_size - 1 || (rank_id == rank_size - 1 && i == rank_size - 2)) {
      *current_insn |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | forward_event << V_M_HEAD_SET_EVENT_OFFSET;
    }
  }

  cp.xn = lhs_->xbuf_;
  cp.xd = xbufs_[rank_id];
  cp.config = DMAConfig(0, 1, (tile_stride_size + 31) >> 5, 0, 0);
  current_insn = insn_ + code_size;
  code_size += vCopy::Encode(current_insn, V_COPY, cp);
  *current_insn |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  *current_insn |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | forward_event << V_HEAD_WAIT_EVENT_OFFSET;
  tail_insn_ = current_insn;
  return code_size;
}

void AllGatherV2Op::Dump(bool verbose, std::ostringstream &oss) { oss << "AllGatherV2"; }
}  // namespace dvm
