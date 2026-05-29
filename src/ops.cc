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
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include "ops.h"
#include "ops_m.h"
#include "kernel.h"
#include "comm.h"

namespace dvm {
namespace {
struct InsnIdTable {
  const char *name;
  vSimdInsnID ids[SIMD_DTYPE_END];
};

static const InsnIdTable unary_id_list[kUnaryTypeEnd] = {
  // must keep consistent order with UnaryType
  {"Sqrt", {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE}},
  {"Abs", {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_ABS_INT32}},
  {"Log", {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE}},
  {"Exp", {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE}},
  {"Reciprocal", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"IsFinite", {V_NONE, V_ISFINITE_FP16, V_ISFINITE_BF16, V_ISFINITE, V_NONE}},
  {"LogicalNot", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"Round", {V_NONE, V_NONE, V_NONE, V_ROUND, V_NONE}},
  {"Floor", {V_NONE, V_NONE, V_NONE, V_FLOOR, V_NONE}},
  {"Ceil", {V_NONE, V_NONE, V_NONE, V_CEIL, V_NONE}},
  {"Trunc", {V_NONE, V_NONE, V_NONE, V_TRUNC, V_NONE}}};

static const InsnIdTable binary_id_list[] = {
  // must keep consistent order with BinaryType
  {"Equal", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"NotEqual", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"Greater", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"GreaterEqual", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"Less", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"LessEqual", {V_NONE, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32}},
  {"Add", {V_NONE, V_ADD_FP16, V_ADD_BF16, V_ADD, V_ADD_INT32}},
  {"Sub", {V_NONE, V_SUB_FP16, V_SUB_BF16, V_SUB, V_SUB_INT32}},
  {"Mul", {V_NONE, V_MUL_FP16, V_MUL_BF16, V_MUL, V_MUL_INT32}},
  {"Div", {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE}},
  {"Pow", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},  // power: individual implement
  {"Maximum", {V_NONE, V_MAX_FP16, V_MAX_BF16, V_MAX, V_MAX_INT32}},
  {"Minimum", {V_NONE, V_MIN_FP16, V_MIN_BF16, V_MIN, V_MIN_INT32}},
  {"LogicalAnd", {V_NONE, V_MIN_FP16, V_MAX_BF16, V_MIN, V_MIN_INT32}},
  {"LogicalOr", {V_NONE, V_MAX_FP16, V_MIN_BF16, V_MAX, V_MAX_INT32}}};

static const InsnIdTable binarys_id_list[] = {
  // must keep consistent order with BinarySOpType
  {"Equal", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"NotEqual", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"Greater", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"GreaterEqual", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"Less", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"LessEqual", {V_NONE, V_CMPS_FP16, V_CMPS_BF16, V_CMPS, V_CMPS_INT32}},
  {"Add", {V_NONE, V_ADDS_FP16, V_ADDS_BF16, V_ADDS, V_ADDS_INT32}},
  {"Mul", {V_NONE, V_MULS_FP16, V_MULS_BF16, V_MULS, V_MULS_INT32}},
  {"Div", {V_NONE, V_DIVS_FP16, V_NONE, V_DIVS, V_NONE}},
  {"ScalarDiv", {V_NONE, V_SDIV_FP16, V_NONE, V_SDIV, V_NONE}},
  {"Maximum", {V_NONE, V_MAXS_FP16, V_MAXS_BF16, V_MAXS, V_MAXS_INT32}},
  {"Minimum", {V_NONE, V_MINS_FP16, V_MINS_BF16, V_MINS, V_MINS_INT32}}};

static const vSimdInsnID cast_id_list[DataType::kDataTypeEnd][DataType::kDataTypeEnd] = {
  {V_NONE, V_CAST_BOOL_TO_FP16, V_NONE, V_NONE, V_NONE, V_NONE},  // V_BOOL
  {V_CAST_FP16_TO_BOOL, V_NONE, V_NONE, V_CAST_FP16_TO_FP32, V_CAST_FP16_TO_INT32, V_NONE},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE, V_CAST_BF16_TO_FP32, V_CAST_BF16_TO_INT32, V_NONE},               // V_BFLOAT16
  {V_NONE, V_CAST_FP32_TO_FP16, V_CAST_FP32_TO_BF16, V_NONE, V_CAST_FP32_TO_INT32,
   V_CAST_FP32_TO_INT64},  // V_FLOAT32
  {V_NONE, V_CAST_INT32_TO_FP16, V_NONE, V_CAST_INT32_TO_FP32, V_NONE, V_CAST_INT32_TO_INT64},  // V_INT32
  {V_NONE, V_NONE, V_NONE, V_CAST_INT64_TO_FP32, V_CAST_INT64_TO_INT32, V_NONE},  // V_INT64
};

static const vSimdInsnID reduce_x_list[][ReduceType::kReduceTypeEnd] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_X_FP16, V_RMIN_X_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_X, V_RMAX_X, V_RMIN_X},          // V_FLOAT32
};

static const vSimdInsnID reduce_y_list[][ReduceType::kReduceTypeEnd] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_Y_FP16, V_RMIN_Y_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_Y, V_RMAX_Y, V_RMIN_Y},          // V_FLOAT32
};

static const scode_t reduce_init_value[][ReduceType::kReduceTypeEnd] = {
  {0, 0, 0},                    // V_BOOL
  {0, 0xFC00, 0x7C00},          // V_FLOAT16
  {0, 0, 0},                    // V_BFLOAT16
  {0, 0xFF800000, 0x7F800000},  // V_FLOAT32
};

uint64_t EmitCopy(bcodeptr_t insn, uint64_t xd, uint64_t xn, uint64_t bytes) {
  if (xd == xn) {
    return vNop::Encode(insn);
  }
  vCopy op;
  op.xd = xd;
  op.xn = xn;
  op.lenburst = CeilDiv(bytes, 32UL);
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

inline ReduceOp *GetStoreReduceOp(NDObject *lhs) {
  if (lhs->obj_id_ == kReduce) {
    return static_cast<ReduceOp *>(lhs);
  }
  if (lhs->obj_id_ == kRemovePad && lhs->lhs_->obj_id_ == kReduce) {
    return static_cast<ReduceOp *>(lhs->lhs_);
  }
  return nullptr;
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

uint64_t SelectSimdWidth(uint64_t iter_size, DataType type_id) {
  uint64_t block_width = ITEM_SIMD_WIDTH_MAX[type_id] >> 3;
  ASSERT(iter_size % block_width == 0);
  uint64_t iter_block = iter_size / block_width;
  if (iter_block <= 8) return iter_size;
  uint64_t simd_block = 8;
  uint64_t repeat = iter_block / simd_block;
  while (repeat * simd_block != iter_block) {
    repeat = iter_block / (--simd_block);
  }
  return simd_block * block_width;
}

uint64_t EmitClearPad(uint64_t *pc, NDObject *op, uint64_t iter_tail, uint64_t simd_width, uint64_t scalar = 0) {
  const static vSimdInsnID id_list[SIMD_DTYPE_END] = {V_NONE, V_CLR_PAD_B16, V_CLR_PAD_B16, V_CLR_PAD, V_CLR_PAD};
  vClearPad clr_op;
  clr_op.xd = op->xbuf_;
  clr_op.iter_size = op->nd_.lead_dim();
  clr_op.iter_stride = op->nd_.lead_stride();
  clr_op.iter_num = op->nd_.stride_back() / clr_op.iter_stride;
  clr_op.simd_width = simd_width;
  clr_op.iter_tail = iter_tail;
  clr_op.scalar = scalar;
  return vClearPad::Encode(pc, id_list[op->type_id_], clr_op);
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

scode_t GetRedInitScalarEncode(int red_op, DataType op_type) {
  ASSERT(op_type == kFloat32 || op_type == kFloat16);
  return reduce_init_value[op_type][red_op];
}

float GetRedInitScalar(int red_op) {
  if (red_op == V_RED_SUM) {
    return 0;
  } else if (red_op == V_RED_MAX) {
    return -std::numeric_limits<float>::infinity();
  } else if (red_op == V_RED_MIN) {
    return std::numeric_limits<float>::infinity();
  }
  return 0;
}
}  // namespace

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

template <int AFFINE>
void BroadReduceFoldProp(const DimArray &small_dim, const DimArray &big_dim, PropRange &range) {
  int state = 0;  // -1 - broadcast/reduce; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  int range_base = range.base;
  if (int small_size = small_dim.size(); range_base >= small_size) {
    if (!small_size) {
      return;
    }
    for (int i = small_size; i < std::min<int>(range_base + 1, big_dim.size()); ++i) {
      if (big_dim[i] != 1) {
        state = -1;
        break;
      }
    }
    range_base = small_size - 1;
    new_depth = range.base - range_base;
  }
  for (int i = range_base; i > range.base - range.depth; --i) {
    if ((state == -1 && small_dim[i] != 1) || (state == 1 && small_dim[i] != big_dim[i])) {
      break;
    }
    if (state == 0) {
      if (small_dim[i] != 1)
        state = 1;
      else if (small_dim[i] != big_dim[i])
        state = -1;
    }
    new_depth++;
  }
  if (state == -1 && range.affine < AFFINE) {
    range.affine = AFFINE;
  }
  range.depth = new_depth;
}
template void BroadReduceFoldProp<PropRange::BROADCAST>(const DimArray &, const DimArray &, PropRange &);
template void BroadReduceFoldProp<PropRange::REDUCE>(const DimArray &, const DimArray &, PropRange &);

template <int AFFINE>
void BroadReduceTileCollect(const DimArray &small_dim, const DimArray &big_dim, TileInfo &info) {
  int state = 0;  // -1 - broadcast/reduce; 1 - elemwise, 0 - undetermined
  int new_depth = 0;
  int depth_size = std::min<int>(info.lead_depth, small_dim.size());
  for (int i = 0; i < depth_size; ++i) {
    if ((state == -1 && small_dim[i] != 1) || (state == 1 && small_dim[i] != big_dim[i])) {
      break;
    }
    if (state == 0) {
      if (small_dim[i] != 1)
        state = 1;
      else if (small_dim[i] != big_dim[i])
        state = -1;
    }
    new_depth++;
  }
  if (state) {
    if (state == -1 && info.lead_affine < AFFINE) {
      info.lead_affine = AFFINE;
    }
    info.lead_depth = new_depth;
  }
}
template void BroadReduceTileCollect<PropRange::BROADCAST>(const DimArray &, const DimArray &, TileInfo &);
template void BroadReduceTileCollect<PropRange::REDUCE>(const DimArray &, const DimArray &, TileInfo &);

vSimdInsnID GetBinaryInsnID(BinaryType op, DataType dtype) { return binary_id_list[op].ids[dtype]; }
vSimdInsnID GetCastInsnID(DataType from, DataType to) { return cast_id_list[from][to]; }

std::ostream &operator<<(std::ostream &oss, const IntArrayRef &shape) {
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

void DumpScalarCode(std::ostringstream &oss, scode_t code, DataType type) {
  switch (type) {
    case kFloat16: {
      oss << Float16(static_cast<uint16_t>(code));
      break;
    }
    case kBFloat16: {
      oss << BFloat16(static_cast<uint16_t>(code));
      break;
    }
    case kFloat32: {
      union {
        float val;
        uint32_t code;
      } data;
      data.code = code;
      oss << data.val;
      break;
    }
    default: {
      oss << code;
      break;
    }
  }
}

MemPool<512, 8192> NDObject::mem_pool_;

static constexpr ObjectMeta GenObjectMeta() {
  struct BaseData {
    CodeGenTmpl tmpl;
    uint32_t flags;
    void (*dim_changed)(NDObject *);
    void (*fold_prop)(NDObject *, PropRange &);
    void (*tile_collect)(NDObject *, TileInfo &);
    void (*shape_prop)(NDObject *, int64_t &);
  };
  constexpr uint32_t F_NS = ObjectMeta::kNddShared;
  constexpr uint32_t F_IP = ObjectMeta::kInplaceProp;
  constexpr uint32_t F_LR = ObjectMeta::kLhsReuse;
  constexpr uint32_t F_RR = ObjectMeta::kRhsReuse;
  constexpr uint32_t F_LD = ObjectMeta::kLhsDom;
  constexpr uint32_t F_DM = ObjectMeta::kDom;
  BaseData data[] = {
    {kGenLoad, 0, nullptr, nullptr, nullptr},                                                        // LoadDummy
    {kGenLoad, 0, nullptr, nullptr, nullptr},                                                        // MultiLoad
    {kGenLoad, F_IP, nullptr, nullptr, nullptr},                                                     // GlobalAccess
    {kGenLoad, F_IP, nullptr, nullptr, NDSimtLoad::TileCollect},                                     // GatherLoad
    {kGenLoad, 0, NDViewLoad::DimChanged, NDViewLoad::FoldProp, NDViewLoad::TileCollect},              // ViewLoad
    {kGenLoad, F_IP, nullptr, nullptr, nullptr},                                                     // Load
    {kGenStore, F_NS | F_LD, nullptr, NDPadStore::FoldProp, NDPadStore::TileCollect},                  // PadStore
    {kGenStore, F_NS | F_LD, NDViewStore::DimChanged, NDViewStore::FoldProp, NDViewStore::TileCollect}, // ViewStore
    {kGenStore, F_IP | F_NS | F_LD, NDStore::DimChanged, nullptr, nullptr},                          // Store
    {kGenComm, F_LR | F_LD, nullptr, ReduceScatterOp::FoldProp, ReduceScatterOp::TileCollect, ReduceScatterOp::ShapeProp},         // ReduceScatter
    {kGenComm, 0, nullptr, AllGatherOp::FoldProp, AllGatherOp::TileCollect, AllGatherOp::ShapeProp},                           // AllGather
    {kGenComm, 0, nullptr, nullptr, CommOp::TileCollect, nullptr},                                                        // AllGatherV2
    {kGenComm, F_LR, nullptr, nullptr, CommOp::TileCollect, nullptr},                                                     // AllReduce
    {kGenSimd1, F_IP | F_LR, nullptr, nullptr, nullptr, ReshapeOp::ShapeProp},                                             // Reshape
    {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                                      // Copy
    {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                                      // Unary
    {kGenSimd2, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr, BinaryOp::ShapeProp},                               // Binary
    {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr},                                             // Cast
    {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr, nullptr},                     // Extract
    {kGenFlex, F_IP | F_NS, nullptr, nullptr, nullptr, nullptr},                                      // Pack
    {kGenSimd1, F_IP | F_NS | F_LR | F_DM, nullptr, nullptr, nullptr},                               // BinaryS
    {kGenSimd1, F_DM, nullptr, _BroadcastOp::FoldProp, _BroadcastOp::TileCollect, BroadcastOp::ShapeProp},                     // BroadcastTo
    {kGenSimd0, F_IP, nullptr, nullptr, nullptr},                                                    // BroadcastS
    {kGenFlex, F_LR | F_LD, _ReduceOp::DimChanged, _ReduceOp::FoldProp, _ReduceOp::TileCollect, ReduceOp::ShapeProp},       // Reduce
    {kGenSimd3, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr},                               // Select
    {kGenSimd1, F_LD, nullptr, nullptr, nullptr},                                                    // ElemAny
    {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr},                                             // RemovePad
    {kGenFlex, F_IP | F_NS, nullptr, nullptr, nullptr, PowerOp::ShapeProp},                                              // Power
    {kGenFlex, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr, CompareOp::ShapeProp},                                // Compare
    {kGenFlex, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},                                       // CompareS
    {kGenSimd1, F_DM, OneHotOp::DimChanged, OneHotOp::FoldProp, OneHotOp::TileCollect, OneHotOp::ShapeProp},                // OneHot
    {kGenSimd1, F_LR, nullptr, nullptr, nullptr, PermuteOp::ShapeProp},                                             // Permute
    {kGenSimd0, 0, nullptr, nullptr, nullptr, CubeOp::ShapeProp},                                                         // CubeOp
    {kGenSimd0, 0, nullptr, nullptr, nullptr, GmmOp::ShapeProp},                                                          // GmmOp
  };

  ObjectMeta meta;
  for (size_t i = 0; i < sizeof(data) / sizeof(BaseData); ++i) {
    meta.flags[i] = data[i].flags;
    meta.tmpl[i] = data[i].tmpl;
    meta.dim_changed[i] = data[i].dim_changed;
    meta.fold_prop[i] = data[i].fold_prop;
    meta.tile_collect[i] = data[i].tile_collect;
    meta.shape_prop[i] = data[i].shape_prop;
  }
  return meta;
}

const ObjectMeta NDObject::meta_ = GenObjectMeta();

class AtomicCleanWrap : public CodeWrap {
 public:
  AtomicCleanWrap(NDAccess *store, int red_op) {
    clear_shape_.data = &clear_shape_data_;
    clear_shape_.size = 1;
    auto type = store->type_id_;
    KernelBuilder b(&kernel_);
    auto op = b.Broadcast(GetRedInitScalar(red_op), &clear_shape_, type);
    store_ = static_cast<NDStore *>(b.Store(store->addr_.gm, op));
  }

  void CodeGen(Code &code, NDAccess *store) {
    clear_shape_data_ = std::accumulate(store->shape_ref_->data, store->shape_ref_->data + store->shape_ref_->size, 1LL,
                                        std::multiplies{});
    if (g_system.deterministic_) {
      clear_shape_data_ += 32 / sizeof(float);
    }
    kernel_.CodeGen();
    code.BindOpFast(store_->addr_, store->addr_);
  }

  int LaunchWrap(void *workspace, void *stream) override {
    kernel_.code_.Launch(nullptr, stream);
    return next_->LaunchWrap(workspace, stream);
  }

  void DasWrap(std::ostringstream &oss) override {
    kernel_.code_.DisAssemble(oss);
    oss << std::endl;
    next_->DasWrap(oss);
  }

  void CollectWrap(std::vector<Code *> &codes) override {
    kernel_.code_.Collect(codes);
    next_->CollectWrap(codes);
  }

 private:
  VKernelD kernel_;
  NDStore *store_;
  IntArrayRef clear_shape_;
  int64_t clear_shape_data_;
};

int64_t NDObject::Size() {
  return std::accumulate(shape_ref_->data, shape_ref_->data + shape_ref_->size, 1LL, std::multiplies{}) *
         ITEM_SIZE[type_id_];
}

void NDObject::Tile(const TileParam &tp) {
  if (auto ndd = Ndd(); ndd != nullptr) {
    auto &dims = ndd->dims;
    bool pointwise = dims[tp.start] != 1;
    for (int i = tp.start + 1; i <= tp.end; ++i) {
      pointwise = pointwise || dims[i] != 1;
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

NDObject *NDObject::Clone(CloneHelper &h) {
  DvmException("[BUG] NDObject::Clone is not inheritted!");
  return nullptr;
}

void NDObject::Dump(bool verbose, std::ostringstream &oss) { oss << "NDObject"; }

uint64_t NDLoadDummy::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  *insn_ = vMakeAccHead(V_LOAD_DUMMY, 0, 1);
  return 1;
}

uint64_t NDGlobalAccess::Emit(VectorKernel &k) { return 0; }

NDObject *NDGlobalAccess::Clone(CloneHelper &h) { return new NDGlobalAccess(addr_.gm, h.GetClone(shape_ref_), type_id_); }

void NDGlobalAccess::Dump(bool verbose, std::ostringstream &oss) { oss << "GlobalAccess"; }

NDObject *NDLoadDummy::Clone(CloneHelper &h) { return new NDLoadDummy(type_id_); }

void NDLoadDummy::Dump(bool verbose, std::ostringstream &oss) { oss << "LoadDummy"; }

void NDLoad::Shard(const ShardParam &sp) {
  if (sp.sink) {
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

uint64_t NDLoad::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }
  int64_t lead_align = ndd_.lead_stride();
  int64_t lead_dim = ndd_.lead_dim();
  uint64_t src_tile_stride_ = ndd_.stride_back() / lead_align * lead_dim;
  if (auto shard = k.shard_) {
    constexpr int SHARD_M = 1;
    constexpr int SHARD_N = 0;
    if (flags_ & OBJ_FLAG_LOAD_FROM_CC_ONCE) {
      *insn_ = vMakeAccHead(V_LOAD_DUMMY, 0, 1);
      vCopy op;
      op.xd = xbuf_;
      op.xn = k.local_mem_size_;
      op.lenburst = GetBlocks(src_tile_stride_);
      return vCopy::Encode(insn_ + 1, V_COPY_CUBE_TILE, op) + 1;
    } else if (flags_ & OBJ_FLAG_LOAD_FROM_CC) {
      vccload op;
      op.xn = xbuf_;
      return vccload::Encode(insn_, vAccInsnID::V_LOAD_CC, op);
    } else if (flags_ & OBJ_FLAG_LOAD_PINGPONG) {
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
  op.pad_size = lead_align * ITEM_SIZE[type_id_] - op.iter_size;
  if (op.body_iter == 1) {
    op.tail_iter = tail_dim_ < 0 ? op.iter_size : tail_size_ * ITEM_SIZE[type_id_];
  } else {
    op.tail_iter = tail_dim_ < 0 ? op.body_iter : op.body_iter / ndd_[tail_dim_] * tail_size_;
    if (op.pad_size == 0) {
      op.tail_iter *= op.iter_size;
      op.iter_size *= op.body_iter;
      op.body_iter = 1;
    }
  }
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

NDObject *NDLoad::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  return new NDLoad(addr_.gm, shape_ref, type_id_);
}

void NDLoad::Dump(bool verbose, std::ostringstream &oss) { oss << "Load"; }

NDGatherLoad::~NDGatherLoad() {
  if (own_index_) {
    delete index_;
  }
}

void NDGatherLoad::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(src_shape_ref_->size >= 1);
  ASSERT(index_ && index_->shape_ref_);
  axis_ = axis_ >= 0 ? axis_ : static_cast<int>(src_shape_ref_->size) + axis_;
  auto *index_shape_ref = index_->shape_ref_;
  shape_.Resize(src_shape_ref_->size - 1 + index_shape_ref->size);
  size_t out_idx = 0;
  gather_size_ = 1;
  inner_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    shape_[out_idx++] = src_shape_ref_->data[i];
  }
  gather_dim_size_ = static_cast<uint64_t>(src_shape_ref_->data[axis_]);
  for (size_t i = 0; i < index_shape_ref->size; ++i) {
    shape_[out_idx++] = index_shape_ref->data[i];
    gather_size_ *= static_cast<uint64_t>(index_shape_ref->data[i]);
  }
  for (size_t i = axis_ + 1; i < src_shape_ref_->size; ++i) {
    shape_[out_idx++] = src_shape_ref_->data[i];
    inner_size_ *= static_cast<uint64_t>(src_shape_ref_->data[i]);
  }
  NDLoad::Normalize(run_ops);
}

uint64_t NDGatherLoad::Emit(VectorKernel &k) {
  ASSERT(k.shard_ == nullptr);
  ndd_.UpdateStride(k.LeadAlign());
  uint64_t rounds[2];
  if (!round_tile_.empty()) {
    BuildDimRounds(round_tile_, rounds);
  }
  const uint64_t lead_align = static_cast<uint64_t>(ndd_.lead_stride());
  const uint64_t lead_dim = static_cast<uint64_t>(ndd_.lead_dim());
  vGatherLoad op;
  op.xn = xbuf_;
  op.from = reinterpret_cast<uint64_t>(addr_.gm);
  op.index = reinterpret_cast<uint64_t>(index_->addr_.gm);
  op.inner_size = inner_size_;
  op.gather_size = gather_size_;
  op.gather_dim_size = gather_dim_size_;
  op.body_iter = static_cast<uint64_t>(ndd_.stride_back()) / lead_align;
  op.iter_size = lead_dim;
  op.pad_size = lead_align - lead_dim;
  op.round_rank = round_tile_.size();
  op.tail_iter = tail_dim_ < 0 ? (op.body_iter == 1 ? op.iter_size : op.body_iter)
                               : (op.body_iter == 1 ? static_cast<uint64_t>(tail_size_)
                                                    : op.body_iter / static_cast<uint64_t>(ndd_[tail_dim_]) *
                                                        static_cast<uint64_t>(tail_size_));
  addr_.Update(insn_ + vGatherLoad::RELOC_OFFSET);
  index_->addr_.Update(insn_ + vGatherLoad::INDEX_RELOC_OFFSET);
  const auto insn_id = ITEM_SIZE[type_id_] == 2 ? vAccInsnID::V_LOAD_GATHER_B16 : vAccInsnID::V_LOAD_GATHER_B32;
  return vGatherLoad::Encode(insn_, insn_id, op, rounds);
}

NDObject *NDGatherLoad::Clone(CloneHelper &h) {
  auto src_shape_ref = h.GetClone(src_shape_ref_);
  auto index = static_cast<NDAccess *>(h.GetClone(index_));
  bool own_index = false;
  if (index == nullptr) {
    index = static_cast<NDAccess *>(index_->CloneUpdate(h));
    own_index = true;
  }
  return new NDGatherLoad(addr_.gm, src_shape_ref, index, axis_, type_id_, own_index);
}

void NDGatherLoad::Dump(bool verbose, std::ostringstream &oss) {
  oss << "GatherLoad";
  if (verbose) {
    oss << "<axis=" << axis_ << ">";
  }
}

void NDViewLoad::DimChanged(NDObject *op) {
  auto view = static_cast<NDViewLoad *>(op);
  auto ref_stride = view->src_stride_ref_->data;
  auto ref_shape = view->shape_ref_->data;
  int ref_idx = view->shape_ref_->size - 1;
  auto &stride = view->src_stride_;
  auto &dims = view->ndd_.dims;
  size_t dim_size = dims.size();
  view->tile_.resize(dim_size);
  stride.resize(dim_size);
  view->tail_dim_ = dim_size;
  int64_t acc_nd = 0;
  int64_t acc_ref = 0;
  int64_t acc_stride = 1;
  for (size_t i = 0; i < dim_size; ++i) {
    if (dims[i] == 1) {
      stride[i] = 0;
    } else if (!acc_nd) {
      acc_stride = ref_stride[ref_idx];
      stride[i] = acc_stride * ITEM_SIZE[view->type_id_];
      if (dims[i] != ref_shape[ref_idx]) {
        acc_nd = dims[i];
        acc_ref = ref_shape[ref_idx];
      }
      ref_idx--;
    } else {
      acc_stride *= dims[i - 1];
      stride[i] = acc_stride * ITEM_SIZE[view->type_id_];
      acc_nd *= dims[i];
      while (acc_ref < acc_nd) {
        acc_ref *= ref_shape[ref_idx--];
      }
      if (acc_nd == acc_ref) {
        acc_nd = 0;
      }
    }
  }
}

void NDViewLoad::TileCollect(NDObject *op, TileInfo &info) {
  auto view = static_cast<NDViewLoad *>(op);
  auto &stride = view->src_stride_;
  if (int max_depth = stride.size(); info.lead_depth > max_depth) {
    info.lead_depth = max_depth;
  }
  auto &dims = view->ndd_.dims;
  for (int d = 1; d < info.lead_depth; ++d) {
    if (stride[d - 1] * dims[d - 1] != stride[d]) {
      info.lead_depth = d;
      break;
    }
  }
  if (g_system.Arch() == AiCoreArch::kAiCore_C220) {
    constexpr uint32_t EXT_WS = 512;
    if (static_cast<uint64_t>(stride[0]) > ITEM_SIZE[view->type_id_] && info.ext_ws < EXT_WS) {
      info.ext_ws = EXT_WS;
    }
    if (info.event_reserve < 2) {
      info.event_reserve = 2;
    }
  }
}

void NDViewLoad::FoldProp(NDObject *op, PropRange &range) {
  auto &stride = static_cast<NDViewLoad *>(op)->src_stride_;
  int stride_size = stride.size();
  if (range.base >= stride_size) {
    if (int max_depth = range.base - stride_size + 1; range.depth > max_depth) {
      range.depth = max_depth;
    }
  } else {
    auto &dims = static_cast<NDViewLoad *>(op)->ndd_.dims;
    for (int d = 1; d < range.depth; ++d) {
      if (stride[range.base - d] * dims[range.base - d] != stride[range.base - d + 1]) {
        range.depth = d;
        break;
      }
    }
  }
}

void NDViewLoad::Normalize(std::vector<NDObject *> &run_ops) {
  auto dim_size = shape_ref_->size;
  ndd_.dims.resize(dim_size);
  src_stride_.resize(dim_size);
  tile_.resize(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    ndd_.dims[i] = shape_ref_->data[dim_size - i - 1];
    src_stride_[i] = ndd_.dims[i] == 1 ? 0 : src_stride_ref_->data[dim_size - i - 1] * ITEM_SIZE[type_id_];
  }
  tail_dim_ = dim_size;
  tail_size_ = 0;
  offset_bytes_ = 0;
}

void NDViewLoad::Tile(const TileParam &tp) {
  if (tp.num > 1) {
    if (auto tile_size = static_cast<size_t>(tp.start) + 1; tile_size >= tile_.size()) {
      for (size_t i = tile_.size(); i < tile_size; ++i) {
        src_stride_[i] = 0;
      }
      src_stride_.resize(tile_size);
      tile_.resize(tile_size);
      tail_dim_ = tile_size;
    }
    tile_[tp.start] = tp.num;
    for (int i = tp.start + 1; i < tail_dim_; ++i) {
      tile_[i] = 0;
    }
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

uint64_t NDViewLoad::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  if (int dim_size = ndd_.size(); tail_dim_ == dim_size) {
    tail_dim_ = dim_size - 1;
  }
  DimArray fold_dim, fold_stride;
  auto lead_idx = ndd_.lead_idx();
  fold_dim[0] = ndd_.dims[lead_idx];
  fold_stride[0] = src_stride_[lead_idx];
  int fold_idx = 0;
  for (int i = lead_idx + 1; i <= tail_dim_; ++i) {
    if (ndd_.dims[i] == 1) continue;
    // TODO: fold optimize for ndd_.lead_stride == ndd.lead_dim
    if (fold_idx == 0 || src_stride_[i - 1] * ndd_.dims[i - 1] != src_stride_[i]) {
      fold_idx++;
      fold_dim[fold_idx] = ndd_.dims[i];
      fold_stride[fold_idx] = src_stride_[i];
    } else {
      fold_dim[fold_idx] *= ndd_.dims[i];
    }
  }
  int tile_start = fold_idx + 1;
  if (int tile_size = static_cast<int>(tile_.size()); tail_dim_ < tile_size) {
    fold_idx++;
    fold_dim[fold_idx] = tile_[tail_dim_];
    fold_stride[fold_idx] = src_stride_[tail_dim_] * ndd_.dims[tail_dim_];
    for (int i = tail_dim_ + 1; i < tile_size; ++i) {
      if (!tile_[i]) continue;
      if (fold_stride[fold_idx] * fold_dim[fold_idx] != src_stride_[i]) {
        fold_idx++;
        fold_dim[fold_idx] = tile_[i];
        fold_stride[fold_idx] = src_stride_[i];
      } else {
        fold_dim[fold_idx] *= tile_[i];
      }
    }
  }
  fold_dim.resize(fold_idx + 1);
  fold_stride.resize(fold_idx + 1);
  int64_t item_size = ITEM_SIZE[type_id_];
  if (fold_stride[0]  > item_size) {
    vViewLoadX op;
    op.xd = xbuf_;
    op.from = addr_.data;
    op.offset = offset_bytes_;
    auto last_store = k.static_ops_.back();
    ASSERT(last_store->IsStore());
    op.ws = last_store->lhs_->xbuf_ + k.tile_size_ * ITEM_SIZE[k.MaxType()];
    auto ws_block_num = (g_system.LocalMemSize() - op.ws) / (SIMD_BLOCK_SIZE * 2);
    op.ws_size = RoundDown(ws_block_num, SIMD_BLOCK_SIZE / item_size);
    op.iter_size = fold_dim[0];
    op.iter_stride= fold_stride[0];
    op.loop_depth = tile_start - 1;
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size_) {
      op.tail_size = op.tail_size / ndd_[tail_dim_] * tail_size_;
    }
    uint64_t *var_insn = insn_ + vViewLoadX::VAR_OFFSET;
    uint64_t dst_stride = ndd_.lead_stride();
    for (int i = 1; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeLoop(fold_dim[i], dst_stride, fold_stride[i]);
      dst_stride *= fold_dim[i];
    }
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewLoadX::RELOC_OFFSET);
    return vViewLoadX::Encode(insn_, item_size == 2 ?  V_LOAD_VIEW_X_B16 : V_LOAD_VIEW_X_B32 , op);
  } else {
    vViewLoad op;
    uint64_t *var_insn = insn_ + vViewLoad::VAR_OFFSET;
    op.xd = xbuf_;
    op.from = addr_.data;
    op.offset = offset_bytes_;
    op.iter_size = fold_dim[0] * item_size;
    // loop space
    int loop_start;
    uint64_t dst_stride = ndd_.lead_stride() * item_size;
    if (tile_start > 1 && fold_stride[0] * fold_dim[0] <= fold_stride[1]) {
      op.src_gap = fold_stride[1] - fold_stride[0] * fold_dim[0];
      op.dst_gap = (ndd_.lead_stride() * item_size - op.iter_size) >> 5;
      op.iter_num = fold_dim[1];
      loop_start = 2;
      dst_stride *= fold_dim[1];
    } else {
      op.src_gap = 0;
      op.dst_gap = 0;
      op.iter_num = 1;
      loop_start = 1;
    }
    op.loop_depth = tile_start - loop_start;
    for (int i = loop_start; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeLoop(fold_dim[i], dst_stride, fold_stride[i]);
      dst_stride *= fold_dim[i];
    }
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size_) {
      op.tail_size = op.tail_size / ndd_[tail_dim_] * tail_size_;
    }
    if (op.loop_depth == 0 && loop_start == 1) {
      op.tail_size *= item_size;
    }
    // tile space
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewLoad::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewLoad::RELOC_OFFSET);
    return vViewLoad::Encode(insn_, V_LOAD_VIEW, op);
  }
}

NDObject *NDViewLoad::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  auto src_stride_ref = h.GetClone(src_stride_ref_);
  return new NDViewLoad(addr_.gm, shape_ref, src_stride_ref, type_id_);
}

void NDViewLoad::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ViewLoad";
  if (verbose) {
    int64_t offset = static_cast<int64_t>(offset_bytes_ / ITEM_SIZE[type_id_]);
    oss << "<" << offset << ", " << *src_stride_ref_ << ">";
  }
}

void NDViewStore::DimChanged(NDObject *op) {
  auto store = static_cast<NDViewStore *>(op);
  auto ref_stride = store->dst_stride_ref_;
  auto ref_shape = store->shape_ref_;
  auto &stride = store->dst_stride_;
  auto &dims = store->nd_;
  size_t dim_size = dims.size();
  store->tile_.resize(dim_size);
  stride.resize(dim_size);
  int64_t acc_nd = 0;
  int64_t acc_ref = 0;
  int64_t acc_stride = 1;
  int ref_idx = ref_shape->size - 1;
  for (size_t i = 0; i < dim_size; ++i) {
    if (dims[i] == 1) {
      stride[i] = 0;
    } else if (!acc_nd) {
      acc_stride = ref_stride->data[ref_idx];
      stride[i] = acc_stride * ITEM_SIZE[store->type_id_];
      if (dims[i] != ref_shape->data[ref_idx]) {
        acc_nd = dims[i];
        acc_ref = ref_shape->data[ref_idx];
      }
      ref_idx--;
    } else {
      acc_stride *= dims[i - 1];
      stride[i] = acc_stride * ITEM_SIZE[store->type_id_];
      acc_nd *= dims[i];
      while (acc_ref < acc_nd) {
        acc_ref *= ref_shape->data[ref_idx--];
      }
      if (acc_nd == acc_ref) {
        acc_nd = 0;
      }
    }
  }
}

void NDViewStore::TileCollect(NDObject *op, TileInfo &info) {
  auto store = static_cast<NDViewStore *>(op);
  auto &stride = store->dst_stride_;
  if (int max_depth = stride.size(); info.lead_depth > max_depth) {
    info.lead_depth = max_depth;
  }
  for (int d = 1; d < info.lead_depth; ++d) {
    if (stride[d - 1] * store->nd_[d - 1] != stride[d]) {
      info.lead_depth = d;
      break;
    }
  }
  if (g_system.Arch() == AiCoreArch::kAiCore_C220) {
    constexpr uint32_t EXT_WS = 512;
    if (static_cast<uint64_t>(stride[0]) > ITEM_SIZE[store->type_id_] && info.ext_ws < EXT_WS) {
      info.ext_ws = EXT_WS;
    }
    if (info.event_reserve < 2) {
      info.event_reserve = 2;
    }
  }
}

void NDViewStore::FoldProp(NDObject *op, PropRange &range) {
  auto store = static_cast<NDViewStore *>(op);
  auto &stride = store->dst_stride_;
  int stride_size = stride.size();
  if (range.base >= stride_size) {
    if (int max_depth = range.base - stride_size + 1; range.depth > max_depth) {
      range.depth = max_depth;
    }
  } else {
    for (int d = 1; d < range.depth; ++d) {
      if (stride[range.base - d] * op->nd_[range.base - d] != stride[range.base - d + 1]) {
        range.depth = d;
        break;
      }
    }
  }
}

void NDViewStore::Normalize(std::vector<NDObject *> &run_ops) {
  nd_ = lhs_->nd_;
  auto dim_size = dst_stride_ref_->size;
  ASSERT(dim_size <= nd_.size());
  dst_stride_.resize(dim_size);
  tile_.resize(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    dst_stride_[i] = nd_[i] == 1 ?  0 : dst_stride_ref_->data[dim_size - i - 1] * ITEM_SIZE[type_id_];
  }
  tail_dim_ = dim_size;
  tail_size_ = 0;
  offset_bytes_ = 0;
}

void NDViewStore::Tile(const TileParam &tp) {
  if (tp.num > 1) {
    if (auto tile_size = static_cast<size_t>(tp.start) + 1; tile_size >= tile_.size()) {
      for (size_t i = tile_.size(); i < tile_size; ++i) {
        dst_stride_[i] = 0;
      }
      dst_stride_.resize(tile_size);
      tile_.resize(tile_size);
      tail_dim_ = tile_size;
    }
    tile_[tp.start] = tp.num;
    for (int i = tp.start + 1; i < tail_dim_; ++i) {
      tile_[i] = 0;
    }
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

uint64_t NDViewStore::Emit(VectorKernel &k) {
  if (int dim_size = dst_stride_ref_->size; tail_dim_ == dim_size) {
    tail_dim_ = dim_size - 1;
  }
  DimArray fold_dim, fold_stride;
  auto lead_idx = nd_.lead_idx();
  fold_dim[0] = nd_[lead_idx];
  fold_stride[0] = dst_stride_[lead_idx];
  int fold_idx = 0;
  for (int i = lead_idx + 1; i <= tail_dim_; ++i) {
    if (nd_[i] == 1) continue;
    if (fold_idx == 0 || dst_stride_[i - 1] * nd_[i - 1] != dst_stride_[i]) {
      fold_idx++;
      fold_dim[fold_idx] = nd_[i];
      fold_stride[fold_idx] = dst_stride_[i];
    } else {
      fold_dim[fold_idx] *= nd_[i];
    }
  }
  int tile_start = fold_idx + 1;
  if (int tile_size = static_cast<int>(tile_.size()); tail_dim_ < tile_size) {
    fold_idx++;
    fold_dim[fold_idx] = tile_[tail_dim_];
    fold_stride[fold_idx] = dst_stride_[tail_dim_] * nd_[tail_dim_];
    for (int i = tail_dim_ + 1; i < tile_size; ++i) {
      if (!tile_[i]) continue;
      if (fold_stride[fold_idx] * fold_dim[fold_idx] != dst_stride_[i]) {
        fold_idx++;
        fold_dim[fold_idx] = tile_[i];
        fold_stride[fold_idx] = dst_stride_[i];
      } else {
        fold_dim[fold_idx] *= tile_[i];
      }
    }
  }
  fold_dim.resize(fold_idx + 1);
  fold_stride.resize(fold_idx + 1);
  int64_t item_size = ITEM_SIZE[type_id_];
  if (fold_stride[0] > item_size) {
    vViewStoreX op;
    op.xn = lhs_->xbuf_;
    op.to = addr_.data;
    op.offset = offset_bytes_;
    auto last_store = k.static_ops_.back();
    ASSERT(last_store->IsStore());
    op.ws = last_store->lhs_->xbuf_ + k.tile_size_ * ITEM_SIZE[k.MaxType()];
    auto ws_block_num = (g_system.LocalMemSize() - op.ws) / (SIMD_BLOCK_SIZE * 2);
    op.ws_size = RoundDown(ws_block_num, SIMD_BLOCK_SIZE / item_size);
    op.iter_size = fold_dim[0];
    op.iter_stride = fold_stride[0];
    op.loop_depth = tile_start - 1;
    op.tail_size = fold_dim[tile_start - 1];
    if (tail_size_) {
      op.tail_size = op.tail_size / nd_[tail_dim_] * tail_size_;
    }
    uint64_t *var_insn = insn_ + vViewStoreX::VAR_OFFSET;
    uint64_t src_stride = nd_.lead_stride() * item_size;
    for (int i = 1; i < tile_start; ++i, ++var_insn) {
      *var_insn = vViewStore::EncodeLoop(fold_dim[i], fold_stride[i], src_stride);
      src_stride *= fold_dim[i];
    }
    uint64_t space = 1;
    while (static_cast<size_t>(tile_start) < fold_dim.size() && fold_stride[tile_start] == 0) {
      space *= fold_dim[tile_start++];
    }
    op.tile_depth = fold_dim.size() - tile_start;
    for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
      *var_insn = vViewStore::EncodeTile(space, fold_stride[i]);
      space *= fold_dim[i];
    }
    addr_.Update(insn_ + vViewStoreX::RELOC_OFFSET);
    return vViewStoreX::Encode(insn_, item_size == 2 ? V_STORE_VIEW_X_B16 : V_STORE_VIEW_X_B32, op);
  }
  vViewStore op;
  uint64_t *var_insn = insn_ + vViewStore::VAR_OFFSET;
  op.xn = lhs_->xbuf_;
  op.to = addr_.data;
  op.offset = offset_bytes_;
  op.iter_size = fold_dim[0] * ITEM_SIZE[type_id_];
  int loop_start;
  uint64_t src_stride = nd_.lead_stride() * ITEM_SIZE[type_id_];
  if (tile_start > 1 && fold_stride[0] * fold_dim[0] <= fold_stride[1]) {
    op.src_gap = (nd_.lead_stride() * ITEM_SIZE[type_id_] - op.iter_size) >> 5;
    op.dst_gap = fold_stride[1] - fold_stride[0] * fold_dim[0];
    op.iter_num = fold_dim[1];
    loop_start = 2;
    src_stride *= fold_dim[1];
  } else {
    op.src_gap = 0;
    op.dst_gap = 0;
    op.iter_num = 1;
    loop_start = 1;
  }
  op.loop_depth = tile_start - loop_start;
  for (int i = loop_start; i < tile_start; ++i, ++var_insn) {
    *var_insn = vViewStore::EncodeLoop(fold_dim[i], fold_stride[i], src_stride);
    src_stride *= fold_dim[i];
  }
  op.tail_size = fold_dim[tile_start - 1];
  if (tail_size_) {
    op.tail_size = op.tail_size / nd_[tail_dim_] * tail_size_;
  }
  if (op.loop_depth == 0 && loop_start == 1) {
    op.tail_size *= ITEM_SIZE[type_id_];
  }
  op.tile_depth = fold_dim.size() - tile_start;
  uint64_t space = 1;
  for (size_t i = tile_start; i < fold_dim.size(); ++i, ++var_insn) {
    *var_insn = vViewStore::EncodeTile(space, fold_stride[i]);
    space *= fold_dim[i];
  }
  addr_.Update(insn_ + vViewStore::RELOC_OFFSET);
  return vViewStore::Encode(insn_, V_STORE_VIEW, op);
}

NDObject *NDViewStore::Clone(CloneHelper &h) {
  auto src = h.GetClone(lhs_);
  auto stride_ref = h.GetClone(dst_stride_ref_);
  return new NDViewStore(addr_.gm, src, stride_ref);
}

void NDViewStore::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ViewStore";
  if (verbose) {
    int64_t offset = static_cast<int64_t>(offset_bytes_ / ITEM_SIZE[type_id_]);
    oss << "<" << offset << ", " << *dst_stride_ref_ << ">";
  }
}

void NDPadStore::Normalize(std::vector<NDObject *> &run_ops) {
  auto size = lhs_->shape_ref_->size;
  shape_.Resize(size);
  for (size_t i = 0; i < size; i++) {
    shape_[i] = lhs_->shape_ref_->data[i];
  }
  shape_[size - 1] += pad_size_;
  nd_ = lhs_->nd_;
}

void NDPadStore::TileCollect(NDObject *op, TileInfo &info) { info.lead_depth = 1; }

void NDPadStore::FoldProp(NDObject *op, PropRange &range) {
  if (range.depth == range.base + 1) range.depth--;
}

uint64_t NDPadStore::Emit(VectorKernel &k) {
  uint64_t lead_align = nd_.lead_stride();
  uint64_t src_tile_stride_ = nd_.stride_back() / lead_align * nd_.lead_dim();
  vSliceSL op;
  auto size = shape_ref_->size;
  op.gm = addr_.gm;
  op.xn = lhs_->xbuf_;
  op.tile_stride = src_tile_stride_;
  op.pad_size = lead_align - nd_.lead_dim();
  op.slice_m = 1;
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
  return vSliceSL::Encode(insn_, vAccInsnID::V_SLICE_STORE, op, nullptr);
}

NDObject *NDPadStore::Clone(CloneHelper &h) { return new NDPadStore(h.GetClone(lhs_), pad_size_); }

void NDPadStore::Dump(bool verbose, std::ostringstream &oss) { oss << "PadStore"; }

void NDConcatStore::Normalize(std::vector<NDObject *> &run_ops) {
  ASSERT(main_->dst_stride_data_.size == lhs_->shape_ref_->size);
  NDViewStore::Normalize(run_ops);
  main_->NormUpdate();
}

uint64_t NDConcatStore::Emit(VectorKernel &k) {
  if (this != main_) {
    k.code_.BindOpFast(addr_, main_->addr_);
  }
  return NDViewStore::Emit(k);
}

NDObject *NDConcatStore::Clone(CloneHelper &h) {
  auto main = static_cast<NDConcatStoreM *>(h.GetClone(main_));
  auto clone = new NDConcatStore(addr_.gm, &main->dst_stride_data_, h.GetClone(lhs_), main);
  if (++main_->group_sync_ == main_->sibling_.size()) {
    main_->group_sync_ = 0;
    for (auto op : main_->sibling_) {
      main->sibling_.push_back(static_cast<NDConcatStore *>(h.GetClone(op)));
    }
  }
  return clone;
}

void NDConcatStore::Dump(bool verbose, std::ostringstream &oss) {
  oss << "ConcatStore";
  if (verbose) {
    int idx;
    if (main_ == this) {
      idx = 0;
    } else {
      idx = -1;
      for (int i = 0; i < static_cast<int>(main_->sibling_.size()); ++i) {
        if (main_->sibling_[i] == this) {
          idx = i + 1;
          break;
        }
      }
    }
    oss << '<' << idx << ", " << main_->concat_dim_ << '>';
  }
}

void NDConcatStoreM::Normalize(std::vector<NDObject *> &run_ops) {
  int ndim = lhs_->shape_ref_->size;
  int concat_dim = concat_dim_ >= 0 ? concat_dim_ : concat_dim_ + ndim;
  shape_.Resize(ndim);
  for (int d = 0; d < ndim; ++d) {
    shape_[d] = lhs_->shape_ref_->data[d];
  }
  shape_[concat_dim] *= static_cast<int64_t>(sibling_.size() + 1);
  dst_stride_data_.Resize(ndim);
  dst_stride_data_[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d) {
    dst_stride_data_[d] = dst_stride_data_[d + 1] * shape_[d + 1];
  }
  NDViewStore::Normalize(run_ops);
}

void NDConcatStoreM::_NormUpdate() {
  int ndim = lhs_->shape_ref_->size;
  int concat_dim = concat_dim_ >= 0 ? concat_dim_ : concat_dim_ + ndim;
  int64_t concat_size = lhs_->shape_ref_->data[concat_dim];
  for (auto op : sibling_) {
    concat_size += op->shape_ref_->data[concat_dim];
  }
  if (shape_[concat_dim] != concat_size) {
    // TODO: DimChanged
    ASSERT(dst_stride_.size() == static_cast<size_t>(ndim));
    shape_[concat_dim] = concat_size;
    for (int d = concat_dim - 1; d >= 0; --d) {
      dst_stride_data_[d] = dst_stride_data_[d + 1] * shape_[d + 1];
      dst_stride_[ndim - 1 - d] = dst_stride_data_[d] * ITEM_SIZE[type_id_];
    }
    for (auto op : sibling_) {
      ASSERT(op->dst_stride_.size() == static_cast<size_t>(ndim));
      for (int d = ndim - concat_dim; d < ndim; ++d) {
        op->dst_stride_[d] = dst_stride_[d];
      }
    }
  }
  offset_bytes_ = 0;
  int64_t offset_elem = lhs_->shape_ref_->data[concat_dim];
  int64_t stride_bytes = dst_stride_data_[concat_dim] * ITEM_SIZE[type_id_];
  for (auto op : sibling_) {
    op->offset_bytes_ = offset_elem * stride_bytes;
    offset_elem += op->shape_ref_->data[concat_dim];
  }
}

NDObject *NDConcatStoreM::Clone(CloneHelper &h) {
  return new NDConcatStoreM(addr_.gm, h.GetClone(lhs_), main_->concat_dim_);
}

void NDStore::Normalize(std::vector<NDObject *> &run_ops) {
  nd_ = lhs_->nd_;
  tail_dim_ = -1;
  tail_size_ = 0;
  round_tile_.resize(0);
  UpdateDimMask();
}

void NDStore::Shard(const ShardParam &sp) {
  if (sp.sink) {
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

uint64_t NDStore::Emit(VectorKernel &k) {
  int64_t lead_align = nd_.lead_stride();
  int64_t lead_dim = nd_.lead_dim();
  bool no_pad = lhs_->obj_id_ == kRemovePad;
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
    if (auto red_op = GetStoreReduceOp(lhs_); red_op != nullptr) {
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
          op.pad_size = no_pad ? 0 : lead_align * ITEM_SIZE[type_id_] - op.iter_size;
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
        op.cum_flag = (!red_op->CheckFlag(OBJ_FLAG_REDUCE_NO_CUM)) && (round_tile_.size() & 1);
        op.iter_size = lead_dim * ITEM_SIZE[type_id_];
        op.pad_size = no_pad ? 0 : lead_align * ITEM_SIZE[type_id_] - op.iter_size;
        op.iter_num = nd_.stride_back() / lead_align;
        if (tail_dim_ < 0 || red_op->InRange(tail_dim_)) {
          op.iter_tail = op.iter_num;
        } else {
          ASSERT(op.iter_num > 1);  // inner reduce is divided. outer reduce is not lead
          op.iter_tail = op.iter_num / nd_[tail_dim_] * tail_size_;
        }
        op.tile_stride = dst_tile_stride_ * ITEM_SIZE[type_id_];
        op.round_rank = round_tile_.size();
        op.red_op = red_op->red_op_;
        op.atmoic_type = (type_id_ == kFloat32 ? V_ATOMIC_FP32 : V_ATOMIC_FP16);
        addr_.Update(insn_ + vStoreAtomic::RELOC_OFFSET);
        code_size = vStoreAtomic::Encode(insn_, V_STORE_ATOMIC, op, rounds);
        auto clean_wrap = red_op->clean_wrap_;
        if (clean_wrap == nullptr) {
          red_op->clean_wrap_ = clean_wrap = new AtomicCleanWrap(this, red_op->red_op_);
        }
        clean_wrap->CodeGen(code, this);
        code.InsertWrap(clean_wrap);
        return code_size;
      }
    }
  }
  if (lhs_->GetObjectType() == kAllGatherV2) {
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
  if (k.shard_) {
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
  uint64_t pad_size = no_pad ? 0 : lead_align * ITEM_SIZE[type_id_] - iter_size;
  uint64_t body_iter = nd_.stride_back() / lead_align;
  uint64_t tail_iter;
  if (body_iter == 1) {
    tail_iter = tail_dim_ < 0 ? iter_size : tail_size_ * ITEM_SIZE[type_id_];
  } else {
    tail_iter = tail_dim_ < 0 ? body_iter : body_iter / nd_[tail_dim_] * tail_size_;
    if (pad_size == 0) {
      tail_iter *= iter_size;
      iter_size *= body_iter;
      body_iter = 1;
    }
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

NDObject *NDStore::Clone(CloneHelper &h) { return new NDStore(h.GetClone(lhs_)); }

void NDStore::Dump(bool verbose, std::ostringstream &oss) { oss << "Store"; }

void NDStore::DimChanged(NDObject *op) { static_cast<NDStore *>(op)->UpdateDimMask(); }

uint64_t CopyOp::Emit(VectorKernel &k) {
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, nd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *CopyOp::Clone(CloneHelper &h) { return new CopyOp(h.GetClone(lhs_)); }

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

uint64_t ReshapeOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  ASSERT(lhs_->nd_.lead_dim() == ndd_.lead_dim());
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *ReshapeOp::Clone(CloneHelper &h) {
  auto dst_shape_ref = h.GetClone(dst_shape_ref_);
  return new ReshapeOp(h.GetClone(lhs_), dst_shape_ref);
}

void ReshapeOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Reshape"; }

bool ReshapeOp::VisitChangeRange(ChangeRange &range) {
  int out_size = ndd_.size();
  int in_size = lhs_->nd_.size();
  while (range.begin < out_size && range.begin < in_size && ndd_[range.begin] == lhs_->nd_[range.begin]) {
    range.begin++;
  }
  int out_idx = range.begin;
  int in_idx = range.begin;
  if (out_idx < out_size && in_idx < in_size) {
    size_t in_prod = lhs_->nd_[in_idx++];
    size_t out_prod = ndd_[out_idx++];
    while (in_prod != out_prod) {
      if (in_prod < out_prod) {
        in_prod *= lhs_->nd_[in_idx++];
      } else {
        out_prod *= ndd_[out_idx++];
      }
    }
  }
  if (in_idx == range.begin || out_idx == range.begin) {
    return false;
  }
  int in_one = 0;
  int out_one = 0;
  while ((in_idx + in_one) < in_size && lhs_->nd_[in_idx + in_one] == 1) in_one++;
  while ((out_idx + out_one) < out_size && ndd_[out_idx + out_one] == 1) out_one++;
  if (in_one != out_one) {
    if (in_one > out_one) {
      in_idx += in_one - out_one;
    } else {
      out_idx += out_one - in_one;
    }
  }
  range.size = out_idx - range.begin;
  range.in_size = in_idx - range.begin;
  return range.size || range.in_size;
}

void ReshapeOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ReshapeOp *>(op);
  auto &shape = self->shape_;
  auto dst_shape = self->dst_shape_ref_;
  shape.Resize(dst_shape->size);
  for (size_t i = 0; i < dst_shape->size; ++i) {
    shape[i] = dst_shape->data[i];
  }
}

void PermuteOp::Normalize(std::vector<NDObject *> &run_ops) {
  size_t dim_size = lhs_->shape_ref_->size;
  for (size_t k = 0; k < dim_size; ++k) {
    perm_[k] = dim_size - 1 - dims_ref_->data[dim_size - 1 - k];
  }
  perm_.resize(dim_size);
  ndd_.dims.resize(dim_size);
  shape_.Resize(dim_size);
  for (size_t k = 0; k < dim_size; ++k) {
    ndd_.dims[k] = lhs_->nd_.data->dims[perm_[k]];
  }
  for (size_t k = 0; k < dim_size; ++k) {
    shape_[dim_size - 1 - k] = ndd_.dims[k];
  }
}

uint64_t PermuteOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
}

NDObject *PermuteOp::Clone(CloneHelper &h) { return new PermuteOp(h.GetClone(lhs_), dims_ref_); }

void PermuteOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Permute"; }

void PermuteOp::FoldProp(NDObject *op, PropRange &range) {
  if (!(op->flags_ & OBJ_FLAG_BROKER_AFFINED)) {
    range.depth = 1;
  }
}

void PermuteOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<PermuteOp *>(op);
  auto &shape = self->shape_;
  auto dims_ref = self->dims_ref_;
  auto input_shape = self->lhs_->shape_ref_;
  shape.Resize(dims_ref->size);
  for (size_t i = 0; i < dims_ref->size; ++i) {
    shape[i] = input_shape->data[dims_ref->data[i]];
  }
}

uint64_t UnaryOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  ASSERT(size_t(op_type_) < sizeof(unary_id_list) / sizeof(InsnIdTable));
  auto id = unary_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

NDObject *UnaryOp::Clone(CloneHelper &h) { return new UnaryOp(op_type_, h.GetClone(lhs_)); }

void UnaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << unary_id_list[op_type_].name; }

uint64_t RemovePadOp::Emit(VectorKernel &k) {
  const static vSimdInsnID id_list[SIMD_DTYPE_END] = {V_NONE, V_REMOVEPAD_U16, V_REMOVEPAD_U16, V_REMOVEPAD, V_REMOVEPAD};
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

NDObject *RemovePadOp::Clone(CloneHelper &h) { return new RemovePadOp(h.GetClone(lhs_)); }

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

uint64_t ElementAnyOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
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
  uint64_t size = 0;
  if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail) {
    size = EmitClearPad(insn_, lhs_, clr_tail, op.simd_width);
    tail_insn_ = insn_ + size;
    insn_num++;
  }
  size += vElementAny::Encode(tail_insn_, V_ELEMENT_ANY, op);

  if (insn_num > 1) {
    *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  }
  return size;
}

NDObject *ElementAnyOp::Clone(CloneHelper &h) { return new ElementAnyOp(h.GetClone(lhs_)); }

void ElementAnyOp::Dump(bool verbose, std::ostringstream &oss) { oss << "ElementAny"; }

uint64_t CastOp::Emit(VectorKernel &k) {
  vUnary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  auto id = cast_id_list[lhs_->type_id_][type_id_];
  ASSERT(id != V_NONE);
  return vUnary::Encode(insn_, id, op);
}

NDObject *CastOp::Clone(CloneHelper &h) { return new CastOp(h.GetClone(lhs_), type_id_); }

void CastOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Cast"; }

void ExtractOp::Normalize(std::vector<NDObject *> &run_ops) { nd_ = lhs_->nd_; }

uint64_t ExtractOp::Emit(VectorKernel &k) {
  vExtract op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = nd_.stride_back();
  op.slot = slot_;
  ASSERT(type_id_ == DataType::kInt32 || type_id_ == DataType::kFloat32);
  return vExtract::Encode(insn_, V_EXTRACT_B32, op);
}

NDObject *ExtractOp::Clone(CloneHelper &h) { return new ExtractOp(h.GetClone(lhs_), slot_, type_id_); }

void ExtractOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "Extract<" << slot_ << '>';
}

uint64_t PackOp::Emit(VectorKernel &k) {
  vPack op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  op.count = nd_.stride_back() * 2;
  op.ws = wss_[0];
  return vPack::Encode(insn_, V_PACK_B32, op);
}

NDObject *PackOp::Clone(CloneHelper &h) { return new PackOp(h.GetClone(lhs_), h.GetClone(rhs_)); }

void PackOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Pack"; }

uint64_t BinaryScalarOp::Emit(VectorKernel &k) {
  vBinaryS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = scalar_;
  ASSERT(size_t(op_type_) < sizeof(binarys_id_list) / sizeof(InsnIdTable));
  auto id = binarys_id_list[op_type_].ids[type_id_];
  ASSERT(id != V_NONE);
  return vBinaryS::Encode(insn_, id, op);
}

NDObject *BinaryScalarOp::Clone(CloneHelper &h) { return new BinaryScalarOp(op_type_, h.GetClone(lhs_), scalar_); }

void BinaryScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << binarys_id_list[op_type_].name;
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

uint64_t CompareScalarOp::Emit(VectorKernel &k) {
  vCompareS op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.count = nd_.stride_back();
  op.scalar = scalar_;
  op.ws = wss_[0];
  op.type = cmp_op_;
  auto id = binarys_id_list[cmp_op_].ids[type_id_];
  return vCompareS::Encode(insn_, id, op);
}

NDObject *CompareScalarOp::Clone(CloneHelper &h) { return new CompareScalarOp(cmp_op_, h.GetClone(lhs_), scalar_); }

void CompareScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "CompareS";
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

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
    if (lhs_axis == rhs_axis) {
      nd[i] = lhs_axis;
    } else if (lhs_axis == 1) {
      nd[i] = rhs_axis;
      lhs_need_broadcast = true;
    } else {
      nd[i] = lhs_axis;
      rhs_need_broadcast = true;
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

uint64_t BinaryOp::Emit(VectorKernel &k) {
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

NDObject *BinaryOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ops_.empty() ? lhs_ : norm_.lhs_stuff_ops_.front()->lhs_;
  auto rhs = norm_.rhs_stuff_ops_.empty() ? rhs_ : norm_.rhs_stuff_ops_.front()->lhs_;
  return new BinaryOp(op_type_, h.GetClone(lhs), h.GetClone(rhs));
}

void BinaryOp::Dump(bool verbose, std::ostringstream &oss) { oss << binary_id_list[op_type_].name; }

static void BinaryShapeProp(const IntArrayRef *lhs, const IntArrayRef *rhs, ShapeWithRef &shape,
                            int64_t &sym_dim_next) {
  if (lhs->size < rhs->size) {
    auto tmp = lhs;
    lhs = rhs;
    rhs = tmp;
  }
  shape.Resize(lhs->size);
  auto diff = lhs->size - rhs->size;
  for (size_t i = 0; i < diff; ++i) {
    shape[i] = lhs->data[i];
  }
  for (size_t i = diff; i < lhs->size; ++i) {
    auto l_dim = lhs->data[i];
    auto r_dim = rhs->data[i - diff];
    if (l_dim == r_dim || r_dim == 1) {
      shape[i] = l_dim;
    } else if (l_dim == 1) {
      shape[i] = r_dim;
    } else {
      shape[i] = sym_dim_next--;
    }
  }
}

void BinaryOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<BinaryOp *>(op)->norm_.shape_,
                         sym_dim_next);
}

uint64_t CompareOp::Emit(VectorKernel &k) {
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

NDObject *CompareOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ops_.empty() ? lhs_ : norm_.lhs_stuff_ops_.front()->lhs_;
  auto rhs = norm_.rhs_stuff_ops_.empty() ? rhs_ : norm_.rhs_stuff_ops_.front()->lhs_;
  return new CompareOp(cmp_op_, h.GetClone(lhs), h.GetClone(rhs));
}

void CompareOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<CompareOp *>(op)->norm_.shape_,
                         sym_dim_next);
}

uint64_t PowerOp::Emit(VectorKernel &k) {
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

NDObject *PowerOp::Clone(CloneHelper &h) {
  auto lhs = norm_.lhs_stuff_ops_.empty() ? lhs_ : norm_.lhs_stuff_ops_.front()->lhs_;
  auto rhs = norm_.rhs_stuff_ops_.empty() ? rhs_ : norm_.rhs_stuff_ops_.front()->lhs_;
  return new PowerOp(h.GetClone(lhs), h.GetClone(rhs));
}

void PowerOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Power"; }

void PowerOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  return BinaryShapeProp(op->lhs_->shape_ref_, op->rhs_->shape_ref_, static_cast<PowerOp *>(op)->norm_.shape_,
                         sym_dim_next);
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
  if (flags_ & OBJ_FLAG_EAGER) {
    auto insert_broadcast = [&run_ops, &nd](NDObject *input) -> NDObject * {
      size_t stuff_idx = run_ops.size();
      return InsertImplicitBroadcast(input, nd, run_ops, stuff_idx);
    };
    if (lhs_bc) lhs_ = insert_broadcast(lhs_);
    if (rhs_bc) rhs_ = insert_broadcast(rhs_);
    if (xhs_bc) xhs_ = insert_broadcast(xhs_);
  } else {
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
  }
  nd_ = lhs_->nd_;
}

SelectOp::~SelectOp() {
  for (size_t j = 0; j < 3; ++j) {
    for (auto op : stuff_ops_[j]) {
      delete op;
    }
  }
}

uint64_t SelectOp::Emit(VectorKernel &k) {
  vSelect op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  const static vSimdInsnID id_list[kDataTypeEnd] = {V_NONE, V_SEL_FP16, V_SEL_BF16, V_SEL, V_SEL_INT32};
  op.count = nd_.stride_back();
  op.xm = rhs_->xbuf_;
  op.cond = xhs_->xbuf_;
  op.ws = wss_[0];
  ASSERT(id_list[type_id_] != V_NONE);
  return vSelect::Encode(insn_, id_list[type_id_], op);
}

NDObject *SelectOp::Clone(CloneHelper &h) {
  auto lhs = stuff_ops_[0].empty() ? lhs_ : stuff_ops_[0].front()->lhs_;
  auto rhs = stuff_ops_[1].empty() ? rhs_ : stuff_ops_[1].front()->lhs_;
  auto xhs = stuff_ops_[2].empty() ? xhs_ : stuff_ops_[2].front()->lhs_;
  return new SelectOp(h.GetClone(xhs), h.GetClone(lhs), h.GetClone(rhs));
}

void SelectOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Select"; }

void _BroadcastOp::FoldProp(NDObject *op, PropRange &range) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<_BroadcastOp *>(op)->ndd_;
  BroadReduceFoldProp<PropRange::BROADCAST>(lhs_nd.data->dims, ndd.dims, range);
}

void _BroadcastOp::TileCollect(NDObject *op, TileInfo &info) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<_BroadcastOp *>(op)->ndd_;
  BroadReduceTileCollect<PropRange::BROADCAST>(lhs_nd.data->dims, ndd.dims, info);
}

uint64_t _BroadcastOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
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
  uint64_t offset;
  if (start_dim == 0 || start_dim == ndd_.lead_idx()) {
    offset = EmitBroadcastX(insn_, end_dim);
  } else {
    offset = EmitBroadcastY(insn_, start_dim, end_dim);
  }
  return offset;
}

uint64_t _BroadcastOp::EmitBroadcastX(uint64_t *p, int end_dim) {
  vBroadcastX op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.count = ndd_.stride(end_dim);
  int64_t rank_size = static_cast<int64_t>(ndd_.size());
  op.lead_num = end_dim + 1 < rank_size ? ndd_[end_dim + 1] : 1;
  op.iter_num = end_dim + 2 < rank_size ? ndd_.stride_back() / ndd_.stride(end_dim + 1) : 1;
  op.lead_pad = lhs_->nd_.lead_stride() - lhs_->nd_.lead_dim();
  const static vSimdInsnID id_list[SIMD_DTYPE_END] = {V_NONE, V_BROADCAST_X_B16, V_BROADCAST_X_B16, V_BROADCAST_X_B32,
                                                      V_BROADCAST_X_B32};
  ASSERT(id_list[type_id_] != V_NONE);
  return vBroadcastX::Encode(p, id_list[type_id_], op);
}

uint64_t _BroadcastOp::EmitBroadcastY(uint64_t *p, int start_dim, int end_dim) {
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

NDObject *BroadcastOp::Clone(CloneHelper &h) {
  auto lhs = stuff_ops_.empty() ? lhs_ : stuff_ops_.front()->lhs_;
  return new BroadcastOp(h.GetClone(lhs), dst_shape_ref_);
}

void BroadcastOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<BroadcastOp *>(op);
  auto &shape = self->shape_;
  auto dst_shape = self->dst_shape_ref_;
  shape.Resize(dst_shape->size);
  for (size_t i = 0; i < dst_shape->size; ++i) {
    shape[i] = dst_shape->data[i];
  }
}

BroadcastScalarOp::~BroadcastScalarOp() {
  if (dummy_load_) {
    delete dummy_load_;
  }
}

void BroadcastScalarOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update nd_ from shape_ref_
  auto dims = shape_ref_->size;
  ndd_.dims.resize(dims);
  for (size_t i = 0; i < dims; ++i) {
    ndd_.dims[i] = shape_ref_->data[dims - i - 1];
  }
  if (flags_ & OBJ_FLAG_EAGER) {
    lhs_ = new NDLoadDummy(type_id_);
    lhs_->SetFlag(OBJ_FLAG_EAGER);
    run_ops.push_back(lhs_);
  } else if (run_ops.empty()) {
    if (dummy_load_ == nullptr) {
      dummy_load_ = new NDLoadDummy(type_id_);
    }
    lhs_ = dummy_load_;
    run_ops.push_back(dummy_load_);
  } else {
    lhs_ = nullptr;
  }
}

uint64_t BroadcastScalarOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  vBroadcastS op;
  op.scalar = scalar_;
  op.xd = xbuf_;
  op.count = ndd_.stride_back();
  return vBroadcastS::Encode(insn_, ITEM_SIZE[type_id_] == sizeof(uint32_t) ? V_BROADCAST_S : V_BROADCAST_S_B16, op);
}

NDObject *BroadcastScalarOp::Clone(CloneHelper &h) {
  auto shape_ref = h.GetClone(shape_ref_);
  return new BroadcastScalarOp(scalar_, shape_ref, type_id_);
}

void BroadcastScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "BroadcastS";
  if (verbose) {
    oss << "<";
    DumpScalarCode(oss, scalar_, type_id_);
    oss << ">";
  }
}

BroadcastInt64ScalarOp::BroadcastInt64ScalarOp(uint64_t scalar, IntArrayRef *shape_ref)
    : BroadcastScalarOp(static_cast<scode_t>(scalar & 0xfffffffful), shape_ref, DataType::kInt64),
      high_(static_cast<scode_t>((scalar >> 32) & 0xfffffffful)) {}

uint64_t BroadcastInt64ScalarOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  vBroadcastS_B64 op;
  op.scalar = scalar_;
  op.high = high_;
  op.xd = xbuf_;
  op.count = ndd_.stride_back();
  return vBroadcastS_B64::Encode(insn_, V_BROADCAST_S_B64, op);
}

NDObject *BroadcastInt64ScalarOp::Clone(CloneHelper &h) {
  uint64_t scalar = (static_cast<uint64_t>(high_) << 32) | scalar_;
  return new BroadcastInt64ScalarOp(scalar, h.GetClone(shape_ref_));
}

void BroadcastInt64ScalarOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "BroadcastS";
  if (verbose) {
    uint64_t scalar = (static_cast<uint64_t>(high_) << 32) | scalar_;
    oss << "<" << static_cast<int64_t>(scalar) << ">";
  }
}

void _ReduceOp::FoldProp(NDObject *op, PropRange &range) {
  auto &lhs_nd = op->lhs_->nd_;
  auto &ndd = static_cast<_ReduceOp *>(op)->ndd_;
  BroadReduceFoldProp<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, range);
}

void _ReduceOp::TileCollect(NDObject *op, TileInfo &info) {
  auto self = static_cast<_ReduceOp *>(op);
  if (self->ws_num_ == 2 && info.event_reserve < 2) {
    info.event_reserve = 2;
  }
  auto &lhs_nd = self->lhs_->nd_;
  auto &ndd = self->ndd_;
  BroadReduceTileCollect<PropRange::REDUCE>(ndd.dims, lhs_nd.data->dims, info);
  info.flags = ObjectMeta::kSimdDim;
}

void _ReduceOp::Tile(const TileParam &tp) {
  if (tp.tail > 0) {
    tail_dim_ = tp.start;
    tail_size_ = tp.tail;
  }
  NDObject::Tile(tp);
}

uint64_t _ReduceOp::Emit(VectorKernel &k) {
  if (ndd_.strides.empty()) {
    ndd_.UpdateStride(k.LeadAlign());
  }
  ASSERT(red_op_ < ReduceType::kReduceTypeEnd);
  if (ndd_.lead_dim() == lhs_->nd_.lead_dim() && ndd_.stride_back() == lhs_->nd_.stride_back()) {
    return EmitCopy(insn_, xbuf_, lhs_->xbuf_, ndd_.stride_back() * ITEM_SIZE[type_id_]);
  } else if (start_dim_ <= lhs_->nd_.lead_idx()) {  // reduce x
    ASSERT(start_dim_ >= 0 && end_dim_ >= 0);
    vReduceX op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.red_size = lhs_->nd_.stride(end_dim_);
    uint64_t clr_tail = 0;
    if (tail_dim_ == -1 || tail_dim_ > end_dim_) {
      op.simd_width = SelectSimdWidth(op.red_size, type_id_);
      op.red_tail = op.red_size;
    } else if (tail_dim_ > lhs_->nd_.lead_idx()) {
      op.simd_width = SelectSimdWidth(lhs_->nd_.stride(tail_dim_ - 1), type_id_);
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * static_cast<uint64_t>(tail_size_);
    } else {
      op.simd_width = SelectSimdWidth(lhs_->nd_.lead_stride(), type_id_);
      auto tail_size = static_cast<uint64_t>(tail_size_);
      op.red_tail = RoundUp(tail_size, op.simd_width);
      if (op.red_tail != tail_size) {
        clr_tail = tail_size;
      }
    }
    op.dup_size = lhs_->nd_.stride_back() / lhs_->nd_.stride(end_dim_);
    if (auto lead_dim = ndd_.lead_dim(); lead_dim > 1) {
      op.dup_block = lead_dim;
      op.dup_pad = ndd_.lead_stride() - lead_dim;
    } else {
      op.dup_block = op.dup_size;
      op.dup_pad = 0;
    }
    uint64_t size = 0;
    uint32_t insn_num = 1;
    if (lhs_->nd_.lead_dim() != lhs_->nd_.lead_stride() || clr_tail > 0) {
      size = EmitClearPad(insn_, lhs_, clr_tail, op.simd_width, GetRedInitScalarEncode(red_op_, type_id_));
      tail_insn_ = insn_ + size;
      insn_num++;
    }
    size += vReduceX::Encode(tail_insn_, reduce_x_list[type_id_][red_op_], op);
    if (insn_num > 1) {
      *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
    }
    return size;
  } else {
    vReduceY op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.iter_size = ndd_.stride(start_dim_);
    op.red_size = lhs_->nd_.stride(end_dim_) / op.iter_size;
    ASSERT(op.red_size > 1);
    if (InRange(tail_dim_)) {
      op.red_tail = op.red_size / lhs_->nd_[tail_dim_] * tail_size_;
    } else {
      op.red_tail = op.red_size;
    }
    op.dup_num = ndd_.stride_back() / ndd_.stride(end_dim_);
    return vReduceY::Encode(insn_, reduce_y_list[type_id_][red_op_], op);
  }
}

void _ReduceOp::Dump(bool verbose, std::ostringstream &oss) { oss << "Reduce"; }

void _ReduceOp::DimChanged(NDObject *op) {
  auto &input_axis = op->lhs_->nd_;
  auto &output_axis = op->nd_;
  ASSERT(input_axis.size() == output_axis.size());
  size_t i = 0;
  while (i < input_axis.size() && input_axis[i] == 1) {
    ++i;
  }
  size_t lead_dim = i;
  while (i < input_axis.size() && input_axis[i] == output_axis[i]) {
    ++i;
  }
  int start = i == lead_dim ? 0 : i;
  while (i < input_axis.size() && input_axis[i] != output_axis[i]) {
    ++i;
  }
  int end = i - 1;
  ASSERT(start <= end);
  reinterpret_cast<_ReduceOp *>(op)->SetRange(start, end);
}

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
  if (visit_) {
    delete visit_;
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
        for (int j = red_start; j <= red_end; ++j) {
          obj->ndd_.dims[j] = 1;
        }
        obj->ndd_.strides.resize(0);  // force update stride
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
  if (visit_) {
    visit_->Clear();
  }
  round_tile_.resize(0);
}

void ReduceOp::Tile(const TileParam &tp) {
  CollectRoundTile(ndd_.dims, tp, round_tile_);
  _ReduceOp::Tile(tp);
}

uint64_t ReduceOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  if (ws_num_ == 2) {
    return EmitDeterm(k);
  }
  if (CheckFlag(OBJ_FLAG_REDUCE_NO_CUM) || !(round_tile_.size() & 1)) {
    return _ReduceOp::Emit(k);
  }
  auto out_xbuf = xbuf_;
  xbuf_ = wss_[0];
  uint64_t size = _ReduceOp::Emit(k);
  vAtomicCum op;
  uint64_t rounds[2];
  BuildDimRounds(round_tile_, rounds);
  op.xd = out_xbuf;
  op.xn = xbuf_;
  xbuf_ = out_xbuf;
  op.count = ndd_.stride_back();
  op.round_rank = round_tile_.size();
  op.red_op = red_op_;
  tail_insn_ = insn_ + size;
  size += vAtomicCum::Encode(tail_insn_, type_id_ == kFloat32 ? V_ATOMICCUM : V_ATOMICCUM_FP16, op, rounds);
  *(tail_insn_) |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
  return size;
}

NDObject *ReduceOp::Clone(CloneHelper &h) {
  NDObject *input = stuff_ops_.empty() ? lhs_ : stuff_ops_.front()->lhs_;
  auto dims_ref = h.GetClone(dims_ref_);
  auto op = new ReduceOp(h.GetClone(input), red_op_, dims_ref, keepdims_);
  if (CheckFlag(OBJ_FLAG_REDUCE_NO_CUM)) {
    op->SetFlag(OBJ_FLAG_REDUCE_NO_CUM);
  }
  return op;
}

void ReduceOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<ReduceOp *>(op);
  auto &shape = self->shape_;
  auto input_shape = self->lhs_->shape_ref_;
  auto keepdims = self->keepdims_;
  if (auto dims_ref = self->dims_ref_; dims_ref->size) {
    auto in_size = input_shape->size;
    for (size_t i = 0; i < in_size; ++i) {
      shape[i] = input_shape->data[i];
    }
    int64_t red_dim = keepdims ? 1 : 0;
    for (size_t i = 0; i < dims_ref->size; i++) {
      auto dim = dims_ref->data[i];
      if (dim < 0) {
        dim += in_size;
      }
      shape[dim] = red_dim;
    }
    if (keepdims) {
      shape.Resize(in_size);
    } else {
      size_t out_size = 0;
      for (size_t i = 0; i < in_size; ++i) {
        if (shape[i] != 0) {
          shape[out_size++] = shape[i];
        }
      }
      shape.Resize(out_size);
    }
  } else if (keepdims) {
    for (size_t i = 0; i < input_shape->size; ++i) {
      shape[i] = 1;
    }
    shape.Resize(input_shape->size);
  } else {
    shape.Resize(0);
  }
}

namespace {
bool GenTileVisit(VectorKernel &k, const DimArray &round_tile, RedVisitCoder &coder) {
  auto round_depth = round_tile.size();
  if (round_depth == 0) {
    return false;
  }
  uint32_t core_limit = k.code_.block_dim_;
  if (round_depth == 1) {
    auto v = reinterpret_cast<vVisitRed1 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    v->head = 0;
    v->r1 = r1;
    v->e = (k.tile_num_ / r1) << 32;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed1);
    coder.visit_id_ = V_VISIT_RED_1;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_);
  } else if (round_depth == 2) {
    auto v = reinterpret_cast<vVisitRed2 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    v->head = 0;
    v->e = (k.tile_num_ / r1) << 32;
    v->e1_r1 = e1 << 32 | r1;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed2);
    coder.visit_id_ = V_VISIT_RED_2;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_);
  } else if (round_depth == 3) {
    auto v = reinterpret_cast<vVisitRed3 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r2 = e1 << 32 | r2;
    v->r1 = r1;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed3);
    coder.visit_id_ = V_VISIT_RED_3;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_ / r2);
  } else {
    ASSERT(round_depth == 4);
    auto v = reinterpret_cast<vVisitRed4 *>(coder.code_);
    uint64_t r1 = round_tile[0];
    uint64_t e1 = round_tile[1];
    uint64_t r2 = round_tile[2];
    uint64_t e2 = round_tile[3];
    v->tidx_head = 0;
    v->e = (k.tile_num_ / (r1 * r2)) << 32;
    v->e1_r1 = e1 << 32 | r1;
    v->e2_r2 = e2 << 32 | r2;
    v->user_cnt = 1;
    coder.code_size_ = sizeof(vVisitRed4);
    coder.visit_id_ = V_VISIT_RED_4;
    coder.block_num_ = std::min<uint32_t>(core_limit, k.tile_num_ / r2);
  }
  return true;
}

void AddTileVisit(size_t round_depth, RedVisitCoder *coder) {
  switch (round_depth) {
    case 1: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_1);
      reinterpret_cast<vVisitRed1 *>(coder->code_)->user_cnt++;
      break;
    }
    case 2: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_2);
      reinterpret_cast<vVisitRed2 *>(coder->code_)->user_cnt++;
      break;
    }
    case 3: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_3);
      reinterpret_cast<vVisitRed3 *>(coder->code_)->user_cnt++;
      break;
    }
    case 4: {
      ASSERT(coder->visit_id_ == V_VISIT_RED_4);
      reinterpret_cast<vVisitRed4 *>(coder->code_)->user_cnt++;
      break;
    }
    default: {
      ASSERT(0);
      break;
    }
  }
}
} // namepsace

uint64_t ReduceOp::EmitDeterm(VectorKernel &k) {
  auto coder = k.GetVisitor<RedVisitCoder>();
  uint64_t ws_offset;
  if (likely(coder == nullptr)) {
    if (!GenTileVisit(k, round_tile_, *visit_)) {
      return _ReduceOp::Emit(k);
    }
    k.AddVisitor(visit_);
    ws_offset = 0;  // TODO: multi workspace
    visit_->ws_size_ = ndd_.stride_back() * sizeof(float) * visit_->block_num_;
  } else {
    AddTileVisit(round_tile_.size(), coder);
    ws_offset = coder->ws_size_;
    coder->ws_size_ += ndd_.stride_back() * sizeof(float) * coder->block_num_;
  }
  auto out_xbuf = xbuf_;
  xbuf_ = wss_[0];
  uint64_t size = _ReduceOp::Emit(k);
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
  k.code_.BindWorkspace(ws_reloc_, ws_offset);
  return size;
}

void OneHotOp::Normalize(std::vector<NDObject *> &run_ops) {
  // update shape_ref
  shape_.Resize(lhs_->shape_ref_->size);
  int64_t depth = depth_->data[0];
  size_t axis = axis_ >= 0 ? axis_ : shape_.size + axis_;
  for (size_t i = 0; i < shape_.size; ++i) {
    shape_[i] = i == axis ? depth : lhs_->shape_ref_->data[i];
  }
  depth_dim_ = shape_.size - axis - 1;
  // update ndd
  ndd_.dims.resize(lhs_->nd_.size());
  for (size_t i = 0; i < ndd_.size(); ++i) {
    ndd_.dims[i] = i == static_cast<size_t>(depth_dim_) ? depth : lhs_->nd_[i];
  }
  tile_dim_ = ndd_.size();
}

void OneHotOp::FoldProp(NDObject *op, PropRange &range) {
  auto depth_dim = static_cast<OneHotOp *>(op)->depth_dim_;
  if (range.base > depth_dim) {
    if (auto depth_len = range.base - depth_dim; depth_len < range.depth) {
      range.depth = depth_len;
    }
  } else if (range.base == depth_dim) {
    range.depth = 1;
  }
}

void OneHotOp::TileCollect(NDObject *op, TileInfo &info) {
  auto depth_dim = static_cast<OneHotOp *>(op)->depth_dim_;
  if (depth_dim == 0) {
    info.lead_depth = 1;
  } else if (info.lead_depth > depth_dim) {
    info.lead_depth = depth_dim;
  }
}

void OneHotOp::Tile(const TileParam &tp) {
  if (tp.start <= tile_dim_) {
    tile_dim_ = tp.start;
    if (depth_dim_ == tp.end) {
      depth_tile_ = tp.num;
    } else {
      depth_tile_ *= tp.num;
    }
  }
  NDObject::Tile(tp);
}

uint64_t OneHotOp::Emit(VectorKernel &k) {
  ndd_.UpdateStride(k.LeadAlign());
  vOneHot op;
  op.xn = lhs_->xbuf_;
  op.xd = xbuf_;
  op.data_size = ndd_.stride_back();
  if (depth_dim_ == 0) {
    if (tile_dim_ > 0) {
      op.depth = ndd_.lead_stride();
      op.iter_num = ndd_.stride_back() / op.depth;
      op.dup_round = 0;
      op.mode = vOneHot::MODE_X;
    } else {
      op.depth = ndd_.lead_dim();
      op.dup_round = depth_tile_;
      op.iter_num = 0;
      op.mode = vOneHot::MODE_X_TILE;
    }
  } else {
    if (tile_dim_ > depth_dim_) {
      op.dup_round = ndd_.stride_back() / ndd_.stride(depth_dim_);
      op.iter_num = ndd_.stride(depth_dim_ - 1);
      op.depth = ndd_[depth_dim_];
      op.mode = vOneHot::MODE_Y;
    } else if (tile_dim_ == depth_dim_) {
      op.dup_round = depth_tile_;
      op.iter_num = ndd_.stride(depth_dim_ - 1);
      op.depth = ndd_[depth_dim_];
      op.mode = vOneHot::MODE_Y_TILE;
    } else {
      op.dup_round = depth_tile_;
      op.iter_num = ndd_.stride_back();
      op.depth = depth_tile_ / shape_[shape_.size - depth_dim_ - 1];
      op.mode = vOneHot::MODE_Y_TILE_2;
    }
  }
  return vOneHot::Encode(insn_, ITEM_SIZE[type_id_] == 2 ? V_ONE_HOT_B16 : V_ONE_HOT, on_value_, off_value_, op);
}

NDObject *OneHotOp::Clone(CloneHelper &h) {
  auto depth = h.GetClone(depth_);
  return new OneHotOp(h.GetClone(lhs_), depth, axis_, on_value_, off_value_, type_id_);
}

void OneHotOp::Dump(bool verbose, std::ostringstream &oss) {
  oss << "OneHot";
  if (verbose) {
    oss << "<" << axis_ << "," << depth_->data[0] << ",";
    DumpScalarCode(oss, on_value_, type_id_);
    oss << ",";
    DumpScalarCode(oss, off_value_, type_id_);
    oss << ">";
  }
}

void OneHotOp::DimChanged(NDObject *op) {
  for (size_t i = 0; i < op->nd_.size(); ++i) {
    if (i >= op->lhs_->nd_.size() || op->lhs_->nd_[i] != op->nd_[i]) {
      static_cast<OneHotOp *>(op)->depth_dim_ = i;
      break;
    }
  }
}

void OneHotOp::ShapeProp(NDObject *op, int64_t &sym_dim_next) {
  auto *self = static_cast<OneHotOp *>(op);
  auto &shape = self->shape_;
  auto depth = self->depth_->data[0];
  auto in_shape = op->lhs_->shape_ref_;
  auto axis = self->axis_;
  if (axis < 0) {
    axis += in_shape->size;
  }
  shape.Resize(in_shape->size);
  for (int i = 0; i < static_cast<int>(shape.size); ++i) {
    shape[i] = i == axis ? depth : in_shape->data[i];
  }
}
}  // namespace dvm
