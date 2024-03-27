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

#include <unordered_map>
#include <vector>
#include <cstring>
#include <stdexcept>
#ifndef VK_SIM_MODEL
#include "acl/acl_base.h"
#endif
#include "code.h"

namespace dvm {
namespace {
std::string GetSocName() {
  std::string res;
  const char *soc_name = getenv("DVM_SOC_NAME");
  if (soc_name == nullptr) {
#ifdef VK_SIM_MODEL
    DvmException("simulator must set environment variable DVM_SOC_NAME");
#else
    soc_name = aclrtGetSocName();
#endif
  }
  if (soc_name == nullptr) {
    return res;
  }
  res = soc_name;
  return res;
}

std::unordered_map<std::string, vCompareType> cmp_insn_id = {
  {"Greater", V_CMP_GT},
  {"Less", V_CMP_LT},
  {"GreaterEqual", V_CMP_GE},
  {"LessEqual", V_CMP_LE},
  {"Equal", V_CMP_EQ},
  {"NotEqual", V_CMP_NE},
};

template <typename T>
void DumpVal(const std::string &name, T val, std::ostringstream &oss) {
  oss << name << "(" << val << ")";
}

struct DumpInfo {
  uint64_t *insn = nullptr;
  uint64_t ext = 0;
  uint64_t simd_width = 0;
  DumpInfo(uint64_t *insn_in, uint64_t ext_in, uint64_t simd_width_in = 0)
      : insn(insn_in), ext(ext_in), simd_width(simd_width_in) {}
};

void DumpRounds(uint64_t rank, uint64_t *rounds, std::ostringstream &oss) {
  oss << "rounds(";
  for (uint64_t i = 0; i < (rank - 1) / 2; ++i) {
    uint64_t round = *rounds++;
    oss << (round & 0xfffffffful) << "," << (round >> 32) << ",";
  }
  uint64_t round = *rounds;
  oss << (round & 0xfffffffful);
  if ((rank & 1ul) == 0) {
    oss << "," << (round >> 32);
  }
  oss << ")";
}

void DumpLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vDMA op;
  vDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "load.u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vDMA::ROUND_OFFSET, oss);
  }
}

void DumpSliceLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSliceLoad op;
  vSliceLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "slice_load " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("body_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
}

void DumpStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vDMA op;
  vDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store.u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.gm) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
}

void DumpLoad2(const DumpInfo &dump_info, std::ostringstream &oss) {
  vLoad op;
  vLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "load2.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vLoad::ROUND_OFFSET, oss);
  }
}

void DumpStore2(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStore *op = reinterpret_cast<vStore *>(dump_info.insn);
  auto tile_stride = dump_info.ext >> V_X_BITS;
  auto xn = dump_info.ext & V_X_MASK;
  auto lead_tiling = op->config >> 62;
  auto pad_size = (op->config >> 54) & 0xfful;
  auto iter_size = (op->config >> 36) & V_X_MASK;
  auto iter_tail = (op->config >> 18) & V_X_MASK;
  auto iter_body = op->config & V_X_MASK;
  oss << "store2.u8." << iter_size << "x" << iter_body;
  oss << " " << reinterpret_cast<void *>(op->to) << ", " << reinterpret_cast<void *>(xn);
  oss << " //";
  DumpVal("tile_stride", tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", pad_size, oss);
  oss << ", ";
  DumpVal("lead_tiling", lead_tiling, oss);
}

void DumpStoreAtomic(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAtomic op;
  vStoreAtomic::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_atomic.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("round", op.round, oss);
  oss << ", ";
  DumpVal("factor", op.factor, oss);
}

void DumpStoreStatus(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreStatus *op = reinterpret_cast<vStoreStatus *>(dump_info.insn);
  auto xn = dump_info.ext & V_X_MASK;
  oss << "store_status." << reinterpret_cast<void *>(op->to) << ", " << reinterpret_cast<void *>(xn);
}

void DumpLoadDummy(const DumpInfo &dump_info, std::ostringstream &oss) { oss << "dummy_load.u8.0"; }

void DumpUnary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vUnary op;
  vUnary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpRemovePad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vRemovePad op;
  vRemovePad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpVal("iter_num", op.iter_num, oss);
  oss << ", ";
  DumpVal("rs", op.rs, oss);
}

template <typename T = float>
void DumpBinaryS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinaryS<T> *op = reinterpret_cast<vBinaryS<T> *>(dump_info.insn);
  auto rs = op->data >> 18;
  auto repeat = op->data & V_X_MASK;
  auto xn = dump_info.ext & V_X_MASK;
  auto xd = (dump_info.ext >> V_X_BITS) & V_X_MASK;
  oss << dump_info.simd_width << "x" << repeat;
  oss << " " << reinterpret_cast<void *>(xd) << ", " << reinterpret_cast<void *>(xn) << ", " << op->scalar << " //";
  DumpVal("rs", rs, oss);
}

void DumpBinary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm);
}

void DumpCompare(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCompare op;
  vCompare::Decode(dump_info.insn, *dump_info.insn, op);
  std::string cmp_op("Unknown");
  for (auto it = cmp_insn_id.begin(); it != cmp_insn_id.end(); ++it) {
    if (it->second == op.type) {
      cmp_op = it->first;
    }
  }
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("cmp_type", cmp_op, oss);
}

template <typename T = float>
void DumpBroadcastS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastS<T> *op = reinterpret_cast<vBroadcastS<T> *>(dump_info.insn);
  uint64_t data = op->data;
  uint64_t rs = data >> 18;
  uint64_t repeat = data & V_X_MASK;
  oss << dump_info.simd_width << "x" << repeat;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << op->scalar << " //";
  DumpVal("rs", rs, oss);
}

void DumpSelect(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSelect *op = reinterpret_cast<vSelect *>(dump_info.insn);
  auto rs = op->data >> 60;
  auto repeat = (op->data >> (V_X_BITS + V_X_BITS)) & V_X_MASK;
  auto cond = op->data & V_X_MASK;
  auto xm = (op->data >> V_X_BITS) & V_X_MASK;
  auto xn = dump_info.ext & V_X_MASK;
  auto xd = (dump_info.ext >> V_X_BITS) & V_X_MASK;
  oss << dump_info.simd_width << "x" << repeat;
  oss << " " << reinterpret_cast<void *>(cond) << ", " << reinterpret_cast<void *>(xd) << ", "
      << reinterpret_cast<void *>(xn);
  oss << ", " << reinterpret_cast<void *>(xm) << " //";
  DumpVal("rs", rs, oss);
}

void DumpBroadcastX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastX op;
  vBroadcastX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "[" << dump_info.simd_width << "x" << op.repeat << "]x" << op.lead_num << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpBroadcastY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastY op;
  vBroadcastY::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "32x" << op.dup_stride << "x[" << op.dup_num << "]x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpReduceX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceX op;
  auto head = *dump_info.insn;
  vReduceX::Decode(dump_info.insn, head, op);
  vReduceX::DecodeBlock(dump_info.insn, op);
  oss << "[" << op.red_size<< "]x" << op.dup_size;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
  oss << ", ";
  DumpVal("dup_block", op.dup_block, oss);
  oss << ", ";
  DumpVal("dup_pad", op.dup_pad, oss);
}

void DumpReduceY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceY op;
  auto head = *dump_info.insn;
  vReduceY::Decode(dump_info.insn, head, op);
  oss << op.iter_size << "x[" << op.red_size << "]x" << op.dup_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
}

void DumpCopy(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCopy *op = reinterpret_cast<vCopy *>(dump_info.insn);
  auto xn = dump_info.ext & V_X_MASK;
  auto xd = (dump_info.ext >> V_X_BITS) & V_X_MASK;
  auto lenburst = (op->config) >> 16 & 0xfffful;
  oss << "32x" << lenburst;
  oss << " " << reinterpret_cast<void *>(xd) << ", " << reinterpret_cast<void *>(xn);
}

void DumpClearPad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vClearPad op;
  vClearPad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_size << "x" << op.iter_num << " " << reinterpret_cast<void *>(op.xd) << " //";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpVal("simd_width", op.simd_width, oss);
}

void DumpElementAny(const DumpInfo &dump_info, std::ostringstream &oss) {
  vElementAny op;
  vElementAny::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat << " " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("rs", op.rs, oss);
  oss << ", ";
  DumpVal("iter_size", op.iter_size, oss);
  oss << ", ";
  DumpVal("tail_size", op.tail_size, oss);
}

using DumpFunc = void(const DumpInfo &, std::ostringstream &oss);

std::unordered_map<uint64_t, DumpFunc *> mem_dump_func_table = {
  {V_LOAD, &DumpLoad},
  {V_STORE, &DumpStore},
  {V_LOAD_2, &DumpLoad2},
  {V_STORE_2, &DumpStore2},
  {V_STORE_ATOMIC, &DumpStoreAtomic},
  {V_STORE_STATUS, &DumpStoreStatus},
  {V_LOAD_DUMMY, &DumpLoadDummy},
  {V_SLICE_LOAD, &DumpSliceLoad},
  {V_SLICE_LOAD_U16, &DumpSliceLoad},
};

std::unordered_map<uint64_t, std::tuple<DumpFunc *, std::string, std::string>> op_dump_info_table = {
  {V_COPY, {&DumpCopy, "Copy", "u8"}},
  {V_BROADCAST_X, {&DumpBroadcastX, "BroadcastX", "fp32"}},
  {V_BROADCAST_X_FP16, {&DumpBroadcastX, "BroadcastX", "fp16"}},
  {V_BROADCAST_X_INT32, {&DumpBroadcastX, "BroadcastX", "int32"}},
  {V_BROADCAST_Y, {&DumpBroadcastY, "BroadcastY", "u8"}},
  {V_BROADCAST_S, {&DumpBroadcastS, "BroadcastS", "fp32"}},
  {V_BROADCAST_S_FP16, {&DumpBroadcastS, "BroadcastS", "fp16"}},
  {V_BROADCAST_S_INT32, {&DumpBroadcastS<int32_t>, "BroadcastS", "int32"}},
  {V_SQRT, {&DumpUnary, "Sqrt", "fp32"}},
  {V_SQRT_FP16, {&DumpUnary, "Sqrt", "fp16"}},
  {V_ABS, {&DumpUnary, "Abs", "fp32"}},
  {V_ABS_FP16, {&DumpUnary, "Abs", "fp16"}},
  {V_LOG, {&DumpUnary, "Log", "fp32"}},
  {V_LOG_FP16, {&DumpUnary, "Log", "fp16"}},
  {V_EXP, {&DumpUnary, "Exp", "fp32"}},
  {V_EXP_FP16, {&DumpUnary, "Exp", "fp16"}},
  {V_REC, {&DumpUnary, "Reciprocal", "fp32"}},
  {V_REC_FP16, {&DumpUnary, "Reciprocal", "fp16"}},
  {V_NOT_INT8, {&DumpUnary, "LogicalNot", "u8"}},
  {V_ISFINITE, {&DumpUnary, "IsFinite", "fp32"}},
  {V_ISFINITE_FP16, {&DumpUnary, "IsFinite", "fp16"}},
  {V_CAST_FP16_TO_FP32, {&DumpUnary, "CastFP16", "fp32"}},
  {V_CAST_INT8_TO_FP16, {&DumpUnary, "CastS8", "fp16"}},
  {V_CAST_FP16_TO_INT8, {&DumpUnary, "CastFP16", "u8"}},
  {V_CAST_FP16_TO_INT32, {&DumpUnary, "CastFP16", "int32"}},
  {V_CAST_FP32_TO_INT32, {&DumpUnary, "CastFP32", "int32"}},
  {V_CAST_FP32_TO_FP16, {&DumpUnary, "CastFP32", "fp16"}},
  {V_CAST_FP32_TO_BF16, {&DumpUnary, "CastFP32", "bf16"}},
  {V_CAST_INT32_TO_FP32, {&DumpUnary, "CastS32", "fp32"}},
  {V_CAST_INT32_TO_FP16, {&DumpUnary, "CastS32", "fp16"}},
  {V_CAST_BF16_TO_FP32, {&DumpUnary, "CastBF16", "fp32"}},
  {V_CAST_BF16_TO_INT32, {&DumpUnary, "CastBF16", "int32"}},
  {V_ADDS, {&DumpBinaryS, "Adds", "fp32"}},
  {V_ADDS_FP16, {&DumpBinaryS, "Adds", "fp16"}},
  {V_ADDS_INT32, {&DumpBinaryS<int32_t>, "Adds", "int32"}},
  {V_MULS, {&DumpBinaryS, "Muls", "fp32"}},
  {V_MULS_FP16, {&DumpBinaryS, "Muls", "fp16"}},
  {V_MULS_INT32, {&DumpBinaryS<int32_t>, "Muls", "int32"}},
  {V_MAXS, {&DumpBinaryS, "Maximums", "fp32"}},
  {V_MAXS_FP16, {&DumpBinaryS, "Maximums", "fp16"}},
  {V_MAXS_INT32, {&DumpBinaryS<int32_t>, "Maximums", "int32"}},
  {V_MINS, {&DumpBinaryS, "Minimums", "fp32"}},
  {V_MINS_FP16, {&DumpBinaryS, "Maximums", "fp16"}},
  {V_MINS_INT32, {&DumpBinaryS<int32_t>, "Maximums", "int32"}},
  {V_ADD, {&DumpBinary, "Add", "fp32"}},
  {V_ADD_FP16, {&DumpBinary, "Add", "fp16"}},
  {V_ADD_INT32, {&DumpBinary, "Add", "int32"}},
  {V_SUB, {&DumpBinary, "Sub", "fp32"}},
  {V_SUB_FP16, {&DumpBinary, "Sub", "fp16"}},
  {V_SUB_INT32, {&DumpBinary, "Sub", "int32"}},
  {V_MUL, {&DumpBinary, "Mul", "fp32"}},
  {V_MUL_FP16, {&DumpBinary, "Mul", "fp16"}},
  {V_MUL_INT32, {&DumpBinary, "Mul", "int32"}},
  {V_DIV, {&DumpBinary, "Div", "fp32"}},
  {V_DIV_FP16, {&DumpBinary, "Div", "fp16"}},
  {V_MAX, {&DumpBinary, "Maximum", "fp32"}},
  {V_MAX_FP16, {&DumpBinary, "Maximum", "fp16"}},
  {V_MAX_INT32, {&DumpBinary, "Maximum", "int32"}},
  {V_MIN, {&DumpBinary, "Minimum", "fp32"}},
  {V_MIN_FP16, {&DumpBinary, "Minimum", "fp16"}},
  {V_MIN_INT32, {&DumpBinary, "Minimum", "int32"}},
  {V_POW, {&DumpBinary, "Pow", "fp32"}},
  {V_POW_FP16, {&DumpBinary, "Pow", "fp16"}},
  {V_CMP, {&DumpCompare, "Cmp", "fp32"}},
  {V_CMP_FP16, {&DumpCompare, "Cmp", "fp16"}},
  {V_AND_INT8, {&DumpBinary, "LogicalAnd", "int8"}},
  {V_OR_INT8, {&DumpBinary, "LogicalOr", "int8"}},
  {V_SEL, {&DumpSelect, "Select", "fp32"}},
  {V_SEL_FP16, {&DumpSelect, "Select", "fp16"}},
  {V_SEL_INT32, {&DumpSelect, "Select", "int32"}},
  {V_RSUM_X, {&DumpReduceX, "SumX", "fp32"}},
  {V_RSUM_Y, {&DumpReduceY, "SumY", "fp32"}},
  {V_CLR_PAD, {&DumpClearPad, "ClrPad", "fp32"}},
  {V_ELEMENT_ANY, {&DumpElementAny, "ElementAny", "fp32"}},
  {V_ELEMENT_ANY_FP16, {&DumpElementAny, "ElementAny", "fp16"}},
  {V_REMOVEPAD, {&DumpRemovePad, "RemovePad", "u32"}},
  {V_REMOVEPAD_U16, {&DumpRemovePad, "RemovePad", "u16"}},
};

size_t DumpInsn(uint64_t *insn, uint64_t simd_width, std::ostringstream &oss) {
  uint64_t head = *insn;
  uint64_t id = (head >> V_HEAD_ID_OFFSET) & V_HEAD_ID_MASK;
  uint64_t ext = (head >> V_HEAD_EXT_OFFSET) & V_HEAD_EXT_MASK;
  uint64_t offset = (head >> V_HEAD_SIZE_OFFSET) & V_HEAD_SIZE_MASK;
  DumpInfo info{insn, ext, simd_width};
  if (head & (1ul << V_HEAD_IS_SIMD_OFFSET)) {
    if (op_dump_info_table.find(id) != op_dump_info_table.end()) {
      auto [dump_func, name, dtype_str] = op_dump_info_table[id];
      oss << name << "." << dtype_str << ".";
      dump_func(info, oss);
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  } else {
    if (mem_dump_func_table.find(id) != mem_dump_func_table.end()) {
      mem_dump_func_table[id](info, oss);
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  }  // end else
  return offset;
}

void DasBody(std::ostringstream &oss, uint8_t *bcode, uint64_t bcode_size, uint64_t simd_width, const std::string &indent) {
  unsigned char *insn = bcode;
  unsigned char *insn_end = bcode + bcode_size;
  std::vector<std::pair<uint64_t*, std::string>> insn_dump;
  while (insn < insn_end) {
    auto op = reinterpret_cast<uint64_t*>(insn);
    std::ostringstream os;
    size_t offset = DumpInsn(op, simd_width, os);
    if (offset == 0) {
      break;
    }
    insn_dump.emplace_back(std::make_pair(op, os.str()));
    insn = insn + offset*8;
  }
  auto find_wait = [&insn_dump](bool is_simd, uint64_t event, int from_idx) -> int {
    for (size_t i = from_idx; i < insn_dump.size(); ++i){
      auto head = *(insn_dump[i].first);
      if ((is_simd != bool(head & (1ul << V_HEAD_IS_SIMD_OFFSET))) && 
          (head & (0x1ul << V_HEAD_WAIT_FLAG_OFFSET)) && 
          (((head >> V_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) == event))
        return i;
    }
    return -1;
  }; 
  std::unordered_map<int, int> wait_map;
  for (size_t i = 0; i < insn_dump.size(); ++i) {
    auto &dump = insn_dump[i];
    auto head = *(dump.first);
    bool is_simd = bool(head & (1ul << V_HEAD_IS_SIMD_OFFSET));
    oss << indent << i << ": " << dump.second << std::endl;
    oss << indent << "  { simd(" << is_simd << ")";
    if (head & (0x1ul << V_HEAD_WAIT_FLAG_OFFSET)) {
      auto it = wait_map.find(i);
      int from = it != wait_map.end() ? it->second : -1;
      oss << ", wait(" << from <<  ", event_" << int((head >> V_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
    }
    if (head & (0x1ul << V_HEAD_BAR_FLAG_OFFSET)) {
      oss << ", bar(1)";
    }
    if (head & (0x1ul << V_HEAD_BACK_WAIT_OFFSET)) {
      oss << ", wait_prev_store(1)";
    }
    if (head & (0x1ul << V_HEAD_SET_FLAG_OFFSET)) {
      auto event = (head >> V_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK;
      auto wait_idx = find_wait(is_simd, event, i + 1);
      if (wait_idx > 0) wait_map[wait_idx] = i;
      oss << ", set(" << wait_idx << ", event_"<< event << ")";
    }
    if (head & (0x1ul << V_HEAD_BACK_SET_OFFSET)) {
      oss << ", set_next_load(1)";
    }
    oss << " }" << std::endl;
  }
}
}  // namespace

void DvmException(const char* error_str) {
  std::ostringstream oss;
  oss << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}

DeviceInfo::DeviceInfo() {
  auto soc_name = GetSocName();
  if (soc_name.find("Ascend910B") != std::string::npos) {
    arch_ = kAiCore_C220;
    local_mem_size_ = 192 * 1024;
    event_num_ = 8;
    core_num_ = (soc_name == "Ascend910B1" || soc_name == "Ascend910B2") ? 48 : 40;
  } else {
    arch_ = kAiCore_C100;
    local_mem_size_ = 256 * 1024;
    event_num_ = 4;
    core_num_ = 32;
  }
  ub_workspace_size_ = 1024;
}

void Code::DisAssemble(std::ostringstream &oss) {
  oss << "vmain(tile_num=" << tile_num_ << ", block_dim=" << block_dim_ <<
      ", simd_width="<<simd_width_ << ", insn_num=" << insn_num_ << ") {" << std::endl;
  DasBody(oss, data_ + HeadSize(), data_size_ - HeadSize(), simd_width_, " "); 
  oss << "}";
}

void CodeP::LinkAll(std::vector<uint64_t> &offsets) {
  atomic_clean_.clear();
  block_dim_ = 0;
  uint64_t code_size = 0;
  for (auto c : children_) {
    block_dim_ += c->block_dim_;
    code_size += ((c->data_size_ - 8 + 31) >> 5) << 5;
  }
  data_size_ = (((block_dim_ + 1) * sizeof(uint64_t) + 31) >> 5) << 5; // config + summary
  uint64_t offset = data_size_;
  data_size_ += code_size;
  Alloc(data_size_);
  uint64_t *data_64 = reinterpret_cast<uint64_t*>(data_);
  uint64_t config = 1ul << 63;
  int summary_idx = 1;
  for (size_t k = 0; k < children_.size(); ++k) {
    Code *code = children_[k];
    ASSERT(code->tile_num_ <= 0xffffful && code->insn_num_ <= 0x3ful);
    // config
    config |= (children_[k]->simd_width_ - 1) << (8 * k);
    // summary
    uint64_t lenburst = (code->data_size_ - sizeof(uint64_t) + 31) / 32;
    uint64_t body_tile_flag = 1ul << 39;
    uint64_t summary = lenburst << 58 | (offset >> 5) << 49 | k << 46 | code->insn_num_ << 40 | body_tile_flag;
    uint64_t tile_per_block = (code->tile_num_ - 1) / code->block_dim_ + 1;
    uint64_t start_idx = 0;
    for (uint64_t i = 0; i < code->block_dim_ - 1; ++i) {
      data_64[summary_idx++] = summary | (tile_per_block - 1) << 20 | start_idx;
      start_idx += tile_per_block;
    }
    summary &= ~body_tile_flag;
    data_64[summary_idx++] = summary | (code->tile_num_ - start_idx - 1) << 20 | start_idx;
    // data
    offsets.push_back(offset - sizeof(uint64_t));
    uint64_t cpy_size = code->data_size_ - sizeof(uint64_t);
    memcpy(data_ + offset, code->data_ + sizeof(uint64_t), cpy_size);
    offset += ((cpy_size + 31) >> 5) << 5;
    // atomic clean
    if (!code->atomic_clean_.empty()) {
      for (auto ac : code->atomic_clean_) {
        atomic_clean_.push_back(ac);
      }
    }
  }
  data_64[0] = config;
}

void CodeP::DisAssemble(std::ostringstream &oss) {
  struct Summary {
    int64_t block_start{-1};
    int64_t block_end{-1};
    int64_t block_step{-1};
    int64_t block_tail{-1};
    uint8_t *bcode{nullptr};
  };
  std::vector<Summary> summays(children_.size());
  uint64_t *data_64 = reinterpret_cast<uint64_t*>(data_ + sizeof(uint64_t));
  for (uint64_t i = 0; i < block_dim_; ++i) {
    uint64_t sum_data = *data_64++;
    uint64_t start_idx = sum_data & 0xffffful;
    uint64_t end_idx = start_idx + ((sum_data >> 20) & 0x7fffful);
    uint64_t ker_idx = (sum_data >> 46) & 0x7ul;
    uint64_t offset = ((sum_data >> 49) & 0x1fful) * 32;
    uint64_t body_flag = (sum_data >> 39) & 0x1ul;
    auto &summary = summays[ker_idx];
    if (summary.block_start == -1) {
      summary.block_start = i;
      summary.block_step = end_idx - start_idx + 1;
      summary.bcode = data_ + offset;
    }
    if (!body_flag) {
      summary.block_end = i;
      summary.block_tail = end_idx - start_idx + 1;
    }
  }
  oss << "vmain.parallel(block_dim=" << block_dim_ << ") {" << std::endl;
  for (uint64_t i = 0; i < children_.size(); ++i) {
    auto code = children_[i];
    auto &summary = summays[i];
    oss << " kernel_" << i << "(tile_num=" << code->tile_num_ << ", simd_width="<<code->simd_width_ << ", insn_num=" << code->insn_num_ <<
          ", block_range=[" << summary.block_start << ", " << summary.block_end << "], block_step=" << summary.block_step <<
          ", block_tail=" << summary.block_tail << ") {" << std::endl;
    DasBody(oss, summary.bcode, code->data_size_ - code->HeadSize(), code->simd_width_, "  ");
    oss << " }" << std::endl;
  }
  oss << "}";
}
}  // namespace dvm
