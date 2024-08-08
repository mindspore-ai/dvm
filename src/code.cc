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
#include "code.h"
#include "ops.h"

#ifndef VK_SIM_MODEL
#include "acl/acl_rt.h"
#else
#define aclrtMallocHost(addr, size) 0; *addr = std::malloc(size)
#define aclrtFreeHost(addr) 0; std::free(addr)
#endif

namespace dvm {
namespace {
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

void DumpSLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSLoad op;
  vSLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sload " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("cube_m", op.slice_m, oss);
  oss << ", ";
  DumpVal("cube_n", op.slice_n, oss);
  oss << ", ";
  DumpVal("src_n", op.src_n, oss);
  oss << ", ";
  DumpVal("tail_m", op.tail_m, oss);
  oss << ", ";
  DumpVal("tail_n", op.tail_n, oss);
}

void DumpSStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSStore op;
  vSStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sstore " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.gm) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("cube_m", op.slice_m, oss);
  oss << ", ";
  DumpVal("cube_n", op.slice_n, oss);
}

void DumpSliceLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSliceSL op;
  vSliceSL::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "slice_load " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
}

void DumpSliceStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSliceSL op;
  vSliceSL::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "slice_store " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("one_flag", op.one_flag, oss);
}

void DumpLoadExit(const DumpInfo &dump_info, std::ostringstream &oss) {
  oss << "exit 0";
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

void DumpPingPongLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPingPongLoad op;
  vPingPongLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "PingPongLoad.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("pingpong_stride", op.pingpong_stride, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vLoad::ROUND_OFFSET, oss);
  }
}

void DumpStore2(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStore *op = reinterpret_cast<vStore *>(dump_info.insn);
  auto tile_stride = dump_info.ext >> V_C_X_BITS;
  auto xn = vDeCompactX(vGetBitRange(dump_info.ext, 0, V_C_X_BITS));
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
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreAtomic::ROUND_OFFSET, oss);
  }
}

void DumpStoreAtomicDeterm(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAtomicDeterm op;
  vStoreAtomicDeterm::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_atomic_determ.u8." << op.base.iter_size << "x" << op.base.iter_num;
  oss << " " << reinterpret_cast<void *>(op.base.to) << ", " << reinterpret_cast<void *>(op.base.xn);
  oss << " //";
  DumpVal("tile_stride", op.base.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.base.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.base.pad_size, oss);
  oss << ", ";
  DumpVal("step", *op.step_addr, oss);
  oss << ", ";
  DumpVal("step_end", *op.step_end_addr, oss);
  oss << ", ";
  DumpVal("core_tile_num", op.core_tile_num, oss);
  oss << ", ";
  DumpVal("tail_tile_num", op.tail_tile_num, oss);
  oss << ", ";
  DumpVal("stride_num", op.stride_num, oss);
  if (op.base.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.base.round_rank, dump_info.insn + vStoreAtomicDeterm::ROUND_OFFSET, oss);
  }
}

void DumpStoreStatus(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreStatus op;
  vStoreStatus::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_status." << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
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
  vBinaryS<T> op;
  vBinaryS<T>::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << ", " << op.scalar;
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
  vBroadcastS<T> op;
  vBroadcastS<T>::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << op.scalar;
}

void DumpSelect(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSelect op;
  vSelect::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.cond) << ", " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm);
}

void DumpBroadcastX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastX op;
  vBroadcastX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "[" << dump_info.simd_width << "x" << op.repeat << "]x" << op.lead_num << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("lead_pad", op.lead_pad, oss);
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
  vCopy op;
  auto head = *dump_info.insn;
  vCopy::Decode(dump_info.insn, head, op);
  auto lenburst = (op.config) >> 16 & 0xfffful;
  oss << "32x" << lenburst;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
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

void DumpReshape(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReshape op;
  vReshape::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.xd_lead << "x" << op.dup_size << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("xd_pad", op.xd_pad, oss);
  oss << ", ";
  DumpVal("xn_lead", op.xn_lead, oss);
  oss << ", ";
  DumpVal("xn_pad", op.xn_pad, oss);
}

using DumpFunc = void(const DumpInfo &, std::ostringstream &oss);

std::unordered_map<uint64_t, DumpFunc *> load_dump_func_table = {
  {V_LOAD, &DumpLoad},
  {V_LOAD_2, &DumpLoad2},
  {V_LOAD_DUMMY, &DumpLoadDummy},
  {V_SLICE_LOAD, &DumpSliceLoad},
  {V_SLOAD, &DumpSLoad},
  {V_PINGPONG_LOAD, &DumpPingPongLoad},
  {V_LOAD_NONE, &DumpLoadExit},
};

std::unordered_map<uint64_t, DumpFunc *> store_dump_func_table = {
  {V_STORE, &DumpStore},
  {V_STORE_2, &DumpStore2},
  {V_STORE_ATOMIC, &DumpStoreAtomic},
  {V_STORE_ATOMIC_DETERM, &DumpStoreAtomicDeterm},
  {V_STORE_STATUS, &DumpStoreStatus},
  {V_SSTORE, &DumpSStore},
  {V_SLICE_STORE, &DumpSliceStore},
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
  {V_REMOVEPAD, {&DumpRemovePad, "RemovePad", "u32"}},
  {V_REMOVEPAD_U16, {&DumpRemovePad, "RemovePad", "u16"}},
  {V_RESHAPE_B32, {&DumpReshape, "Reshape", "u32"}},
  {V_RESHAPE_B16, {&DumpReshape, "Reshape", "u16"}},
};

size_t DumpInsn(uint64_t *insn, uint64_t simd_width, std::ostringstream &oss) {
  auto convert_id = [](const uint64_t offsets[], uint64_t none_idx, uint64_t id) -> uint64_t {
    for (uint64_t i = 0; i <= none_idx; ++i) {
      if (offsets[i] == id) {
        return i;
      }
    }
    ASSERT(0);
    return 0;
  };
  uint64_t head = *insn;
  uint64_t id = (head >> V_HEAD_ID_OFFSET) & V_HEAD_ID_MASK;
  uint64_t offset = 0;
  if (head & (1ul << V_HEAD_SIMD_FLAG_OFFSET)) {
    uint64_t ext = (head >> V_HEAD_EXT_OFFSET) & V_HEAD_EXT_MASK;
    offset = (head >> V_HEAD_SIZE_OFFSET) & V_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext, simd_width};
    id = convert_id(g_simd_func_offset, V_NONE, id);
    if (op_dump_info_table.find(id) != op_dump_info_table.end()) {
      auto [dump_func, name, dtype_str] = op_dump_info_table[id];
      oss << name << "." << dtype_str << ".";
      dump_func(info, oss);
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  } else if (head & (1ul << V_HEAD_LOAD_FLAG_OFFSET)) {
    uint64_t ext = (head >> V_M_HEAD_EXT_OFFSET) & V_M_HEAD_EXT_MASK;
    offset = (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext, simd_width};
    id = convert_id(g_load_func_offset, V_LOAD_NONE, id);
    if (load_dump_func_table.find(id) != load_dump_func_table.end()) {
      load_dump_func_table[id](info, oss);
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  } else {
    uint64_t ext = (head >> V_M_HEAD_EXT_OFFSET) & V_M_HEAD_EXT_MASK;
    offset = (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext, simd_width};
    id = convert_id(g_store_func_offset, V_STORE_NONE, id);
    if (store_dump_func_table.find(id) != store_dump_func_table.end()) {
      store_dump_func_table[id](info, oss);
#ifdef DEBUG
      uint64_t debug_size = *(insn + (((head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK) - 1));
      oss << ", ";
      DumpVal("dbg_size", debug_size, oss);
#endif
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  }
  return offset;
}

void DasBody(std::ostringstream &oss, uint8_t *bcode, uint64_t bcode_size, uint64_t simd_width, const std::string &indent) {
  bcodeptr_t insn = reinterpret_cast<bcodeptr_t>(bcode);
  bcodeptr_t insn_end = reinterpret_cast<bcodeptr_t>(bcode + bcode_size);
  uint64_t insn_idx = 0;
  while (insn < insn_end) {
    auto head = *insn;
    oss << indent << insn_idx << ": ";
    insn_idx++;
    auto offset = DumpInsn(insn, simd_width, oss);
    oss << "\n";
    if (offset == 0) break;
    insn = insn + offset;
    oss <<  indent << "  {";
    bool load_flag = bool(head & (1ul << V_HEAD_LOAD_FLAG_OFFSET));
    if (head & (1ul << V_HEAD_SIMD_FLAG_OFFSET)) {
      oss << "simd";
      if (head & (0x1ul << V_HEAD_BAR_FLAG_OFFSET)) {
        oss << ", bar(1)";
      }
      if (head & (0x1ul << V_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", load_simd_sync(wait, " << int((head >> V_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_BACK_WAIT_OFFSET)) {
        oss << ", store_simd_sync(wait, " << int((head >> V_HEAD_B_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_SET_FLAG_OFFSET)) {
        oss << ", simd_store_sync(set, " << int((head >> V_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_HEAD_BACK_SET_OFFSET)) {
        oss << ", simd_load_sync(set, " << int((head >> V_HEAD_B_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    } else if (load_flag) {
      oss << "load";
      if (head & (0x1ul << V_M_HEAD_SET_FLAG_OFFSET)) {
        oss << ", load_simd_sync(set, " << int((head >> V_M_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", simd_load_sync(wait, " << int((head >> V_M_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    } else {
      oss << "store";
      if (head & (0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET)) {
        oss << ", simd_store_sync(wait, " << int((head >> V_M_HEAD_WAIT_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
      if (head & (0x1ul << V_M_HEAD_SET_FLAG_OFFSET)) {
        oss << ", store_simd_sync(set, " << int((head >> V_M_HEAD_SET_EVENT_OFFSET) & V_HEAD_EVENT_MASK) << ")";
      }
    }
    oss << " }" << std::endl;
  }
}
}  // namespace

class DisAssembler {
 public:
  DisAssembler(std::ostringstream &oss_) : oss(oss_) {}

  void Run(Code *code, const char *prefix) {
    void* ffts = *reinterpret_cast<void**>(code->data_);
    uint64_t entry = *reinterpret_cast<uint64_t*>(code->data_ + sizeof(uint64_t));
    uint8_t *bcode = code->data_ + code->HeadSize();
    uint64_t bcode_size = code->data_size_ - code->HeadSize();
    if (!code->atomic_clean_.empty()) {
      for (auto ac : code->atomic_clean_) {
        Run(ac, "atomic_clean");
        oss << std::endl;
      }
    }
    oss << "// target=" << code->target_ << ", block_dim=" << code->block_dim_ << ", ffts_addr=" << ffts << std::endl;
    oss << prefix << ".";
    if (entry & V_ENTRY_FLAG_NEXT_STAGE) {
      DasStages(entry, bcode, bcode_size, "");
    } else if (entry & V_ENTRY_FLAG_PARALLEL) {
      DasParallel(entry, bcode, bcode_size, "");
    } else if (entry & V_ENTRY_FLAG_MIX) {
      if (bcode_size == sizeof(vCubeOp)) {
        DasCube(entry, bcode, bcode_size, "");
      } else {
        DasMix(entry, bcode, bcode_size, "");
      }
    } else {
      DasVec(entry, bcode, bcode_size, "");
    }
  }

  void DasVecBody(uint8_t *bcode, uint64_t bcode_size, uint64_t simd_width, const std::string &indent) {
    DasBody(oss, bcode, bcode_size, simd_width, indent);
  }

  void DasCubeBody(vCubeOp *op, const std::string &indent) {
    oss << indent << "MatMul." << op->m_real << "x" << op->k_real << "x" << op->n_real << " " << reinterpret_cast<void*>(op->gm_c) <<
        " " << reinterpret_cast<void*>(op->gm_a) << " " << reinterpret_cast<void*>(op->gm_b);
    oss << " //";
    DumpVal("m0", op->m0, oss);
    oss << ", ";
    DumpVal("k0", op->k0, oss);
    oss << ", ";
    DumpVal("n0", op->n0, oss);
    oss << ", ";
    DumpVal("trans_a", bool(op->flags & V_CUBE_FLAG_TRANS_A), oss);
    oss << ", ";
    DumpVal("trans_b", bool(op->flags & V_CUBE_FLAG_TRANS_B), oss);
    oss << ", ";
    DumpVal("swizzle", op->swizzle, oss);
  }

  void DasVec(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto tile_num = vGetBitRange(entry, V_ENTRY_TILE_NUM_OFFSET, V_ENTRY_TILE_NUM_BITS);
    auto simd_width = vGetBitRange(entry, V_ENTRY_SIMD_WIDTH_OFFSET, V_ENTRY_SIMD_WIDTH_BITS);
    oss << indent << "aiv(tile_num=" << tile_num << ", simd_width=" << simd_width;
    if (entry & V_ENTRY_FLAG_MIX) {
      oss <<", mix=1";
    }
    if (entry & V_ENTRY_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    oss << ") {" << std::endl;
    DasVecBody(bcode, bcode_size, simd_width, indent + "  ");
    oss << indent << "}";
  }

  void DasCube(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    oss << indent << "aic(mix=" << bool(entry & V_ENTRY_FLAG_MIX);
    vCubeOp *cube = reinterpret_cast<vCubeOp*>(bcode);
    if (cube->flags & V_CUBE_FLAG_GROUP_SET) {
      oss << ", sub_tile_num=[" << (cube->subtilenum & 0xfffffffful) << ", " << (cube->subtilenum >> 32) << "]";
      oss << ", group_set=1";
    }
    if (cube->flags & V_CUBE_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    if (cube->flags & V_CUBE_FLAG_PINGPONG_STORE) {
      oss << ", pingpong_store=1";
    }
    oss << ") {" << std::endl;
    DasCubeBody(cube, indent + "  ");
    oss << std::endl << indent << "}";
  }

  void DasMix(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    oss << indent << "mix(tile_num=" << vGetBitRange(entry, V_ENTRY_TILE_NUM_OFFSET, V_ENTRY_TILE_NUM_BITS) << ") {" << std::endl;
    DasCube(entry, bcode, sizeof(vCubeOp), indent + "  ");
    oss << std::endl;
    DasVec(entry, bcode + sizeof(vCubeOp), bcode_size - sizeof(vCubeOp), indent + "  ");
    oss << std::endl << indent << "}";
  }

  void DasParallel(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    struct Summary {
      Summary(uint8_t *code = nullptr) : bcode(code) {}
      int64_t block_start{-1};
      int64_t block_end{-1};
      int64_t block_step{-1};
      int64_t block_tail{-1};
      int64_t simd_width{-1};
      uint8_t *bcode;
      int64_t code_size;
    };
    std::vector<Summary> summays;
    Summary *current = nullptr;
    uint64_t *summaries = reinterpret_cast<uint64_t*>(bcode);
    auto block_dim = vGetBitRange(entry, V_ENTRY_TILE_NUM_OFFSET, V_ENTRY_TILE_NUM_BITS);
    for (uint64_t i = 0; i < block_dim; ++i) {
      uint64_t sum_data = *summaries++;
      uint64_t start_idx = sum_data & 0xffffful;
      uint64_t tile_loops = (sum_data >> 20) & 0xffffful;
      uint64_t offset = ((sum_data >> 49) & 0x1fful) * 32;
      uint64_t tail_flag = (sum_data >> 40) & 0x1ul;
      auto vec_code = bcode + offset;
      if (current == nullptr || current->bcode != vec_code) {
        summays.emplace_back(vec_code);
        current = &(summays.back());
        current->block_start = i;
        current->block_step = tile_loops;
        current->bcode = vec_code;
        current->simd_width = (sum_data >> 41) & 0xfful;
        current->code_size = (sum_data >> 58) * 32;
      }
      if (tail_flag) {
        current->block_end = i;
        current->block_tail = start_idx + tile_loops;
      }
    }
    oss << indent << "parallel() {" << std::endl;
    for (uint64_t i = 0; i < summays.size(); ++i) {
      auto &summary = summays[i];
      oss << " kernel_" << i << "(simd_width="<< summary.simd_width <<
            ", block_range=[" << summary.block_start << ", " << summary.block_end << "], block_step=" << summary.block_step <<
            ", block_tail=" << summary.block_tail << ") {" << std::endl;
      DasVecBody(summary.bcode, summary.code_size, summary.simd_width, indent + "  ");
      oss << indent << " }" << std::endl;
    }
    oss << indent << "}";
  }

  void DasStages(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    oss << indent << "stages() {" << std::endl;
    int stage_idx = 0;
    uint8_t *bcode_end = bcode + bcode_size;
    while (bcode < bcode_end) {
      uint64_t stage_size = vGetBitRange(entry, V_ENTRY_CODE_SIZE_OFFSET, V_ENTRY_CODE_SIZE_BITS) * sizeof(uint64_t);
      oss << "  // stage " << stage_idx << std::endl;
      stage_idx++;
      if (entry & V_ENTRY_FLAG_MIX) {
        if (stage_size == sizeof(vCubeOp)) {
          DasCube(entry, bcode, stage_size, indent + "  ");
        } else {
          DasMix(entry, bcode, stage_size, indent + "  ");
        }
      } else if (entry & V_ENTRY_FLAG_PARALLEL) {
        DasParallel(entry, bcode, stage_size, indent + "  ");
      } else {
        DasVec(entry, bcode, stage_size, indent + "  ");
      }
      oss << std::endl;
      if ((entry & V_ENTRY_FLAG_NEXT_STAGE) == 0) break;
      bcode += stage_size;
      entry = *reinterpret_cast<uint64_t*>(bcode);
      bcode += sizeof(uint64_t);
    }
    oss << indent << "}";
  }

 private:
  std::ostringstream &oss;
};

Code::~Code() {
  if (data_) {
    if (mem_size_ <= PARAM_TABLE_LIMIT) {
      std::free(data_);
    } else {
      auto ret = aclrtFreeHost(data_);
      EXCEPTION_IF(ret != 0, "aclrtFreeHost error");
    }
  }
}

void Code::Alloc(size_t size) {
  if (size <= mem_size_) {
    return;
  }
  if (size <= PARAM_TABLE_LIMIT) {
    if (data_) {
      data_ = static_cast<unsigned char *>(std::realloc(data_, size));
    } else {
      data_ = static_cast<unsigned char *>(std::malloc(size));
    }
  } else {
    if (data_) {
      if (mem_size_ <= PARAM_TABLE_LIMIT) {
        std::free(data_);
      } else {
        auto ret = aclrtFreeHost(data_);
        EXCEPTION_IF(ret != 0, "Alloc aclrtFreeHost error");
      }
    }
    constexpr uint32_t RT_MEM_ALIGN = 1024 * 1024;
    size = (size + RT_MEM_ALIGN - 1) / RT_MEM_ALIGN * RT_MEM_ALIGN;
    auto ret = aclrtMallocHost(reinterpret_cast<void **>(&data_), size);
    EXCEPTION_IF(ret != 0, "Alloc aclrtMallocHost error");
  }
  mem_size_ = size;
}

void Code::DisAssemble(std::ostringstream &oss) {
  DisAssembler(oss).Run(this, "vmain");
}

void Code::LinkBody(uint64_t offset, const Code &code, const std::vector<NDAccess*> &ios, uint64_t ws_offset) {
  std::memcpy(data_ + offset, code.data_ + HeadSize(), code.data_size_ - HeadSize());
  uint64_t *new_base = reinterpret_cast<uint64_t*>(data_ + offset);
  uint64_t *old_base = reinterpret_cast<uint64_t*>(code.data_ + HeadSize());
  for (auto a :  ios) {
    a->reloc_addr_ = a->reloc_addr_ - old_base + new_base;
  }
  if (!code.reloc_workspaces_.empty()) {
    for (auto &[dst, offset]: code.reloc_workspaces_) {
      reloc_workspaces_.emplace_back(dst - old_base + new_base, offset + ws_offset);
    }
  }
  if (!code.reloc_reuse_.empty()) {
    for (auto &[old_dst, old_src] : code.reloc_reuse_) {
      uint64_t *src = old_src - old_base + new_base;
      uint64_t *dst = old_dst >= old_base && old_dst < reinterpret_cast<uint64_t *>(code.data_ + code.data_size_)
                        ? old_dst - old_base + new_base
                        : old_dst;
      reloc_reuse_.emplace_back(dst, src);
    }
  }
  if (!code.atomic_clean_.empty()) {
    for (auto a : code.atomic_clean_) {
      atomic_clean_.push_back(a);
    }
  }
}

int Code::LaunchAtomicClean(void* stream) {
  for (auto a : atomic_clean_) {
    uint8_t* a_stub = System::Instance().StubFunc(a->target_);
    auto ret = System::Instance().launch_func_(a_stub, a->block_dim_, a->data_, a->data_size_, nullptr, stream);
    if (ret != RT_ERROR_NONE) return ret;
  }
  return 0;
}

int Code::LaunchEx(void *workspace, void* stream) {
#ifdef VK_SIM_MODEL
  return -1;
#else
  auto data_dev = reinterpret_cast<uint8_t*>(workspace) + extern_code_;
  auto ret = aclrtMemcpyAsync(data_dev, data_size_, data_, data_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  EXCEPTION_IF(ret != 0, "aclrtMemcpyAsync error");
  uint64_t args[] = {reinterpret_cast<uint64_t>(data_dev), *(reinterpret_cast<uint64_t*>(data_) + 1)};
  auto stub_func = System::Instance().StubFunc(target_);
  return System::Instance().launch_func_(stub_func, block_dim_, args, sizeof(args), nullptr, stream);
#endif
}

StageLinker::StageLinker(Code &code, int64_t stage_num, int64_t total_size) : code_(code) {
  constexpr int64_t ffts_size = sizeof(uint64_t);
  code.Alloc(total_size + ffts_size - ffts_size * stage_num);
  code_.target_ = Code::kTargetMix;
  code_.block_dim_ = 0;
  *reinterpret_cast<uint64_t*>(code_.data_) = 0; //ffts
  code_.data_size_ = ffts_size;
}

int StageLinker::Add(const Code &code, int64_t ws_offset, const std::vector<NDAccess*> &ios) {
  constexpr int64_t ffts_size = sizeof(uint64_t);
  int64_t code_offset = code_.data_size_;
  code_.data_size_ += code.data_size_ - ffts_size;
  if (code.target_ == Code::kTargetVec) {
    auto group_num = (code.block_dim_ + 1) / 2;
    if (group_num > code_.block_dim_) code_.block_dim_ = group_num;
  } else if (code.block_dim_ > code_.block_dim_) {
    code_.block_dim_ = code.block_dim_;
  }
  code_.LinkBody(code_offset + sizeof(uint64_t), code, ios, ws_offset);
  auto cur_entry = *reinterpret_cast<uint64_t*>(code.data_ + ffts_size);
  if (!code_offsets_.empty()) { // add sync
    auto pre_code = code_.data_ + code_offsets_.back();
    auto pre_entry = *reinterpret_cast<uint64_t*>(pre_code);
    pre_entry |= V_ENTRY_FLAG_NEXT_STAGE;
    if ((pre_entry & V_ENTRY_FLAG_MIX) &&
        !(reinterpret_cast<vCubeOp*>(pre_code + sizeof(uint64_t))->flags & V_CUBE_FLAG_GROUP_SET)) { // cube->vector/cube/mix
      if (!(cur_entry & V_ENTRY_FLAG_MIX)) {
        cur_entry |= V_ENTRY_FLAG_PRE_WAIT;
      }
    } else if (cur_entry & V_ENTRY_FLAG_MIX) { // vector/mix->cube/mix
      auto cube = reinterpret_cast<vCubeOp*>(code_.data_ + code_offset + sizeof(uint64_t));
      cube->flags |= V_CUBE_FLAG_PRE_WAIT;
    }
    *reinterpret_cast<uint64_t*>(pre_code) = pre_entry;
  }
  *reinterpret_cast<uint64_t*>(code_.data_ + code_offset) = cur_entry;
  int index = code_offsets_.size();
  code_offsets_.push_back(code_offset);
  return index;
}

void StageLinker::RelocWorkspace(NDAccess* op, int64_t ws_offset) {
  code_.reloc_workspaces_.emplace_back(op->reloc_addr_, ws_offset);
}

void StageLinker::RelocReuse(NDAccess* op, NDAccess* reuse) {
  auto reloc_addr = op->reloc_addr_;
  for (auto it = code_.reloc_reuse_.begin(); it != code_.reloc_reuse_.end(); ++it) {
    if (it->second == reloc_addr) {
      code_.reloc_reuse_.emplace(it, reloc_addr, reuse->reloc_addr_);
      return;
    }
  }
  code_.reloc_reuse_.emplace_back(reloc_addr, reuse->reloc_addr_);
}
}  // namespace dvm
