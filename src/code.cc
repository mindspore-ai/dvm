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

#include <dlfcn.h>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <stdexcept>
#ifndef VK_SIM_MODEL
#include "acl/acl_base.h"
#endif
#include "code.h"
#include "ops.h"

// rts_runtime
#if defined(__cplusplus)
extern "C" {
#endif
#define RT_DEV_BINARY_MAGIC_ELF        0x43554245U
#define RT_DEV_BINARY_MAGIC_ELF_AICPU  0x41415243U
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC  0x41415246U
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);
rtError_t rtDevBinaryUnRegister(void *hdl);
rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                         const void *kernelInfoExt, uint32_t funcMode);
rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                         rtSmDesc_t *smDesc, rtStream_t stm);
rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
#if defined(__cplusplus)
}
#endif

extern uint64_t g_simd_func_offset[];
extern uint64_t g_load_func_offset[];
extern uint64_t g_store_func_offset[];
extern const unsigned char g_vkernel_bin[];
extern unsigned int  g_vkernel_bin_len;
extern const unsigned char g_vkernel_910b_bin[];
extern unsigned int  g_vkernel_910b_bin_len;

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
  auto convert_id = [](uint64_t offsets[], uint64_t none_idx, uint64_t id) -> uint64_t {
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

void DvmException(const char* error_str) {
  std::ostringstream oss;
  oss << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}

DeviceInfo::DeviceInfo() {
  auto soc_name = GetSocName();
  if (soc_name.find("Ascend910B") != std::string::npos || soc_name.find("Ascend910C") != std::string::npos) {
    arch_ = kAiCore_C220;
    local_mem_size_ = 192 * 1024;
    event_num_ = 8;
    if (soc_name == "Ascend910B1" || soc_name == "Ascend910B2" || soc_name == "Ascend910C1" ||
        soc_name == "Ascend910C2") {
      vector_core_num_ = 48;
      cube_core_num_ = 24;
    } else {
      vector_core_num_ = 40;
      cube_core_num_ = 20;
    }
    l2_size_ = (soc_name == "Ascend910B4" || soc_name == "Ascend910C4") ? (96 * 1024 * 1024) : (192 * 1024 * 1024);
    l1_size_ = 512 * 1024;
    l0c_size_ = 128 * 1024;
  } else {
    arch_ = kAiCore_C100;
    local_mem_size_ = 256 * 1024;
    event_num_ = 4;
    vector_core_num_ = 32;
    cube_core_num_ = vector_core_num_;
    l2_size_ = 32 * 1024 * 1024;
    l1_size_ = 1024 * 1024;
    l0c_size_ = 256 * 1024;
    auto set_func_ids = [](uint64_t offsets[], uint64_t size) {
      for (uint64_t i = 0; i <= size; ++i) {
        offsets[i] = i;
      }
    };
    set_func_ids(g_simd_func_offset, V_NONE);
    set_func_ids(g_load_func_offset, V_LOAD_NONE);
    set_func_ids(g_store_func_offset, V_STORE_NONE);
  }
  ub_workspace_size_ = 1024;
  std::unordered_map<std::string, SocType> soc_name_map = {{"Ascend910B1", kAscend910B1},
                                                           {"Ascend910B2", kAscend910B2},
                                                           {"Ascend910B3", kAscend910B3},
                                                           {"Ascend910B4", kAscend910B4}};
  if (const auto &iter = soc_name_map.find(soc_name); iter != soc_name_map.end()) {
    soc_name_ = iter->second;
  }

#ifdef VK_SIM_MODEL
  auto rt_binary_register = rtDevBinaryRegister;
  auto rt_function_register = rtFunctionRegister;
  launch_func_ = rtKernelLaunch;
#else
  void *handle = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(handle == nullptr, "Load libruntime.so failed");
  auto rt_binary_register =
    reinterpret_cast<rtError_t (*)(const rtDevBinary_t *, void **)>(dlsym(handle, "rtDevBinaryRegister"));
  EXCEPTION_IF(rt_binary_register == nullptr, "load rt_binary_register symbol failed");
  auto rt_function_register =
    reinterpret_cast<rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t)>(
      dlsym(handle, "rtFunctionRegister"));
  EXCEPTION_IF(rt_function_register == nullptr, "load rt_function_register symbol failed");
  launch_func_ = reinterpret_cast<rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t)>(
    dlsym(handle, "rtKernelLaunch"));
  EXCEPTION_IF(launch_func_ == nullptr, "load rt_kernel_launch symbol failed");
#endif
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  if (arch_ == kAiCore_C100) {
    dev_bin.data = g_vkernel_bin;
    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    dev_bin.length = g_vkernel_bin_len;
    err = rt_binary_register(&dev_bin, &module);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg binary failed");
    uint8_t* stub_func = reinterpret_cast<uint8_t*>(this) + Code::kTargetVec;
    err = rt_function_register(module, stub_func, "vmain_mix_aiv",  "vmain_mix_aiv", 0);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg function failed");
  } else {
    dev_bin.data = g_vkernel_910b_bin;
    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    dev_bin.length = g_vkernel_910b_bin_len;
    err = rt_binary_register(&dev_bin, &module);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
    uint8_t* stub_func = reinterpret_cast<uint8_t*>(this) + Code::kTargetVec;
    err = rt_function_register(module, stub_func, "vmain_mix_aiv",  "vmain_mix_aiv", 0);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
    err = rt_binary_register(&dev_bin, &module);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
    stub_func = reinterpret_cast<uint8_t*>(this) + Code::kTargetCube;
    err = rt_function_register(module, stub_func, "vmain_mix_aic",  "vmain_mix_aic", 0);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

    dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    err = rt_binary_register(&dev_bin, &module);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
    stub_func = reinterpret_cast<uint8_t*>(this) + Code::kTargetMix;
    err = rt_function_register(module, stub_func, "vmain",  "vmain", 0);
    EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");

#ifdef VK_SIM_MODEL
    get_c2c_addr_func_ = rtGetC2cCtrlAddr;
#else
    get_c2c_addr_func_ = reinterpret_cast<rtError_t(*)(uint64_t*, uint32_t*)>(dlsym(handle, "rtGetC2cCtrlAddr"));
#endif
  }
}

class DisAssembler {
 public:
  DisAssembler(std::ostringstream &oss_) : oss(oss_) {}

  void Run(Code *code) {
    void* ffts = *reinterpret_cast<void**>(code->data_);
    uint64_t entry = *reinterpret_cast<uint64_t*>(code->data_ + sizeof(uint64_t));
    uint8_t *bcode = code->data_ + code->HeadSize();
    uint64_t bcode_size = code->data_size_ - code->HeadSize();
    oss << "// block_dim=" << code->block_dim_ << ", ffts_addr=" << ffts << std::endl;
    oss << "vmain.";
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
    if (entry & V_ENTRY_FLAG_GROUP) {
      oss <<", group=1";
    }
    if (entry & V_ENTRY_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    if (entry & V_ENTRY_FLAG_POST_SET) {
      oss << ", post_set=1";
    }
    if (entry & V_ENTRY_FLAG_POST_BAR) {
      oss << ", post_bar=1";
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
    if (cube->flags & V_CUBE_FLAG_POST_SET) {
      oss << ", post_set=1";
    }
    if (cube->flags & V_CUBE_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    if (cube->flags & V_CUBE_FLAG_POST_BAR) {
      oss << ", post_bar=1";
    }
    oss << ") {" << std::endl;
    DasCubeBody(cube, indent + "  ");
    oss << std::endl << indent << "}";
  }

  void DasMix(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    oss << indent << "mix(tile_num=" << vGetBitRange(entry, V_ENTRY_TILE_NUM_OFFSET, V_ENTRY_TILE_NUM_BITS);
    if (entry & V_ENTRY_FLAG_GROUP) {
      oss <<", group=1";
    }
    oss << ") {" << std::endl;
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

void Code::DisAssemble(std::ostringstream &oss) {
  DisAssembler(oss).Run(this);
}

void Code::LinkBody(uint64_t offset, const Code &code, const std::vector<NDAccess*> &ios, uint64_t ws_offset) {
  std::memcpy(data_ + offset, code.data_ + HeadSize(), code.data_size_ - HeadSize());
  uint64_t *new_base = reinterpret_cast<uint64_t*>(data_ + offset);
  uint64_t *old_base = reinterpret_cast<uint64_t*>(code.data_ + HeadSize());
  for (auto a :  ios) {
    a->reloc_addr_ = a->reloc_addr_ - old_base + new_base;
  }
  if (!code.reloc_workspaces_.empty()) {
    for (auto &r: code.reloc_workspaces_) {
      reloc_workspaces_.emplace_back(std::make_pair(r.first - old_base + new_base, r.second + ws_offset));
    }
  }
  if (!code.reloc_reuse_.empty()) {
    for (auto &r: code.reloc_reuse_) {
      uint64_t *src = r.second - old_base + new_base;
      uint64_t *dst = r.first >= old_base && r.first < reinterpret_cast<uint64_t*>(code.data_ + code.data_size_)
                    ? r.first - old_base + new_base : r.first;
      reloc_reuse_.emplace_back(std::make_pair(dst, src));
    }
  }
  if (!code.atomic_clean_.empty()) {
    for (auto a : code.atomic_clean_) {
      atomic_clean_.push_back(a);
    }
  }
}
}  // namespace dvm
