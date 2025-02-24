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
#define aclrtMallocHost(addr, size) \
  0;                                \
  *addr = std::malloc(size)
#define aclrtFreeHost(addr) \
  0;                        \
  std::free(addr)
#endif

namespace dvm {
namespace {
std::unordered_map<std::string, vCompareType> cmp_insn_id = {
  {"Greater", V_CMP_GT},   {"Less", V_CMP_LT},  {"GreaterEqual", V_CMP_GE},
  {"LessEqual", V_CMP_LE}, {"Equal", V_CMP_EQ}, {"NotEqual", V_CMP_NE},
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

void DumpSLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSLoad op;
  vSLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sload " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", "
      << reinterpret_cast<void *>(op.gm);
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
  oss << ", ";
  DumpVal("flags", op.flags, oss);
}

void DumpSStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSStore op;
  vSStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "sstore " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.gm) << ", "
      << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("cube_m", op.slice_m, oss);
  oss << ", ";
  DumpVal("cube_n", op.slice_n, oss);
}

void DumpAtomicCum(const DumpInfo &dump_info, std::ostringstream &oss) {
  vAtomicCum op;
  vAtomicCum::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vAtomicCum::ROUND_OFFSET, oss);
  }
}

void DumpSliceLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSliceSL op;
  vSliceSL::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "slice_load " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", "
      << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("offset", op.offset, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vSliceSL::ROUND_OFFSET, oss);
  }
}

void DumpSliceStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSliceSL op;
  vSliceSL::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "slice_store " << op.type_size << "x" << op.tile_stride << " " << reinterpret_cast<void *>(op.xn) << ", "
      << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("one_flag", op.one_flag, oss);
}

void DumpLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vLoad op;
  vLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "load.u8." << op.iter_size << "x" << op.body_iter;
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

void DumpMultiLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vMultiLoad op;
  vMultiLoad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "multi_load.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tile_stride2", op.tile_stride2, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("gap", op.gap, oss);
  oss << ", ";
  DumpVal("xbuf_size", op.xbuf_size, oss);
  oss << ", ";
  DumpVal("peer_mem", reinterpret_cast<void *>(op.peer_mem), oss);
  oss << ", ";
  DumpVal("multi_size", op.multi_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
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

void DumpPingpongPeerLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPingPongPeerLoad p_load;
  vPingPongPeerLoad::Decode(dump_info.insn, *dump_info.insn, p_load);
  vPingPongLoad &op = p_load.base;
  oss << "PingPongPeerLoad.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.from);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
  oss << ", ";
  DumpVal("pingpong_stride", op.pingpong_stride, oss);
  oss << ", ";
  DumpVal("offset", p_load.peer_mem_offset, oss);
  oss << ", ";
  DumpVal("set_flag", reinterpret_cast<void *>(p_load.set_flag), oss);
  oss << ", ";
  DumpVal("wait_flag", reinterpret_cast<void *>(p_load.wait_flag), oss);
  oss << ", ";
  DumpVal("event_id", reinterpret_cast<void *>(p_load.event_id), oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vPingPongPeerLoad::ROUND_OFFSET, oss);
  }
}

void DumpStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStore op;
  vStore::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStore::ROUND_OFFSET, oss);
  }
}

void DumpStoreCond(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreCond op;
  vStoreCond::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_cond.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("cond_offset", op.cond_offset, oss);
  oss << ", ";
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreCond::ROUND_OFFSET, oss);
  }
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

void DumpUnaryWS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " // ";
  DumpVal("ws", reinterpret_cast<void *>(op.xm), oss);
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
  vBinaryS op;
  vBinaryS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << ", "
      << vBinaryS::GetScalar<T>(dump_info.insn);
}

void DumpBinary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm);
}

void DumpBinaryWS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinaryWS op;
  vBinaryWS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("ws0", reinterpret_cast<void *>(op.ws0), oss);
  oss << ", ";
  DumpVal("ws1", reinterpret_cast<void *>(op.ws1), oss);
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
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

template <typename T>
void DumpCompareS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCompareS op;
  vCompareS::Decode(dump_info.insn, *dump_info.insn, op);
  std::string cmp_op("Unknown");
  for (auto it = cmp_insn_id.begin(); it != cmp_insn_id.end(); ++it) {
    if (it->second == op.type) {
      cmp_op = it->first;
    }
  }
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << vCompareS::GetScalar<T>(dump_info.insn) << " //";
  DumpVal("cmp_type", cmp_op, oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
}

void DumpBroadcastS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastS op;
  vBroadcastS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << op.scalar;
}

void DumpSelect(const DumpInfo &dump_info, std::ostringstream &oss) {
  vSelect op;
  vSelect::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.cond) << ", "
      << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm) << " //";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
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
  oss << "[" << op.red_size << "]x" << op.dup_size;
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

void DumpReduceJoin(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceJoin op;
  vReduceJoin::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.repeat << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("xs", reinterpret_cast<void *>(op.xs), oss);
  oss << ", ";
  DumpVal("ws", reinterpret_cast<void *>(op.ws), oss);
  oss << ", ";
  DumpVal("seg_tile_rel", op.seg_tile_rel, oss);
}

void DumpCopy(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCopy op;
  auto head = *dump_info.insn;
  vCopy::Decode(dump_info.insn, head, op);
  auto lenburst = (op.config) >> 16 & 0xfffful;
  oss << "32x" << lenburst;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

void DumpNop(const DumpInfo &dump_info, std::ostringstream &oss){
  oss << "0";
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
  oss << op.xd_lead << "x" << op.dup_size << " " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("xd_pad", op.xd_pad, oss);
  oss << ", ";
  DumpVal("xn_lead", op.xn_lead, oss);
  oss << ", ";
  DumpVal("xn_pad", op.xn_pad, oss);
}

void DumpStoreAG(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreAG op;
  vStoreAG::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_ag.u8.32x" << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_num", op.iter_num, oss);
  oss << ", ";
  DumpVal("iter_size", op.iter_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
  oss << ", ";
  DumpVal("rank_size", op.rank_size, oss);
  oss << ", ";
  DumpVal("shard_stride", op.shard_stride, oss);
}

void DumpStoreRS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreRS op;
  vStoreRS::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store_rs.u8." << op.iter_size << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.to) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("iter_tail", op.iter_tail, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("rank_id", op.rank_id, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vStoreRS::ROUND_OFFSET, oss);
  }
}

const char name_peer_load[] = "peer_load";
const char name_peer_load_mix[] = "peer_load_mix";
const char name_peer_store[] = "peer_store";
const char name_peer_store_mix[] = "peer_store_mix";

template <char const *name>
void DumpPeerDMA(const DumpInfo &dump_info, std::ostringstream &oss) {
  vPeerDMA op;
  vPeerDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << name;
  oss << ".u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.peer_mem);
  oss << " //";
  DumpVal("rank_id", op.rank_id, oss);
  oss << ", ";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
  oss << ", ";
  DumpVal("flag_mem", reinterpret_cast<void *>(op.flag_mem), oss);
  oss << ", ";
  DumpVal("set_flag", op.set_flag, oss);
  oss << ", ";
  DumpVal("wait_flag", op.wait_flag, oss);
  oss << ", ";
  DumpVal("set_event_id", op.event_id, oss);
  if (op.round_rank > 0) {
    oss << ", ";
    DumpRounds(op.round_rank, dump_info.insn + vPeerDMA::ROUND_OFFSET, oss);
  }
}

using DumpFunc = void(const DumpInfo &, std::ostringstream &oss);

std::unordered_map<uint64_t, DumpFunc *> acc_dump_func_table = {
  {V_LOAD, &DumpLoad},
  {V_LOAD_DUMMY, &DumpLoadDummy},
  {V_SLICE_LOAD, &DumpSliceLoad},
  {V_SLOAD, &DumpSLoad},
  {V_MULTI_LOAD, &DumpMultiLoad},
  {V_PINGPONG_LOAD, &DumpPingPongLoad},
  {V_PINGPONG_PEER_LOAD, &DumpPingpongPeerLoad},
  {V_PEER_LOAD, &DumpPeerDMA<name_peer_load>},
  {V_PEER_LOAD_MIX, &DumpPeerDMA<name_peer_load_mix>},
  {V_STORE, &DumpStore},
  {V_STORE_ATOMIC, &DumpStoreAtomic},
  {V_STORE_STATUS, &DumpStoreStatus},
  {V_STORE_COND, &DumpStoreCond},
  {V_SSTORE, &DumpSStore},
  {V_STORE_AG, &DumpStoreAG},
  {V_STORE_RS, &DumpStoreRS},
  {V_PEER_STORE, &DumpPeerDMA<name_peer_store>},
  {V_PEER_STORE_MIX, &DumpPeerDMA<name_peer_store_mix>},
  {V_SLICE_STORE, &DumpSliceStore},
};

std::unordered_map<uint64_t, std::tuple<DumpFunc *, std::string, std::string>> op_dump_info_table = {
  {V_COPY, {&DumpCopy, "Copy", "u8"}},
  {V_NOP, {&DumpNop, "Nop", "u8"}},
  {V_BROADCAST_X_B32, {&DumpBroadcastX, "BroadcastX", "b32"}},
  {V_BROADCAST_X_B16, {&DumpBroadcastX, "BroadcastX", "b16"}},
  {V_BROADCAST_Y, {&DumpBroadcastY, "BroadcastY", "u8"}},
  {V_BROADCAST_S, {&DumpBroadcastS, "BroadcastS", "b32"}},
  {V_BROADCAST_S_B16, {&DumpBroadcastS, "BroadcastS", "b16"}},
  {V_SQRT, {&DumpUnary, "Sqrt", "fp32"}},
  {V_SQRT_FP16, {&DumpUnary, "Sqrt", "fp16"}},
  {V_ABS, {&DumpUnary, "Abs", "fp32"}},
  {V_ABS_FP16, {&DumpUnary, "Abs", "fp16"}},
  {V_LOG, {&DumpUnary, "Log", "fp32"}},
  {V_LOG_FP16, {&DumpUnary, "Log", "fp16"}},
  {V_EXP, {&DumpUnary, "Exp", "fp32"}},
  {V_EXP_FP16, {&DumpUnary, "Exp", "fp16"}},
  {V_ROUND, {&DumpUnary, "Round", "fp32"}},
  {V_ROUND_FP16, {&DumpUnary, "Round", "fp16"}},
  {V_FLOOR, {&DumpUnary, "Floor", "fp32"}},
  {V_FLOOR_FP16, {&DumpUnary, "Floor", "fp16"}},
  {V_CEIL, {&DumpUnary, "Ceil", "fp32"}},
  {V_CEIL_FP16, {&DumpUnary, "Ceil", "fp16"}},
  {V_TRUNC, {&DumpUnary, "Trunc", "fp32"}},
  {V_TRUNC_FP16, {&DumpUnary, "Trunc", "fp16"}},
  {V_ISFINITE, {&DumpUnary, "IsFinite", "fp32"}},
  {V_ISFINITE_FP16, {&DumpUnaryWS, "IsFinite", "fp16"}},
  {V_CAST_FP16_TO_FP32, {&DumpUnary, "CastFP16", "fp32"}},
  {V_CAST_BOOL_TO_FP16, {&DumpUnary, "CastS8", "fp16"}},
  {V_CAST_FP16_TO_BOOL, {&DumpUnary, "CastFP16", "bool"}},
  {V_CAST_FP16_TO_INT32, {&DumpUnary, "CastFP16", "int32"}},
  {V_CAST_FP32_TO_INT32, {&DumpUnary, "CastFP32", "int32"}},
  {V_CAST_FP32_TO_FP16, {&DumpUnary, "CastFP32", "fp16"}},
  {V_CAST_FP32_TO_BF16, {&DumpUnary, "CastFP32", "bf16"}},
  {V_CAST_INT32_TO_FP32, {&DumpUnary, "CastS32", "fp32"}},
  {V_CAST_INT32_TO_FP16, {&DumpUnary, "CastS32", "fp16"}},
  {V_CAST_BF16_TO_FP32, {&DumpUnary, "CastBF16", "fp32"}},
  {V_CAST_BF16_TO_INT32, {&DumpUnary, "CastBF16", "int32"}},
  {V_ADDS, {&DumpBinaryS, "Adds", "fp32"}},
  {V_ADDS_FP16, {&DumpBinaryS<Float16>, "Adds", "fp16"}},
  {V_ADDS_INT32, {&DumpBinaryS<int32_t>, "Adds", "int32"}},
  {V_MULS, {&DumpBinaryS, "Muls", "fp32"}},
  {V_MULS_FP16, {&DumpBinaryS<Float16>, "Muls", "fp16"}},
  {V_SDIV, {&DumpBinaryS, "SDiv", "fp32"}},
  {V_SDIV_FP16, {&DumpBinaryS<Float16>, "Divs", "fp16"}},
  {V_MULS_INT32, {&DumpBinaryS<int32_t>, "Muls", "int32"}},
  {V_MAXS, {&DumpBinaryS, "Maximums", "fp32"}},
  {V_MAXS_FP16, {&DumpBinaryS<Float16>, "Maximums", "fp16"}},
  {V_MAXS_INT32, {&DumpBinaryS<int32_t>, "Maximums", "int32"}},
  {V_MINS, {&DumpBinaryS, "Minimums", "fp32"}},
  {V_MINS_FP16, {&DumpBinaryS<Float16>, "Maximums", "fp16"}},
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
  {V_POW, {&DumpBinaryWS, "Pow", "fp32"}},
  {V_CMP, {&DumpCompare, "Compare", "fp32"}},
  {V_CMP_FP16, {&DumpCompare, "Compare", "fp16"}},
  {V_CMP_INT32, {&DumpCompare, "Compare", "int32"}},
  {V_CMPS, {&DumpCompareS<float>, "CompareS", "fp32"}},
  {V_CMPS_FP16, {&DumpCompareS<Float16>, "CompareS", "fp16"}},
  {V_SEL, {&DumpSelect, "Select", "fp32"}},
  {V_SEL_FP16, {&DumpSelect, "Select", "fp16"}},
  {V_SEL_INT32, {&DumpSelect, "Select", "int32"}},
  {V_RSUM_X, {&DumpReduceX, "SumX", "fp32"}},
  {V_RSUM_Y, {&DumpReduceY, "SumY", "fp32"}},
  {V_RSUM_JOIN, {&DumpReduceJoin, "SumJoin", "fp32"}},
  {V_CLR_PAD, {&DumpClearPad, "ClrPad", "fp32"}},
  {V_ELEMENT_ANY, {&DumpElementAny, "ElementAny", "fp32"}},
  {V_REMOVEPAD, {&DumpRemovePad, "RemovePad", "u32"}},
  {V_REMOVEPAD_U16, {&DumpRemovePad, "RemovePad", "u16"}},
  {V_ATOMICCUM, {&DumpAtomicCum, "AtomicCum", "fp32"}},
  {V_RESHAPE_B32, {&DumpReshape, "Reshape", "u32"}},
  {V_RESHAPE_B16, {&DumpReshape, "Reshape", "u16"}},
};

size_t DumpInsn(uint64_t *insn, uint64_t simd_width, std::ostringstream &oss, uint64_t &pipe) {
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
    pipe = V_PIPE_SIMD;
  } else {
    uint64_t ext = (head >> V_M_HEAD_EXT_OFFSET) & V_M_HEAD_EXT_MASK;
    offset = (head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK;
    DumpInfo info{insn, ext, simd_width};
    id = convert_id(g_access_func_offset, V_ACCESS_NONE, id);
    if (acc_dump_func_table.find(id) != acc_dump_func_table.end()) {
      acc_dump_func_table[id](info, oss);
      pipe = id >= V_STORE ? V_PIPE_STORE : V_PIPE_LOAD;
#ifdef DEBUG
      if (pipe == V_PIPE_STORE ) {
        uint64_t debug_size = *(insn + (((head >> V_M_HEAD_SIZE_OFFSET) & V_M_HEAD_SIZE_MASK) - 1));
        oss << ", ";
        DumpVal("dbg_size", debug_size, oss);
      }
#endif
    } else {
      oss << "Unknown insn: ";
      DumpVal("id", id, oss);
    }
  }
  return offset;
}

void DasBody(std::ostringstream &oss, uint8_t *bcode, uint64_t bcode_size, uint64_t simd_width,
             const std::string &indent) {
  bcodeptr_t insn = reinterpret_cast<bcodeptr_t>(bcode);
  bcodeptr_t insn_end = reinterpret_cast<bcodeptr_t>(bcode + bcode_size);
  uint64_t insn_idx = 0;
  uint64_t pipe = 0;
  while (insn < insn_end) {
    auto head = *insn;
    if (head == 0) break;
    oss << indent << insn_idx << ": ";
    insn_idx++;
    auto offset = DumpInsn(insn, simd_width, oss, pipe);
    oss << "\n";
    insn = insn + offset;
    oss << indent << "    {";
    if (pipe == V_PIPE_SIMD) {
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
    } else if (pipe == V_PIPE_LOAD) {
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
    void *ffts = *reinterpret_cast<void **>(code->data_);
    uint64_t entry = *reinterpret_cast<uint64_t *>(code->data_ + sizeof(uint64_t));
    uint8_t *bcode = code->data_ + code->HeadSize();
    uint64_t bcode_size = code->data_size_ - code->HeadSize();
    if (!code->sub_codes_.empty()) {
      for (auto ac : code->sub_codes_) {
        Run(ac, "_sub");
        oss << std::endl;
      }
    }
    oss << "// target=" << code->target_ << ", block_dim=" << code->block_dim_ << ", ffts_addr=" << ffts << std::endl;
    oss << prefix << ".";
    auto ktype = entry & V_ENTRY_MASK_TYPE;
    if (ktype == V_ENTRY_TYPE_V) {
      DasVec(entry, bcode, bcode_size, "");
    } else if (ktype == V_ENTRY_TYPE_VE) {
      DasVecEx(entry, bcode, bcode_size, "");
    } else if (ktype == V_ENTRY_TYPE_VP) {
      DasParallel(entry, bcode, bcode_size, "");
    } else if (ktype == V_ENTRY_TYPE_C) {
      DasCube(entry, bcode, bcode_size, "");
    } else if (ktype == V_ENTRY_TYPE_MIX) {
      DasMix(entry, bcode, bcode_size, "");
    } else {
      ASSERT(0);  // removed
      DasStages(entry, bcode, bcode_size, "");
    }
  }

  void DasVecBody(uint8_t *bcode, uint64_t bcode_size, uint64_t simd_width, const std::string &indent) {
    DasBody(oss, bcode, bcode_size, simd_width, indent);
  }

  void DasCubeBody(vCubeOp *op, const std::string &indent) {
    oss << indent << "MatMul." << op->m_real << "x" << op->k_real << "x" << op->n_real << " " << op->m_align << "x"
        << op->k_align << "x" << op->n_align << " " << reinterpret_cast<void *>(op->gm_c) << " "
        << reinterpret_cast<void *>(op->gm_a) << " " << reinterpret_cast<void *>(op->gm_b);
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
    if (op->flags & V_CUBE_FLAG_WITH_BIAS) {
      oss << ", ";
      DumpVal("gm_bias", reinterpret_cast<void *>(op->gm_bias), oss);
      oss << ", ";
      DumpVal("bias_type", (op->flags & V_CUBE_FLAG_BIAS_FP16) ? "fp16" : "fp32", oss);
    }
  }

  void DasVec(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto tile_body = vGetBitRange(entry, V_ENTRY_V_TILE_BODY_OFFSET, V_ENTRY_V_TILE_BODY_BITS);
    auto tile_tail = vGetBitRange(entry, V_ENTRY_V_TILE_TAIL_OFFSET, V_ENTRY_V_TILE_TAIL_BITS);
    auto block_num = vGetBitRange(entry, V_ENTRY_V_BLOCK_NUM_OFFSET, V_ENTRY_V_BLOCK_NUM_BITS);
    auto simd_width = vGetBitRange(entry, V_ENTRY_SIMD_WIDTH_OFFSET, V_ENTRY_SIMD_WIDTH_BITS);
    oss << indent << "aiv(tile_num=" << tile_body << 'x' << block_num << '-' << tile_tail
        << ", simd_width=" << simd_width;
    oss << ") {" << std::endl;
    DasVecBody(bcode, bcode_size, simd_width, indent + "  ");
    oss << indent << "}";
  }

  void DasVecEx(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto visit_id = vGetBitRange(entry, V_ENTRY_VE_VISIT_ID_OFFSET, V_ENTRY_VE_VISIT_ID_BITS);
    auto offset = vGetBitRange(entry, V_ENTRY_VE_VISIT_OFFSET_OFFSET, V_ENTRY_VE_VISIT_OFFSET_BITS);
    auto simd_width = vGetBitRange(entry, V_ENTRY_SIMD_WIDTH_OFFSET, V_ENTRY_SIMD_WIDTH_BITS);
    oss << indent << "aiv(visit={";
    auto visit_addr = bcode + offset * sizeof(uint64_t);
    constexpr uint64_t MASK_32 = 0xfffffffful;
    for (uint64_t i = 0; i < V_VISIT_NONE; ++i) {
      if (g_visit_func_offset[i] == visit_id) {
        visit_id = i;
        break;
      }
    }
    switch (visit_id) {
      case V_VISIT_RED_1: {
        auto v = reinterpret_cast<vVisitRed1 *>(visit_addr);
        oss << "e:" << (v->e >> 32) << ", r1:" << (v->r1);
        break;
      }
      case V_VISIT_RED_2: {
        auto v = reinterpret_cast<vVisitRed2 *>(visit_addr);
        oss << "e:" << (v->e >> 32) << ", r1:" << (v->e1_r1 & MASK_32) << ", e1:" << (v->e1_r1 >> 32);
        break;
      }
      case V_VISIT_RED_3: {
        auto v = reinterpret_cast<vVisitRed3 *>(visit_addr);
        oss << "e:" << (v->e >> 32) << ", r1:" << (v->r1) << ", e1:" << (v->e1_r2 >> 32) << ", r2:" << (v->e1_r2 & MASK_32);
        break;
      }
      case V_VISIT_RED_4: {
        auto v = reinterpret_cast<vVisitRed4 *>(visit_addr);
        oss << "e:" << (v->e >> 32) << ", r1:" << (v->e1_r1 & MASK_32) << ", e1:" << (v->e1_r1 >> 32)
            << ", r2:" << (v->e2_r2 & MASK_32) << ", e2:" << (v->e2_r2 >> 32);
        break;
      }
      default:
        break;
    }
    oss << "}, simd_width=" << simd_width << ") {" << std::endl;
    DasVecBody(bcode, bcode_size, simd_width, indent + "  ");
    oss << indent << "}";
  }

  void DasCube(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto group_num = vGetBitRange(entry, V_ENTRY_M_GROUP_NUM_OFFSET, V_ENTRY_M_GROUP_NUM_BITS);
    oss << indent << "aic(group_num=" << group_num << ") {" << std::endl;
    vCubeOp *cube = reinterpret_cast<vCubeOp *>(bcode);
    DasCubeBody(cube, indent + "  ");
    oss << std::endl << indent << "}";
  }

  void DasMix(uint64_t entry, uint8_t *bcode, uint64_t bcode_size, const std::string &indent) {
    auto group_num = vGetBitRange(entry, V_ENTRY_M_GROUP_NUM_OFFSET, V_ENTRY_M_GROUP_NUM_BITS);
    oss << indent << "mix(group_num=" << group_num << ") {" << std::endl;
    auto sub_indent = indent + "  ";
    vCubeOp *cube = reinterpret_cast<vCubeOp *>(bcode);
    ASSERT(cube->flags & V_CUBE_FLAG_GROUP_SET);
    oss << sub_indent << "aic(group_set=1";
    if (cube->flags & V_CUBE_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    if (cube->flags & V_CUBE_FLAG_PINGPONG_STORE) {
      oss << ", pingpong_store=1";
    }
    if (cube->flags & V_CUBE_FLAG_PEER_STORE) {
      oss << ", peer_store=1";
    }
    oss << ") {" << std::endl;
    DasCubeBody(cube, sub_indent + "  ");
    oss << std::endl << sub_indent << "}" << std::endl;
    oss << sub_indent << "aiv(sub_tile_num=[" << (cube->subtilenum & 0xfffffffful) << ", " << (cube->subtilenum >> 32)
        << "]";
    auto simd_width = vGetBitRange(entry, V_ENTRY_SIMD_WIDTH_OFFSET, V_ENTRY_SIMD_WIDTH_BITS);
    oss << ", simd_width=" << simd_width;
    if (entry & V_ENTRY_FLAG_PRE_WAIT) {
      oss << ", pre_wait=1";
    }
    oss << ") {" << std::endl;
    DasVecBody(bcode + sizeof(vCubeOp), bcode_size - sizeof(vCubeOp), simd_width, sub_indent + "  ");
    oss << sub_indent << "}" << std::endl;
    oss << indent << "}";
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
    uint64_t *summaries = reinterpret_cast<uint64_t *>(bcode);
    auto block_dim = vGetBitRange(entry, V_ENTRY_VP_BLOCK_SUM_OFFSET, V_ENTRY_VP_BLOCK_SUM_BITS);
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
    oss << indent << "parallel(block_num=" << block_dim << ") {" << std::endl;
    for (uint64_t i = 0; i < summays.size(); ++i) {
      auto &summary = summays[i];
      oss << " kernel_" << i << "(simd_width=" << summary.simd_width << ", block_range=[" << summary.block_start << ", "
          << summary.block_end << "], block_step=" << summary.block_step << ", block_tail=" << summary.block_tail
          << ") {" << std::endl;
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
      auto ktype = entry & V_ENTRY_MASK_TYPE;
      if (ktype == V_ENTRY_TYPE_V) {
        DasVec(entry, bcode, stage_size, indent + "  ");
      } else if (ktype == V_ENTRY_TYPE_VE) {
        DasVecEx(entry, bcode, stage_size, indent + "  ");
      } else if (ktype == V_ENTRY_TYPE_VP) {
        DasParallel(entry, bcode, stage_size, indent + "  ");
      } else if (ktype == V_ENTRY_TYPE_C) {
        DasCube(entry, bcode, stage_size, indent + "  ");
      } else {
        ASSERT(ktype == V_ENTRY_TYPE_MIX);
        DasMix(entry, bcode, stage_size, indent + "  ");
      }
      oss << std::endl;
      if ((entry & V_ENTRY_FLAG_NEXT_STAGE) == 0) break;
      bcode += stage_size;
      entry = *reinterpret_cast<uint64_t *>(bcode);
      bcode += sizeof(uint64_t);
    }
    oss << indent << "}";
  }

 private:
  std::ostringstream &oss;
};

std::atomic<uint32_t> Code::unique_id_ = 0;

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

Code &Code::operator=(Code &&other) {
  MoveCode(other);
  sub_codes_ = std::move(other.sub_codes_);
  unique_ids_ = std::move(other.unique_ids_);
  bind_wss_ = other.bind_wss_;
  bind_ops_ = other.bind_ops_;
  return *this;
}

void Code::MoveCode(Code &other) {
  ASSERT(data_ == nullptr);
  data_ = other.data_;
  data_size_ = other.data_size_;
  block_dim_ = other.block_dim_;
  target_ = other.target_;
  mem_size_ = other.mem_size_;

  other.data_ = nullptr;
  other.data_size_ = 0;
  other.mem_size_ = 0;
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

void Code::DisAssemble(std::ostringstream &oss) { DisAssembler(oss).Run(this, "vmain"); }

void Code::Combine(const Code &code, uint64_t ws_base) {
  if (auto op = code.bind_wss_) {
    for (auto next = op->bind_list_; next != nullptr; next = next->bind_list_) {
      BindWorkspace(*op, op->ws + ws_base);
      op = next;
    }
    BindWorkspace(*op, op->ws + ws_base);
  }
  if (auto op = code.bind_ops_) {
    for (auto next = op->bind_list_; next != nullptr; next = next->bind_list_) {
      BindOp(*op, *(op->op));
      op = next;
    }
    BindOp(*op, *(op->op));
  }
  if (!code.unique_ids_.empty()) {
    for (auto id : code.unique_ids_) {
      unique_ids_.push_back(id);
    }
  }
  if (!code.sub_codes_.empty()) {
    for (auto a : code.sub_codes_) {
      sub_codes_.push_back(a);
    }
  }
}

void Code::BindOp(RelocAddr &op, const RelocAddr &target) {
  op.op = &target;
  for (auto x = bind_ops_; x != nullptr; x = x->bind_list_) {
    if (x == op.op) {
      InsertBind(x->bind_list_, op);
      return;
    }
  }
  InsertBind(bind_ops_, op);
}

int Code::LaunchEx(void *workspace, void *stream) {
#ifdef VK_SIM_MODEL
  return -1;
#else
  auto data_dev = reinterpret_cast<uint8_t *>(workspace);
  auto ret = aclrtMemcpyAsync(data_dev, data_size_, data_, data_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  EXCEPTION_IF(ret != 0, "aclrtMemcpyAsync error");
  uint64_t args[] = {reinterpret_cast<uint64_t>(data_dev), *(reinterpret_cast<uint64_t *>(data_) + 1)};
  auto stub_func = System::Instance().StubFunc(target_);
  return System::Instance().rtKernelLaunch(stub_func, block_dim_, args, sizeof(args), stream);
#endif
}

uint64_t Code::ReserveCodeSpace(uint64_t workspace_size) {
  uint64_t *head = reinterpret_cast<uint64_t *>(data_);
  head[1] |= V_ENTRY_FLAG_EXTERN_CODE;
  auto offset = RoundUp<uint64_t>(data_size_, 512);
  for (auto op = bind_wss_; op != nullptr; op = op->bind_list_) {
    op->ws += offset;
  }
  return workspace_size + offset;
}
}  // namespace dvm
