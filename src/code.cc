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
#include "acl/acl_base.h"
#include "code.h"

namespace dvm {
namespace {
std::string GetSocName() {
  const char *soc_name = getenv("VK_SOC_NAME");
  if (soc_name == nullptr) {
    soc_name = aclrtGetSocName();
  }
  std::string res;
  if (soc_name == nullptr) {
    return res;
  }
  res = soc_name;
  return res;
}
}  // namespace

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


static std::unordered_map<std::string, vCompareType> cmp_insn_id = {
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

static void DumpLoad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vDMA op;
  vDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "load.u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " << reinterpret_cast<void *>(op.gm);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
  if (op.has_round) {
    vDMA::DecodeRound(dump_info.insn, op);
    oss << ", ";
    DumpVal("factor", op.factor, oss);
    oss << ", ";
    DumpVal("round", op.round, oss);
  }
}

static void DumpStore(const DumpInfo &dump_info, std::ostringstream &oss) {
  vDMA op;
  vDMA::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "store.u8.32x" << op.lenburst;
  oss << " " << reinterpret_cast<void *>(op.gm) << ", " << reinterpret_cast<void *>(op.xn);
  oss << " //";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("tail_lenburst", op.tail_lenburst, oss);
}

static void DumpLoad2(const DumpInfo &dump_info, std::ostringstream &oss) {
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
  if (op.has_round) {
    vLoad::DecodeRound(dump_info.insn, op);
    oss << ", ";
    DumpVal("factor", op.factor, oss);
    oss << ", ";
    DumpVal("round", op.round, oss);
  }
}

static void DumpStore2(const DumpInfo &dump_info, std::ostringstream &oss) {
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

static void DumpStoreAtomic(const DumpInfo &dump_info, std::ostringstream &oss) {
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
  oss << ", ";
  DumpVal("type", op.type, oss);
}

static void DumpStoreStatus(const DumpInfo &dump_info, std::ostringstream &oss) {
  vStoreStatus *op = reinterpret_cast<vStoreStatus *>(dump_info.insn);
  auto xn = dump_info.ext & V_X_MASK;
  oss << "store_status." << reinterpret_cast<void *>(op->to) << ", " << reinterpret_cast<void *>(xn);
}

static void DumpLoadDummy(const DumpInfo &dump_info, std::ostringstream &oss) { oss << "dummy_load.u8.0"; }

static void DumpUnary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vUnary op;
  vUnary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

static void DumpBinaryS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinaryS *op = reinterpret_cast<vBinaryS *>(dump_info.insn);
  auto rs = op->data >> 18;
  auto repeat = op->data & V_X_MASK;
  auto xn = dump_info.ext & V_X_MASK;
  auto xd = (dump_info.ext >> V_X_BITS) & V_X_MASK;
  oss << dump_info.simd_width << "x" << repeat;
  oss << " " << reinterpret_cast<void *>(xd) << ", " << reinterpret_cast<void *>(xn) << ", " << op->scalar << " //";
  DumpVal("rs", rs, oss);
}

static void DumpBinary(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
  oss << ", " << reinterpret_cast<void *>(op.xm);
}

static void DumpCompare(const DumpInfo &dump_info, std::ostringstream &oss) {
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

static void DumpBroadcastS(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastS *op = reinterpret_cast<vBroadcastS *>(dump_info.insn);
  uint64_t data = op->data;
  uint64_t rs = data >> 18;
  uint64_t repeat = data & V_X_MASK;
  oss << dump_info.simd_width << "x" << repeat;
  oss << " " << reinterpret_cast<void *>(dump_info.ext) << ", " << op->scalar << " //";
  DumpVal("rs", rs, oss);
}

static void DumpSelect(const DumpInfo &dump_info, std::ostringstream &oss) {
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

static void DumpBroadcastX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastX op;
  vBroadcastX::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "[" << dump_info.simd_width << "x" << op.repeat << "]x" << op.lead_num << "x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

static void DumpBroadcastY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vBroadcastY op;
  vBroadcastY::Decode(dump_info.insn, *dump_info.insn, op);
  oss << "32x" << op.dup_stride << "x[" << op.dup_num << "]x" << op.iter_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn);
}

static void DumpReduceX(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceX op;
  vReduceX::Decode(dump_info.insn, *dump_info.insn, op);
  vReduceX::DecodeBlock(dump_info.insn, op);
  oss << "[" << op.red_size<< "]x" << op.dup_size;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
  oss << ", ";
  DumpVal("dup_block", op.dup_block, oss);
  oss << ", ";
  DumpVal("dup_pad", op.dup_pad, oss);
}

static void DumpReduceY(const DumpInfo &dump_info, std::ostringstream &oss) {
  vReduceY op;
  vReduceY::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_size << "x[" << op.red_size << "]x" << op.dup_num;
  oss << " " << reinterpret_cast<void *>(op.xd) << ", " << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("red_tail", op.red_tail, oss);
}

static void DumpCopy(const DumpInfo &dump_info, std::ostringstream &oss) {
  vCopy *op = reinterpret_cast<vCopy *>(dump_info.insn);
  auto xn = dump_info.ext & V_X_MASK;
  auto xd = (dump_info.ext >> V_X_BITS) & V_X_MASK;
  auto lenburst = (op->config) >> 16 & 0xfffful;
  oss << "32x" << lenburst;
  oss << " " << reinterpret_cast<void *>(xd) << ", " << reinterpret_cast<void *>(xn);
}

static void DumpClearPad(const DumpInfo &dump_info, std::ostringstream &oss) {
  vClearPad op;
  vClearPad::Decode(dump_info.insn, *dump_info.insn, op);
  oss << op.iter_size << "x" << op.iter_num << " " << reinterpret_cast<void *>(op.xd) << " //";
  DumpVal("iter_stride", op.iter_stride, oss);
  oss << ", ";
  DumpVal("simd_width", op.simd_width, oss);
}

static void DumpElementAny(const DumpInfo &dump_info, std::ostringstream &oss) {
  vElementAny op;
  vElementAny::Decode(dump_info.insn, *dump_info.insn, op);
  oss << dump_info.simd_width << "x" << op.repeat << " " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn) << " //";
  DumpVal("rs", op.rs, oss);
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
};

std::unordered_map<uint64_t, std::tuple<DumpFunc *, std::string, std::string>> op_dump_info_table = {
  {V_COPY, {&DumpCopy, "Copy", "u8"}},
  {V_BROADCAST_X, {&DumpBroadcastX, "BroadcastX", "fp32"}},
  {V_BROADCAST_X_FP16, {&DumpBroadcastX, "BroadcastX", "fp16"}},
  {V_BROADCAST_Y, {&DumpBroadcastY, "BroadcastY", "u8"}},
  {V_BROADCAST_S, {&DumpBroadcastS, "BroadcastS", "fp32"}},
  {V_BROADCAST_S_FP16, {&DumpBroadcastS, "BroadcastS", "fp16"}},
  {V_SQRT, {&DumpUnary, "Sqrt", "fp32"}},
  {V_SQRT_FP16, {&DumpUnary, "Sqrt", "fp16"}},
  {V_RSQRT, {&DumpUnary, "Rsqrt", "fp32"}},
  {V_RSQRT_FP16, {&DumpUnary, "Rsqrt", "fp16"}},
  {V_ABS, {&DumpUnary, "Abs", "fp32"}},
  {V_ABS_FP16, {&DumpUnary, "Abs", "fp16"}},
  {V_LOG, {&DumpUnary, "Log", "fp32"}},
  {V_LOG_FP16, {&DumpUnary, "Log", "fp16"}},
  {V_EXP, {&DumpUnary, "Exp", "fp32"}},
  {V_EXP_FP16, {&DumpUnary, "Exp", "fp16"}},
  {V_REC, {&DumpUnary, "Reciprocal", "fp32"}},
  {V_REC_FP16, {&DumpUnary, "Reciprocal", "fp16"}},
  {V_NOT, {&DumpUnary, "LogicalNot", "fp32"}},
  {V_NOT_FP16, {&DumpUnary, "LogicalNot", "fp16"}},
  {V_NOT_INT8, {&DumpUnary, "LogicalNot", "u8"}},
  {V_ISFINITE, {&DumpUnary, "IsFinite", "fp32"}},
  {V_ISFINITE_FP16, {&DumpUnary, "IsFinite", "fp16"}},
  {V_CAST_FP16_TO_FP32, {&DumpUnary, "CastFP16", "fp32"}},
  {V_CAST_INT8_TO_FP16, {&DumpUnary, "CastS8", "fp16"}},
  {V_CAST_FP16_TO_INT8, {&DumpUnary, "CastFP16", "u8"}},
  {V_CAST_FP16_TO_INT32, {&DumpUnary, "CastFP16", "int32"}},
  {V_CAST_FP32_TO_INT32, {&DumpUnary, "CastFP32", "int32"}},
  {V_CAST_FP32_TO_FP16, {&DumpUnary, "CastFP32", "fp16"}},
  {V_CAST_INT32_TO_FP32, {&DumpUnary, "CastS32", "fp32"}},
  {V_CAST_INT32_TO_FP16, {&DumpUnary, "CastS32", "fp16"}},
  {V_ADDS, {&DumpBinaryS, "Adds", "fp32"}},
  {V_ADDS_FP16, {&DumpBinaryS, "Adds", "fp16"}},
  {V_MULS, {&DumpBinaryS, "Muls", "fp32"}},
  {V_MULS_FP16, {&DumpBinaryS, "Muls", "fp16"}},
  {V_MAXS, {&DumpBinaryS, "Maximums", "fp32"}},
  {V_MAXS_FP16, {&DumpBinaryS, "Maximums", "fp16"}},
  {V_MINS, {&DumpBinaryS, "Minimums", "fp32"}},
  {V_MINS_FP16, {&DumpBinaryS, "Maximums", "fp32"}},
  {V_ADD, {&DumpBinary, "Add", "fp32"}},
  {V_ADD_FP16, {&DumpBinary, "Add", "fp16"}},
  {V_SUB, {&DumpBinary, "Sub", "fp32"}},
  {V_SUB_FP16, {&DumpBinary, "Sub", "fp16"}},
  {V_MUL, {&DumpBinary, "Mul", "fp32"}},
  {V_MUL_FP16, {&DumpBinary, "Mul", "fp16"}},
  {V_DIV, {&DumpBinary, "Div", "fp32"}},
  {V_DIV_FP16, {&DumpBinary, "Div", "fp16"}},
  {V_MAX, {&DumpBinary, "Maximum", "fp32"}},
  {V_MAX_FP16, {&DumpBinary, "Maximum", "fp16"}},
  {V_MIN, {&DumpBinary, "Minimum", "fp32"}},
  {V_MIN_FP16, {&DumpBinary, "Minimum", "fp16"}},
  {V_POW, {&DumpBinary, "Pow", "fp32"}},
  {V_POW_FP16, {&DumpBinary, "Pow", "fp16"}},
  {V_CMP, {&DumpCompare, "Cmp", "fp32"}},
  {V_CMP_FP16, {&DumpCompare, "Cmp", "fp16"}},
  {V_AND, {&DumpBinary, "LogicalAnd", "fp32"}},
  {V_AND_FP16, {&DumpBinary, "LogicalAnd", "fp16"}},
  {V_AND_INT8, {&DumpBinary, "LogicalAnd", "int8"}},
  {V_OR, {&DumpBinary, "LogicalOr", "fp32"}},
  {V_OR_FP16, {&DumpBinary, "LogicalOr", "fp16"}},
  {V_OR_INT8, {&DumpBinary, "LogicalOr", "int8"}},
  {V_SEL, {&DumpSelect, "Select", "fp32"}},
  {V_SEL_FP16, {&DumpSelect, "Select", "fp16"}},
  {V_RSUM_X, {&DumpReduceX, "SumX", "fp32"}},
  {V_RSUM_X_FP16, {&DumpReduceX, "SumX", "fp16"}},
  {V_RSUM_Y, {&DumpReduceY, "SumY", "fp32"}},
  {V_RSUM_Y_FP16, {&DumpReduceY, "SumY", "fp16"}},
  {V_CLR_PAD, {&DumpClearPad, "ClrPad", "fp32"}},
  {V_ELEMENT_ANY, {&DumpElementAny, "ElementAny", "fp32"}},
  {V_ELEMENT_ANY_FP16, {&DumpElementAny, "ElementAny", "fp16"}},
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

void Code::DisAssemble(std::ostringstream &oss) {
  unsigned char *insn = data + HeadSize();
  unsigned char *insn_end = data + size;
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
  oss << "vkernel.main(tile_num=" << tile_num << ", block_dim=" << BlockDim() <<
      ", simd_width="<<simd_width << ", insn_num=" << insn_num << ") {" << std::endl;
  for (size_t i = 0; i < insn_dump.size(); ++i) {
    auto &dump = insn_dump[i];
    auto head = *(dump.first);
    bool is_simd = bool(head & (1ul << V_HEAD_IS_SIMD_OFFSET));
    oss << " " << i << ": " << dump.second << std::endl;
    oss << "   { simd(" << is_simd << ")";
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
  oss << "}";
}
}  // namespace dvm
