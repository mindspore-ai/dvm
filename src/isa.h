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

#ifndef _DVM_ISA_H_
#define _DVM_ISA_H_

#include <stdint.h>

#ifdef _CCE_KERNEL_
#define INSN_ATTR __attribute__((device_immutable))
#define __aicore_inline__ static inline [aicore]
#define bcodeptr_t __ubuf__ uint64_t* __restrict__
#else
#define __gm__
#define INSN_ATTR
#define __aicore_inline__ static inline
#define bcodeptr_t uint64_t*
#endif

enum vPipe {
  V_PIPE_LOAD = 0,
  V_PIPE_STORE,
  V_PIPE_SIMD,
  V_PIPE_ALL,
};

enum vLoadInsnID {
  V_LOAD = 0,
  V_LOAD_2,
  V_LOAD_DUMMY,
  V_SLICE_LOAD,
  V_EXIT,
};

enum vStoreInsnID {
  V_STORE = 0,
  V_STORE_2,
  V_STORE_ATOMIC,
  V_STORE_STATUS,
};

enum vSimdInsnID {
  V_COPY = 0,
  V_BROADCAST_X,
  V_BROADCAST_Y,
  V_BROADCAST_S,
  V_SQRT,
  V_ABS,
  V_LOG,
  V_EXP,
  V_REC,
  V_ADDS,
  V_MULS,
  V_MAXS,
  V_MINS,
  V_ADD,
  V_SUB,
  V_MUL,
  V_DIV,
  V_MIN,
  V_MAX,
  V_CMP,
  V_CAST_FP32_TO_FP16,
  V_CAST_FP32_TO_INT32,
  V_CAST_FP32_TO_BF16,
  V_RSUM_X,
  V_RSUM_Y,
  V_SEL,
  V_POW,
  V_ISFINITE,
  V_CLR_PAD,
  V_ELEMENT_ANY,
  V_REMOVEPAD,
  // float16
  V_BROADCAST_X_FP16,
  V_BROADCAST_S_FP16,
  V_SQRT_FP16,
  V_ABS_FP16,
  V_LOG_FP16,
  V_EXP_FP16,
  V_REC_FP16,
  V_ADDS_FP16,
  V_MULS_FP16,
  V_MAXS_FP16,
  V_MINS_FP16,
  V_ADD_FP16,
  V_SUB_FP16,
  V_MUL_FP16,
  V_DIV_FP16,
  V_MIN_FP16,
  V_MAX_FP16,
  V_CAST_FP16_TO_INT8,
  V_CAST_FP16_TO_FP32,
  V_CAST_FP16_TO_INT32,
  V_CMP_FP16,
  V_SEL_FP16,
  V_POW_FP16,
  V_ISFINITE_FP16,
  V_ELEMENT_ANY_FP16,
  V_REMOVEPAD_U16,
  // int8
  V_CAST_INT8_TO_FP16,
  V_NOT_INT8,
  V_OR_INT8,
  V_AND_INT8,
  // int32
  V_BROADCAST_X_INT32,
  V_BROADCAST_S_INT32,
  V_ADDS_INT32,
  V_MULS_INT32,
  V_MAXS_INT32,
  V_MINS_INT32,
  V_ADD_INT32,
  V_SUB_INT32,
  V_MUL_INT32,
  V_MIN_INT32,
  V_MAX_INT32,
  V_SEL_INT32,
  V_CAST_INT32_TO_FP16,
  V_CAST_INT32_TO_FP32,
  // bfloat16
  V_CAST_BF16_TO_FP32,
  V_CAST_BF16_TO_INT32,
  V_NONE,
};

// head: x(36) << 28 | len(4) << 24 | wait_event(4) << 20 | set_event(4) << 16 | id(9) << 7 |
//       back_wait(1) << 6 | back_set(1) << 5 | wait_flag(1) << 4 | set_flag(1) << 3 |
//       bar_flag(1) << 2 | store_flag(1) << 1 | load_flag(1)
#define V_HEAD_SIMD_FLAG_OFFSET  0
#define V_HEAD_LOAD_FLAG_OFFSET  1
#define V_HEAD_BAR_FLAG_OFFSET   2
#define V_HEAD_SET_FLAG_OFFSET   3
#define V_HEAD_WAIT_FLAG_OFFSET  4
#define V_HEAD_BACK_SET_OFFSET   5
#define V_HEAD_BACK_WAIT_OFFSET  6
#define V_HEAD_ID_OFFSET         7
#define V_HEAD_SET_EVENT_OFFSET  16
#define V_HEAD_WAIT_EVENT_OFFSET 20 
#define V_HEAD_SIZE_OFFSET       24
#define V_HEAD_EXT_OFFSET        28

#define V_HEAD_EXT_MASK          0xffffffffful
#define V_HEAD_SIZE_MASK         0xful
#define V_HEAD_EVENT_MASK        0xful
#define V_HEAD_ID_MASK           0x1fful

__aicore_inline__ uint64_t vMakeHead(uint64_t id, uint64_t ext, uint64_t len, vPipe pipe) {
  return ext << V_HEAD_EXT_OFFSET | len << V_HEAD_SIZE_OFFSET | id << V_HEAD_ID_OFFSET |
  (pipe == V_PIPE_LOAD ? 1 : 0) << V_HEAD_LOAD_FLAG_OFFSET | (pipe == V_PIPE_SIMD ? 1 : 0) << V_HEAD_SIMD_FLAG_OFFSET;
}

// common mask
// ub address, loop ext, stride should not extent ub size limit
#define V_X_BITS                 18
#define V_X_MASK                 0x3fffful
#define V_RS_MASK                0xful   // repeat stride

struct vUnary {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  // pc[0]: xd
  // pc[1]: xn(18) << 32 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vUnary &op) {
    op.xd = head >> V_HEAD_EXT_OFFSET;
    uint64_t data = pc[1];
    op.repeat = data & 0xfffful;
    op.xn = data >> 32;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vUnary &op) {
    uint32_t size = 2;
    pc[0] = vMakeHead(id, op.xd, size, V_PIPE_SIMD);
    pc[1] = op.xn << 32 | op.repeat;
    return size;
  }
};

struct vRemovePad {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  uint64_t iter_num;
  uint64_t rs;
  // pc[0]: xd
  // pc[1]: xn(18) << 46 | rs(8) << 32 | iter_num(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vRemovePad &op) {
    op.xd = head >> V_HEAD_EXT_OFFSET;
    uint64_t data = pc[1];
    op.repeat = data & 0xfffful;
    op.iter_num = (data >> 16) & 0xfffful;
    op.xn = data >> 46;
    op.rs = (data >> 32) & 0xfful;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vRemovePad &op) {
    uint32_t size = 2;
    pc[0] = vMakeHead(id, op.xd, size, V_PIPE_SIMD);
    pc[1] = op.xn << 46 | op.rs << 32 | op.iter_num << 16 | op.repeat;
    return size;
  }
};

template<typename T>
struct vBinaryS {
  uint64_t head; // ext: xd(18) << 18 | xn(18)
  T scalar;
  uint32_t data; // rs(4) << 18 | repeat(18)
}INSN_ATTR;

struct vBinary {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t repeat;
  // pc[0]: xd(18) << 18 | xn(18)
  // pc[1]: xm(18) << 32 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinary &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
    uint64_t data = pc[1];
    op.xm = data >> 32;
    op.repeat = data & 0xfffful;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBinary &op) {
    uint32_t size = 2;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.xm << 32 | op.repeat;
    return size;
  }
};

enum vCompareType {
  V_CMP_EQ = 0,
  V_CMP_NE,
  V_CMP_GT,
  V_CMP_GE,
  V_CMP_LT,
  V_CMP_LE,
  V_CMP_ALL,
};

struct vCompare {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t type;
  uint64_t repeat;
  // pc[0]: xd << 18 | xn
  // pc[1]: xm(18) << 32 | op(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCompare &op) {
    uint64_t data1 = pc[1];
    op.repeat = data1 & 0xfffful;
    op.type = (data1 >> 16) & 0xfffful;
    op.xm = data1 >> 32;
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vCompare &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.xm << 32 | op.type << 16 | op.repeat;
    return size;
  }
};

template <typename T>
struct vBroadcastS {
  uint64_t head; // ext: xd(18)
  T scalar;
  uint32_t data;  // rs(4) << 16) | repeat(18)
}INSN_ATTR;

struct vSelect { //16B
  uint64_t head; // ext: xd(18) << 18 | xn(18)
  uint64_t data; // rs (4) << 60 | repeat(18) << 36 | xm(18) << 18 | cond(18)
}INSN_ATTR;

// [iter_num, lead_num, repeat*simd_width] = Broadcast([iter_num, lead_num + pad, 1])
struct vBroadcastX {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  uint64_t lead_num;
  uint64_t iter_num;
  // pc[0]:  xd(18) << 18 | xn(18)
  // pc[1]:  iter_num(16) << 48 | lead_num(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastX &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
    uint64_t data = pc[1];
    op.repeat = data & 0xfffful;
    op.lead_num = (data >> 16) & 0xfffful;
    op.iter_num = data >> 48;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastX &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.iter_num << 48 | op.lead_num << 16 | op.repeat;
    return size;
  }
};

// [iter_num, dup_num, dup_stride] = Broadcast([iter_num, 1, dup_stride])
struct vBroadcastY {
  uint64_t xd;
  uint64_t xn;
  uint64_t iter_num;
  uint64_t dup_num;
  uint64_t dup_stride;
  // pc[0]:  xd(18) << 18 | xn(18)
  // pc[1]:  iter_num(16) << 48 | dup_num(16) << 16 | dup_stride(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastY &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
    uint64_t data = pc[1];
    op.dup_stride = data & 0xfffful;
    op.dup_num = (data >> 16) & 0xfffful;
    op.iter_num = data >> 48;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastY &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.iter_num << 48 | op.dup_num << 16 | op.dup_stride;
    return size;
  }
};

// [dup_size/dup_block, dup_block/dup_pad, 1] = reduce([dup_size, red_size/red_tail])
struct vReduceX {
  uint64_t xd;
  uint64_t xn;
  uint64_t red_size;
  uint64_t red_tail;
  uint64_t dup_size;
  uint64_t dup_pad;
  uint64_t dup_block;
  // pc[0]: xd << 18 | xn
  // pc[1]: dup_size(16) << 32 | red_tail(16) << 16 | red_size(16)
  // pc[2]: dup_pad << 16 | dup_block
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceX &op) {
    uint64_t data1 = pc[1];
    op.red_size = data1 & 0xfffful;
    op.red_tail = (data1 >> 16) & 0xfffful;
    op.dup_size = data1 >> 32;
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
  }
  __aicore_inline__ void DecodeBlock(bcodeptr_t pc, vReduceX &op) {
    uint64_t data2 = pc[2];
    op.dup_block = data2 & 0xfffful;
    op.dup_pad = data2 >> 16;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vReduceX &op) {
    uint64_t size = 3;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.dup_size << 32 | op.red_tail << 16 | op.red_size;
    pc[2] = op.dup_pad << 16 | op.dup_block;
    return size;
  }
};

// [dup_num, 1, iter_size] = reduce([dup_num, red_size/red_tail, iter_size])
struct vReduceY {
  uint64_t xd;
  uint64_t xn;
  uint64_t iter_size;
  uint64_t red_size;
  uint64_t red_tail;
  uint64_t dup_num;
  // pc[0]: xd << 18 | xn
  // pc[1]: dup_num(16) << 48 | red_tail(16) << 16 | red_size(16) << 16 | iter_size(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceY &op) {
    uint64_t data = pc[1];
    op.iter_size = data & 0xfffful;
    op.red_size = (data >> 16) & 0xfffful;
    op.red_tail = (data >> 32) & 0xfffful;
    op.dup_num = data >> 48;
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.xd = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vReduceY &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xd << V_X_BITS | op.xn, size, V_PIPE_SIMD);
    pc[1] = op.dup_num << 48 | op.red_tail << 32 | op.red_size << 16 | op.iter_size;
    return size;
  }
}INSN_ATTR;

struct vCopy {
  uint64_t head;  // ext: xd(18) << 18 | xn(18)
  uint64_t config;
} INSN_ATTR;

// [iter_num, iter_size/iter_stride]
struct vClearPad {
  uint64_t xd;
  uint64_t iter_num;
  uint64_t iter_size;
  uint64_t iter_stride;
  uint64_t simd_width;
  // pc[0]: xd
  // pc[1]: simd_width(8) << 48 | iter_stride(16) << 32 | iter_size(16) << 16 | iter_num(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vClearPad &op) {
    op.xd = head >> V_HEAD_EXT_OFFSET;
    uint64_t data = pc[1];
    op.simd_width = data >> 48;
    op.iter_stride = (data >> 32) & 0xfffful;
    op.iter_size = (data >> 16) & 0xfffful;
    op.iter_num = data & 0xfffful;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vClearPad &op) {
    uint32_t size = 2;
    pc[0] = vMakeHead(id, op.xd, size, V_PIPE_SIMD);
    pc[1] = op.simd_width << 48 | op.iter_stride << 32 | op.iter_size << 16 | op.iter_num;
    return size;
  }
};

struct vDMA {
  enum { ROUND_OFFSET = 3 };
  enum { RELOC_OFFSET = 1 };
  __gm__ uint8_t *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t lenburst;
  uint64_t tail_lenburst;
  uint64_t round_rank;
  // pc[0]: round_rank(4) << 20 | xn(18)
  // pc[1]: gm
  // pc[2]: tile_stride(32) << 32 | tail_lenburst(16) << 16 | lenburst(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vDMA &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.round_rank = head >> (V_HEAD_EXT_OFFSET + 20);
    op.gm = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    uint64_t data = pc[2];
    op.lenburst = data & 0xfffful;
    op.tail_lenburst = (data >> 16) & 0xfffful;
    op.tile_stride = data >> 32;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, vPipe pipe, const vDMA &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vDMA::ROUND_OFFSET + round_size;
    pc[0] = vMakeHead(id, op.round_rank << 20 | op.xn, size, pipe);
    pc[1] = reinterpret_cast<uint64_t>(op.gm);
    pc[2] = op.tile_stride << 32 | op.tail_lenburst << 16 | op.lenburst;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vDMA::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

struct vSliceLoad {
  enum { RELOC_OFFSET = 1 };
  __gm__ uint8_t *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t offset;
  uint64_t src_n;
  uint64_t src_m;
  uint64_t slice_n;
  uint64_t slice_m;
  uint64_t pad_size;
  uint64_t slice_k;
  uint64_t type_size;

  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSliceLoad &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.gm = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    op.tile_stride = head >> (V_HEAD_EXT_OFFSET + 18);
    uint64_t data = pc[2];
    op.src_n = data & 0xfffful;
    op.src_m = (data >> 16) & 0xfffful;
    op.slice_n = (data >> 32) & 0xfffful;
    op.slice_m = (data >> 48) & 0xfffful;
    data = pc[3];
    op.type_size = data & 0xfffful;
    op.pad_size = (data >> 32) & 0xfffful;
    op.slice_k = (data >> 48) & 0xfffful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vSliceLoad &op) {
    uint64_t size = 4;
    pc[0] = vMakeHead(id, op.tile_stride << 18 | op.xn, size, V_PIPE_LOAD);
    pc[1] = reinterpret_cast<uint64_t>(op.gm);
    pc[2] = op.slice_m << 48 | op.slice_n << 32 | op.src_m << 16 | op.src_n;
    pc[3] = op.slice_k << 48 | op.pad_size << 32 | op.type_size;
    return size;
  }
};

struct vLoad {
  enum { ROUND_OFFSET = 3 };
  enum { RELOC_OFFSET = 1 };
  __gm__ void *from;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t body_iter;
  uint64_t tail_iter;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t round_rank;
  // pc[0]: tile_stride(18) << 18 | xn(18)
  // pc[1]: from
  // pc[2]: // round_rank(4) << 60 | pad_size(8) << 50 | iter_size(18) << 32 | tail_iter(16) << 16 | body_iter(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vLoad &op) {
    op.tile_stride = head >> (V_HEAD_EXT_OFFSET + 18);
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.from = reinterpret_cast<__gm__ void *>(pc[1]);
    uint64_t data = pc[2];
    op.round_rank = data >> 60;
    op.pad_size = (data >> 50) & 0xfful;
    op.iter_size = (data >> 32) & 0x3fffful;
    op.tail_iter = (data >> 16) & 0xfffful;
    op.body_iter = data & 0xfffful;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vLoad &op, const uint64_t *rounds) {
    uint64_t round_size = (op.round_rank + 1) / 2;
    uint64_t size = vLoad::ROUND_OFFSET + round_size;
    pc[0] = vMakeHead(id, op.tile_stride << 18 | op.xn, size, V_PIPE_LOAD);
    pc[1] = reinterpret_cast<uint64_t>(op.from);
    pc[2] = op.round_rank << 60 | op.pad_size << 50 | op.iter_size << 32 | op.tail_iter << 16 | op.body_iter;
    for (uint64_t i = 0; i < round_size; ++i) {
      pc[vDMA::ROUND_OFFSET + i] = rounds[i];
    }
    return size;
  }
};

// complete lead_dim: [iter_num/iter_tail, iter_size+pad_size]
// tiling lead_dim:   [iter_size/iter_tail+pad_size]
struct vStore {
  enum { RELOC_OFFSET = 2 };
  uint64_t head;   // tile_stride(18) << 18 | xn(18)
  uint64_t config; // lead_tiling(1) << 62 | pad_size(8) << 54 | iter_size(18) << 36 | iter_tail(18) << 18 | iter_num(18)
  __gm__ void *to;
}INSN_ATTR;

// [iter_num/iter_tail, iter_size+pad_size]
struct vStoreAtomic {
  enum { RELOC_OFFSET = 3 };
  uint64_t to;
  uint64_t xn;
  uint64_t iter_size;
  uint64_t pad_size;
  uint64_t iter_num;
  uint64_t iter_tail;
  uint64_t tile_stride;
  uint64_t round;
  uint64_t factor;
  // pc[0]: tile_stride(18) << 18 | xn(18)
  // pc[1]: pad_size(8) << 50 | iter_size(18) << 32 | iter_tail(16) << 16 | iter_num(16)
  // pc[2]: round(32) << 32 | factor(32)
  // pc[3]: to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreAtomic &op) {
    op.tile_stride = head >> (V_HEAD_EXT_OFFSET + V_X_BITS);
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.pad_size = (data >> 50) & 0xfful;
    op.iter_size = (data >> 32) & 0x3fffful;
    op.iter_tail = (data >> 16) & 0xfffful;
    op.iter_num = data & 0xfffful;
    data = pc[2];
    op.round = data >> 32;
    op.factor = data & 0xfffffffful;
    op.to = pc[3];
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vStoreAtomic &op) {
    uint32_t size = 4;
    uint64_t ext = op.tile_stride << V_X_BITS | op.xn;
    pc[0] = vMakeHead(id, ext, size, V_PIPE_STORE);
    pc[1] = op.pad_size << 50 | op.iter_size << 32 | op.iter_tail << 16 | op.iter_num;
    pc[2] = op.round << 32 | op.factor;
    pc[3] = op.to;
    return size;
  }
};

struct vElementAny {
  uint64_t xd;
  uint64_t xn;
  uint64_t rs;
  uint64_t iter_size;
  uint64_t repeat;
  uint64_t tail_size;

  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vElementAny &op) {
    op.xd = (head >> (V_HEAD_EXT_OFFSET + V_X_BITS)) & V_X_MASK;
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.rs = data >> 60;
    op.tail_size = (data >> 32) & 0xfffful;
    op.iter_size = (data >> 16) & 0xfffful;
    op.repeat = data & 0xfffful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vElementAny &op) {
    uint64_t size = 2;
    uint64_t ext = op.xd << V_X_BITS | op.xn;
    pc[0] = vMakeHead(id, ext, size, V_PIPE_SIMD);
    pc[1] = op.rs << 60 | op.tail_size  << 32 | op.iter_size << 16 | op.repeat;
    return size;
  }
}INSN_ATTR;

struct vStoreStatus {
  enum { RELOC_OFFSET = 1 };
  uint64_t head;   // tile_stride(18) << 18 | xn(18)
  __gm__ void *to;
}INSN_ATTR;

#define V_INSN_SIZE_MAX   (4 * sizeof(uint64_t))

#endif // _DVM_ISA_H_
