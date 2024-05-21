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
#ifdef AICORE_ARCH_C100
#define __bcode__ __ubuf__
#else
#define __bcode__ __gm__
#endif
#define bcodeptr_t __bcode__ uint64_t* __restrict__
#else
#define __gm__
#define INSN_ATTR
#define __bcode__
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
  V_SLOAD,
  V_LOAD_NONE,
};

enum vStoreInsnID {
  V_STORE = 0,
  V_STORE_2,
  V_STORE_ATOMIC,
  V_STORE_STATUS,
  V_SSTORE,
  V_STORE_NONE,
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
  V_ADD,
  V_SUB,
  V_MUL,
  V_DIV,
  V_MIN,
  V_MAX,
  V_CMP,
  V_CAST_FP32_TO_FP16,
  V_CAST_FP32_TO_INT32,
  V_RSUM_X,
  V_RSUM_Y,
  V_SEL,
  V_POW,
  V_CLR_PAD,
  V_ELEMENT_ANY,
  V_BROADCAST_X_FP16,
  V_BROADCAST_S_FP16,
  V_SQRT_FP16,
  V_ABS_FP16,
  V_LOG_FP16,
  V_EXP_FP16,
  V_REC_FP16,
  V_ADDS_FP16,
  V_MULS_FP16,
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
  V_CAST_INT8_TO_FP16,
  V_NOT_INT8,
  V_OR_INT8,
  V_AND_INT8,
  V_BROADCAST_X_INT32,
  V_BROADCAST_S_INT32,
  V_ADD_INT32,
  V_SUB_INT32,
  V_MUL_INT32,
  V_MIN_INT32,
  V_MAX_INT32,
  V_SEL_INT32,
  V_CAST_INT32_TO_FP16,
  V_CAST_INT32_TO_FP32,
  V_RESHAPE_B32,
  V_RESHAPE_B16,
  // ASCEND 910B
  V_ISFINITE,
  V_MAXS,
  V_MINS,
  V_MAXS_FP16,
  V_MINS_FP16,
  V_ADDS_INT32,
  V_MULS_INT32,
  V_MAXS_INT32,
  V_MINS_INT32,
  V_REMOVEPAD_U16,
  V_REMOVEPAD,
  V_CAST_FP32_TO_BF16,
  V_CAST_BF16_TO_FP32,
  V_CAST_BF16_TO_INT32,
  V_NONE,
};

// head(simd):
//  ID(16) << 48 | ext(26) << 22 | b_wait_event(3) << 19 | b_set_event(3) << 16 | wait_event(3) << 13 | set_event(3) << 10 | len(3) << 7 |
//  back_wait(1) << 6 | back_set(1) << 5 | wait_flag(1) << 4 | set_flag(1) << 3 | bar_flag(1) << 2 | LOAD_FLAG(1) << 1 | SIMD_FLAG(1)
// head(load/store):
//  ID(16) << 48 | ext(34) << 14 | wait_event(3) << 11 | set_event(3) << 8 | len(4) << 4 |
//  wait_flag(1) << 3 | set_flag(1) << 2 | LOAD_FLAG(1) << 1 | SIMD_FLAG(1)

// common area
#define V_HEAD_SIMD_FLAG_OFFSET    0
#define V_HEAD_LOAD_FLAG_OFFSET    1
#define V_HEAD_ID_OFFSET           48
#define V_HEAD_ID_MASK             0xfffful
#define V_HEAD_EVENT_MASK          0x7ul

// simd
#define V_HEAD_BAR_FLAG_OFFSET     2
#define V_HEAD_SET_FLAG_OFFSET     3
#define V_HEAD_WAIT_FLAG_OFFSET    4
#define V_HEAD_BACK_SET_OFFSET     5
#define V_HEAD_BACK_WAIT_OFFSET    6
#define V_HEAD_SIZE_OFFSET         7
#define V_HEAD_SET_EVENT_OFFSET    10
#define V_HEAD_WAIT_EVENT_OFFSET   13
#define V_HEAD_B_SET_EVENT_OFFSET  16
#define V_HEAD_B_WAIT_EVENT_OFFSET 19
#define V_HEAD_EXT_OFFSET          22
#define V_HEAD_EXT_MASK            0x3fffffful
#define V_HEAD_SIZE_MASK           0x7ul

// load/store
#define V_M_HEAD_SET_FLAG_OFFSET   2
#define V_M_HEAD_WAIT_FLAG_OFFSET  3
#define V_M_HEAD_SIZE_OFFSET       4
#define V_M_HEAD_SET_EVENT_OFFSET  8
#define V_M_HEAD_WAIT_EVENT_OFFSET 11
#define V_M_HEAD_EXT_OFFSET        14
#define V_M_HEAD_EXT_MASK          0x3fffffffful
#define V_M_HEAD_SIZE_MASK         0xful

// common mask
// ub address, loop ext, stride should not extent ub size limit
#define V_X_BITS                 18
#define V_C_X_BITS               13
#define V_X_MASK                 0x3fffful
#define V_RS_MASK                0xful   // repeat stride

#define vCompactX(x)    ((x) >> 5)
#define vDeCompactX(x)  ((x) << 5)

#ifndef _CCE_KERNEL_
extern uint64_t g_simd_func_offset[];
extern uint64_t g_load_func_offset[];
extern uint64_t g_store_func_offset[];
__aicore_inline__ uint64_t vMakeHead(uint64_t id, uint64_t ext, uint64_t len, vPipe pipe) {
  if (pipe == V_PIPE_SIMD) {
    return ext << V_HEAD_EXT_OFFSET | len << V_HEAD_SIZE_OFFSET | g_simd_func_offset[id] << V_HEAD_ID_OFFSET | 1 << V_HEAD_SIMD_FLAG_OFFSET;
  } else if (pipe == V_PIPE_LOAD) {
    return ext << V_M_HEAD_EXT_OFFSET | len << V_M_HEAD_SIZE_OFFSET | g_load_func_offset[id] << V_HEAD_ID_OFFSET | 1 << V_HEAD_LOAD_FLAG_OFFSET;
  } else {
    return ext << V_M_HEAD_EXT_OFFSET | len << V_M_HEAD_SIZE_OFFSET | g_store_func_offset[id] << V_HEAD_ID_OFFSET;
  }
}
#else
__aicore_inline__ uint64_t vMakeHead(uint64_t id, uint64_t ext, uint64_t len, vPipe pipe) {
  return 0;
}
#endif

__aicore_inline__ uint64_t vGetBitRange(uint64_t x, uint64_t offset, uint64_t len) {
  return x << (64 - len - offset) >> (64 - len);
}

__aicore_inline__ void vClrBitRange(uint64_t &x, uint64_t offset, uint64_t len) {
  x &= (~(((1ul << len) - 1) << offset));
}

template <typename T>
__aicore_inline__ uint32_t EncodeScalar(T scalar) {
  union Scalar {
    T val;
    uint32_t encode;
  } data;
  data.val = scalar;
  return data.encode;
}

template <typename T>
__aicore_inline__ T DecodeScalar(__bcode__ uint32_t *scalr_offset) {
  return *(__bcode__ T *)(scalr_offset);
}

struct vUnary {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  // pc[0]: xd
  // pc[1]: xn(18) << 32 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vUnary &op) {
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
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
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
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

template <typename T>
struct vBinaryS {
  uint64_t xn;
  uint64_t xd;
  uint64_t repeat;
  T scalar;
  // pc[0]: xn
  // pc[1]: scalar(32) << 32 | c_xd(13) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinaryS &op) {
    uint32_t data = *((__bcode__ uint32_t *)pc + 2);
    op.repeat = data & 0xffffu;
    op.xd = vDeCompactX(data >> 16);
    op.scalar = DecodeScalar<T>((__bcode__ uint32_t *)pc + 3);
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBinaryS &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_SIMD);
    pc[1] = (uint64_t)EncodeScalar(op.scalar) << 32 | vCompactX(op.xd) << 16 | op.repeat;
    return size;
  }
};

struct vBinary {
  uint64_t xd;
  uint64_t xn;
  uint64_t xm;
  uint64_t repeat;
  // pc[0]: xn(18)
  // pc[1]: repeat(48) | xd(18) << 18 | xm(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBinary &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.xm = data & V_X_MASK;
    op.xd = (data >> 18) & V_X_MASK;
    op.repeat = data >> 48;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBinary &op) {
    uint32_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_SIMD);
    pc[1] = op.repeat << 48 | op.xd << 18 | op.xm;
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
  // pc[0]: xn
  // pc[1]: xd(18) << 46 | xm(18) << 28 | op(8) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCompare &op) {
    uint64_t data1 = pc[1];
    op.repeat = data1 & 0xfffful;
    op.type = (data1 >> 16) & 0xfful;
    op.xm = (data1 >> 28) & V_X_MASK;
    op.xd = data1 >> 46;
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vCompare &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_SIMD);
    pc[1] = op.xd << 46 | op.xm << 28 | op.type << 16 | op.repeat;
    return size;
  }
};

template <typename T>
struct vBroadcastS {
  uint64_t xd;
  uint64_t repeat;
  T scalar;
  // pc[0]: xd
  // pc[1]: val << 32 | repeat
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastS &op) {
    op.repeat = *((__bcode__ uint32_t *)pc + 2);
    op.scalar = DecodeScalar<T>((__bcode__ uint32_t *)pc + 3);
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastS &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xd, size, V_PIPE_SIMD);
    pc[1] = (uint64_t)EncodeScalar(op.scalar) << 32 | op.repeat;
    return size;
  }
};

struct vSelect { //24B
  uint64_t xn;
  uint64_t xd;
  uint64_t repeat;
  uint64_t xm; 
  uint64_t cond;
  // pc[0]: xn
  // pc[1]: repeat(15) << 49 | c_xd(13) << 36 | xm(18) << 18 | cond(18)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSelect &op) {
    uint64_t data1 = pc[1];
    op.repeat = data1 >> 49;
    op.xm = (data1 >> 18) & V_X_MASK;
    op.cond = data1 & V_X_MASK;
    op.xd = vDeCompactX(vGetBitRange(data1, 36, 13));
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vSelect &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_SIMD);
    pc[1] = op.repeat << 49 | vCompactX(op.xd) << 36 | op.xm << 18 | op.cond;
    return size;
  }
};

// [iter_num, lead_num, repeat*simd_width] = Broadcast([iter_num, lead_num + pad, 1])
struct vBroadcastX {
  uint64_t xd;
  uint64_t xn;
  uint64_t repeat;
  uint64_t lead_num;
  uint64_t iter_num;
  uint64_t lead_pad;
  // pc[0]:  lead_pad(8) << 18 | xn(18)
  // pc[1]:  c_xd(16) << 48 | iter_num(16) << 32 | lead_num(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastX &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    op.lead_pad = (head >> (V_HEAD_EXT_OFFSET + 18)) & 0xfful;
    uint64_t data = pc[1];
    op.repeat = data & 0xfffful;
    op.lead_num = (data >> 16) & 0xfffful;
    op.iter_num = (data >> 32) & 0xfffful;
    op.xd = vDeCompactX(data >> 48);
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastX &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.lead_pad << 18 | op.xn, size, V_PIPE_SIMD);
    pc[1] = vCompactX(op.xd) << 48 | op.iter_num << 32 | op.lead_num << 16 | op.repeat;
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
  // pc[0]:  xn(18)
  // pc[1]:  c_xd(16) << 48 | iter_num(16) << 32 | dup_num(16) << 16 | dup_stride(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vBroadcastY &op) {
    op.xn = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
    uint64_t data = pc[1];
    op.dup_stride = data & 0xfffful;
    op.dup_num = (data >> 16) & 0xfffful;
    op.iter_num = (data >> 32) & 0xfffful;
    op.xd = vDeCompactX(data >> 48);
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vBroadcastY &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_SIMD);
    pc[1] = vCompactX(op.xd) << 48 | op.iter_num << 32 | op.dup_num << 16 | op.dup_stride;
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
  // pc[0]: xd
  // pc[1]: dup_size(16) << 32 | red_tail(16) << 16 | red_size(16)
  // pc[2]: xn << 32 | dup_pad << 16 | dup_block
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceX &op) {
    uint64_t data1 = pc[1];
    op.red_size = data1 & 0xfffful;
    op.red_tail = (data1 >> 16) & 0xfffful;
    op.dup_size = data1 >> 32;
    op.xn = pc[2] >> 32;
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ uint64_t GetXd(bcodeptr_t pc, uint64_t head) {
    return (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
  }
  __aicore_inline__ void DecodeBlock(bcodeptr_t pc, vReduceX &op) {
    uint64_t data2 = pc[2];
    op.dup_block = data2 & 0xfffful;
    op.dup_pad = (data2 >> 16) & 0xfffful;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vReduceX &op) {
    uint64_t size = 3;
    pc[0] = vMakeHead(id, op.xd, size, V_PIPE_SIMD);
    pc[1] = op.dup_size << 32 | op.red_tail << 16 | op.red_size;
    pc[2] = op.xn << 32 | op.dup_pad << 16 | op.dup_block;
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
  // pc[0]: c_xd(13) << 13 | c_xn(13)
  // pc[1]: dup_num(16) << 48 | red_tail(16) << 16 | red_size(16) << 16 | iter_size(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReduceY &op) {
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    op.xn = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.iter_size = data & 0xfffful;
    op.red_size = (data >> 16) & 0xfffful;
    op.red_tail = (data >> 32) & 0xfffful;
    op.dup_num = data >> 48;
  }
  __aicore_inline__ uint64_t GetXd(bcodeptr_t pc, uint64_t head) {
    return vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vReduceY &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, vCompactX(op.xd) << V_C_X_BITS | vCompactX(op.xn), size, V_PIPE_SIMD);
    pc[1] = op.dup_num << 48 | op.red_tail << 32 | op.red_size << 16 | op.iter_size;
    pc[2] = op.xn;
    return size;
  }
};

struct vCopy {
  uint64_t xd;
  uint64_t xn;
  uint64_t config;
  // pc[0]: c_xd(13) << 13 | c_xn(13)
  // pc[1]: simd_width(8) << 48 | iter_stride(16) << 32 | iter_size(16) << 16 | iter_num(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vCopy &op) {
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    op.xn = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET, V_C_X_BITS));
    op.config = pc[1];
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vCopy &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(V_COPY, vCompactX(op.xd) << V_C_X_BITS | vCompactX(op.xn), size, V_PIPE_SIMD);
    pc[1] = op.config;
    return size;
  }
};

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
    op.xd = (head >> V_HEAD_EXT_OFFSET) & V_X_MASK;
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

struct vElementAny {
  uint64_t xd;
  uint64_t xn;
  uint64_t rs;
  uint64_t iter_size;
  uint64_t repeat;
  uint64_t tail_size;
  // pc[0]: c_xd(13) << 13 | c_xn(13)
  // pc[1]: rs(4) << 60 | tail_size(16) << 32 | iter_size(16) << 16 | repeat(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vElementAny &op) {
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    op.xn = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.rs = data >> 60;
    op.tail_size = (data >> 32) & 0xfffful;
    op.iter_size = (data >> 16) & 0xfffful;
    op.repeat = data & 0xfffful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vElementAny &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, vCompactX(op.xd) << V_C_X_BITS | vCompactX(op.xn), size, V_PIPE_SIMD);
    pc[1] = op.rs << 60 | op.tail_size  << 32 | op.iter_size << 16 | op.repeat;
    return size;
  }
};

// [dup_size, xd_lead+xd_pad] = Reshape([*, xn_lead+xn_pad])
struct vReshape {
  uint64_t xd;
  uint64_t xn;
  uint64_t xn_lead;
  uint64_t xd_lead;
  uint64_t xn_pad;
  uint64_t xd_pad;
  uint64_t dup_size;
  // pc[0]: c_xd(13) << 13 | c_xn(13)
  // pc[1]: xd_pad(8) << 56 | xn_pad(8) << 48 | dup_size(16) << 32 | xd_lead(16) << 16 | xn_lead(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vReshape &op) {
    op.xd = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET + V_C_X_BITS, V_C_X_BITS));
    op.xn = vDeCompactX(vGetBitRange(head, V_HEAD_EXT_OFFSET, V_C_X_BITS));
    uint64_t data = pc[1];
    op.xn_lead = data & 0xfffful;
    op.xd_lead = (data >> 16) & 0xfffful;
    op.dup_size = (data >> 32) & 0xfffful;
    op.xn_pad = (data >> 48) & 0xfful;
    op.xd_pad = data >> 56;
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vReshape &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, vCompactX(op.xd) << V_C_X_BITS | vCompactX(op.xn), size, V_PIPE_SIMD);
    pc[1] = op.xd_pad << 56 | op.xn_pad << 48 | op.dup_size << 32 | op.xd_lead << 16 | op.xn_lead;
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
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.round_rank = (head >> (V_M_HEAD_EXT_OFFSET + 20)) & 0xful;
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

struct vSLoad {
  enum { RELOC_OFFSET = 1 };
  __gm__ uint8_t *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t src_n;
  uint64_t slice_n;
  uint64_t slice_m;
  uint64_t pad_size;
  uint64_t type_size;
  // pc[0]: xn(18)
  // pc[1]: src
  // pc[2]: slice_n(16) << 48 | slice_m(16) << 32 | src_n(16) << 16 | pad_size(16);
  // pc[2]: tile_stride(32) << 32 | type_size(8)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSLoad &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.gm = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    uint64_t data = pc[2];
    op.src_n = (data >> 16) & 0xfffful;
    op.slice_m = (data >> 32) & 0xfffful;
    op.slice_n = (data >> 48) & 0xfffful;
    op.pad_size = data & 0xfffful;
    data = pc[3];
    op.tile_stride = (data >> 32) & 0xfffffffful;
    op.type_size = data & 0xfful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vSLoad &op) {
    uint64_t size = 4;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_LOAD);
    pc[1] = reinterpret_cast<uint64_t>(op.gm);
    pc[2] = op.slice_n << 48 | op.slice_m << 32 | op.src_n << 16 | op.pad_size;
    pc[3] = op.tile_stride << 32 | op.type_size;
    return size;
  }
};

struct vSStore {
  enum { RELOC_OFFSET = 1 };
  __gm__ uint8_t *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t src_n;
  uint64_t slice_n;
  uint64_t slice_m;
  uint64_t pad_size;
  uint64_t type_size;
  // pc[0]: xn(18)
  // pc[1]: dst
  // pc[2]: slice_n(16) << 48 | slice_m(16) << 32 | src_n(16) << 16 | pad_size(16);
  // pc[2]: tile_stride(32) << 32 | type_size(8)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSStore &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.gm = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    uint64_t data = pc[2];
    op.src_n = (data >> 16) & 0xfffful;
    op.slice_m = (data >> 32) & 0xfffful;
    op.slice_n = (data >> 48) & 0xfffful;
    op.pad_size = data & 0xfffful;
    data = pc[3];
    op.tile_stride = (data >> 32) & 0xfffffffful;
    op.type_size = data & 0xfful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vSStore &op) {
    uint64_t size = 4;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_STORE);
    pc[1] = reinterpret_cast<uint64_t>(op.gm);
    pc[2] = op.slice_n << 48 | op.slice_m << 32 | op.src_n << 16 | op.pad_size;
    pc[3] = op.tile_stride << 32 | op.type_size;
    return size;
  }
};

struct vSliceLoad {
  enum { RELOC_OFFSET = 1 };
  __gm__ uint8_t *gm;
  uint64_t xn;
  uint64_t tile_stride;
  uint64_t src_m;
  uint64_t src_n;
  uint64_t slice_m;
  uint64_t slice_n;
  uint64_t pad_size;
  uint64_t slice_k;
  uint64_t type_size;

  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vSliceLoad &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.gm = reinterpret_cast<__gm__ uint8_t *>(pc[1]);
    uint64_t data = pc[2];
    op.src_m = data & 0xfffful;
    op.src_n = (data >> 16) & 0xfffful;
    op.slice_m = (data >> 32) & 0xfffful;
    op.slice_n = (data >> 48) & 0xfffful;
    data = pc[3];
    op.type_size = data & 0xfful;
    op.tile_stride = (data >> 8) & 0xfffffful;
    op.pad_size = (data >> 32) & 0xfffful;
    op.slice_k = (data >> 48) & 0xfffful;
  }

  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vSliceLoad &op) {
    uint64_t size = 4;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_LOAD);
    pc[1] = reinterpret_cast<uint64_t>(op.gm);
    pc[2] = op.slice_n << 48 | op.slice_m << 32 | op.src_n << 16 | op.src_m;
    pc[3] = op.slice_k << 48 | op.pad_size << 32 | op.tile_stride << 8 | op.type_size;
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
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: from
  // pc[2]: // round_rank(4) << 60 | pad_size(8) << 50 | iter_size(18) << 32 | tail_iter(16) << 16 | body_iter(16)
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vLoad &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + 13, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, 13));
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
    pc[0] = vMakeHead(id, op.tile_stride << 13 | vCompactX(op.xn), size, V_PIPE_LOAD);
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
  uint64_t head;   // tile_stride(18) << 13 | c_xn(13)
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
  // pc[0]: tile_stride(18) << 13 | c_xn(13)
  // pc[1]: pad_size(8) << 50 | iter_size(18) << 32 | iter_tail(16) << 16 | iter_num(16)
  // pc[2]: round(32) << 32 | factor(32)
  // pc[3]: to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreAtomic &op) {
    op.tile_stride = vGetBitRange(head, V_M_HEAD_EXT_OFFSET + V_C_X_BITS, 18);
    op.xn = vDeCompactX(vGetBitRange(head, V_M_HEAD_EXT_OFFSET, V_C_X_BITS));
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
    uint64_t ext = op.tile_stride << 13 | vCompactX(op.xn);
    pc[0] = vMakeHead(id, ext, size, V_PIPE_STORE);
    pc[1] = op.pad_size << 50 | op.iter_size << 32 | op.iter_tail << 16 | op.iter_num;
    pc[2] = op.round << 32 | op.factor;
    pc[3] = op.to;
    return size;
  }
};

struct vStoreStatus {
  enum { RELOC_OFFSET = 1 };
  uint64_t xn;
  uint64_t to;
  // pc[0]: // xn(18)
  // pc[1]: // to
  __aicore_inline__ void Decode(bcodeptr_t pc, uint64_t head, vStoreStatus &op) {
    op.xn = (head >> V_M_HEAD_EXT_OFFSET) & V_X_MASK;
    op.to = pc[1];
  }
  __aicore_inline__ uint32_t Encode(bcodeptr_t pc, uint64_t id, const vStoreStatus &op) {
    uint64_t size = 2;
    pc[0] = vMakeHead(id, op.xn, size, V_PIPE_STORE);
    pc[1] = op.to;
    return size;
  }
};

#define V_INSN_SIZE_MAX   (4 * sizeof(uint64_t))

#define V_CUBE_FLAG_TRANS_A  1
#define V_CUBE_FLAG_TRANS_B  2
#define V_CUBE_FLAG_GROUP_SET 4
#define V_CUBE_FLAG_PRE_WAIT  8
#define V_CUBE_FLAG_POST_SET  16
#define V_CUBE_FLAG_POST_BAR  32

struct vCubeOp {
  enum {FP16, BF16};

  uint32_t flags;
  uint32_t m, n, k;
  // shape_a: [batch_a0, batch_a1, m, k], shape_b: [batch_b0, batch_b1, k, n]
  uint32_t batch_a0, batch_a1, batch_b0, batch_b1;
  uint32_t m0, n0, k0;
  // swizzle_dir << 16 | swizzle_cnt
  uint32_t swizzle;
  uint32_t dtype;
  uint32_t post_set;
  uint32_t pre_wait;
  uint32_t group_set;
  uint64_t gm_a;
  uint64_t gm_b;
  uint64_t gm_c;
  // for aiv
  uint64_t subtilenum; // subblockid1 << 32 | subblockid0

  __aicore_inline__ uint64_t GetCubeOffset(__gm__ vCubeOp *__restrict__ op, uint32_t block_tile) {
    int64_t start_m, start_n;
    uint64_t swizzle_dir = op->swizzle >> 16;
    uint64_t swizzle_cnt = op->swizzle & 0xffff;
    uint64_t m_loop = (op->m + op->m0 - 1) / op->m0;
    uint64_t n_loop = (op->n + op->n0 - 1) / op->n0;
    TileMap(block_tile, m_loop, n_loop, swizzle_dir, swizzle_cnt, start_m, start_n);
    start_m *= op->m0;
    start_n *= op->n0;
    uint64_t batch_offset = block_tile / (m_loop * n_loop) * op->n * op->m;
    return start_m * op->n + start_n + batch_offset;
  }

  __aicore_inline__ void TileMap(uint32_t tile, uint64_t m_loop, uint64_t n_loop, uint64_t swizzle_dir,
                                        uint64_t swizzle_cnt, int64_t &midx, int64_t &nidx) {
    tile = tile % (m_loop * n_loop);
    if (swizzle_dir == 0) {
      uint32_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint32_t tile_block_idx = tile / (swizzle_cnt * n_loop);
      uint32_t in_tile_block_idx = tile - tile_block_idx * (swizzle_cnt * n_loop);

      uint32_t n_row = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_row = m_loop - swizzle_cnt * tile_block_idx;
      }
      midx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
      nidx = in_tile_block_idx / n_row;
      if (tile_block_idx % 2 != 0) {
        nidx = n_loop - nidx - 1;
      }
    } else {
      uint32_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
      uint32_t tile_block_idx = tile / (swizzle_cnt * m_loop);
      uint32_t in_tile_block_idx = tile - tile_block_idx * (swizzle_cnt * m_loop);

      uint32_t n_col = swizzle_cnt;
      if (tile_block_idx == tile_block_loop - 1) {
        n_col = n_loop - swizzle_cnt * tile_block_idx;
      }
      midx = in_tile_block_idx / n_col;
      nidx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
      if (tile_block_idx % 2 != 0) {
        midx = m_loop - midx - 1;
      }
    }
  }
};

// [entry]
// tilenum(20) << 44 | simd_width(8) << 36 | code_size_8B(12) << 24 | post_flag(12) << 12 | pre_flag(4) << 8 |
// post_sync(1) << 6 | next_stage(1) << 5 | group(1) << 4 | mix(1) << 3 | parallel(1) << 2 | post_set(1) << 1 | pre_wait(1)
#define V_ENTRY_FLAG_PRE_WAIT            1
#define V_ENTRY_FLAG_POST_SET            2
#define V_ENTRY_FLAG_PARALLEL            4
#define V_ENTRY_FLAG_MIX                 8
#define V_ENTRY_FLAG_GROUP               16
#define V_ENTRY_FLAG_NEXT_STAGE          32
#define V_ENTRY_FLAG_POST_BAR            64

#define V_ENTRY_PRE_WAIT_OFFSET          8
#define V_ENTRY_POST_SET_OFFSET          12
#define V_ENTRY_CODE_SIZE_OFFSET         24
#define V_ENTRY_SIMD_WIDTH_OFFSET        36
#define V_ENTRY_TILE_NUM_OFFSET          44

#define V_ENTRY_PRE_WAIT_BITS            4
#define V_ENTRY_POST_SET_BITS            12
#define V_ENTRY_CODE_SIZE_BITS           12
#define V_ENTRY_SIMD_WIDTH_BITS          8
#define V_ENTRY_TILE_NUM_BITS            20

#endif // _DVM_ISA_H_
