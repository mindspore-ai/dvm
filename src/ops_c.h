/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef _DVM_OPS_C_H_
#define _DVM_OPS_C_H_

#include "ops.h"

namespace dvm {
// split input to `multi_size` parts, everytime load a piece from all parts
class NDMultiLoad : public NDLoad {
 public:
  NDMultiLoad(uint8_t *src, IntArrayRef *shape_ref, DataType type_id, const Communicator *comm)
      : NDLoad(src, shape_ref, type_id), comm_(comm) {
    obj_id_ = ObjectType::kMultiLoad;
  }
  void Normalize(std::vector<NDObject *> &run_ops) override;
  // void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  // uint32_t multi_size_{1}; // only support up to 1023(2^10)
  uint64_t gap_{0};
  uint32_t xbuf_size_{0};
  const Communicator *comm_;
};

class CommIdWrap : public CodeWrap {
 public:
  int LaunchWrap(void *workspace, void *stream) override;
  std::vector<uint32_t *> ids_;  // used to ensure softsync work, not affected by last kernel
 private:
  static std::atomic<uint32_t> unique_id_;  // each kernel has a unique id
};

class CubeOp;
class CommOp : public NDObject {
 public:
  CommOp(NDObject *input, const Communicator *comm, ObjectType obj_id)
      : NDObject(input, nullptr, input->type_id_, obj_id), comm_(comm) {
    shape_ref_ = input->shape_ref_;
    max_type_ = type_id_;
    nd_.data = &ndd_;
  }
  ~CommOp() override = default;
  // Extra space needed to store expanded instructions
  uint64_t CodeReserve() { return code_reserve_; }
  int XbufReserve() { return xbuf_reserve_; }
  void SetXbufSize(uint32_t size) { xbuf_size_ = size; }
  void SetCubeOp(CubeOp *op) { cube_op_ = op; }

 public:
  std::vector<uint64_t> xbufs_;
  std::vector<uint64_t> forward_events_;
  std::vector<uint64_t> backward_events_;
  bool mix_{false};
  DataType max_type_;
  const Communicator *comm_;
  CommIdWrap id_wrap_;

 protected:
  int xbuf_reserve_{0};  // static
  uint64_t code_reserve_{0};
  CubeOp *cube_op_{nullptr};
  uint32_t xbuf_size_{0};
  bool store_lhs_{true};
  NDSpaceData ndd_;
};

// Not Support (rank_size, 1)
class ReduceScatterOp : public CommOp {
 public:
  ReduceScatterOp(NDObject *input, const Communicator *comm);
  ~ReduceScatterOp() override;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Dump(bool verbose, std::ostringstream &oss) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  uint64_t MultiLoadEmit(VectorKernel &k);

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);
  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);
  bool multi_load_;

 private:
  int tail_dim_{-1};
  int tail_size_{0};
  ShapeWithRef shape_;
  vSimdInsnID add_id_;
  DimArray round_tile_;
};

class AllReduceOpBase : public CommOp {
 public:
  AllReduceOpBase(int op_type, NDObject *input, const Communicator *comm);
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};
  bool use_twoshot_{false};
  vSimdInsnID insn_id_;
};

// Design: AllReduce is used before codegen, then codegen will generate PeerLoad and PeerStore
template <bool is_bf16>
class AllReduceOp : public AllReduceOpBase {
 public:
  AllReduceOp(int op_type, NDObject *input, const Communicator *comm) : AllReduceOpBase(op_type, input, comm) {};
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  uint64_t MatmulEmit(VectorKernel &k);  // used when lhs_ is Matmul
};

class AllGatherOp : public CommOp {
 public:
  AllGatherOp(NDObject *input, const Communicator *comm);
  ~AllGatherOp() override = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  void Tile(const TileParam &tp) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

  static void TileCollect(NDObject *op, TileInfo &info);
  static void FoldProp(NDObject *op, PropRange &range);
  static void ShapeProp(NDObject *op, int64_t &sym_dim_next);

  DimArray round_tile_;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};

 private:
  ShapeWithRef shape_;
};

class AllGatherV2Op : public CommOp {
 public:
  AllGatherV2Op(NDObject *input, const Communicator *comm);
  ~AllGatherV2Op() override = default;
  void Normalize(std::vector<NDObject *> &run_ops) override;
  uint64_t Emit(VectorKernel &k) override;
  NDObject *Clone(CloneHelper &h) override;
  void Dump(bool verbose, std::ostringstream &oss) override;

 protected:
  int tail_dim_{-1};
  int tail_size_{0};

 private:
  ShapeWithRef shape_;
};
}  // namespace dvm
#endif  // _DVM_OPS_C_H_