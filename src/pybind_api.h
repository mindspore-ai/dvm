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

#ifndef _DVM_PYBIND_API_H_
#define _DVM_PYBIND_API_H_
#include <vector>
#include "pybind11/pybind11.h"
#include "code.h"
#include "dvm.h"

namespace dvm {
namespace py = pybind11;
class NDObjectPy {
 public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() {
    py::tuple out(obj_->shape_ref_->size);
    for (size_t i = 0; i < obj_->shape_ref_->size; ++i) {
      out[i] = py::cast(obj_->shape_ref_->data[i]);
    }
    return out;
  }
  std::string GetDType() const;
  NDObject *Get() const { return obj_; }

 private:
  NDObject *obj_;
};

struct NDSymInt {
  NDSymInt(int64_t data) : data_(data) {}
  int64_t data_;
};

struct NDSymFloat {
  NDSymFloat(float data) : data_(data) {}
  float data_;
};

class ShapeRefPy {
 public:
  ShapeRefPy() { shape_ref_ = new ShapeRef(shape_); }
  explicit ShapeRefPy(const std::vector<int64_t> &shape) : shape_(shape) { shape_ref_ = new ShapeRef(shape_); }
  ~ShapeRefPy() { delete shape_ref_; }
  void Update(const py::object &shape);
  ShapeRef *Get() const { return shape_ref_; }

  py::object GetShape() {
    py::tuple out(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
      out[i] = py::cast(shape_[i]);
    }
    return out;
  }

 private:
  ShapeRef *shape_ref_;
  std::vector<int64_t> shape_;
};

class KernelRunner;
class KernelPy {
 public:
  using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
  using NDSymIntPtr = std::shared_ptr<NDSymInt>;
  using NDSymFloatPtr = std::shared_ptr<NDSymFloat>;
  using ShapeRefPyPtr = std::shared_ptr<ShapeRefPy>;
  friend class ShapeRefPy;

  KernelPy(const std::string &ker_type, const std::string &run_type, int dev_id);
  ~KernelPy();

  py::object Load(const py::object &shape, const std::string &type);
  py::object ViewLoad(const py::object &shape, const py::object &stride, int64_t offset, const std::string &type);
  py::object SliceLoad(const py::object &shape, const py::object &start, const py::object &size,
                       const std::string &type);
  py::object StridedSliceLoad(const py::object &shape, const py::object &start, const py::object &end,
                              const py::object &step, const std::string &type);
  py::object MultiLoad(const py::object &shape, const std::string &type);
  py::object Store(const py::object &obj);
  py::object PadStore(const py::object &obj, int64_t pad_shape);
  void SetStoreInplace(const py::object &store);
  template <UnaryType op_type>
  py::object Unary(const py::object &input);
  template <BinaryType op_type>
  py::object Binary(const py::object &lhs, const py::object &rhs);
  py::object Broadcast(const py::object &input, const py::object &shape, const std::string &dtype);
  py::object Reshape(const py::object &input, const py::object &shape);
  py::object Cast(const py::object &input, const std::string &type);
  template <ReduceType op_type>
  py::object Reduce(const py::object &input, const py::object &dims, bool keepdims);
  py::object Select(const py::object &cond, const py::object &lhs, const py::object &rhs);
  py::object ElementAny(const py::object &input);
  py::object Copy(const py::object &input);
  py::object OneHot(const py::object &indices, int depth, int axis, const py::object &on_value,
                    const py::object &off_value, const std::string &dtype);
  py::object AllReduce(const std::string &type, const py::object &input);
  py::object AllGather(const py::object &input);
  py::object AllGatherV2(const py::object &input);
  py::object ReduceScatter(const py::object &input);
  py::object MatMul(const py::object &lhs, const py::object &rhs, bool trans_a, bool trans_b, const py::object &bias);
  py::object GroupedMatMul(const py::object &lhs, const py::object &rhs, bool trans_a, bool trans_b,
                           const py::object &bias, const py::object &group_list, int64_t group_type,
                           int64_t group_list_type);
  py::object ConvertToBF16(const py::object &input);
  py::object ConvertFromBF16(const py::object &input);
  py::object MakeFloatScalar();
  py::object MakeIntScalar();
  void ParallelNext();
  void SpecNext() { kernel_.SpecNext(); }
  void SequenceAdd(const std::string &ker_type);

  void Reset();

  void Input(const py::object &load, const py::object &array);
  py::object Output(const py::object &store);
  void ClearStoreMemory(const py::object &store);
  void Tile(int start, int end, int64_t num, int64_t factor);
  void CodeGen(const py::object &pass_names);
  void Run();

  void DryRun(int core_idx, bool cube_core);
  py::object DisAssemble();
  py::object DumpGraph();
  py::object Perf();
  py::object Msprof(const std::string &path, int64_t test_num);

  static void SetDeterm(bool enable);
  static void SetTuning(bool enable);
  static void SetCubeStoreType(int type) { g_system.SetCubeStoreType((CubeStoreType)type); }

  static void InitComm(int rank_id, int rank_size, const std::string &comm_type);
  static void Fork(int size, const std::string &comm_type);
  static void Join();
  static void Barrier();
  static int RankId();
  static int RankSize();

  static std::string Arch() {
    static const char *soc_names[] = {"AscendC220"};
    return soc_names[g_system.Arch()];
  }
  static int CoreNum() { return g_system.CoreNum(); }
  static std::string SocName() {
    static const char *soc_names[] = {"Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4", "Unknow"};
    return soc_names[g_system.SocName()];
  }

  struct LoadInfo {
    NDObject *op;
    void *dev{nullptr};
    std::vector<int64_t> shape;
  };

  struct StoreInfo {
    NDObject *op;
    void *host{nullptr};
    void *dev{nullptr};
    size_t size{0};
    bool clear_mem{false};
  };

 protected:
  ShapeRef *GetShapeRef(const py::object &shape);
  void PrepareIO();

  Kernel kernel_;
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<ShapeRef *> shape_;
  std::vector<LoadInfo> loads_;
  std::vector<StoreInfo> stores_;
  std::vector<std::vector<float>> f32s_;      // store f32 converted from bf16
  std::vector<std::vector<uint16_t>> bf16s_;  // store bf16 converted from f32
  std::vector<NDSymIntPtr> scalars_;

  void *workspace_{nullptr};
  KernelRunner *runner_;
};
}  // namespace dvm
#endif  // _DVM_PYBIND_API_H_
