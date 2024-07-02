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

#ifndef _DVM_PYBIND_API_H_
#define _DVM_PYBIND_API_H_
#include <vector>
#include <unordered_map>
#include "pybind11/pybind11.h"
#include "code.h"
#include "dvm.h"

namespace dvm {
namespace py = pybind11;

class NDObjectPy {
 public:
  NDObjectPy(NDObject *obj): obj_(obj) {}
  py::object GetShape() {
    py::tuple out(obj_->shape_ref_->size);
    for (size_t i = 0; i < obj_->shape_ref_->size; ++i) {
      out[i] = py::cast(obj_->shape_ref_->data[i]);
    }
    return out;
  }
  std::string GetDType() const;
  NDObject* Get() const { return obj_; }
 private:
  NDObject *obj_;
};

class ShapeRefPy{
 public:
  ShapeRefPy() {
    shape_ref_ = new ShapeRef(shape_);
  }
  ShapeRefPy(const std::vector<int64_t> &shape): shape_(shape) {
    shape_ref_ = new ShapeRef(shape_);
  }
  ~ShapeRefPy() { delete shape_ref_; }
  void Update(const py::object &shape);
  ShapeRef* Get() const { return shape_ref_; }

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

class KernelPy {
 public:
  using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
  using ShapeRefPyPtr = std::shared_ptr<ShapeRefPy>;
  friend class ShapeRefPy;

  KernelPy(int dev_id, const std::string &ker_type);
  ~KernelPy();

  py::object Load(const py::object &shape, const std::string &type);
  py::object SliceLoad(const py::object &shape, const py::object &start, const py::object &size, const std::string &type);
  py::object StridedSliceLoad(const py::object &shape, const py::object &start, const py::object &end, const py::object &step, const std::string &type);
  py::object Store(const py::object &obj);
  py::object PadStore(const py::object &obj, const py::object &pad_shape);
  py::object Unary(const std::string &op_name, const py::object &input);
  py::object Binary(const std::string &op_name, const py::object &lhs, const py::object &rhs);
  py::object Broadcast(const py::object &input, const py::object &shape, const std::string &dtype, bool dummy_load);
  py::object Reshape(const py::object &input, const py::object &shape);
  py::object Cast(const py::object &input, const std::string &type);
  py::object Reduce(const std::string &type, const py::object &input, const py::object &dims, bool keepdims);
  py::object Select(const py::object &cond, const py::object &lhs, const py::object &rhs);
  py::object ElementAny(const py::object &input);
  py::object Copy(const py::object &input);
  py::object MatMul(const py::object &lhs, const py::object &rhs, bool trans_a, bool trans_b);
  py::object ConvertToBF16(const py::object &input);
  py::object ConvertFromBF16(const py::object &input);
  void ParallelNext();

  void StageSwitch(const std::string &ker_type);
  py::object StageLoad(const py::object &store);
  py::object StageStore(const py::object &input);
  py::object StagePadStore(const py::object &input, const py::object &pad_shape);

  void Input(const py::object &load, const py::object &array);
  py::object Output(const py::object &store);
  void ClearStoreMemory(const py::object &store);
  void Tile(int start, int end, int64_t num);
  void CodeGen(const py::object& pass_names);
  void Run();

  py::object DisAssemble();
  py::object DumpGraph();
  py::object Measure();
  py::object Perf();

  static void SetDeterm(bool enable) {
    SetDeterministic(enable);
  }

 protected:
  ShapeRef* GetShapeRef(const py::object &shape);
  void PrepareOutput();

  struct LoadInfo {
    std::vector<int64_t> shape;
    void *dev{nullptr};
  };

  struct StoreInfo {
    void *host{nullptr};
    void *dev{nullptr};
    size_t size{0};
    bool clear_mem{false};
  };

  Kernel kernel_;
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<ShapeRef*> shape_;
  std::unordered_map<NDObject*, LoadInfo> loads_;
  std::unordered_map<NDObject*, StoreInfo> stores_;
  std::vector<std::vector<float>> f32s_; // store f32 converted from bf16
  std::vector<std::vector<uint16_t>> bf16s_; // store bf16 converted from f32

  int dev_id_{0};
  void *workspace_{nullptr};
};
}
#endif // _DVM_PYBIND_API_H_
