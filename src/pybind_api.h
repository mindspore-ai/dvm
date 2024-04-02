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
  NDObject* Get() const { return obj_; }
 private:
  NDObject *obj_;
};

class KernelPy {
 public:
  using NDOpPyPtr = std::shared_ptr<NDObjectPy>;

  KernelPy(int dev_id, const std::string &ker_type);
  ~KernelPy();

  py::object Load(const py::object &array);
  py::object SliceLoad(const py::object &array, const py::object &start, const py::object &size);
  py::object StridedSliceLoad(const py::object &array, const py::object &start, const py::object &end, const py::object &step);
  py::object Store(const py::object &obj);
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
  void ParallelNext();

  void Tile(int start, int end, int64_t num);
  void Optimize();
  py::object CodeGen(const std::string &path);
  py::object DisAssemble();
  py::object DumpGraph();
  py::object Perf();
  py::object Measure();
  void Run();

  void ResetPasses(const py::object& pass_names);

  struct StoreInfo {
    NDObject* obj;
    void *host;
    void *dev;
    std::vector<int64_t> shape;
  };

 protected:
  void *ToDev(void *host, size_t size);

  CodeBase* GetCode();
  bool codegen_{false};
  Kernel kernel_;
  std::vector<StoreInfo> stores_;
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<ShapeRef*> shape_;
  std::vector<void*> dev_mem_;
  int dev_id_{0};
};
}
#endif // _DVM_PYBIND_API_H_
