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

#ifndef _DVM_PY_API_H_
#define _DVM_PY_API_H_
#include <memory>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "dvm.h"

namespace py = pybind11;

namespace dvm {
class RtKernelPy;
namespace pyapi {
class NDObjectPy {
 public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() const;
  std::string GetDType() const;
  NDObject *Get() const { return obj_; }

 private:
  NDObject *obj_;
};

class ShapeRefPy {
 public:
  ShapeRefPy() : shape_ref_(shape_) {}
  explicit ShapeRefPy(const std::vector<int64_t> &shape) : shape_(shape), shape_ref_(shape_) {}
  ~ShapeRefPy() = default;
  void Update(py::object shape);
  py::object GetShape() const;
  ShapeRef *Get() { return &shape_ref_; }

 private:
  std::vector<int64_t> shape_;
  ShapeRef shape_ref_;
};

class ScalarRefPy {
 public:
  ScalarRefPy() = default;
  void Update(py::object val);
  ScalarRef data_;
};

DType StringToTypeID(const std::string &type);

class KernelPy {
 public:
  KernelPy();
  virtual ~KernelPy();

  virtual py::object Load(py::object shape, const std::string &type) = 0;
  virtual py::object ViewLoad(py::object shape, py::object stride, int64_t offset, const std::string &type) = 0;
  virtual py::object Store(py::object obj) = 0;
  virtual ShapeRef *GetShapeRef(py::object shape) = 0;

  template <UnaryOpType op_type>
  py::object Unary(py::object input);

  template <BinaryOpType op_type>
  py::object Binary(py::object lhs, py::object rhs);

  template <ReduceOpType op_type>
  py::object Reduce(py::object input, py::object dims, bool keepdims);

  py::object Cast(py::object input, const std::string &type);
  py::object Select(py::object cond, py::object lhs, py::object rhs);
  py::object Full(py::object scalar, py::object shape, const std::string &dtype);
  py::object Reshape(py::object input, py::object shape);
  py::object Copy(py::object input);
  py::object Broadcast(py::object input, py::object shape);
  py::object ElementAny(py::object input);
  py::object MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias);
  py::object GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                           py::object group_list, int64_t group_type, int64_t group_list_type);
  py::object DisAssemble();
  py::object DumpGraph();
  void ParallelNext();
  void SpecNext();
  static void SetDeterm(bool enable);
  static void SetTuning(bool enable);
  py::object MakeIntArray() { return py::cast(std::make_shared<ShapeRefPy>()); }
  py::object MakeScalar() { return py::cast(std::make_shared<ScalarRefPy>()); }

 protected:
  Kernel kernel_;
};
using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
using ScalarRefPyPtr = std::shared_ptr<ScalarRefPy>;

void RegBaseApi(const py::module &m);
void RegKernelApi(const py::module &m);
}  // namespace pyapi
}  // namespace dvm
#endif  // _DVM_PY_API_H_
