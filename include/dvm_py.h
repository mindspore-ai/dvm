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

namespace dvm {
namespace py = pybind11;
extern const char *DTYPE_NAMES[];
void DvmException(const char *error_str);
class NDObjectPy {
 public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() const {
    const size_t size = obj_->shape_ref_->size;
    py::tuple out(size);
    for (size_t i = 0; i < size; ++i) {
      out[i] = py::cast(obj_->shape_ref_->data[i]);
    }
    return out;
  }
  std::string GetDType() const { return DTYPE_NAMES[obj_->type_id_]; }
  NDObject *Get() const { return obj_; }

 private:
  NDObject *obj_;
};

class IntArrayRefPy {
 public:
  IntArrayRefPy() : shape_ref_(shape_) {}
  explicit IntArrayRefPy(const std::vector<int64_t> &shape) : shape_(shape), shape_ref_(shape_) {}
  ~IntArrayRefPy() = default;
  void Update(py::object shape) {
    shape_ = py::cast<std::vector<int64_t>>(shape);
    shape_ref_ = shape_;
  }
  py::object GetShape() const {
    const size_t size = shape_.size();
    py::tuple out(size);
    for (size_t i = 0; i < size; ++i) {
      out[i] = py::cast(shape_[i]);
    }
    return out;
  }
  IntArrayRef *Get() { return &shape_ref_; }

 private:
  std::vector<int64_t> shape_;
  IntArrayRef shape_ref_;
};

class ScalarRefPy {
 public:
  ScalarRefPy() = default;
  void Update(py::object val) {
    if (py::isinstance<py::int_>(val)) {
      data_ = val.cast<int>();
    } else if (py::isinstance<py::float_>(val)) {
      data_ = val.cast<float>();
    } else {
      DvmException("unsupport type");
    }
  }
  ScalarRef data_;
};

using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
using ScalarRefPyPtr = std::shared_ptr<ScalarRefPy>;

class KernelPy {
 public:
  KernelPy() = default;
  virtual ~KernelPy() {}

  virtual py::object Load(py::object shape, DataType type) = 0;
  virtual py::object ViewLoad(py::object shape, py::object stride, int64_t offset, DataType type) = 0;
  virtual py::object Store(py::object obj) = 0;
  virtual IntArrayRef *GetShapeRef(py::object shape) = 0;

  template <UnaryOpType op_type>
  py::object Unary(py::object input) {
    return ObjToPy(kernel_.Unary<op_type>(PyToObj(input)));
  }

  template <BinaryOpType op_type>
  py::object Binary(py::object lhs, py::object rhs) {
    NDObject *op;
    if (py::isinstance<py::int_>(lhs)) {
      op = kernel_.Binary<op_type>(lhs.cast<int>(), PyToObj(rhs));
    } else if (py::isinstance<py::float_>(lhs)) {
      op = kernel_.Binary<op_type>(lhs.cast<float>(), PyToObj(rhs));
    } else if (py::isinstance<py::int_>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), rhs.cast<int>());
    } else if (py::isinstance<py::float_>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), rhs.cast<float>());
    } else if (py::isinstance<ScalarRefPy>(lhs)) {
      op = kernel_.Binary<op_type>(PyToScalar(lhs), PyToObj(rhs));
    } else if (py::isinstance<ScalarRefPy>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), PyToScalar(rhs));
    } else {
      op = kernel_.Binary<op_type>(PyToObj(lhs), PyToObj(rhs));
    }
    return ObjToPy(op);
  }

  template <ReduceOpType op_type>
  py::object Reduce(py::object input, py::object dims, bool keepdims) {
    return ObjToPy(kernel_.Reduce<op_type>(PyToObj(input), GetShapeRef(dims), keepdims));
  }

  py::object Cast(py::object input, DataType type) {
    return ObjToPy(kernel_.Cast(PyToObj(input), type));
  }
  py::object Select(py::object cond, py::object lhs, py::object rhs) {
    return ObjToPy(kernel_.Select(PyToObj(cond), PyToObj(lhs), PyToObj(rhs)));
  }
  py::object Full(py::object scalar, py::object shape, DataType dtype) {
    auto shape_ref = GetShapeRef(shape);
    NDObject *op = nullptr;
    if (py::isinstance<py::int_>(scalar)) {
      op = kernel_.Broadcast(scalar.cast<int>(), shape_ref, dtype);
    } else if (py::isinstance<py::float_>(scalar)) {
      op = kernel_.Broadcast(scalar.cast<float>(), shape_ref, dtype);
    } else if (py::isinstance<ScalarRefPy>(scalar)) {
      op = kernel_.Broadcast(PyToScalar(scalar), shape_ref, dtype);
    } else {
      DvmException("Unsupported scalar type for full: expected int, float, NDSymInt, or NDSymFloat.");
    }
    return ObjToPy(op);
  }
  py::object Reshape(py::object input, py::object shape) {
    return ObjToPy(kernel_.Reshape(PyToObj(input), GetShapeRef(shape)));
  }
  py::object Copy(py::object input) { return ObjToPy(kernel_.Copy(PyToObj(input))); }
  py::object Broadcast(py::object input, py::object shape) {
    return ObjToPy(kernel_.Broadcast(PyToObj(input), GetShapeRef(shape)));
  }
  py::object ElementAny(py::object input) { return ObjToPy(kernel_.ElemAny(PyToObj(input))); }
  py::object MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias) {
    auto op = kernel_.MatMul(PyToObj(lhs), PyToObj(rhs), trans_a, trans_b, bias.is_none() ? nullptr : PyToObj(bias));
    return ObjToPy(op);
  }
  py::object GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                           py::object group_list, int64_t group_type, int64_t group_list_type) {
    NDObject *bias_obj = bias.is_none() ? nullptr : PyToObj(bias);
    NDObject *group_list_obj = group_list.is_none() ? nullptr : PyToObj(group_list);
    auto op = kernel_.GroupedMatMul(PyToObj(lhs), PyToObj(rhs), trans_a, trans_b, bias_obj, group_list_obj,
                                    GroupType(group_type), GroupListType(group_list_type));
    return ObjToPy(op);
  }
  py::object DisAssemble() { return py::cast(kernel_.Das()); }
  py::object DumpGraph() { return py::cast(kernel_.Dump()); }
  void ParallelNext() { kernel_.ParallelNext(); }
  void SpecNext() { kernel_.SpecNext(); }
  py::object MakeIntArray() { return py::cast(std::make_shared<IntArrayRefPy>()); }
  py::object MakeScalar() { return py::cast(std::make_shared<ScalarRefPy>()); }

 protected:
  NDObject *PyToObj(py::object obj) { return obj.cast<NDOpPyPtr>()->Get(); }
  py::object ObjToPy(NDObject *obj) { return py::cast(std::make_shared<NDObjectPy>(obj)); }
  ScalarRef *PyToScalar(py::object scalar) { return &(scalar.cast<ScalarRefPyPtr>()->data_); }
  Kernel kernel_;
};

static inline void RegDvmPy(const py::module &m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject")
    .def("shape", &NDObjectPy::GetShape, "get shape")
    .def("dtype", &NDObjectPy::GetDType, "get dtype");

  auto dtype = py::enum_<DataType>(m, "DataType")
    .value("bool", kBool)
    .value("float16", kFloat16)
    .value("bfloat16", kBFloat16)
    .value("float32", kFloat32)
    .value("int32", kInt32)
    .value("int64", kInt64);
  m.attr("bool") = dtype.attr("bool");
  m.attr("float16") = dtype.attr("float16");
  m.attr("bfloat16") = dtype.attr("bfloat16");
  m.attr("float32") = dtype.attr("float32");
  m.attr("int32") = dtype.attr("int32");
  m.attr("int64") = dtype.attr("int64");

  (void)py::class_<IntArrayRefPy, std::shared_ptr<IntArrayRefPy>>(m, "IntArrayRef")
    .def(py::init<>())
    .def(py::init<const std::vector<int64_t> &>())
    .def("shape", &IntArrayRefPy::GetShape, "get shape")
    .def("update", &IntArrayRefPy::Update, "update shape");

  (void)py::class_<ScalarRefPy, std::shared_ptr<ScalarRefPy>>(m, "ScalarRef")
    .def("update", &ScalarRefPy::Update, "update value");

  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "KernelBase")
    .def("load", &KernelPy::Load, "load array")
    .def("view_load", &KernelPy::ViewLoad, "load array")
    .def("store", &KernelPy::Store, "store array")
    .def("scalar", &KernelPy::MakeScalar, "create scalar")
    .def("int_array", &KernelPy::MakeIntArray, "create int array")
    .def("sqrt", &KernelPy::Unary<UnaryOpType::kSqrt>, "emit sqrt")
    .def("abs", &KernelPy::Unary<UnaryOpType::kAbs>, "emit abs")
    .def("log", &KernelPy::Unary<UnaryOpType::kLog>, "emit log")
    .def("exp", &KernelPy::Unary<UnaryOpType::kExp>, "emit exp")
    .def("reciprocal", &KernelPy::Unary<UnaryOpType::kReciprocal>, "emit reciprocal")
    .def("isfinite", &KernelPy::Unary<UnaryOpType::kIsFinite>, "emit isfinite")
    .def("logical_not", &KernelPy::Unary<UnaryOpType::kLogicalNot>, "emit logical_not")
    .def("round", &KernelPy::Unary<UnaryOpType::kRound>, "emit round")
    .def("floor", &KernelPy::Unary<UnaryOpType::kFloor>, "emit floor")
    .def("ceil", &KernelPy::Unary<UnaryOpType::kCeil>, "emit ceil")
    .def("trunc", &KernelPy::Unary<UnaryOpType::kTrunc>, "emit trunc")
    .def("cast", &KernelPy::Cast, "emit cast op")
    .def("element_any", &KernelPy::ElementAny, "emit element_any op")
    .def("equal", &KernelPy::Binary<BinaryOpType::kEqual>, "emit equal")
    .def("not_equal", &KernelPy::Binary<BinaryOpType::kNotEqual>, "emit equal")
    .def("greater", &KernelPy::Binary<BinaryOpType::kGreater>, "emit greater")
    .def("greater_equal", &KernelPy::Binary<BinaryOpType::kGreaterEqual>, "emit greater_equal")
    .def("less", &KernelPy::Binary<BinaryOpType::kLess>, "emit less")
    .def("less_equal", &KernelPy::Binary<BinaryOpType::kLessEqual>, "emit less_equal")
    .def("add", &KernelPy::Binary<BinaryOpType::kAdd>, "emit add")
    .def("sub", &KernelPy::Binary<BinaryOpType::kSub>, "emit sub")
    .def("mul", &KernelPy::Binary<BinaryOpType::kMul>, "emit mul")
    .def("div", &KernelPy::Binary<BinaryOpType::kDiv>, "emit div")
    .def("pow", &KernelPy::Binary<BinaryOpType::kPow>, "emit pow")
    .def("maximum", &KernelPy::Binary<BinaryOpType::kMaximum>, "emit maximum")
    .def("minimum", &KernelPy::Binary<BinaryOpType::kMinimum>, "emit minimum")
    .def("logical_and", &KernelPy::Binary<BinaryOpType::kLogicalAnd>, "emit logical_add")
    .def("logical_or", &KernelPy::Binary<BinaryOpType::kLogicalOr>, "emit logical_or")
    .def("select", &KernelPy::Select, "emit select op")
    .def("broadcast", &KernelPy::Broadcast, "emit broadcast op")
    .def("full", &KernelPy::Full, "emit broadcast op", py::arg("input"), py::arg("shape"), py::arg("dtype"))
    .def("reshape", &KernelPy::Reshape, "emit reshape op")
    .def("sum", &KernelPy::Reduce<ReduceOpType::kSum>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit sum")
    .def("max", &KernelPy::Reduce<ReduceOpType::kMax>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit max")
    .def("min", &KernelPy::Reduce<ReduceOpType::kMin>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit min")
    .def("copy", &KernelPy::Copy, "emit copy op")
    .def("matmul", &KernelPy::MatMul, "emit matmul op", py::arg("lhs"), py::arg("rhs"), py::arg("trans_a"),
         py::arg("trans_b"), py::arg("bias") = py::none())
    .def("grouped_matmul", &KernelPy::GroupedMatMul, "emit grouped_matmul op", py::arg("lhs"), py::arg("rhs"),
         py::arg("trans_a"), py::arg("trans_b"), py::arg("bias"), py::arg("group_list"), py::arg("group_type"),
         py::arg("group_list_type") = 0)
    .def("das", &KernelPy::DisAssemble, "disassemble code")
    .def("dump", &KernelPy::DumpGraph, "dump graph")
    .def("p_next", &KernelPy::ParallelNext, "parallel next")
    .def("spec_next", &KernelPy::SpecNext, "spec next");
}
}  // namespace dvm
#endif  // _DVM_PY_API_H_
