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

#include <unordered_map>
#include "pybind11/stl.h"
#include "py_api.h"
#include "ops.h"

namespace dvm {
namespace pyapi {

DType StringToTypeID(const std::string &type) {
  static const std::unordered_map<std::string, DType> map = {
    {"bool", kBool},       {"float16", kFloat16}, {"bfloat16", kBFloat16},
    {"float32", kFloat32}, {"int32", kInt32},     {"int64", kInt64},
  };
  auto it = map.find(type);
  if (it == map.end()) {
    std::string msg =
      "Unsupported dtype: " + type + ". Supported dtypes: bool, float16, bfloat16, float32, int32, int64.";
    DvmException(msg.c_str());
  }
  return it->second;
}

py::object NDObjectPy::GetShape() const {
  const size_t size = obj_->shape_ref_->size;
  py::tuple out(size);
  for (size_t i = 0; i < size; ++i) {
    out[i] = py::cast(obj_->shape_ref_->data[i]);
  }
  return out;
}

std::string NDObjectPy::GetDType() const { return DTYPE_NAMES[obj_->type_id_]; }

void ShapeRefPy::Update(py::object shape) {
  shape_ = py::cast<std::vector<int64_t>>(shape);
  shape_ref_ = shape_;
}

py::object ShapeRefPy::GetShape() const {
  const size_t size = shape_.size();
  py::tuple out(size);
  for (size_t i = 0; i < size; ++i) {
    out[i] = py::cast(shape_[i]);
  }
  return out;
}

KernelPy::KernelPy() = default;

KernelPy::~KernelPy() {}

template <UnaryOpType op_type>
py::object KernelPy::Unary(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Unary<op_type>(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

template <ReduceOpType op_type>
py::object KernelPy::Reduce(py::object input, py::object dims, bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto dims_ref = GetShapeRef(dims);
  auto op = kernel_.Reduce<op_type>(in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

template <BinaryOpType op_type>
py::object KernelPy::Binary(py::object lhs, py::object rhs) {
  NDObject *op;
  if (py::isinstance<py::int_>(lhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<int>(), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::float_>(lhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<float>(), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<py::int_>(rhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<int>());
  } else if (py::isinstance<py::float_>(rhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<float>());
  } else if (py::isinstance<NDSymInt>(lhs)) {
    op = kernel_.Binary<op_type>(&(lhs.cast<NDSymIntPtr>()->data_), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<NDSymInt>(rhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<NDOpPyPtr>()->Get(), &(rhs.cast<NDSymIntPtr>()->data_));
  } else if (py::isinstance<NDSymFloat>(lhs)) {
    op = kernel_.Binary<op_type>(&(lhs.cast<NDSymFloatPtr>()->data_), rhs.cast<NDOpPyPtr>()->Get());
  } else if (py::isinstance<NDSymFloat>(rhs)) {
    op = kernel_.Binary<op_type>(lhs.cast<NDOpPyPtr>()->Get(), &(rhs.cast<NDSymFloatPtr>()->data_));
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_.Binary<op_type>(input1, input2);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(py::object input, const std::string &type) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Cast(in_obj, StringToTypeID(type));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(py::object cond, py::object lhs, py::object rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Select(input0, input1, input2);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::BroadcastScalar(py::object scalar, py::object shape, const std::string &dtype) {
  auto shape_ref = GetShapeRef(shape);
  auto type_id = StringToTypeID(dtype);
  NDObject *op = nullptr;
  if (py::isinstance<py::int_>(scalar)) {
    op = kernel_.Broadcast(scalar.cast<int>(), shape_ref, type_id);
  } else if (py::isinstance<py::float_>(scalar)) {
    op = kernel_.Broadcast(scalar.cast<float>(), shape_ref, type_id);
  } else if (py::isinstance<NDSymInt>(scalar)) {
    op = kernel_.Broadcast(&(scalar.cast<NDSymIntPtr>()->data_), shape_ref, type_id);
  } else if (py::isinstance<NDSymFloat>(scalar)) {
    op = kernel_.Broadcast(&(scalar.cast<NDSymFloatPtr>()->data_), shape_ref, type_id);
  } else {
    DvmException("Unsupported scalar type for full: expected int, float, NDSymInt, or NDSymFloat.");
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(py::object input, py::object shape) {
  auto shape_ref = GetShapeRef(shape);
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Broadcast(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(py::object input, py::object shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto shape_ref = GetShapeRef(shape);
  auto op = kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ElementAny(py::object input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  auto op =
    kernel_.MatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get());
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                                   py::object group_list, int64_t group_type, int64_t group_list_type) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  NDObject *bias_obj = bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get();
  NDObject *group_list_obj = group_list.is_none() ? nullptr : group_list.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.GroupedMatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias_obj, group_list_obj,
                                  dvm::GroupType(group_type), dvm::GroupListType(group_list_type));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::DisAssemble() { return py::cast(kernel_.Das()); }

py::object KernelPy::DumpGraph() { return py::cast(kernel_.Dump()); }

void KernelPy::ParallelNext() { kernel_.ParallelNext(); }

void KernelPy::SpecNext() { kernel_.SpecNext(); }

void KernelPy::SetDeterm(bool enable) {
  auto &conf = Config::Instance();
  if (enable) {
    conf.SetDeterm();
  } else {
    conf.UnsetDeterm();
  }
}

void KernelPy::SetTuning(bool enable) {
  auto &conf = Config::Instance();
  if (enable) {
    conf.SetOnlineTuner().SetLazyTuner();
  } else {
    conf.UnsetOnlineTuner().UnsetLazyTuner();
  }
}

template py::object KernelPy::Unary<UnaryOpType::kSqrt>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kAbs>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kLog>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kExp>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kReciprocal>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kIsFinite>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kLogicalNot>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kRound>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kFloor>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kCeil>(py::object);
template py::object KernelPy::Unary<UnaryOpType::kTrunc>(py::object);

template py::object KernelPy::Binary<BinaryOpType::kEqual>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kNotEqual>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kGreater>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kGreaterEqual>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kLess>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kLessEqual>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kAdd>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kSub>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kMul>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kDiv>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kPow>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kMaximum>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kMinimum>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kLogicalAnd>(py::object, py::object);
template py::object KernelPy::Binary<BinaryOpType::kLogicalOr>(py::object, py::object);

template py::object KernelPy::Reduce<ReduceOpType::kSum>(py::object, py::object, bool);
template py::object KernelPy::Reduce<ReduceOpType::kMax>(py::object, py::object, bool);
template py::object KernelPy::Reduce<ReduceOpType::kMin>(py::object, py::object, bool);

void RegBaseApi(const py::module &m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject")
    .def("shape", &NDObjectPy::GetShape, "get shape")
    .def("dtype", &NDObjectPy::GetDType, "get dtype");

  (void)py::class_<ShapeRefPy, std::shared_ptr<ShapeRefPy>>(m, "ShapeRef")
    .def(py::init<>())
    .def(py::init<const std::vector<int64_t> &>())
    .def("shape", &ShapeRefPy::GetShape, "get shape")
    .def("update", &ShapeRefPy::Update, "update shape");

  (void)py::class_<NDSymInt, std::shared_ptr<NDSymInt>>(m, "NDSymInt")
    .def("update", [](NDSymInt &self, int64_t v) { self.data_ = v; }, py::arg("value"), "Set the int scalar value");
  (void)py::class_<NDSymFloat, std::shared_ptr<NDSymFloat>>(m, "NDSymFloat")
    .def("update", [](NDSymFloat &self, float v) { self.data_ = v; }, py::arg("value"), "Set the float scalar value");
}
void RegKernelApi(const py::module &m) {
  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "KernelBase")
    .def("load", &KernelPy::Load, "load array")
    .def("view_load", &KernelPy::ViewLoad, "load array")
    .def("store", &KernelPy::Store, "store array")
    .def("make_int", &KernelPy::MakeIntScalar, "create int scalar")
    .def("make_float", &KernelPy::MakeFloatScalar, "create float scalar")
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
    .def("broadcast", &KernelPy::Broadcast, "emit broadcast op", py::arg("input"), py::arg("shape"))
    .def("full", &KernelPy::BroadcastScalar, "emit broadcast op", py::arg("input"), py::arg("shape"),
         py::arg("dtype") = "float32")
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
    .def("spec_next", &KernelPy::SpecNext, "spec next")
    .def_static("set_determ", &KernelPy::SetDeterm, "set deterministic")
    .def_static("set_online_tuning", &KernelPy::SetTuning, "set online tuning");
}
}  // namespace pyapi
}  // namespace dvm
