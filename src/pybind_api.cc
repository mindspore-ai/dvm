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
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "acl_ext.h"
#include "acl/acl_rt.h"
#include "kernel.h"
#include "pybind_api.h"

#define ASCEND_CALL(func)                                                                               \
  do {                                                                                                  \
    rtError_t err = (func);                                                                             \
    if (err != RT_ERROR_NONE) {                                                                         \
      std::cerr << "Ascend error in function " << #func << " : " << static_cast<int>(err) << std::endl; \
      exit(0);                                                                                          \
    }                                                                                                   \
  } while (0)

namespace dvm {
namespace py = pybind11;

static int64_t GetTimeX() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

DType StringToTypeID(const std::string type) {
  const static std::unordered_map<std::string, DType> map = {
    {"float32", DType::kFloat32}, {"float16", DType::kFloat16}, {"bool", DType::kInt8}, {"int32", DType::kInt32}};
  return map.at(type);
}

std::string GetBufferFormat(const DType type) {
  std::vector<std::string> map{py::format_descriptor<bool>::format(), "e", py::format_descriptor<float>::format(),
                               py::format_descriptor<int32_t>::format()};
  return map[type];
}

DType GetTypeID(py::buffer_info &info) {
  if (info.format == py::format_descriptor<float>::format()) {
    return DType::kFloat32;
  } else if (info.format == py::format_descriptor<int32_t>::format()) {
    return DType::kInt32;
  } else if (info.format == py::format_descriptor<bool>::format()) {
    return DType::kInt8;
  } else if (info.itemsize == 2) {
    return DType::kFloat16;
  }
  return DType::kFloat32;
}

std::vector<int64_t> GetVector(const py::object &shape) {
  py::list shape_list = py::cast<py::list>(shape);
  size_t size = shape_list.size();
  std::vector<int64_t> shape_vec(size);
  for (size_t i = 0; i < size; ++i) {
    shape_vec[i] = py::cast<int64_t>(shape_list[i]);
  }
  return shape_vec;
}

std::pair<bool, float> GetScalar(const py::object &obj) {
  if (py::isinstance<py::int_>(obj)) {
    return {true, static_cast<float>(py::cast<int64_t>(obj))};
  } else if (py::isinstance<py::float_>(obj)) {
    return {true, py::cast<float>(obj)};
  }
  return {false, 0.0};
}

static std::unordered_map<std::string, UnaryOpType> unary_map = {{"Abs", UnaryOpType::kAbs},
                                                                 {"Exp", UnaryOpType::kExp},
                                                                 {"IsFinite", UnaryOpType::kIsFinite},
                                                                 {"Log", UnaryOpType::kLog},
                                                                 {"LogicalNot", UnaryOpType::kLogicalNot},
                                                                 {"Reciprocal", UnaryOpType::kReciprocal},
                                                                 {"Sqrt", UnaryOpType::kSqrt},
                                                                 {"Rsqrt", UnaryOpType::kRsqrt}};

static std::unordered_map<std::string, BinaryOpType> binary_map = {{"Add", BinaryOpType::kAdd},
                                                                   {"Sub", BinaryOpType::kSub},
                                                                   {"Mul", BinaryOpType::kMul},
                                                                   {"Div", BinaryOpType::kDiv},
                                                                   {"Pow", BinaryOpType::kPow},
                                                                   {"RealDiv", BinaryOpType::kDiv},
                                                                   {"Maximum", BinaryOpType::kMaximum},
                                                                   {"Minimum", BinaryOpType::kMinimum},
                                                                   {"Equal", BinaryOpType::kEqual},
                                                                   {"Greater", BinaryOpType::kGreater},
                                                                   {"GreaterEqual", BinaryOpType::kGreaterEqual},
                                                                   {"Less", BinaryOpType::kLess},
                                                                   {"LessEqual", BinaryOpType::kLessEqual},
                                                                   {"NotEqual", BinaryOpType::kNotEqual},
                                                                   {"LogicalAnd", BinaryOpType::kLogicalAnd},
                                                                   {"LogicalOr", BinaryOpType::kLogicalOr}};

void VKernelPy::Tile(int start, int end, int64_t num) {
  kernel_.GetImpl()->SetTile(start, end, num);
}

void* VKernelPy::ToDev(void* host, size_t size) {
  void* dev = nullptr;
  auto it = host_dev_map_.find(host);
  if (it != host_dev_map_.end()) {
    dev = it->second;
  } else {
    ASCEND_CALL(aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    host_dev_map_[host] = dev;
  }
  ASCEND_CALL(aclrtMemcpy(dev, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE));
  return dev;
}

void VKernelPy::FromDev(void* host, size_t size) {
  auto it = host_dev_map_.find(host);
  ASSERT(it != host_dev_map_.end());
  ASCEND_CALL(aclrtMemcpy(host, size, it->second, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

py::object DvmKernelBuilderPy::Unary(const std::string &op_name, const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->kernel_.Unary(unary_map[op_name], in_obj);
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Cast(const py::object &input, const std::string &type) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->kernel_.Cast(in_obj, StringToTypeID(type));
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Select(const py::object &cond, const py::object &lhs, const py::object &rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->kernel_.Select(input0, input1, input2);
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Reduce(const std::string &type, const py::object &input, const py::object &dims,
                                    bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  std::vector<int64_t> &dim_vec = kernel_->shape_vec_.emplace_back(GetVector(dims));
  auto dims_ref = kernel_->shape_.emplace_back(new ShapeRef(dim_vec));
  auto op = kernel_->kernel_.Reduce(ReduceOpType::kSum, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Binary(const std::string &op_name, const py::object &lhs, const py::object &rhs) {
  NDObject *op;
  auto [lhs_is_scalar, lhs_scalar] = GetScalar(lhs);
  auto [rhs_is_scalar, rhs_scalar] = GetScalar(rhs);
  if (lhs_is_scalar) {
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_->kernel_.Binary(binary_map[op_name], lhs_scalar, input2);
  } else if (rhs_is_scalar) {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    op = kernel_->kernel_.Binary(binary_map[op_name], input1, rhs_scalar);
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_->kernel_.Binary(binary_map[op_name], input1, input2);
  }
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Broadcast(const py::object &input, const py::object &shape, const std::string &dtype,
                                       bool dummy_load) {
  std::vector<int64_t> &shape_vec = kernel_->shape_vec_.emplace_back(GetVector(shape));
  auto shape_ref = kernel_->shape_.emplace_back(new ShapeRef(shape_vec));
  NDObject *op;
  auto [is_scalar, scalar] = GetScalar(input);
  if (is_scalar) {
    auto type_id = StringToTypeID(dtype);
    op = kernel_->kernel_.Broadcast(scalar, shape_ref, type_id, dummy_load);
  } else {
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    op = kernel_->kernel_.Broadcast(in_obj, shape_ref);
  }
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Reshape(const py::object &input, const py::object &shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  std::vector<int64_t> &shape_vec = kernel_->shape_vec_.emplace_back(GetVector(shape));
  auto shape_ref = kernel_->shape_.emplace_back(new ShapeRef(shape_vec));
  auto op = kernel_->kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Copy(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDOpPy>(op));
}

VKernelPy::VKernelPy(int dev_id) {
  uint32_t dev_count = 0;
  ASCEND_CALL(aclrtGetDeviceCount(&dev_count));
  ASSERT(static_cast<uint32_t>(dev_id) < dev_count);
  ASCEND_CALL(aclrtSetDevice(dev_id));
  dev_id_ = dev_id_;
  kernel_.Reset(kStaticShape);
}

VKernelPy::~VKernelPy() {
  for (auto it = host_dev_map_.begin(); it != host_dev_map_.end(); ++it) {
    ASCEND_CALL(aclrtFree(it->second));
  }
#ifdef VK_SIM_MODEL
  aclrtResetDevice(dev_id_);
#endif
  for (auto &s : stores_) {
    if (s.host != nullptr) {
      std::free(s.host);
    }
  }
  for (auto ref : shape_) {
    delete ref;
  }
}

py::object VKernelPy::CodeGen(const std::string &path) {
  auto code = GetCode();
  if (!path.empty()) {
    std::ofstream file(path);
    file.write(reinterpret_cast<char *>(code->data), code->size);
    file.close();
  }
  return py::cast(code->BlockDim());
}

py::object VKernelPy::DisAssemble() {
  GetCode();
  std::string data = kernel_.GetImpl()->DisAssemble();
  return py::cast(data);
}

py::object VKernelPy::DumpGraph() {
  std::string data = kernel_.GetImpl()->DumpGraph();
  return py::cast(data);
}

void VKernelPy::Run() {
  GetCode();
  ASCEND_CALL(kernel_.Launch(nullptr));
  ASCEND_CALL(aclrtSynchronizeStream(nullptr));
  for (auto &s : stores_) {
    size_t size = ITEM_SIZE[s.obj->type_id_];
    for (size_t i = 0; i < s.shape.size(); ++i) {
      size *= s.shape[i];
    }
    FromDev(s.host, size);
  }
}

py::object VKernelPy::Perf() {
#define TEST_NUM   10
#ifdef VK_SIM_MODEL
  return py::none();
#else
  GetCode();
  // warm up
  ASCEND_CALL(kernel_.Launch(nullptr));
  ASCEND_CALL(aclrtSynchronizeStream(nullptr));
  float min_us = 1e6;
  float max_us = 0.0f;
  float total_us = 0.0f;
  aclrtEvent start, end;
  ASCEND_CALL(aclrtCreateEvent(&start));
  ASCEND_CALL(aclrtCreateEvent(&end));
  for (int i = 0; i < TEST_NUM; i++) {
    for (auto &s : stores_) {
      size_t size = ITEM_SIZE[s.obj->type_id_];
      for (size_t i = 0; i < s.shape.size(); ++i) {
        size *= s.shape[i];
      }
      ToDev(s.host, size);
    }
    ASCEND_CALL(aclrtRecordEvent(start, nullptr));
    ASCEND_CALL(kernel_.Launch(nullptr));
    ASCEND_CALL(aclrtRecordEvent(end, nullptr));
    ASCEND_CALL(aclrtSynchronizeStream(nullptr));
    float time_us = 0.0f;
    ASCEND_CALL(aclrtEventElapsedTime(&time_us, start, end));
    time_us *= 1000.0;
    if (time_us < min_us) {
      min_us = time_us;
    }
    if (time_us > max_us) {
      max_us = time_us;
    }
    total_us += time_us;
  }
  ASCEND_CALL(aclrtDestroyEvent(start));
  ASCEND_CALL(aclrtDestroyEvent(end));
  return py::make_tuple(py::float_(min_us), py::float_(max_us), py::float_(total_us / float(TEST_NUM)));
#endif
}

Code *VKernelPy::GetCode() {
  if (!codegen_) {
    auto begin = GetTimeX();
    kernel_.GetImpl()->CodeGen();
    auto end = GetTimeX();
    std::cout << "codegen time(us): " << end - begin << std::endl;
    codegen_ = true;
  }
  return kernel_.GetImpl()->GetCode();
}

py::object DvmKernelBuilderPy::Load(const py::object &array) {
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  void *addr = kernel_->ToDev(buf.ptr, buf.itemsize * buf.size);
  std::vector<int64_t> &shape_vec = kernel_->shape_vec_.emplace_back(buf.ndim);
  for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
    shape_vec[i] = buf.shape[i];
  }
  auto shape_ref = kernel_->shape_.emplace_back(new ShapeRef(shape_vec));
  auto op = kernel_->kernel_.Load(reinterpret_cast<void *>(addr), shape_ref, GetTypeID(buf));
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Store(const py::object &obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto src = static_cast<NDObject*>(in_obj);
  VKernelPy::Store store;
  size_t size = ITEM_SIZE[src->type_id_];
  store.shape.resize(src->nd_.size());
  for (size_t i =0 ;i< src->nd_.size();i++) {
    size *= src->nd_[i];
    store.shape[src->nd_.size() - i - 1] = src->nd_[i];
  }
  store.host = std::malloc(size);
  std::memset(store.host, 0, size);
  store.dev = kernel_->ToDev(store.host, size);
  auto op = kernel_->kernel_.Store(reinterpret_cast<void *>(store.dev), in_obj);
  store.obj = op;

  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;
  ssize_t itemsize = ITEM_SIZE[src->type_id_];
  ssize_t ndim = store.shape.size();
  for (size_t i = 0; i < static_cast<size_t>(ndim); ++i) {
    shape.push_back(store.shape[i]);
    auto stride = itemsize;
    for (size_t j = i + 1; j < static_cast<size_t>(ndim); ++j) {
      stride *= store.shape[j];
    }
    strides.push_back(stride);
  }
  py::buffer_info info(store.host, itemsize, GetBufferFormat(src->type_id_), ndim, shape, strides);
  kernel_->stores_.emplace_back(std::move(store));
  return py::array(py::dtype(info), info.shape, info.strides, info.ptr, obj);
}

py::object DvmKernelBuilderPy::ElementAny(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_->kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDOpPy>(op));
}

py::object DvmKernelBuilderPy::Get() {
  return py::cast(kernel_);
}

PYBIND11_MODULE(builder, m) {
  (void)py::class_<NDOpPy, std::shared_ptr<NDOpPy>>(m, "NDOpPy").def("shape", &NDOpPy::GetShape, "get shape");

  (void)py::class_<VKernelPy, std::shared_ptr<VKernelPy>>(m, "VKernel")
      .def("tile", &VKernelPy::Tile, "set tiling")
      .def("codegen", &VKernelPy::CodeGen, "generate code")
      .def("das", &VKernelPy::DisAssemble, "disassemble code")
      .def("dump", &VKernelPy::DumpGraph, "dump graph")
      .def("perf", &VKernelPy::Perf, "perf test")
      .def("run", &VKernelPy::Run, "run kernel");

  (void)py::class_<DvmKernelBuilderPy, std::shared_ptr<DvmKernelBuilderPy>>(m, "DvmKernelBuilder")
      .def(py::init([](int dev_id) { return std::make_shared<DvmKernelBuilderPy>(dev_id); }))
      .def("load", &DvmKernelBuilderPy::Load, "load array")
      .def("store", &DvmKernelBuilderPy::Store, "store array")
      .def("unary", &DvmKernelBuilderPy::Unary, "emit unary op")
      .def("cast", &DvmKernelBuilderPy::Cast, "emit cast op")
      .def("element_any", &DvmKernelBuilderPy::ElementAny, "emit element_any op")
      .def("binary", &DvmKernelBuilderPy::Binary, "emit binary op")
      .def("select", &DvmKernelBuilderPy::Select, "emit select op")
      .def("broadcast", &DvmKernelBuilderPy::Broadcast, "emit broadcast op", py::arg("input"), py::arg("shape"),
        py::arg("dtype") = "float32", py::arg("dummy_load") = true)
      .def("reshape", &DvmKernelBuilderPy::Reshape, "emit reshape op")
      .def("reduce", &DvmKernelBuilderPy::Reduce, "emit reduce op")
      .def("copy", &DvmKernelBuilderPy::Copy, "emit copy op")
      .def("get", &DvmKernelBuilderPy::Get, "get vm kernel");
}
}  // namespace dvm
