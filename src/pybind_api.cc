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

#include <memory>
#include <fstream>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "acl/acl_rt.h"
#include "kernel.h"
#include "pybind_api.h"

#define ASCEND_CALL(func)                                                                               \
  do {                                                                                                  \
    auto err = (func);                                                                                  \
    if (err != 0) {                                                                                     \
      std::cerr << "Ascend error in function " << #func << " : " << static_cast<int>(err) << std::endl; \
      exit(0);                                                                                          \
    }                                                                                                   \
  } while (0)

namespace dvm {
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
  const std::string formats[kTypeEnd] = {py::format_descriptor<bool>::format(), "e", "", py::format_descriptor<float>::format(),
                                         py::format_descriptor<int32_t>::format()};
  return formats[type];
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

template <typename T>
std::pair<bool, T> GetScalar(const py::object &obj) {
  if (py::isinstance<py::int_>(obj)) {
    return {true, static_cast<T>(py::cast<int64_t>(obj))};
  } else if (py::isinstance<py::float_>(obj)) {
    return {true, py::cast<T>(obj)};
  }
  return {false, (T)0};
}

static std::unordered_map<std::string, UnaryOpType> unary_map = {{"Abs", UnaryOpType::kAbs},
                                                                 {"Exp", UnaryOpType::kExp},
                                                                 {"IsFinite", UnaryOpType::kIsFinite},
                                                                 {"Log", UnaryOpType::kLog},
                                                                 {"LogicalNot", UnaryOpType::kLogicalNot},
                                                                 {"Reciprocal", UnaryOpType::kReciprocal},
                                                                 {"Sqrt", UnaryOpType::kSqrt}};

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

KernelPy::KernelPy(int dev_id,  const std::string &type_str) {
  KernelType type = type_str != "parallel" ? kStaticShape : kStaticParallel;
  uint32_t dev_count = 0;
  ASCEND_CALL(aclrtGetDeviceCount(&dev_count));
  ASSERT(static_cast<uint32_t>(dev_id) < dev_count);
  ASCEND_CALL(aclrtSetDevice(dev_id));
  dev_id_ = dev_id_;
  kernel_.Reset(type);
}

KernelPy::~KernelPy() {
  for (auto dev : dev_mem_) {
    ASCEND_CALL(aclrtFree(dev));
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

py::object KernelPy::Unary(const std::string &op_name, const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Unary(unary_map[op_name], in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(const py::object &input, const std::string &type) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Cast(in_obj, StringToTypeID(type));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(const py::object &cond, const py::object &lhs, const py::object &rhs) {
  auto input1 = lhs.cast<NDOpPyPtr>()->Get();
  auto input2 = rhs.cast<NDOpPyPtr>()->Get();
  auto input0 = cond.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Select(input0, input1, input2);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reduce(const std::string &type, const py::object &input, const py::object &dims,
                                    bool keepdims) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  std::vector<int64_t> &dim_vec = shape_vec_.emplace_back(GetVector(dims));
  auto dims_ref = shape_.emplace_back(new ShapeRef(dim_vec));
  auto op = kernel_.Reduce(ReduceOpType::kSum, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Binary(const std::string &op_name, const py::object &lhs, const py::object &rhs) {
  NDObject *op;
  auto [lhs_is_scalar, lhs_scalar] = GetScalar<float>(lhs);
  auto [rhs_is_scalar, rhs_scalar] = GetScalar<float>(rhs);
  if (lhs_is_scalar) {
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    if (kernel_.GetDType(input2) == dvm::kInt32) {
      op = kernel_.Binary(binary_map[op_name], GetScalar<int>(lhs).second, input2);
    } else {
      op = kernel_.Binary(binary_map[op_name], lhs_scalar, input2);
    }
  } else if (rhs_is_scalar) {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    if (kernel_.GetDType(input1) == dvm::kInt32) {
      op = kernel_.Binary(binary_map[op_name], input1, GetScalar<int>(rhs).second);
    } else {
      op = kernel_.Binary(binary_map[op_name], input1, rhs_scalar);
    }
  } else {
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    op = kernel_.Binary(binary_map[op_name], input1, input2);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(const py::object &input, const py::object &shape, const std::string &dtype,
                                       bool dummy_load) {
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(GetVector(shape));
  auto shape_ref = shape_.emplace_back(new ShapeRef(shape_vec));
  NDObject *op;
  auto [is_scalar, scalar] = GetScalar<float>(input);
  if (is_scalar) {
    auto type_id = StringToTypeID(dtype);
    op = kernel_.Broadcast(scalar, shape_ref, type_id, dummy_load);
  } else {
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    op = kernel_.Broadcast(in_obj, shape_ref);
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(const py::object &input, const py::object &shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(GetVector(shape));
  auto shape_ref = shape_.emplace_back(new ShapeRef(shape_vec));
  auto op = kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(const py::object &array) {
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  void *addr = ToDev(buf.ptr, buf.itemsize * buf.size);
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(buf.ndim);
  for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
    shape_vec[i] = buf.shape[i];
  }
  auto shape_ref = shape_.emplace_back(new ShapeRef(shape_vec));
  auto op = kernel_.Load(reinterpret_cast<void *>(addr), shape_ref, GetTypeID(buf));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::SliceLoad(const py::object &array, const py::object &start, const py::object &size) {
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  void *addr = ToDev(buf.ptr, buf.itemsize * buf.size);
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(buf.ndim);
  for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
    shape_vec[i] = buf.shape[i];
  }
  auto shape_ref = shape_.emplace_back(new ShapeRef(shape_vec));

  std::vector<int64_t> &start_vec = shape_vec_.emplace_back(GetVector(start));
  auto start_ref = shape_.emplace_back(new ShapeRef(start_vec));

  std::vector<int64_t> &size_vec = shape_vec_.emplace_back(GetVector(size));
  auto size_ref = shape_.emplace_back(new ShapeRef(size_vec));

  auto op =
    kernel_.SliceLoad(reinterpret_cast<void *>(addr), shape_ref, start_ref, size_ref, GetTypeID(buf));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StridedSliceLoad(const py::object &array, const py::object &start, const py::object &end, const py::object &step) {
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  void *addr = ToDev(buf.ptr, buf.itemsize * buf.size);
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(buf.ndim);
  for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
    shape_vec[i] = buf.shape[i];
  }
  auto shape_ref = shape_.emplace_back(new ShapeRef(shape_vec));

  std::vector<int64_t> &start_vec = shape_vec_.emplace_back(GetVector(start));
  auto start_ref = shape_.emplace_back(new ShapeRef(start_vec));

  std::vector<int64_t> &end_vec = shape_vec_.emplace_back(GetVector(end));
  auto end_ref = shape_.emplace_back(new ShapeRef(end_vec));

  std::vector<int64_t> &step_vec = shape_vec_.emplace_back(GetVector(step));
  auto step_ref = shape_.emplace_back(new ShapeRef(step_vec));

  auto op =
    kernel_.StridedSliceLoad(reinterpret_cast<void *>(addr), shape_ref, start_ref, end_ref, step_ref, GetTypeID(buf));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(const py::object &obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto src = static_cast<NDObject*>(in_obj);
  KernelPy::StoreInfo store;
  size_t size = ITEM_SIZE[src->type_id_];
  store.shape.resize(src->nd_.size());
  for (size_t i =0 ;i< src->nd_.size();i++) {
    size *= src->nd_[i];
    store.shape[src->nd_.size() - i - 1] = src->nd_[i];
  }
  store.host = std::malloc(size);
  std::memset(store.host, 0, size);
  store.dev = ToDev(store.host, size);
  auto op = kernel_.Store(reinterpret_cast<void *>(store.dev), in_obj);
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
  stores_.emplace_back(std::move(store));
  return py::array(py::dtype(info), info.shape, info.strides, info.ptr, obj);
}

py::object KernelPy::ElementAny(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::ParallelNext() {
  kernel_.ParallelNext();
}

void KernelPy::Tile(int start, int end, int64_t num) {
  static_cast<VKernelBase*>(kernel_.GetImpl())->SetTile(start, end, num);
}

void *KernelPy::ToDev(void *host, size_t size) {
  void *dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ASCEND_CALL(aclrtMemcpy(dev, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE));
  dev_mem_.push_back(dev);
  return dev;
}

void KernelPy::Optimize() {
  if (kernel_.GetImpl()->KType() == KernelType::kStaticShape) {
    static_cast<VKernelS*>(kernel_.GetImpl())->Optimize();
  }
}

py::object KernelPy::CodeGen(const std::string &path) {
  auto code = GetCode();
  if (!path.empty()) {
    std::ofstream file(path);
    file.write(reinterpret_cast<char *>(code->data_), code->data_size_);
    file.close();
  }
  return py::cast(code->block_dim_);
}

py::object KernelPy::DisAssemble() {
  GetCode();
  std::string data = kernel_.GetImpl()->DisAssemble();
  return py::cast(data);
}

py::object KernelPy::DumpGraph() {
  std::string data = kernel_.GetImpl()->DumpGraph();
  return py::cast(data);
}

void KernelPy::Run() {
  GetCode();
  ASCEND_CALL(kernel_.Launch(nullptr));
  ASCEND_CALL(aclrtSynchronizeStream(nullptr));
  for (auto &s : stores_) {
    size_t size = ITEM_SIZE[s.obj->type_id_];
    for (size_t i = 0; i < s.shape.size(); ++i) {
      size *= s.shape[i];
    }
    ASCEND_CALL(aclrtMemcpy(s.host, size, s.dev, size, ACL_MEMCPY_DEVICE_TO_HOST));
  }
}

py::object KernelPy::Perf() {
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
      ASCEND_CALL(aclrtMemcpy(s.dev, size, s.host, size, ACL_MEMCPY_HOST_TO_DEVICE));
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

py::object KernelPy::Measure() {
  if (kernel_.GetImpl()->KType() == KernelType::kStaticParallel) {
    return py::none();
  }
  Metrics met;
  VKernelBase* base_kernel = static_cast<VKernelBase*>(kernel_.GetImpl());
  base_kernel->CollectMetrics(met);
  py::dict ret = py::dict();
  ret["core_usage"] = py::float_(met.core_usage);
  ret["simd_usage"] = py::float_(met.simd_usage);
  ret["mem_usage"] = py::float_(met.mem_usage);
  return ret;
}

CodeBase *KernelPy::GetCode() {
  if (!codegen_) {
    auto begin = GetTimeX();
    kernel_.GetImpl()->CodeGen();
    auto end = GetTimeX();
    std::cout << "codegen time(us): " << end - begin << std::endl;
    codegen_ = true;
  }
  return kernel_.GetImpl()->GetCode();
}

void KernelPy::ResetPasses(const py::object &pass_names) {
  const static std::unordered_map<std::string, pass::Pass> pass_map = {
    {"PrintPeakLive", pass::PrintPeakLive},
    {"ReorderStore", pass::ReorderStore},
    {"CompactPeakLiveness", pass::CompactPeakLiveness},
    {"EliminateReshape", pass::EliminateReshape},
    {"InsertRemovePad", pass::InsertRemovePad}};
  pass::passes.clear();
  auto names = py::cast<py::list>(pass_names).cast<std::vector<std::string>>();
  for (auto name : names) {
    pass::passes.push_back(pass_map.at(name));
  }
}

class DevicePy {
 public:
  static std::string Arch() {
    static const char* soc_names[] = {"AscendC100", "AscendC220"};
    return soc_names[DeviceInfo::Instance().Arch()];
  }
  static int CoreNum() {
    return DeviceInfo::Instance().CoreNum();
  }
};

PYBIND11_MODULE(_dvm_py, m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject").def("shape", &NDObjectPy::GetShape, "get shape");

  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "Kernel")
      .def(py::init([](int dev_id, const std::string &ker_type) { return std::make_shared<KernelPy>(dev_id, ker_type); }))
      .def("load", &KernelPy::Load, "load array")
      .def("slice_load", &KernelPy::SliceLoad, "load array")
      .def("stridedslice_load", &KernelPy::StridedSliceLoad, "load array")
      .def("store", &KernelPy::Store, "store array")
      .def("unary", &KernelPy::Unary, "emit unary op")
      .def("cast", &KernelPy::Cast, "emit cast op")
      .def("element_any", &KernelPy::ElementAny, "emit element_any op")
      .def("binary", &KernelPy::Binary, "emit binary op")
      .def("select", &KernelPy::Select, "emit select op")
      .def("broadcast", &KernelPy::Broadcast, "emit broadcast op", py::arg("input"), py::arg("shape"),
        py::arg("dtype") = "float32", py::arg("dummy_load") = true)
      .def("reshape", &KernelPy::Reshape, "emit reshape op")
      .def("reduce", &KernelPy::Reduce, "emit reduce op")
      .def("copy", &KernelPy::Copy, "emit copy op")
      .def("p_next", &KernelPy::ParallelNext, "parallel next")
      .def("tile", &KernelPy::Tile, "set tiling")
      .def("optimize", &KernelPy::Optimize, "optimize code")
      .def("codegen", &KernelPy::CodeGen, "generate code")
      .def("das", &KernelPy::DisAssemble, "disassemble code")
      .def("dump", &KernelPy::DumpGraph, "dump graph")
      .def("perf", &KernelPy::Perf, "perf test")
      .def("measure", &KernelPy::Measure, "measure metrics")
      .def("run", &KernelPy::Run, "run kernel")
      .def("reset_passes", &KernelPy::ResetPasses, "reset passes");

  (void)py::class_<DevicePy, std::shared_ptr<DevicePy>>(m, "Device")
      .def_static("arch", &DevicePy::Arch, "Get system architecture")
      .def_static("core_num", &DevicePy::CoreNum, "Get soc core number");
}
}  // namespace dvm
