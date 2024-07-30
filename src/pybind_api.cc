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

#include <algorithm>
#include <memory>
#include <fstream>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "acl/acl_rt.h"
#include "kernel.h"
#include "pybind_api.h"
#include "bf16.h"

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

DType StringToTypeID(const std::string &type) {
  const static std::unordered_map<std::string, DType> map = {{"float32", DType::kFloat32},
                                                             {"float16", DType::kFloat16},
                                                             {"bfloat16", DType::kBFloat16},
                                                             {"bool", DType::kInt8},
                                                             {"int32", DType::kInt32}};
  return map.at(type);
}

std::string TypeIDToString(DType type) {
  const static std::string map[] = {"bool", "float16", "bfloat16", "float32", "int32"};
  return map[type];
}

std::string GetBufferFormat(const DType type) {
  const std::string formats[kTypeEnd] = {py::format_descriptor<bool>::format(), "e",
                                         py::format_descriptor<uint16_t>::format(), py::format_descriptor<float>::format(),
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

static std::unordered_map<std::string, KernelType> kernel_type_map = {
  {"", kStaticShape}, {"static", kStaticShape}, {"dyn", kDynShape}, {"mix", kStaticMix},
  {"parallel", kStaticParallel}, {"stages", kStaticStages}};

std::string NDObjectPy::GetDType() const { return TypeIDToString(obj_->type_id_); }

void ShapeRefPy::Update(const py::object &shape){
  shape_ = GetVector(shape);
  *shape_ref_ = shape_;
}

KernelPy::KernelPy(int dev_id,  const std::string &type_str) {
  auto it = kernel_type_map.find(type_str);
  KernelType type = it != kernel_type_map.end() ? it->second : kStaticShape;
  uint32_t dev_count = 0;
  ASCEND_CALL(aclrtGetDeviceCount(&dev_count));
  ASSERT(static_cast<uint32_t>(dev_id) < dev_count);
  ASCEND_CALL(aclrtSetDevice(dev_id));
  dev_id_ = dev_id_;
  kernel_.Reset(type);
  (void)System::Instance(); // early construct System
}

KernelPy::~KernelPy() {
  for (auto &it : loads_) {
    if (it.second.dev) {
      ASCEND_CALL(aclrtFree(it.second.dev));
    }
  }
  for (auto &it : stores_) {
    if (it.second.dev) {
      ASCEND_CALL(aclrtFree(it.second.dev));
    }
    if (it.second.host) {
      std::free(it.second.host);
    }
  }
  if (workspace_) {
    ASCEND_CALL(aclrtFree(workspace_));
  }
#ifdef VK_SIM_MODEL
  aclrtResetDevice(dev_id_);
#endif
  for (auto ref : shape_) {
    delete ref;
  }
}

ShapeRef* KernelPy::GetShapeRef(const py::object &shape) {
  if (py::isinstance<ShapeRefPy>(shape)) {
    auto shape_ptr = shape.cast<ShapeRefPyPtr>();
    return shape_ptr->Get();
  }
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(GetVector(shape));
  return shape_.emplace_back(new ShapeRef(shape_vec));
}

py::object KernelPy::Unary(const std::string &op_name, const py::object &input) {
  if (unary_map.count(op_name) == 0) {
    std::string err_msg = "Could not find op: " + op_name;
    throw std::invalid_argument(err_msg);
  }
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
  auto dims_ref = GetShapeRef(dims);
  auto op = kernel_.Reduce(ReduceOpType::kSum, in_obj, dims_ref, keepdims);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Binary(const std::string &op_name, const py::object &lhs, const py::object &rhs) {
  if (binary_map.count(op_name) == 0) {
    std::string err_msg = "Could not find op: " + op_name;
    throw std::invalid_argument(err_msg);
  }
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
  auto shape_ref = GetShapeRef(shape);
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
  auto shape_ref = GetShapeRef(shape);
  auto op = kernel_.Reshape(in_obj, shape_ref);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Copy(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(const py::object &shape, const std::string &type) {
  LoadInfo info;
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto op = kernel_.Load(nullptr, shape_ref, StringToTypeID(type));
  loads_[op] = std::move(info);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::SliceLoad(const py::object &shape, const py::object &start, const py::object &size, const std::string &type) {
  LoadInfo info;
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto size_ref = GetShapeRef(size);
  auto op = kernel_.SliceLoad(nullptr, shape_ref, start_ref, size_ref, StringToTypeID(type));
  loads_[op] = std::move(info);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StridedSliceLoad(const py::object &shape, const py::object &start, const py::object &end, const py::object &step, const std::string &type) {
  LoadInfo info;
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto end_ref = GetShapeRef(end);
  auto step_ref = GetShapeRef(step);
  auto op = kernel_.StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref, step_ref, StringToTypeID(type));
  loads_[op] = std::move(info);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(const py::object &obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Store(nullptr, in_obj);
  stores_[op] = StoreInfo();
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::PadStore(const py::object &obj, const py::object &pad_shape) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.PadStore(nullptr, in_obj, GetShapeRef(pad_shape));
  stores_[op] = StoreInfo();
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ElementAny(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ElemAny(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(const py::object &lhs, const py::object &rhs, bool trans_a, bool trans_b) {
  auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
  auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.MatMul(lhs_obj, rhs_obj, trans_a, trans_b);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ConvertToBF16(const py::object &input) {
  auto array = py::array(input);
  py::buffer_info buf = array.request();
  ASSERT(buf.itemsize == 4);  // input should be array of f32
  size_t size = buf.size;
  std::vector<uint16_t> bf16(size);
  F32ToBF16(reinterpret_cast<float *>(buf.ptr), bf16.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride /= 2; });
  py::buffer_info new_buf(bf16.data(), 2, py::format_descriptor<uint16_t>::format(), buf.ndim, buf.shape,
                          buf.strides);
  bf16s_.push_back(std::move(bf16));
  return py::array(new_buf);
}

py::object KernelPy::ConvertFromBF16(const py::object &input) {
  auto array = py::array(input);
  py::buffer_info buf = array.request();
  ASSERT(buf.itemsize == 2);  // input should be array of bf16
  size_t size = buf.size;
  std::vector<float> float_data(size);
  BF16ToF32(reinterpret_cast<uint16_t *>(buf.ptr), float_data.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride *= 2; });
  py::buffer_info new_buf(float_data.data(), 4, py::format_descriptor<float>::format(), buf.ndim, buf.shape,
                          buf.strides);
  f32s_.push_back(std::move(float_data));
  return py::array(new_buf);
}

void KernelPy::ParallelNext() {
  kernel_.ParallelNext();
}

void KernelPy::StageSwitch(const std::string &ker_type) {
  auto it = kernel_type_map.find(ker_type);
  KernelType type = it != kernel_type_map.end() ? it->second : kStaticShape;
  kernel_.StageSwitch(type);
}

py::object KernelPy::StageLoad(const py::object &store) {
  auto in_obj = store.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.StageLoad(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StageStore(const py::object &input) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.StageStore(in_obj);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::StagePadStore(const py::object &input, const py::object &pad_shape) {
  auto in_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.StagePadStore(in_obj, GetShapeRef(pad_shape));
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::Tile(int start, int end, int64_t num) {
  static_cast<VectorKernel*>(kernel_.GetImpl())->SetTile(start, end, num);
}

void KernelPy::CodeGen(const py::object &pass_names) {
  const static std::unordered_map<std::string, pass::Pass> pass_map = {
    {"PrintPeakLive", pass::PrintPeakLive},
    {"ReorderStore", pass::ReorderStore},
    {"ReorderLoad", pass::ReorderLoad},
    {"CompactPeakLiveness", pass::CompactPeakLiveness},
    {"EliminateReshape", pass::EliminateReshape},
    {"InsertRemovePad", pass::InsertRemovePad}};
  std::vector<pass::Pass> old_passes;
  std::swap(old_passes, pass::passes);
  if (py::isinstance<py::list>(pass_names)) {
    auto names = py::cast<py::list>(pass_names).cast<std::vector<std::string>>();
    for (auto name : names) {
      pass::passes.push_back(pass_map.at(name));
    }
  }
  auto begin = GetTimeX();
  auto workspace_size = kernel_.CodeGen();
  auto end = GetTimeX();
  std::cout << "codegen time(us): " << end - begin << std::endl;
  std::swap(old_passes, pass::passes);
  if (workspace_) {
    ASCEND_CALL(aclrtFree(workspace_));
    workspace_ = nullptr;
  }
  if (workspace_size > 0) {
    ASCEND_CALL(aclrtMalloc(&workspace_, workspace_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  }
}

py::object KernelPy::DisAssemble() {
  std::string data = kernel_.GetImpl()->DisAssemble();
  return py::cast(data);
}

py::object KernelPy::DumpGraph() {
  std::string data = kernel_.GetImpl()->DumpGraph();
  return py::cast(data);
}

void KernelPy::Run() {
  PrepareOutput();
  ASCEND_CALL(kernel_.Launch(workspace_, nullptr));
  auto ret = aclrtSynchronizeStream(nullptr);
  if (ret != 0) {
    std::cerr << kernel_.GetImpl()->DumpGraph() << std::endl;
    std::cerr << kernel_.GetImpl()->DisAssemble() << std::endl;
    std::cerr << "******** Kernel Execute Exception: " << ret << " ********" << std::endl;
    exit(0);
  }
}

py::object KernelPy::Perf() {
#define TEST_NUM   10
#ifdef VK_SIM_MODEL
  return py::none();
#else
  PrepareOutput();
  // warm up
  ASCEND_CALL(kernel_.Launch(workspace_, nullptr));
  ASCEND_CALL(aclrtSynchronizeStream(nullptr));
  float min_us = 1e6;
  float max_us = 0.0f;
  float total_us = 0.0f;
  aclrtEvent start, end;
  ASCEND_CALL(aclrtCreateEvent(&start));
  ASCEND_CALL(aclrtCreateEvent(&end));
  for (int i = 0; i < TEST_NUM; i++) {
    for (auto &s : stores_) {
      auto info = s.second;
      ASCEND_CALL(aclrtMemcpy(info.dev, info.size, info.host, info.size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ASCEND_CALL(aclrtRecordEvent(start, nullptr));
    ASCEND_CALL(kernel_.Launch(workspace_, nullptr));
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
  VectorKernel* base_kernel = static_cast<VectorKernel*>(kernel_.GetImpl());
  base_kernel->CollectMetrics(met);
  py::dict ret = py::dict();
  ret["core_usage"] = py::float_(met.core_usage);
  ret["simd_usage"] = py::float_(met.simd_usage);
  ret["mem_usage"] = py::float_(met.mem_usage);
  return ret;
}

void KernelPy::Input(const py::object &load, const py::object &array) {
  auto op = static_cast<NDAccess*>(load.cast<NDOpPyPtr>()->Get());
  auto it = loads_.find(op);
  ASSERT(it != loads_.end());
  auto &info = it->second;
  if (info.dev) {
    ASCEND_CALL(aclrtFree(info.dev));
  }
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  size_t size = buf.itemsize  * buf.size;
  ASCEND_CALL(aclrtMalloc(&info.dev, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ASCEND_CALL(aclrtMemcpy(info.dev, size, buf.ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
  op->gm_ = reinterpret_cast<uint8_t*>(info.dev);
  if (op->reloc_addr_) {
    op->Reloc(info.dev);
  }
  if (kernel_.GetImpl()->KType() == kDynShape) {
    info.shape.resize(buf.ndim);
    for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
      info.shape[i] = buf.shape[i];
    }
    *(op->shape_ref_) = info.shape;
  }
}

py::object KernelPy::Output(const py::object &store) {
  auto op = store.cast<NDOpPyPtr>()->Get();
  auto it = stores_.find(op);
  ASSERT(it != stores_.end());
  auto &info = it->second;
  ASSERT(info.dev);
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;
  ssize_t itemsize = ITEM_SIZE[op->type_id_];
  ssize_t ndim = op->shape_ref_->size;
  size_t size = itemsize;
  for (size_t i = 0; i < static_cast<size_t>(ndim); ++i) {
    size *= op->shape_ref_->data[i];
    shape.push_back(op->shape_ref_->data[i]);
    auto stride = itemsize;
    for (size_t j = i + 1; j < static_cast<size_t>(ndim); ++j) {
      stride *= op->shape_ref_->data[j];
    }
    strides.push_back(stride);
  }
  ASCEND_CALL(aclrtMemcpy(info.host, size, info.dev, size, ACL_MEMCPY_DEVICE_TO_HOST));
  py::buffer_info buf(info.host, itemsize, GetBufferFormat(op->type_id_), ndim, shape, strides);
  return py::array(py::dtype(buf), buf.shape, buf.strides, buf.ptr, store);
}

void KernelPy::ClearStoreMemory(const py::object &store) {
  auto op = store.cast<NDOpPyPtr>()->Get();
  auto it = stores_.find(op);
  ASSERT(it != stores_.end());
  it->second.clear_mem = true;
}

void KernelPy::PrepareOutput() {
  for (auto &it : stores_) {
    auto op = it.first;
    auto &info = it.second;
    if (info.host) {
      std::free(info.host);
    }
    if (info.dev) {
      ASCEND_CALL(aclrtFree(info.dev));
    }
    info.size = ITEM_SIZE[op->type_id_];
    for (size_t i = 0; i < op->shape_ref_->size; i++) {
      info.size *= op->shape_ref_->data[i];
    }
    info.host = std::malloc(info.size);
    const uint64_t reserve_mem = 512;
    ASCEND_CALL(aclrtMalloc(&info.dev, info.size + reserve_mem, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    if (info.clear_mem) {
      std::memset(info.host, 0, info.size);
      ASCEND_CALL(aclrtMemcpy(info.dev, info.size, info.host, info.size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    static_cast<NDAccess*>(op)->Reloc(info.dev);
  }
}

class DevicePy {
 public:
  static std::string Arch() {
    static const char* soc_names[] = {"AscendC220"};
    return soc_names[System::Instance().Arch()];
  }
  static int CoreNum() {
    return System::Instance().CoreNum();
  }
  static std::string SocName() {
    static const char* soc_names[] = {"Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4", "Unknow"};
    return soc_names[System::Instance().SocName()];
  }
};

PYBIND11_MODULE(_dvm_py, m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject")
    .def("shape", &NDObjectPy::GetShape, "get shape")
    .def("dtype", &NDObjectPy::GetDType, "get dtype");

  (void)py::class_<ShapeRefPy, std::shared_ptr<ShapeRefPy>>(m, "ShapeRef")
    .def(py::init<>())
    .def(py::init<const std::vector<int64_t>&>())
    .def("shape", &ShapeRefPy::GetShape, "get shape")
    .def("update", &ShapeRefPy::Update, "update shape");

  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "Kernel")
      .def(py::init([](int dev_id, const std::string &ker_type) { return std::make_shared<KernelPy>(dev_id, ker_type); }))
      .def("load", &KernelPy::Load, "load array")
      .def("slice_load", &KernelPy::SliceLoad, "load array")
      .def("stridedslice_load", &KernelPy::StridedSliceLoad, "load array")
      .def("store", &KernelPy::Store, "store array")
      .def("pad_store", &KernelPy::PadStore, "pad store array")
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
      .def("matmul", &KernelPy::MatMul, "emit matmul op")
      .def("convert_to_bf16", &KernelPy::ConvertToBF16, "convert f32 array to bf16 array")
      .def("convert_from_bf16", &KernelPy::ConvertFromBF16, "convert bf16 array to f32 array")
      .def("p_next", &KernelPy::ParallelNext, "parallel next")
      .def("stage_switch", &KernelPy::StageSwitch, "stage switch")
      .def("stage_load", &KernelPy::StageLoad, "stage load")
      .def("stage_store", &KernelPy::StageStore, "stage store")
      .def("stage_pad_store", &KernelPy::StagePadStore, "stage store")
      .def("input", &KernelPy::Input, "get ouput array")
      .def("output", &KernelPy::Output, "get ouput array")
      .def("clear_store_memory", &KernelPy::ClearStoreMemory, "clear store memory")
      .def("tile", &KernelPy::Tile, "set tiling")
      .def("codegen", &KernelPy::CodeGen, "generate code")
      .def("das", &KernelPy::DisAssemble, "disassemble code")
      .def("dump", &KernelPy::DumpGraph, "dump graph")
      .def("perf", &KernelPy::Perf, "perf test")
      .def("measure", &KernelPy::Measure, "measure metrics")
      .def("run", &KernelPy::Run, "run kernel")
      .def_static("set_determ", &KernelPy::SetDeterm, "set deterministic");

  (void)py::class_<DevicePy, std::shared_ptr<DevicePy>>(m, "Device")
      .def_static("arch", &DevicePy::Arch, "Get system architecture")
      .def_static("core_num", &DevicePy::CoreNum, "Get soc core number")
      .def_static("soc_name", &DevicePy::SocName, "Get soc name");
}
}  // namespace dvm
