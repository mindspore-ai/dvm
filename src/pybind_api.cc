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

#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <securec.h>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "acl/acl_rt.h"
#include "acl/acl_prof.h"
#include "kernel.h"
#include "pybind_api.h"
#include "msprof.h"
#include "xkernel.h"

namespace dvm {
using namespace pyapi;
namespace {
const size_t TEST_NUM = 10;

class ProfileMgr {
 public:
  ProfileMgr(const std::string &path) {
    aclprofInit(path.c_str(), path.length());
    int32_t device_id;
    aclrtGetDevice(&device_id);
    uint32_t device_list[1] = {static_cast<uint32_t>(device_id)};
    uint32_t device_num = 1;
    uint64_t mask =
      ACL_PROF_ACL_API | ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME | ACL_PROF_TRAINING_TRACE | ACL_PROF_AICPU;
    acl_config_ = aclprofCreateConfig(device_list, device_num, ACL_AICORE_ARITHMETIC_UTILIZATION, nullptr, mask);
  }
  ~ProfileMgr() {
    aclprofDestroyConfig(acl_config_);
    aclprofFinalize();
  }
  void ProfStart() { aclprofStart(acl_config_); }
  void ProfStop() { aclprofStop(acl_config_); }

 private:
  aclprofConfig *acl_config_;
};

inline void F32ToBF16(float *input, uint16_t *output, uint32_t size) {
  while (size-- != 0) {
    *output++ = BFloat16(*input++).int_value();
  }
}

inline void BF16ToF32(uint16_t *input, float *output, uint32_t size) {
  while (size-- != 0) {
    *output++ = static_cast<float>(BFloat16(*input++));
  }
}

std::string GetBufferFormat(const DataType type) {
  const std::string formats[kDataTypeEnd] = {
    py::format_descriptor<bool>::format(),     "e",
    py::format_descriptor<uint16_t>::format(), py::format_descriptor<float>::format(),
    py::format_descriptor<int32_t>::format(),  py::format_descriptor<int64_t>::format()};
  return formats[type];
}

std::vector<int64_t> GetVector(py::object shape) {
  py::list shape_list = py::cast<py::list>(shape);
  size_t size = shape_list.size();
  std::vector<int64_t> shape_vec(size);
  for (size_t i = 0; i < size; ++i) {
    shape_vec[i] = py::cast<int64_t>(shape_list[i]);
  }
  return shape_vec;
}

inline void ResetStoreMemory(const std::vector<RtKernelPy::StoreInfo> &stores) {
  for (auto &info : stores) {
    ERROR_CHECK(aclrtMemcpy(info.dev, info.size, info.host, info.size, ACL_MEMCPY_HOST_TO_DEVICE));
  }
}

std::unordered_map<std::string, KernelType> kernel_type_map = {
  {"", KernelType::kVector},     {"vector", KernelType::kVector},     {"cube", KernelType::kCube},
  {"mix", KernelType::kMix},     {"parallel", KernelType::kParallel}, {"seq", KernelType::kSequence},
  {"split", KernelType::kSplit}, {"eager", KernelType::kEager}};

std::pair<KernelType, uint32_t> ParseKernelType(const std::string &ker_type) {
  auto pos = ker_type.find(':', 0);
  auto type_name = ker_type.substr(0, pos);
  uint32_t flags = 0;
  while (pos != std::string::npos) {
    pos++;
    auto end = ker_type.find(',', pos);
    auto flag_name = ker_type.substr(pos, end == std::string::npos ? end : end - pos);
    if (flag_name == "dyn") {
      flags |= KernelFlag::kDynamic;
    } else if (flag_name == "unify_ws") {
      flags |= KernelFlag::kUnifyWS;
    } else if (flag_name == "spec") {
      flags |= KernelFlag::kSpeculate;
    } else {
      DvmException("kernel flag error");
    }
    pos = end;
  }
  auto it = kernel_type_map.find(type_name);
  if (it == kernel_type_map.end()) {
    DvmException("kernel type error");
  }
  return std::make_pair(it->second, flags);
}

template <typename T>
T &FindVectorInfo(std::vector<T> &infos, NDObject *op) {
  for (auto &info : infos) {
    if (info.op == op) {
      return info;
    }
  }
  ASSERT(0);
  return infos.front();
}

struct MpCtx {
  pid_t pids[32];
  int rank_size{1};
  int rank_id{0};
  int shmid{0};
  volatile int64_t *bars{nullptr};
  Comm comm;
};
MpCtx g_mpc;
}  // end namespace

class KernelRunner : public WsAllocator {
 public:
  using LoadInfo = RtKernelPy::LoadInfo;
  using StoreInfo = RtKernelPy::StoreInfo;

  virtual ~KernelRunner() {}
  virtual void AllocLoad(const py::buffer_info &buf, LoadInfo &load) = 0;
  virtual void AllocStore(StoreInfo &store) = 0;
  virtual void Reset() = 0;
  virtual int Run(Kernel &kernel, void *workspace, bool sync) = 0;
};

class DevRunner : public KernelRunner {
 public:
  DevRunner(int dev_id) { dev_id_ = dev_id; }
  ~DevRunner() override {
    Reset();
    aclrtResetDevice(dev_id_);
  }

  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override {
    size_t size = buf.itemsize * buf.size;
    if (size == 0) return;
    ERROR_CHECK(aclrtMalloc(&load.dev, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ERROR_CHECK(aclrtMemcpy(load.dev, size, buf.ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
    dev_mem_.push_back(load.dev);
  }

  void AllocStore(StoreInfo &store) override {
    if (store.size == 0) return;
    store.host = std::malloc(store.size);
    const uint64_t reserve_mem = 512;
    ERROR_CHECK(aclrtMalloc(&store.dev, store.size + reserve_mem, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    if (store.clear_mem) {
      memset_s(store.host, store.size, 0, store.size);
      ERROR_CHECK(aclrtMemcpy(store.dev, store.size, store.host, store.size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    host_mem_.push_back(store.host);
    dev_mem_.push_back(store.dev);
  }

  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ERROR_CHECK(aclrtMalloc(&ws, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      dev_mem_.push_back(ws);
    }
    return ws;
  }

  int Run(Kernel &kernel, void *workspace, bool sync) override {
    if (kernel.GetImpl()->IsSplit()) {
      kernel.Launch(nullptr);
    } else {
      ERROR_CHECK(kernel.Launch(nullptr, 0, workspace, nullptr));
    }
    return sync ? aclrtSynchronizeStream(nullptr) : 0;
  }

  void Reset() override {
    for (auto mem : host_mem_) {
      std::free(mem);
    }
    host_mem_.clear();
    for (auto mem : dev_mem_) {
      ERROR_CHECK(aclrtFree(mem));
    }
    dev_mem_.clear();
  }

  std::vector<void *> host_mem_;
  std::vector<void *> dev_mem_;
  int dev_id_{0};
};

void DryRunEntry(uint64_t core_idx, bool is_cube, int target);
void DryRunExit();

class DryRunner : public KernelRunner {
 public:
  DryRunner(int core_id) : core_id_(core_id) {}
  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override { load.dev = buf.ptr; }
  void AllocStore(StoreInfo &store) override {
    store.host = std::malloc(store.size);
    if (store.clear_mem) {
      memset_s(store.host, store.size, 0, store.size);
    }
    host_mem_.push_back(store.host);
    store.dev = store.host;
  }
  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ws = std::malloc(size);
      host_mem_.push_back(ws);
    }
    return ws;
  }
  void Reset() override {
    for (auto mem : host_mem_) {
      std::free(mem);
    }
    host_mem_.clear();
  }
  int Run(Kernel &kernel, void *workspace, bool sync) override {
    DryRun(kernel, workspace, core_id_, false);
    return 0;
  }

  void DryRun(Kernel &kernel, void *workspace, int core_id, bool is_cube) {
    int target = kernel.GetImpl()->code_.target_;
    DryRunEntry(core_id, is_cube || target == Code::kTargetCube, target);
    if (kernel.GetImpl()->IsSplit()) {
      kernel.Launch(nullptr);
    } else {
      ERROR_CHECK(kernel.Launch(nullptr, 0, workspace, nullptr));
    }
    DryRunExit();
  }

 private:
  int core_id_;
  std::vector<void *> host_mem_;
};

class DasRunner : public KernelRunner {
 public:
  DasRunner() { addr_ = reinterpret_cast<uint8_t *>(0x10); }
  void AllocLoad(const py::buffer_info &buf, LoadInfo &load) override {
    load.dev = addr_;
    addr_ += buf.itemsize * buf.size;
  }
  void AllocStore(StoreInfo &store) override {
    store.host = std::malloc(store.size);
    host_mem_.push_back(store.host);
    store.dev = addr_;
    addr_ += store.size;
  }
  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      ws = addr_;
      addr_ += size;
    }
    return ws;
  }
  void Reset() override {
    addr_ = reinterpret_cast<uint8_t *>(0x10);
    for (auto mem : host_mem_) {
      std::free(mem);
    }
    host_mem_.clear();
  }
  int Run(Kernel &kernel, void *workspace, bool sync) override {
    auto &code = kernel.GetImpl()->code_;
    code.RelocBinds(workspace);
    return 0;
  }

 private:
  std::vector<void *> host_mem_;
  uint8_t *addr_;
};

class RunnerManager {
 public:
  ~RunnerManager() {
    delete dev_;
    delete dry_;
    delete das_;
  }

  static RunnerManager &Instance() {
    static RunnerManager mng;
    return mng;
  }

  void ResetDevRuner() {
    delete dev_;
    dev_ = nullptr;
  }

  KernelRunner *Get(const std::string type, int dev_id) {
    if (type == "dev") {
      if (dev_) {
        if (dev_->dev_id_ == dev_id) {
          return dev_;
        }
        delete dev_;
      }
      dev_ = new DevRunner(dev_id);
      return dev_;
    } else if (type == "dry") {
      if (!dry_) {
        dry_ = new DryRunner(dev_id);
      }
      return dry_;
    } else if (type == "das") {
      if (!das_) {
        das_ = new DasRunner();
      }
      return das_;
    }
    ASSERT(0);
    return nullptr;
  }

 private:
  DevRunner *dev_;
  DryRunner *dry_;
  DasRunner *das_;
};

RtKernelPy::RtKernelPy(const std::string &ker_type, const std::string &run_type, int dev_id) {
  uint32_t dev_count = 0;
  ERROR_CHECK(aclrtGetDeviceCount(&dev_count));
  ASSERT(static_cast<uint32_t>(dev_id) < dev_count);
  ERROR_CHECK(aclrtSetDevice(dev_id));
  runner_ = RunnerManager::Instance().Get(run_type, dev_id);
  auto [type, flags] = ParseKernelType(ker_type);
  kernel_.Reset(type, flags);
}

RtKernelPy::~RtKernelPy() {
  runner_->Reset();
#ifdef VK_SIM_MODEL
  RunnerManager::Instance().ResetDevRuner();
#endif
  for (auto *ref : shape_) {
    delete ref;
  }
}

ShapeRef *RtKernelPy::GetShapeRef(py::object shape) {
  if (py::isinstance<ShapeRefPy>(shape)) {
    auto shape_ptr = shape.cast<std::shared_ptr<ShapeRefPy>>();
    return shape_ptr->Get();
  }
  std::vector<int64_t> &shape_vec = shape_vec_.emplace_back(GetVector(shape));
  return shape_.emplace_back(new ShapeRef(shape_vec));
}

py::object RtKernelPy::OneHot(py::object indices, int depth, int axis, py::object on_value, py::object off_value,
                              const std::string &dtype) {
  auto indices_obj = indices.cast<NDOpPyPtr>()->Get();
  auto depth_ref = shape_.emplace_back(new ShapeRef(shape_vec_.emplace_back(1, depth)));
  auto type_id = StringToTypeID(dtype);
  NDObject *op;
  if (type_id == kInt32) {
    op = kernel_.OneHot(indices_obj, depth_ref, axis, py::cast<int32_t>(on_value), py::cast<int32_t>(off_value));
  } else {
    auto on_float = py::cast<float>(on_value);
    auto off_float = py::cast<float>(off_value);
    if (type_id == kFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, Float16(on_float), Float16(off_float));
    } else if (type_id == kBFloat16) {
      op = kernel_.OneHot(indices_obj, depth_ref, axis, BFloat16(on_float), BFloat16(off_float));
    } else {
      ASSERT(type_id == kFloat32);
      op = kernel_.OneHot(indices_obj, depth_ref, axis, on_float, off_float);
    }
  }
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::Load(py::object shape, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto op = kernel_.Load(nullptr, shape_ref, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::ViewLoad(py::object shape, py::object stride, int64_t offset, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto stride_ref = GetShapeRef(stride);
  const int64_t *offset_ptr = nullptr;
  if (offset != 0) {
    auto &offset_vec = shape_vec_.emplace_back(1, offset);
    offset_ptr = &offset_vec[0];
  }
  auto op = kernel_.Load(nullptr, shape_ref, stride_ref, offset_ptr, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::SliceLoad(py::object shape, py::object start, py::object size, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto size_ref = GetShapeRef(size);
  auto op = kernel_.SliceLoad(nullptr, shape_ref, start_ref, size_ref, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::StridedSliceLoad(py::object shape, py::object start, py::object end, py::object step,
                                        const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto start_ref = GetShapeRef(start);
  auto end_ref = GetShapeRef(end);
  auto step_ref = GetShapeRef(step);
  auto op = kernel_.StridedSliceLoad(nullptr, shape_ref, start_ref, end_ref, step_ref, StringToTypeID(type));
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::MultiLoad(py::object shape, const std::string &type) {
  auto &info = loads_.emplace_back();
  info.shape = GetVector(shape);
  auto shape_ref = shape_.emplace_back(new ShapeRef(info.shape));
  auto op = kernel_.MultiLoad(nullptr, shape_ref, StringToTypeID(type), &g_mpc.comm);
  info.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::Store(py::object obj) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.Store(nullptr, in_obj);
  auto &store = stores_.emplace_back();
  store.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::PadStore(py::object obj, int64_t pad_size) {
  auto in_obj = obj.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.PadStore(nullptr, in_obj, pad_size);
  auto &store = stores_.emplace_back();
  store.op = op;
  return py::cast(std::make_shared<NDObjectPy>(op));
}

void RtKernelPy::SetStoreInplace(py::object obj) {
  auto store = obj.cast<NDOpPyPtr>()->Get();
  kernel_.SetStoreInplace(store);
}

py::object RtKernelPy::AllReduce(const std::string &type, py::object input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  ReduceType reduce_type{ReduceType::kReduceTypeEnd};
  if (type == "sum") {
    reduce_type = ReduceType::kSum;
  } else if (type == "max") {
    reduce_type = ReduceType::kMax;
  } else {
    throw py::value_error("Unsupported AllReduce type: " + type);
  }
  auto op = kernel_.AllReduce(reduce_type, input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::ReduceScatter(py::object input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.ReduceScatter(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::AllGather(py::object input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.AllGather(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::AllGatherV2(py::object input) {
  auto input_obj = input.cast<NDOpPyPtr>()->Get();
  auto op = kernel_.AllGatherV2(input_obj, &g_mpc.comm);
  return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object RtKernelPy::ConvertToBF16(py::object input) {
  auto array = py::array(input);
  py::buffer_info buf = array.request();
  ASSERT(buf.itemsize == 4);  // input should be array of f32
  size_t size = buf.size;
  std::vector<uint16_t> bf16(size);
  F32ToBF16(reinterpret_cast<float *>(buf.ptr), bf16.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride /= 2; });
  py::buffer_info new_buf(bf16.data(), 2, py::format_descriptor<uint16_t>::format(), buf.ndim, buf.shape, buf.strides);
  bf16s_.push_back(std::move(bf16));
  return py::array(new_buf);
}

py::object RtKernelPy::ConvertFromBF16(py::object input) {
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

void RtKernelPy::SequenceAdd(const std::string &ker_type) {
  auto [type, flags] = ParseKernelType(ker_type);
  kernel_.SequenceAdd(type, flags);
}

void RtKernelPy::Tile(int start, int end, int64_t num, int64_t factor) {
  static_cast<VectorKernel *>(kernel_.GetImpl())->SetTile(start, end, num, factor);
}

void RtKernelPy::CodeGen(py::object pass_names) {
  const static std::unordered_map<std::string, pass::Pass> pass_map = {
    {"PrintPeakLive", pass::PrintPeakLive},       {"ReorderStore", pass::ReorderStore},
    {"ReorderLoad", pass::ReorderLoad},           {"CompactPeakLiveness", pass::CompactPeakLiveness},
    {"EliminateReshape", pass::EliminateReshape}, {"InsertRemovePad", pass::InsertRemovePad}};
  if (kernel_.GetImpl()->IsSplit()) {
    std::vector<RelocEntry> relocs;
    relocs.reserve(loads_.size() + stores_.size());
    for (auto &info : loads_) {
      relocs.emplace_back(info.op, info.dev);
    }
    kernel_.Infer();
    for (auto &info : stores_) {
      auto op = info.op;
      info.size = ITEM_SIZE[op->type_id_];
      for (size_t i = 0; i < op->shape_ref_->size; i++) {
        info.size *= op->shape_ref_->data[i];
      }
      runner_->AllocStore(info);
      relocs.emplace_back(op, info.dev);
    }
    kernel_.CodeGen(relocs.data(), relocs.size(), runner_);
    return;
  }
  uint64_t workspace_size;
  if (py::isinstance<py::list>(pass_names)) {
    std::vector<pass::Pass> old_passes;
    std::swap(old_passes, pass::passes);
    auto names = py::cast<py::list>(pass_names).cast<std::vector<std::string>>();
    for (auto name : names) {
      pass::passes.push_back(pass_map.at(name));
    }
    workspace_size = kernel_.CodeGen();
    std::swap(old_passes, pass::passes);
  } else {
    workspace_size = kernel_.CodeGen();
  }
  workspace_ = runner_->Alloc(workspace_size);
}

void RtKernelPy::InitComm(int rank_id, int rank_size, const std::string &comm_type) {
  int type_id;
  if (comm_type == "hccl") {
    type_id = Comm::kHccl;
  } else if (comm_type == "dummy") {
    type_id = Comm::kDummy;
  } else {
    type_id = Comm::kMemory;
  }
  g_mpc.rank_id = rank_id;
  g_mpc.rank_size = rank_size;
  g_mpc.comm.Init(rank_id, rank_size, type_id);
}

void RtKernelPy::Run() {
  if (!kernel_.GetImpl()->IsSplit()) {
    PrepareIO();
  }
  auto ret = runner_->Run(kernel_, workspace_, true);
  if (ret != 0) {
    std::cerr << kernel_.GetImpl()->DumpGraph() << std::endl;
    std::cerr << kernel_.GetImpl()->DisAssemble() << std::endl;
    std::cerr << "******** Kernel Execute Exception: " << ret << " ********" << std::endl;
    exit(0);
  }
}

void RtKernelPy::DryRun(int core_idx, bool cube_core) {
  static_cast<DryRunner *>(runner_)->DryRun(kernel_, workspace_, core_idx, cube_core);
}

py::object RtKernelPy::Perf() {
  // warm up
  if (!kernel_.GetImpl()->IsSplit()) {
    PrepareIO();
  }
  runner_->Run(kernel_, workspace_, true);
  RepeatProfiler profiler;
  profiler.Reset();
  for (size_t i = 0; i < TEST_NUM; i++) {
    ResetStoreMemory(stores_);
    profiler.RecordStart(nullptr);
    runner_->Run(kernel_, workspace_, false);
    profiler.RecordEnd(nullptr);
  }
  return py::make_tuple(py::float_(profiler.min_us_), py::float_(profiler.max_us_),
                        py::float_(profiler.total_us_ / float(TEST_NUM)));
}

py::object RtKernelPy::Msprof(const std::string &path, int64_t test_num) {
  ProfileMgr mgr(path);
  if (kernel_.GetImpl()->IsSplit()) {
    runner_->Run(kernel_, workspace_, true);
    mgr.ProfStart();
    while (test_num--) {
      ResetStoreMemory(stores_);
      kernel_.Launch(nullptr);
    }
  } else {
    constexpr const char *kTempFusionOp = "DvmOp";
    kernel_.SetNameHint(kTempFusionOp, kTempFusionOp);
    PrepareIO();
    runner_->Run(kernel_, workspace_, true);
    mgr.ProfStart();
    std::vector<RelocEntry> relocs;
    relocs.reserve(loads_.size() + stores_.size());
    std::vector<void *> inputs_addr;
    std::vector<void *> outputs_addr;
    std::vector<NDObject *> inputs;
    std::vector<NDObject *> outputs;
    for (auto &info : loads_) {
      relocs.emplace_back(info.op, info.dev);
    }
    for (auto &info : stores_) {
      relocs.emplace_back(info.op, info.dev);
    }
    while (test_num--) {
      ResetStoreMemory(stores_);
      ERROR_CHECK(kernel_.Launch(relocs.data(), relocs.size(), workspace_, nullptr));
    }
  }
  aclrtSynchronizeStream(nullptr);
  mgr.ProfStop();
  return py::none();
}

void RtKernelPy::Input(py::object load, py::object array) {
  auto op = static_cast<NDAccess *>(load.cast<NDOpPyPtr>()->Get());
  auto &info = FindVectorInfo(loads_, op);
  auto input = py::array(array);
  py::buffer_info buf = input.request();
  runner_->AllocLoad(buf, info);
  if (kernel_.GetImpl()->IsDynamic()) {
    info.shape.resize(buf.ndim);
    for (size_t i = 0; i < static_cast<size_t>(buf.ndim); ++i) {
      info.shape[i] = buf.shape[i];
    }
    *(op->shape_ref_) = info.shape;
  }
}

py::object RtKernelPy::Output(py::object store) {
  auto op = store.cast<NDOpPyPtr>()->Get();
  auto &info = FindVectorInfo(stores_, op);
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
  if (size > 0) {
    ERROR_CHECK(aclrtMemcpy(info.host, size, info.dev, size, ACL_MEMCPY_DEVICE_TO_HOST));
  }
  py::buffer_info buf(info.host, itemsize, GetBufferFormat(op->type_id_), ndim, shape, strides);
  return py::array(py::dtype(buf), buf.shape, buf.strides, buf.ptr, store);
}

void RtKernelPy::ClearStoreMemory(py::object store) {
  auto op = store.cast<NDOpPyPtr>()->Get();
  auto &info = FindVectorInfo(stores_, op);
  info.clear_mem = true;
}

void RtKernelPy::PrepareIO() {
  for (auto &info : loads_) {
    static_cast<NDAccess *>(info.op)->addr_.Reloc(info.dev);
  }
  for (auto &info : stores_) {
    auto op = info.op;
    info.size = ITEM_SIZE[op->type_id_];
    for (size_t i = 0; i < op->shape_ref_->size; i++) {
      info.size *= op->shape_ref_->data[i];
    }
    runner_->AllocStore(info);
    static_cast<NDAccess *>(op)->addr_.Reloc(info.dev);
  }
}

void RtKernelPy::Reset() {
  runner_->Reset();
  if (kernel_.GetImpl()->KType() == kEager) {
    kernel_.Clear();
    loads_.clear();
    stores_.clear();
    for (auto ref : shape_) {
      delete ref;
    }
    shape_.clear();
    shape_vec_.clear();
  }
}

void RtKernelPy::Fork(int size, const std::string &comm_type) {
  ASSERT(size <= static_cast<int>(sizeof(g_mpc.pids) / sizeof(pid_t)));
  g_mpc.rank_size = size;
  auto shm_size = 1024;
  g_mpc.shmid = ::shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0600);
  for (int i = 1; i < size; ++i) {
    g_mpc.rank_id = i;
    auto pid = ::fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
      goto INIT_COMM;
    }
    g_mpc.pids[i - 1] = pid;
  }
  g_mpc.rank_id = 0;
INIT_COMM:
  int rank_id = g_mpc.rank_id;
  g_mpc.bars = (int64_t *)shmat(g_mpc.shmid, nullptr, 0);
  g_mpc.bars[rank_id] = 0;
  ERROR_CHECK(aclrtSetDevice(rank_id));
  g_mpc.comm.Init(rank_id, size);
}

void RtKernelPy::Join() {
  // TODO: free g_mpc.comm
  ::shmdt((void *)(g_mpc.bars));
  if (g_mpc.rank_id > 0) {
    ::exit(0);
  }
  for (int i = 0; i < g_mpc.rank_size - 1; ++i) {
    ::wait(nullptr);
  }
  shmctl(g_mpc.shmid, IPC_RMID, nullptr);
}

void RtKernelPy::Barrier() {
  if (g_mpc.rank_size > 1 && g_mpc.bars) {
    auto cur_cnt = g_mpc.bars[g_mpc.rank_id] + 1;
    g_mpc.bars[g_mpc.rank_id] = cur_cnt;
    for (int i = 0; i < g_mpc.rank_size; ++i) {
      while (g_mpc.bars[i] != cur_cnt) {
        sleep(1);
      }
    }
  }
}

int RtKernelPy::RankId() { return g_mpc.rank_id; }

int RtKernelPy::RankSize() { return g_mpc.rank_size; }

class DevicePy {
 public:
  static std::string Arch() {
    g_system.Init();
    static const char *soc_names[] = {"AscendC220", "AscendC310"};
    return soc_names[g_system.Arch()];
  }
  static int CoreNum() {
    g_system.Init();
    return g_system.CoreNum();
  }
  static std::string SocName() {
    g_system.Init();
    static const char *soc_names[] = {"Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4", "Unknow"};
    return soc_names[g_system.SocName()];
  }
};

PYBIND11_MODULE(_dvm_py, m) {
  pyapi::RegBaseApi(m);
  pyapi::RegKernelApi(m);
  py::class_<RtKernelPy, KernelPy, std::shared_ptr<RtKernelPy>>(m, "Kernel")
    .def(py::init<const std::string &, const std::string &, int>())
    .def("slice_load", &RtKernelPy::SliceLoad, "load array")
    .def("stridedslice_load", &RtKernelPy::StridedSliceLoad, "load array")
    .def("multi_load", &RtKernelPy::MultiLoad, "load array(for reducescatter)")
    .def("pad_store", &RtKernelPy::PadStore, "pad store array")
    .def("set_store_inplace", &RtKernelPy::SetStoreInplace, "store inplace")
    .def("one_hot", &RtKernelPy::OneHot, "emit onehot op")
    .def("allreduce", &RtKernelPy::AllReduce, "emit allreduce op")
    .def("allgather", &RtKernelPy::AllGather, "emit allgather op")
    .def("allgatherv2", &RtKernelPy::AllGatherV2, "emit allgatherv2 op")
    .def("reducescatter", &RtKernelPy::ReduceScatter, "emit reducescatter op")
    .def("convert_to_bf16", &RtKernelPy::ConvertToBF16, "convert f32 array to bf16 array")
    .def("convert_from_bf16", &RtKernelPy::ConvertFromBF16, "convert bf16 array to f32 array")
    .def("seq_add", &RtKernelPy::SequenceAdd, "add new sequence Kernel")
    .def("reset", &RtKernelPy::Reset, "reset eager")
    .def("input", &RtKernelPy::Input, "get ouput array")
    .def("output", &RtKernelPy::Output, "get ouput array")
    .def("clear_store_memory", &RtKernelPy::ClearStoreMemory, "clear store memory")
    .def("tile", &RtKernelPy::Tile, "set tiling", py::arg("start"), py::arg("end"), py::arg("num"),
         py::arg("factor") = 0)
    .def("codegen", &RtKernelPy::CodeGen, "generate code")
    .def("perf", &RtKernelPy::Perf, "perf test")
    .def("msprof", &RtKernelPy::Msprof, "perf test")
    .def("run", &RtKernelPy::Run, "run kernel")
    .def("dry_run", &RtKernelPy::DryRun, "dry run vm")
    .def_static("init_comm", &RtKernelPy::InitComm, "init communicatior")
    .def_static("set_cube_store_type", &RtKernelPy::SetCubeStoreType, "set sync type")
    .def_static("fork", &RtKernelPy::Fork, "fork process", py::arg("size"), py::arg("comm_type") = "")
    .def_static("join", &RtKernelPy::Join, "join process")
    .def_static("barrier", &RtKernelPy::Barrier, "barrier process")
    .def_static("rank_id", &RtKernelPy::RankId, "get current rank id")
    .def_static("rank_size", &RtKernelPy::RankSize, "get current rank size");

  (void)py::class_<DevicePy, std::shared_ptr<DevicePy>>(m, "Device")
    .def_static("arch", &DevicePy::Arch, "Get system architecture")
    .def_static("core_num", &DevicePy::CoreNum, "Get soc core number")
    .def_static("soc_name", &DevicePy::SocName, "Get soc name");
}
}  // namespace dvm
