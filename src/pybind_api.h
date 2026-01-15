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
#include "dvm_py.h"

namespace dvm {
class KernelRunner;
class RtKernelPy : public KernelPy {
 public:
  friend class ShapeRefPy;

  RtKernelPy(const std::string &ker_type, const std::string &run_type, int dev_id);
  ~RtKernelPy();

  py::object Load(py::object shape, const std::string &type) override;
  py::object ViewLoad(py::object shape, py::object stride, int64_t offset, const std::string &type) override;
  py::object Store(py::object obj) override;

  py::object SliceLoad(py::object shape, py::object start, py::object size, const std::string &type);
  py::object StridedSliceLoad(py::object shape, py::object start, py::object end, py::object step,
                              const std::string &type);
  py::object MultiLoad(py::object shape, const std::string &type);
  py::object PadStore(py::object obj, int64_t pad_shape);
  void SetStoreInplace(py::object store);
  py::object OneHot(py::object indices, int depth, int axis, py::object on_value, py::object off_value,
                    const std::string &dtype);
  py::object AllReduce(const std::string &type, py::object input);
  py::object AllGather(py::object input);
  py::object AllGatherV2(py::object input);
  py::object ReduceScatter(py::object input);
  py::object ConvertToBF16(py::object input);
  py::object ConvertFromBF16(py::object input);
  void SequenceAdd(const std::string &ker_type);

  void Reset();
  IntArrayRef *GetShapeRef(py::object shape) override;
  py::object Clone(py::object base, py::object remap);

  void Input(py::object obj, py::object val);
  py::object Output(py::object store);
  void ClearStoreMemory(py::object store);
  void Tile(int start, int end, int64_t num, int64_t factor);
  void CodeGen(py::object pass_names);
  void Run();

  void DryRun(int core_idx, bool cube_core);
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
  void PrepareIO();
  std::vector<std::vector<int64_t>> shape_vec_;
  std::vector<IntArrayRef *> shape_;
  std::vector<LoadInfo> loads_;
  std::vector<StoreInfo> stores_;
  std::vector<std::vector<float>> f32s_;      // store f32 converted from bf16
  std::vector<std::vector<uint16_t>> bf16s_;  // store bf16 converted from f32

  void *workspace_{nullptr};
  KernelRunner *runner_;
};
using RtKernelPyPtr = std::shared_ptr<RtKernelPy>;
}  // namespace dvm
#endif  // _DVM_PYBIND_API_H_
