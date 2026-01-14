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

#ifndef _DVM_MSPROF_H_
#define _DVM_MSPROF_H_

#include <vector>
#include "dvm.h"
#include "system.h"
#include "acl/acl_rt.h"
#ifdef __CANN_85__
#include "profiling/prof_api.h"
#else
#include "experiment/msprof/toolchain/prof_api.h"
#include "experiment/msprof/toolchain/prof_data_config.h"
#endif

namespace dvm {
struct TensorInfoWrapper {
  MsprofAdditionalInfo tensor_info;
  uint64_t tensor_num;
};

struct ProfNodeAdditionInfo {
  MsprofCompactInfo node_basic_info;
  MsprofAdditionalInfo context_id_info;
  std::vector<TensorInfoWrapper> tensor_info_wrappers;
  MsprofApi api;
};

// reference: msprof/analysis/csrc/viewer/database/drafts/number_mapping.cpp
enum TensorDtypeMs : int {
  FLOAT = 0,
  FLOAT16 = 1,
  INT8 = 2,
  INT32 = 3,
  UINT8 = 4,
  INT16 = 6,
  UINT16 = 7,
  UINT32 = 8,
  INT64 = 9,
  UINT64 = 10,
  DOUBLE = 11,
  BOOL = 12,
  STRING = 13,
  DUAL_SUB_INT8 = 14,
  DUAL_SUB_UINT8 = 15,
  COMPLEX64 = 16,
  COMPLEX128 = 17,
  QINT8 = 18,
  QINT16 = 19,
  QINT32 = 20,
  QUINT8 = 21,
  QUINT16 = 22,
  RESOURCE = 23,
  STRING_REF = 24,
  DUAL = 25,
  DT_VARIANT = 26,
  DT_BF16 = 27,
  UNDEFINED = 28,
  DT_INT4 = 29,
  DT_UINT1 = 30,
  DT_INT2 = 31,
  DT_UINT2 = 32,
  DT_COMPLEX32 = 33,
  DT_MAX = 34,
  NUMBER_TYPE_BEGIN_ = 229,
  BOOL_ = 230,
  INT_ = 231,
  INT8_ = 232,
  INT16_ = 233,
  INT32_ = 234,
  INT64_ = 235,
  UINT_ = 236,
  UINT8_ = 237,
  UINT16_ = 238,
  UINT32_ = 239,
  UINT64_ = 240,
  FLOAT_ = 241,
  FLOAT16_ = 242,
  FLOAT32_ = 243,
  FLOAT64_ = 244,
  COMPLEX_ = 245,
  NUMBER_TYPE_END_ = 246,
};
// reference: msprof/analysis/csrc/viewer/database/drafts/number_mapping.cpp
enum FormatMs : int {
  UNKNOWN_ = 200,
  DEFAULT_ = 201,
  NC1KHKWHWC0_ = 202,
  ND_ = 203,
  NCHW_ = 204,
  NHWC_ = 205,
  HWCN_ = 206,
  NC1HWC0_ = 207,
  FRAC_Z_ = 208,
  C1HWNCOC0_ = 209,
  FRAC_NZ_ = 210,
  NC1HWC0_C04_ = 211,
  FRACTAL_Z_C04_ = 212,
  NDHWC_ = 213,
  FRACTAL_ZN_LSTM_ = 214,
  FRACTAL_ZN_RNN_ = 215,
  ND_RNN_BIAS_ = 216,
  NDC1HWC0_ = 217,
  NCDHW_ = 218,
  FRACTAL_Z_3D_ = 219,
  DHWNC_ = 220,
  DHWCN_ = 221,
};
extern const TensorDtypeMs MAP_DTYPE_TO_MSDTYPE[];

struct NodeInfo {
  const char *op_name;
  const char *op_fullname;
  uint64_t input_size{0};
  uint64_t output_size{0};
  uint32_t block_dim;
  std::vector<ShapeRef *> shapes;
  std::vector<TensorDtypeMs> data_types;

  void AppendInput(NDObject *op) {
    shapes.emplace_back(op->shape_ref_);
    data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[op->type_id_]);
    input_size++;
  }

  void AppendOutput(NDObject *op) {
    shapes.emplace_back(op->shape_ref_);
    data_types.emplace_back(MAP_DTYPE_TO_MSDTYPE[op->type_id_]);
    output_size++;
  }
};

template <typename T>
class ScopedValueGuard {
 public:
  ScopedValueGuard(T &target, T new_value) : target_(target), old_value_(target) { target_ = std::move(new_value); }

  ~ScopedValueGuard() { target_ = std::move(old_value_); }

 private:
  T &target_;
  T old_value_;
};

class MsprofHelper {
 public:
  MsprofHelper() = default;
  ~MsprofHelper() = default;

  void InitReportNode();
  void UpdateReportNode(uint32_t block_dim);
  void Update(uint32_t kernel_target);
  void ReportTask();

  NodeInfo info_;

 private:
  void InitProfTensorData(const size_t index, const uint64_t offset_idx, MsprofTensorInfo *tensor_info);
  void BuildSingleTensorInfo(const uint64_t opName_hash_id, const size_t index_begin, const size_t index_end,
                             TensorInfoWrapper *tensor_info_wrapper);
  void UpdateTensorShape(const size_t index_begin, const size_t index_end, TensorInfoWrapper *tensor_info_wrapper);

  ProfNodeAdditionInfo addition_info_;
};

class TimeProfiler {
 public:
  TimeProfiler();
  ~TimeProfiler();
  void RecordStart(void *stream) { ERROR_CHECK(aclrtRecordEvent(start_, stream)); }
  float RecordEnd(void *stream) {
    ERROR_CHECK(aclrtRecordEvent(end_, stream));
    ERROR_CHECK(aclrtSynchronizeStream(stream));
    float time_us = 0.0f;
    ERROR_CHECK(aclrtEventElapsedTime(&time_us, start_, end_));
    return time_us;
  }
 protected:
  aclrtEvent start_, end_;
};

class RepeatProfiler : public TimeProfiler {
 public:
  void Reset() {
    min_us_ = 1e6;
    max_us_ = 0.0f;
    total_us_ = 0.0f;
  }
  void RecordEnd(void *stream) {
    float time_us = TimeProfiler::RecordEnd(stream);
    time_us *= 1000.0;
    min_us_ = std::min(min_us_, time_us);
    max_us_ = std::max(max_us_, time_us);
    total_us_ += time_us;
  }
  float min_us_;
  float max_us_;
  float total_us_;
};
}  // namespace dvm
#endif  // _DVM_MSPROF_H_
