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

#include <map>
#include <algorithm>
#include <dlfcn.h>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include "kernel.h"
#include "msprof.h"

using ShapeVector = std::vector<int64_t>;

// GE task info task_type
enum class TaskInfoTaskType {
  TASK_TYPE_AI_CORE = 0,
  TASK_TYPE_AI_CPU = 1,
  TASK_TYPE_AIV = 2,
  TASK_TYPE_WRITE_BACK = 3,
  TASK_TYPE_MIX_AIC = 4,
  TASK_TYPE_MIX_AIV = 5,
  TASK_TYPE_FFTS_PLUS = 6,
  TASK_TYPE_DSA = 7,
  TASK_TYPE_DVPP = 8,
  TASK_TYPE_HCCL = 9,
  MSPROF_RTS = 11,
  MSPROF_UNKNOWN_TYPE = 1000,
};

constexpr uint32_t kTensorInfoBytes = 44UL;
constexpr uint32_t kTensorInfoBytesWithCap = 56U;

namespace dvm {
const TensorDtypeMs MAP_DTYPE_TO_MSDTYPE[DType::kTypeEnd + 1] = {BOOL_,    FLOAT16_, DT_BF16,
                                                                 FLOAT32_, INT32_,   NUMBER_TYPE_END_};
class MsProfHolder {
 public:
  MsProfHolder();
  ~MsProfHolder() = default;

  static MsProfHolder &Instance() {
    static MsProfHolder instance;
    return instance;
  }
  uint64_t (*msprof_sys_cycle_time_)();
  uint64_t (*msprof_get_hash_id_)(const char *hashInfo, size_t length);
  int32_t (*msprof_report_api_)(uint32_t agingFlag, const MsprofApi *api);
  int32_t (*msprof_report_compact_info_)(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
  int32_t (*msprof_report_additional_info_)(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
};

MsProfHolder::MsProfHolder() {
#ifndef VK_SIM_MODEL
  void *handle = dlopen("libprofapi.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(handle == nullptr, "Load libprofapi.so failed");
  msprof_sys_cycle_time_ = reinterpret_cast<uint64_t (*)()>(dlsym(handle, "MsprofSysCycleTime"));
  EXCEPTION_IF(msprof_sys_cycle_time_ == nullptr, "load msprof_sys_cycle_time symbol failed");
  msprof_get_hash_id_ =
    reinterpret_cast<uint64_t (*)(const char *hashInfo, size_t length)>(dlsym(handle, "MsprofGetHashId"));
  EXCEPTION_IF(msprof_get_hash_id_ == nullptr, "load msprof_get_hash_id symbol failed");
  msprof_report_api_ =
    reinterpret_cast<int32_t (*)(uint32_t agingFlag, const MsprofApi *api)>(dlsym(handle, "MsprofReportApi"));
  EXCEPTION_IF(msprof_report_api_ == nullptr, "load msprof_report_api symbol failed");
  msprof_report_compact_info_ = reinterpret_cast<int32_t (*)(uint32_t agingFlag, const VOID_PTR data, uint32_t length)>(
    dlsym(handle, "MsprofReportCompactInfo"));
  EXCEPTION_IF(msprof_report_compact_info_ == nullptr, "load msprof_report_compact_info symbol failed");
  msprof_report_additional_info_ =
    reinterpret_cast<int32_t (*)(uint32_t agingFlag, const VOID_PTR data, uint32_t length)>(
      dlsym(handle, "MsprofReportAdditionalInfo"));
  EXCEPTION_IF(msprof_report_additional_info_ == nullptr, "load msprof_report_additional_info symbol failed");
#endif
}

void InitLaunchApi(const uint64_t name_hash, MsprofApi *api) {
  const auto kernel_type_hash = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  api->type = kernel_type_hash;
  api->level = MSPROF_REPORT_NODE_LEVEL;
  api->itemId = name_hash;
}
uint64_t GetMsprofHashId(const char *info) {
  uint64_t hash_id = MsProfHolder::Instance().msprof_get_hash_id_(info, strlen(info));
  return hash_id;
}

void MsProfHelper::BuildSingleTensorInfo(const uint64_t opName_hash_id, const size_t index_begin,
                                         const size_t index_end, TensorInfoWrapper *tensor_info_wrapper) {
  auto &tensor_info = tensor_info_wrapper->tensor_info;
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = MSPROF_REPORT_NODE_LEVEL;
  tensor_info_wrapper->tensor_num = index_end - index_begin;
  tensor_info.dataLen =
    kTensorInfoBytesWithCap + kTensorInfoBytes * (static_cast<uint32_t>(tensor_info_wrapper->tensor_num) - 1U);
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  prof_tensor_data->opName = opName_hash_id;
  prof_tensor_data->tensorNum = tensor_info_wrapper->tensor_num;
  for (size_t tensor_index = index_begin; tensor_index < index_end; tensor_index++) {
    size_t k = tensor_index - index_begin;
    prof_tensor_data->tensorData[k].tensorType =
      tensor_index < info_.input_size ? MSPROF_GE_TENSOR_TYPE_INPUT : MSPROF_GE_TENSOR_TYPE_OUTPUT;
    prof_tensor_data->tensorData[k].format = FormatMs::ND_;
    prof_tensor_data->tensorData[k].dataType = info_.data_types[tensor_index];
    auto shape_size =
      std::min(static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), info_.shapes[tensor_index]->size);
    memset(prof_tensor_data->tensorData[k].shape, 0, sizeof(prof_tensor_data->tensorData[k].shape));
    (void)std::transform(info_.shapes[tensor_index]->data, info_.shapes[tensor_index]->data + shape_size,
                         prof_tensor_data->tensorData[k].shape,
                         [](uint64_t value) { return static_cast<uint32_t>(value); });
  }
}

void MsProfHelper::UpdateTensorShape(const size_t index_begin, const size_t index_end,
                                     TensorInfoWrapper *tensor_info_wrapper) {
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info_wrapper->tensor_info.data);
  for (size_t tensor_index = index_begin; tensor_index < index_end; tensor_index++) {
    size_t k = tensor_index - index_begin;
    auto shape_size =
      std::min(static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), info_.shapes[tensor_index]->size);
    memset(prof_tensor_data->tensorData[k].shape, 0, sizeof(prof_tensor_data->tensorData[k].shape));
    (void)std::transform(info_.shapes[tensor_index]->data, info_.shapes[tensor_index]->data + shape_size,
                         prof_tensor_data->tensorData[k].shape,
                         [](uint64_t value) { return static_cast<uint32_t>(value); });
  }
}

void MsProfHelper::InitReportNode() {
  MsprofCompactInfo &basic_info = addition_info_.node_basic_info;
  basic_info.level = MSPROF_REPORT_NODE_LEVEL;
  basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  auto &prof_node_basic_info = basic_info.data.nodeBasicInfo;
  uint64_t opName_hash_id = GetMsprofHashId(info_.op_fullname);
  prof_node_basic_info.opName = opName_hash_id;
  prof_node_basic_info.blockDim = info_.block_dim;
  prof_node_basic_info.opType = GetMsprofHashId(info_.op_name);
  prof_node_basic_info.taskType = (info_.kernel_type == kStaticMix || info_.kernel_type == kStaticStages)
                                    ? static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_MIX_AIC)
                                    : static_cast<uint32_t>(TaskInfoTaskType::TASK_TYPE_AI_CORE);
  size_t total_size = info_.input_size + info_.output_size;
  for (size_t i = 0U; i < total_size; i += MSPROF_GE_TENSOR_DATA_NUM) {
    TensorInfoWrapper tensor_info_wrapper;
    BuildSingleTensorInfo(opName_hash_id, i, std::min(total_size, (i + MSPROF_GE_TENSOR_DATA_NUM)),
                          &tensor_info_wrapper);
    addition_info_.tensor_info_wrappers.emplace_back(tensor_info_wrapper);
  }
  InitLaunchApi(opName_hash_id, &addition_info_.api);
}

void MsProfHelper::UpdateReportNode(uint32_t block_dim) {
  addition_info_.node_basic_info.data.nodeBasicInfo.blockDim = block_dim;
  size_t total_size = info_.input_size + info_.output_size;
  for (size_t i = 0U; i < total_size; i += MSPROF_GE_TENSOR_DATA_NUM) {
    UpdateTensorShape(i, std::min(total_size, (i + MSPROF_GE_TENSOR_DATA_NUM)),
                      &addition_info_.tensor_info_wrappers[i / MSPROF_GE_TENSOR_DATA_NUM]);
  }
}

void MsProfHelper::UpdateBeginTime() {
  addition_info_.api.beginTime = MsProfHolder::Instance().msprof_sys_cycle_time_();
}

void MsProfHelper::ReportTask() {
  const uint64_t prof_time = MsProfHolder::Instance().msprof_sys_cycle_time_();
  addition_info_.node_basic_info.timeStamp = prof_time;
  auto tid = syscall(SYS_gettid);
  addition_info_.node_basic_info.threadId = static_cast<uint32_t>(tid);

  auto compact_ret = MsProfHolder::Instance().msprof_report_compact_info_(false, &addition_info_.node_basic_info,
                                                                          sizeof(MsprofCompactInfo));
  EXCEPTION_IF(compact_ret != MSPROF_ERROR_NONE, "MsprofReportCompactInfo failed.");

  for (auto &tensor_info_wrapper : addition_info_.tensor_info_wrappers) {
    tensor_info_wrapper.tensor_info.timeStamp = prof_time;
    tensor_info_wrapper.tensor_info.threadId = static_cast<uint32_t>(tid);
    auto addition_ret = MsProfHolder::Instance().msprof_report_additional_info_(false, &tensor_info_wrapper.tensor_info,
                                                                                sizeof(MsprofAdditionalInfo));
    EXCEPTION_IF(addition_ret != MSPROF_ERROR_NONE, "MsprofReportAdditionalInfo failed.");
  }
  addition_info_.api.endTime = prof_time;
  addition_info_.api.threadId = static_cast<uint32_t>(tid);
  auto api_ret = MsProfHolder::Instance().msprof_report_api_(false, &addition_info_.api);
  EXCEPTION_IF(api_ret != MSPROF_ERROR_NONE, "MsprofReportAdditionalInfo failed.");
}
}  // namespace dvm