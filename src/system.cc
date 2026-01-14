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

#include <dlfcn.h>
#include <stdexcept>
#include <sstream>
#include "acl/acl_rt.h"
#include "system.h"
#include "code.h"
#include "tuning.h"
#ifdef __CANN_85__
#include "profiling/prof_api.h"
#else
#include "experiment/msprof/toolchain/prof_api.h"
#include "experiment/msprof/toolchain/prof_data_config.h"
#endif

// rts_runtime
#if defined(__cplusplus)
extern "C" {
#endif
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243U
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246U
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

typedef struct tagRtDevBinary {
  uint32_t magic;    // magic number
  uint32_t version;  // version of binary
  const void *data;  // binary data
  uint64_t length;   // binary length
} rtDevBinary_t;

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);
rtError_t rtDevBinaryUnRegister(void *hdl);
rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName, const void *kernelInfoExt,
                             uint32_t funcMode);
rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc,
                         rtStream_t stm);
rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
#if defined(__cplusplus)
}
#endif

#ifdef VK_SIM_MODEL
#include "hccl/hccl.h"

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo) { return HCCL_SUCCESS; }
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm) {
  return HCCL_SUCCESS;
}
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank) { return HCCL_SUCCESS; }
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize) { return HCCL_SUCCESS; }
HcclResult HcclCommDestroy(HcclComm comm) { return HCCL_SUCCESS; }

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream) {
  return HCCL_SUCCESS;
}
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                         aclrtStream stream) {
  return HCCL_SUCCESS;
}
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream) {
  return HCCL_SUCCESS;
}
#endif

extern const uint64_t g_visit_func_offset_c310[];
extern const uint64_t g_simd_func_offset_c310[];
extern const uint64_t g_access_func_offset_c310[];

extern const uint64_t g_visit_func_offset_c220[];
extern const uint64_t g_simd_func_offset_c220[];
extern const uint64_t g_access_func_offset_c220[];

extern const unsigned char g_vkernel_c310_bin[];
extern unsigned int g_vkernel_c310_bin_len;

extern const unsigned char g_vkernel_c220_bin[];
extern unsigned int g_vkernel_c220_bin_len;
namespace dvm {
// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kDataTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t), sizeof(int64_t)};
const char *DTYPE_NAMES[dvm::kDataTypeEnd] = {"bool", "float16", "bfloat16", "float32", "int32", "int64"};
const uint64_t ITEM_SIMD_WIDTH_MAX[kDataTypeEnd] = {128, 128, 128, 64, 64};

static std::string GetSocName() {
  const char *soc_name = getenv("DVM_SOC_NAME");
  if (soc_name == nullptr) {
    soc_name = aclrtGetSocName();
    ASSERT(soc_name != nullptr);
  }
  return soc_name;
}

void DvmException(const char *error_str) {
  std::ostringstream oss;
  oss << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}

using ProfCommandHandle = int32_t (*)(uint32_t type, VOID_PTR data, uint32_t len);
int32_t ProfCommandHandler(uint32_t type, VOID_PTR data, uint32_t len) {
  if (data == nullptr) {
    return -1;
  }
  if (type != PROF_CTRL_SWITCH) {
    return -1;
  }
  if (len < sizeof(MsprofCommandHandle)) {
    return -1;
  }
  MsprofCommandHandle *profilerConfig = static_cast<MsprofCommandHandle *>(data);
  const uint64_t profSwitch = profilerConfig->profSwitch;
  const uint32_t profType = profilerConfig->type;
  if (profType == PROF_COMMANDHANDLE_TYPE_START) {
    g_system.enable_profile_ = true;
  }
  if (profType == PROF_COMMANDHANDLE_TYPE_STOP) {
    g_system.enable_profile_ = false;
  }
  if ((profSwitch & PROF_TASK_TIME_MASK) != 0) {
    g_system.profiler_level_ = Level0;
  }
  if ((profSwitch & PROF_TASK_TIME_L1_MASK) != 0) {
    g_system.profiler_level_ = Level1;
  }
  return 0;
}

struct SocConfig {
  const char *name;
  SocType type;
  AiCoreArch arch;
  uint64_t aicore_num;
  uint64_t l2_size;
};

constexpr uint64_t MB = 1024 * 1024;
const SocConfig soc_configs[] = {
  {"Ascend910B1", kAscend910B1, kAiCore_C220, 25, 192 * MB},
  {"Ascend910B2", kAscend910B2, kAiCore_C220, 24, 192 * MB},
  {"Ascend910B2C", kAscend910B2, kAiCore_C220, 24, 192 * MB},
  {"Ascend910B3", kAscend910B3, kAiCore_C220, 20, 192 * MB},
  {"Ascend910B4", kAscend910B4, kAiCore_C220, 20, 96 * MB},
  {"Ascend910B4-1", kAscend910B4, kAiCore_C220, 20, 96 * MB},
  {"Ascend910_9391", kAscend910_9391, kAiCore_C220, 25, 192 * MB},
  {"Ascend910_9392", kAscend910_9392, kAiCore_C220, 25, 192 * MB},
  {"Ascend910_9381", kAscend910_9381, kAiCore_C220, 24, 192 * MB},
  {"Ascend910_9382", kAscend910_9382, kAiCore_C220, 24, 192 * MB},
  {"Ascend910_9372", kAscend910_9372, kAiCore_C220, 20, 192 * MB},
  {"Ascend910_9361", kAscend910_9361, kAiCore_C220, 20, 96 * MB},
  {"Ascend910_9589", kAscend910_9589, kAiCore_C310, 32, 128 * MB},
  {"Ascend910_9599", kAscend910_9599, kAiCore_C310, 36, 128 * MB},
};

void System::DoInit() {
  inited_ = true;
  const SocConfig *config = nullptr;
  auto soc_name = GetSocName();
  for (const SocConfig &c : soc_configs) {
    if (soc_name == c.name) {
      config = &c;
      break;
    }
  }
  EXCEPTION_IF(config == nullptr, "Unrecognized SoC Version.");
  soc_name_ = config->type;

  arch_ = config->arch;
  event_num_ = 8;
  cube_core_num_ = config->aicore_num;
  vector_core_num_ = cube_core_num_ * 2;
  l2_size_ = config->l2_size;
  l1_size_ = 512 * 1024;
  const unsigned char *g_vkernel_bin = nullptr;
  unsigned int g_vkernel_bin_len = 0;
  if (arch_ == kAiCore_C220) {
    ub_workspace_size_ = 512;
    g_access_func_offset_ = g_access_func_offset_c220;
    g_simd_func_offset_ = g_simd_func_offset_c220;
    g_visit_func_offset_ = g_visit_func_offset_c220;
    g_vkernel_bin = g_vkernel_c220_bin;
    g_vkernel_bin_len = g_vkernel_c220_bin_len;
    l0c_size_ = 128 * 1024;
    local_mem_size_ = 192 * 1024 - ub_workspace_size_;
  } else if (arch_ == kAiCore_C310) {
    ub_workspace_size_ = 256;
    g_access_func_offset_ = g_access_func_offset_c310;
    g_simd_func_offset_ = g_simd_func_offset_c310;
    g_visit_func_offset_ = g_visit_func_offset_c310;
    g_vkernel_bin = g_vkernel_c310_bin;
    g_vkernel_bin_len = g_vkernel_c310_bin_len;
    l0c_size_ = 256 * 1024;
    local_mem_size_ = 256 * 1024 - ub_workspace_size_;
  }
#ifdef VK_SIM_MODEL
  auto rt_binary_register = rtDevBinaryRegister;
  auto rt_function_register = rtFunctionRegister;
  rt_kernel_launch_ = ::rtKernelLaunch;
  rt_get_c2c_addr_ = ::rtGetC2cCtrlAddr;
#else
  rt_handle_ = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(rt_handle_ == nullptr, "Load libruntime.so failed");
  auto rt_binary_register =
    reinterpret_cast<rtError_t (*)(const rtDevBinary_t *, void **)>(dlsym(rt_handle_, "rtDevBinaryRegister"));
  EXCEPTION_IF(rt_binary_register == nullptr, "load rt_binary_register symbol failed");
  auto rt_function_register =
    reinterpret_cast<rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t)>(
      dlsym(rt_handle_, "rtFunctionRegister"));
  EXCEPTION_IF(rt_function_register == nullptr, "load rt_function_register symbol failed");
  rt_kernel_launch_ =
    reinterpret_cast<rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t)>(
      dlsym(rt_handle_, "rtKernelLaunch"));
  EXCEPTION_IF(!rt_kernel_launch_, "load rt_kernel_launch symbol failed");
  rt_get_c2c_addr_ = reinterpret_cast<rtError_t (*)(uint64_t *, uint32_t *)>(dlsym(rt_handle_, "rtGetC2cCtrlAddr"));
  EXCEPTION_IF(!rt_get_c2c_addr_, "load rt_get_c2c_addr_ symbol failed");

  auto prof_handle = dlopen("libprofapi.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(prof_handle == nullptr, "Load libprofapi.so failed");
  msprof_sys_cycle_time_ = reinterpret_cast<uint64_t (*)()>(dlsym(prof_handle, "MsprofSysCycleTime"));
  EXCEPTION_IF(msprof_sys_cycle_time_ == nullptr, "load msprof_sys_cycle_time symbol failed");
  msprof_get_hash_id_ =
    reinterpret_cast<uint64_t (*)(const char *hashInfo, size_t length)>(dlsym(prof_handle, "MsprofGetHashId"));
  EXCEPTION_IF(msprof_get_hash_id_ == nullptr, "load msprof_get_hash_id symbol failed");
  msprof_report_api_ =
    reinterpret_cast<int32_t (*)(uint32_t agingFlag, const MsprofApi *api)>(dlsym(prof_handle, "MsprofReportApi"));
  EXCEPTION_IF(msprof_report_api_ == nullptr, "load msprof_report_api symbol failed");
  msprof_report_compact_info_ = reinterpret_cast<int32_t (*)(uint32_t agingFlag, const VOID_PTR data, uint32_t length)>(
    dlsym(prof_handle, "MsprofReportCompactInfo"));
  EXCEPTION_IF(msprof_report_compact_info_ == nullptr, "load msprof_report_compact_info symbol failed");
  msprof_report_additional_info_ =
    reinterpret_cast<int32_t (*)(uint32_t agingFlag, const VOID_PTR data, uint32_t length)>(
      dlsym(prof_handle, "MsprofReportAdditionalInfo"));
  EXCEPTION_IF(msprof_report_additional_info_ == nullptr, "load msprof_report_additional_info symbol failed");
  auto msprof_register_callback_ = reinterpret_cast<int32_t (*)(uint32_t moduleId, ProfCommandHandle prof_handle)>(
    dlsym(prof_handle, "MsprofRegisterCallback"));
  EXCEPTION_IF(msprof_register_callback_ == nullptr, "load msprof_register_callback symbol failed");

  auto ret = msprof_register_callback_(0, ProfCommandHandler);
  EXCEPTION_IF(ret != MSPROF_ERROR_NONE, "MsprofRegisterCallBack failed.");
#endif
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  dev_bin.data = g_vkernel_bin;
  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  dev_bin.length = g_vkernel_bin_len;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
  uint8_t *stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetVec;
  err = rt_function_register(module, stub_func, "dvm_mix_aiv", "dvm_mix_aiv", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetCube;
  err = rt_function_register(module, stub_func, "dvm_mix_aic", "dvm_mix_aic", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetMix;
  err = rt_function_register(module, stub_func, "dvm", "dvm", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");
}

System::~System() {
  if (comm_stream_) {
    aclrtDestroyStream(comm_stream_);
  }
  if (online_tuner_) {
    delete online_tuner_;
  }
  if (lazy_tuner_) {
    delete lazy_tuner_;
  }
#ifndef VK_SIM_MODEL
  if (rt_handle_) {
    dlclose(rt_handle_);
  }
#endif
}

void *System::CreateStream() {
  aclrtStream stream;
  auto ret = aclrtCreateStream(&stream);
  EXCEPTION_IF(ret != 0, "aclrtCreateStream");
  return stream;
}

Config &System::SetDeterm() {
  deterministic_ = true;
  return *this;
}

Config &System::UnsetDeterm() {
  deterministic_ = false;
  return *this;
}

Config &System::SetOnlineTuner() {
  if (online_tuner_ == nullptr) {
    online_tuner_ = new OnlineCubeTuner();
  }
  return *this;
}

Config &System::UnsetOnlineTuner() {
  delete online_tuner_;
  online_tuner_ = nullptr;
  return *this;
}

Config &System::SetLazyTuner() {
  if (lazy_tuner_ == nullptr) {
    lazy_tuner_ = new LazyCubeTuner();
  }
  return *this;
}

Config &System::UnsetLazyTuner() {
  delete lazy_tuner_;
  lazy_tuner_ = nullptr;
  return *this;
}

System g_system;
}  // namespace dvm
