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
#include <fstream>
#include "acl/acl_rt.h"
#include "system.h"
#include "code.h"
#include "ops_m.h"
#include "profiling/prof_api.h"
#include "pass.h"

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

extern const uint64_t g_visit_func_offset_c310[];
extern const uint64_t g_simd_func_offset_c310[];
extern const uint64_t g_access_func_offset_c310[];

extern const uint64_t g_visit_func_offset_c220[];
extern const uint64_t g_simd_func_offset_c220[];
extern const uint64_t g_access_func_offset_c220[];

extern const unsigned char g_vkernel_c310_bin[];
extern unsigned int g_vkernel_c310_bin_len;
extern const uint64_t g_mix_symbols_c310[];
extern const unsigned int g_mix_symbol_len_c310;

extern const unsigned char g_vkernel_c220_bin[];
extern unsigned int g_vkernel_c220_bin_len;
extern const uint64_t g_mix_symbols_c220[];
extern const unsigned int g_mix_symbol_len_c220;

extern const unsigned int g_meta_aiv_type_value_offset_c310;

namespace dvm {
// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t), sizeof(int64_t)}
const uint64_t ITEM_SIZE[DataType::kDataTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t), sizeof(int64_t)};
const char *DTYPE_NAMES[DataType::kDataTypeEnd] = {"bool", "float16", "bfloat16", "float32", "int32", "int64"};
const uint64_t ITEM_SIMD_WIDTH_MAX[DataType::kDataTypeEnd] = {128, 128, 128, 64, 64, 32};

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
  {"Ascend950PR_9579", kAscend950PR_9579, kAiCore_C310, 28, 128 * MB},
  {"Ascend950PR_9589", kAscend950PR_9589, kAiCore_C310, 32, 128 * MB},
  {"Ascend950PR_9599", kAscend950PR_9599, kAiCore_C310, 36, 128 * MB},
  {"Ascend950PR_958b", kAscend950PR_958b, kAiCore_C310, 32, 112 * MB},
  {"Ascend950PR_957b", kAscend950PR_957b, kAiCore_C310, 28, 112 * MB},
  {"Ascend950PR_957c", kAscend950PR_957c, kAiCore_C310, 28, 112 * MB},
  {"Ascend950PR_957d", kAscend950PR_957d, kAiCore_C310, 28, 96 * MB},
  {"Ascend950PR_950z", kAscend950PR_950z, kAiCore_C310, 4, 16 * MB},
  {"Ascend950DT_950x", kAscend950DT_950x, kAiCore_C310, 8, 32 * MB},
  {"Ascend950DT_950y", kAscend950DT_950y, kAiCore_C310, 8, 32 * MB},
  {"Ascend950DT_9571", kAscend950DT_9571, kAiCore_C310, 28, 128 * MB},
  {"Ascend950DT_9572", kAscend950DT_9572, kAiCore_C310, 28, 128 * MB},
  {"Ascend950DT_9573", kAscend950DT_9573, kAiCore_C310, 28, 112 * MB},
  {"Ascend950DT_9574", kAscend950DT_9574, kAiCore_C310, 28, 112 * MB},
  {"Ascend950DT_9575", kAscend950DT_9575, kAiCore_C310, 28, 128 * MB},
  {"Ascend950DT_9576", kAscend950DT_9576, kAiCore_C310, 28, 128 * MB},
  {"Ascend950DT_9577", kAscend950DT_9577, kAiCore_C310, 28, 112 * MB},
  {"Ascend950DT_9578", kAscend950DT_9578, kAiCore_C310, 28, 112 * MB},
  {"Ascend950DT_9581", kAscend950DT_9581, kAiCore_C310, 32, 128 * MB},
  {"Ascend950DT_9582", kAscend950DT_9582, kAiCore_C310, 32, 128 * MB},
  {"Ascend950DT_9583", kAscend950DT_9583, kAiCore_C310, 32, 112 * MB},
  {"Ascend950DT_9584", kAscend950DT_9584, kAiCore_C310, 32, 112 * MB},
  {"Ascend950DT_9585", kAscend950DT_9585, kAiCore_C310, 32, 128 * MB},
  {"Ascend950DT_9586", kAscend950DT_9586, kAiCore_C310, 32, 128 * MB},
  {"Ascend950DT_9587", kAscend950DT_9587, kAiCore_C310, 32, 112 * MB},
  {"Ascend950DT_9588", kAscend950DT_9588, kAiCore_C310, 32, 112 * MB},
  {"Ascend950DT_9591", kAscend950DT_9591, kAiCore_C310, 36, 128 * MB},
  {"Ascend950DT_9592", kAscend950DT_9592, kAiCore_C310, 36, 128 * MB},
  {"Ascend950DT_9595", kAscend950DT_9595, kAiCore_C310, 36, 128 * MB},
  {"Ascend950DT_9596", kAscend950DT_9596, kAiCore_C310, 36, 128 * MB},
  {"Ascend950DT_95A1", kAscend950DT_95A1, kAiCore_C310, 36, 128 * MB},
  {"Ascend950DT_95A2", kAscend950DT_95A2, kAiCore_C310, 36, 128 * MB},
};

static void GetAclCoreCount(int device_id, uint64_t *aic, uint64_t *aiv) {
  int64_t reported_aic = 0;
  auto aic_ret = aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_AI_CORE_NUM, &reported_aic);
  if (aic_ret != ACL_SUCCESS || reported_aic <= 0) return;
  *aic = static_cast<uint64_t>(reported_aic);
  *aiv = *aic * 2;
}

const char *AiCoreArchName(AiCoreArch arch) {
  static const char *arch_names[] = {"AscendC220", "AscendC310"};
  return arch_names[arch];
}

const char *SocTypeName(SocType type) {
  for (const auto &config : soc_configs) {
    if (config.type == type) {
      return config.name;
    }
  }
  return "Unknow";
}

#ifndef VK_SIM_MODEL
using RtDevBinaryRegisterFunc = rtError_t (*)(const rtDevBinary_t *, void **);
using RtFunctionRegisterFunc = rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t);
static void RegKernelWithRT(RtDevBinaryRegisterFunc reg_binary, RtFunctionRegisterFunc reg_function,
                            const unsigned char *bin_data, unsigned int bin_len, void *func_handles[3]) {
  func_handles[Code::kTargetVec] = reinterpret_cast<uint8_t *>(&g_system) + Code::kTargetVec;
  func_handles[Code::kTargetSimtVec] = func_handles[Code::kTargetVec];
  func_handles[Code::kTargetCube] = reinterpret_cast<uint8_t *>(&g_system) + Code::kTargetCube;
  func_handles[Code::kTargetMix] = reinterpret_cast<uint8_t *>(&g_system) + Code::kTargetMix;
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  dev_bin.data = bin_data;
  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  dev_bin.length = bin_len;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
  err = reg_function(module, func_handles[Code::kTargetVec], "dvm_mix_aiv", "dvm_mix_aiv", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
  err = reg_function(module, func_handles[Code::kTargetCube], "dvm_mix_aic", "dvm_mix_aic", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
  err = reg_function(module, func_handles[Code::kTargetMix], "dvm", "dvm", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");
}
#endif

static int CodeLaunchNone(const System &self, const Code *code, void *extern_ws, void *stream) { return 0; }

int System::CodeLaunchRT(const System &self, const Code *code, void *extern_ws, void *stream) {
  typedef rtError_t (*GetFftsFunc)(uint64_t *addr, uint32_t *len);
  typedef rtError_t (*LaunchKernelFunc)(const void *func, uint32_t blockdim, void *args, uint32_t argssize,
                                        rtSmDesc_t *, rtStream_t);
  if (code->target_ == Code::kTargetMix) {
    auto get_ffts = reinterpret_cast<GetFftsFunc>(self.get_ffts_addr_func_);
    uint32_t len = 0;
    auto err = get_ffts(reinterpret_cast<uint64_t *>(code->data_), &len);
    EXCEPTION_IF(err != 0, "get_ffts failed");
  }
  auto func_handle = static_cast<const void *>(reinterpret_cast<const uint8_t *>(&self) + code->target_);
  auto launch = reinterpret_cast<LaunchKernelFunc>(self.kernel_launch_func_);
  if (likely(code->data_size_ <= PARAM_TABLE_LIMIT)) {
    return launch(func_handle, code->block_dim_, code->data_, code->data_size_, nullptr, stream);
  }
  auto data_dev = reinterpret_cast<uint8_t *>(extern_ws);
  auto ret =
    aclrtMemcpyAsync(data_dev, code->data_size_, code->data_, code->data_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  EXCEPTION_IF(ret != 0, "aclrtMemcpyAsync error");
  uint64_t args[] = {reinterpret_cast<uint64_t>(data_dev), *(reinterpret_cast<uint64_t *>(code->data_) + 1)};
  return launch(func_handle, code->block_dim_, args, sizeof(args), nullptr, stream);
}

int System::CodeLaunchACL_C220(const System &self, const Code *code, void *extern_ws, void *stream) {
  typedef int (*GetFftsFunc)(void **addr);
  typedef int (*LaunchHostArgFunc)(const void *func_handle, uint32_t blockdim, void *stream, void *cfg, void *hostargs,
                                   size_t argssize, void *placeHolder, size_t placehoderNum);
  if (code->target_ == Code::kTargetMix) {
    auto get_ffts = reinterpret_cast<GetFftsFunc>(self.get_ffts_addr_func_);
    auto err = get_ffts(reinterpret_cast<void **>(code->data_));
    EXCEPTION_IF(err != 0, "get_ffts failed");
  }
  auto func_handle = self.func_handles_[code->target_];
  auto launch = reinterpret_cast<LaunchHostArgFunc>(self.kernel_launch_func_);
  if (likely(code->data_size_ <= PARAM_TABLE_LIMIT)) {
    return launch(func_handle, code->block_dim_, stream, nullptr, code->data_, code->data_size_, nullptr, 0);
  }
  auto data_dev = reinterpret_cast<uint8_t *>(extern_ws);
  auto ret =
    aclrtMemcpyAsync(data_dev, code->data_size_, code->data_, code->data_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  EXCEPTION_IF(ret != 0, "aclrtMemcpyAsync error");
  uint64_t args[] = {reinterpret_cast<uint64_t>(data_dev), *(reinterpret_cast<uint64_t *>(code->data_) + 1)};
  return launch(func_handle, code->block_dim_, stream, nullptr, args, sizeof(args), nullptr, 0);
}

int System::CodeLaunchACL_C310(const System &self, const Code *code, void *extern_ws, void *stream) {
  typedef int (*LaunchHostArgFunc)(const void *func_handle, uint32_t blockdim, void *stream, void *cfg, void *hostargs,
                                   size_t argssize, void *placeHolder, size_t placehoderNum);
  auto func_handle = self.func_handles_[code->target_];
  auto launch = reinterpret_cast<LaunchHostArgFunc>(self.kernel_launch_func_);
  aclrtLaunchKernelAttr attr;
  aclrtLaunchKernelCfg cfg;
  aclrtLaunchKernelCfg *cfg_ptr = nullptr;
  if (code->target_ == Code::kTargetSimtVec) {
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attr.value.dynUBufSize = self.local_mem_size_ - self.SimtWorkspace();
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cfg_ptr = &cfg;
  }
  if (likely(code->data_size_ <= PARAM_TABLE_LIMIT)) {
    return launch(func_handle, code->block_dim_, stream, cfg_ptr, code->data_, code->data_size_, nullptr, 0);
  }
  auto data_dev = reinterpret_cast<uint8_t *>(extern_ws);
  auto ret =
    aclrtMemcpyAsync(data_dev, code->data_size_, code->data_, code->data_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  EXCEPTION_IF(ret != 0, "aclrtMemcpyAsync error");
  uint64_t args[] = {reinterpret_cast<uint64_t>(data_dev), *(reinterpret_cast<uint64_t *>(code->data_) + 1)};
  return launch(func_handle, code->block_dim_, stream, cfg_ptr, args, sizeof(args), nullptr, 0);
}

void System::GetSocConfig() {
  std::string soc_name;
  if (const char *env_config = getenv("DVM_SOC_NAME")) {
    soc_name = env_config;
    if (size_t pos1 = soc_name.find(':'); pos1 != std::string::npos) {
      // verbose format: "CustomAscend:220,25,192"
      size_t pos2 = soc_name.find(',', pos1 + 1);
      size_t pos3 = soc_name.find(',', pos2 + 1);
      EXCEPTION_IF(pos2 == std::string::npos || pos3 == std::string::npos, "Invalid DVM_SOC_NAME config.");
      soc_name_ = kSocUnknow;
      arch_ = soc_name.substr(pos1 + 1, pos2 - pos1 - 1) == "220" ? kAiCore_C220 : kAiCore_C310;
      cube_core_num_ = std::stoull(soc_name.substr(pos2 + 1, pos3 - pos2 - 1));
      l2_size_ = std::stoull(soc_name.substr(pos3 + 1)) * MB;
      return;
    }
  } else {
    soc_name = aclrtGetSocName();
  }
  for (const SocConfig &c : soc_configs) {
    if (soc_name == c.name) {
      soc_name_ = c.type;
      arch_ = c.arch;
      cube_core_num_ = c.aicore_num;
      l2_size_ = c.l2_size;
      return;
    }
  }
  std::string soc_err = std::string("Unrecognized SoC Version: ") + soc_name;
  DvmException(soc_err.c_str());
}

void System::DoInit() {
#ifdef VK_SIM_MODEL
  // The model runtime must be available before the first ACL query.  System
  // configuration can be queried before DevRunner is constructed.
  void *sim_handle = dlopen("libruntime_camodel.so", RTLD_NOW | RTLD_GLOBAL);
  EXCEPTION_IF(sim_handle == nullptr, dlerror());
#endif
  GetSocConfig();
  event_num_ = 8;
  vector_core_num_ = cube_core_num_ * 2;
  l1_size_ = 512 * 1024;
  const unsigned char *g_vkernel_bin = nullptr;
  unsigned int g_vkernel_bin_len = 0;
  if (arch_ == kAiCore_C220) {
    bt_size_ = 1024;
    ub_workspace_size_ = 512;
    g_access_func_offset_ = g_access_func_offset_c220;
    g_simd_func_offset_ = g_simd_func_offset_c220;
    g_visit_func_offset_ = g_visit_func_offset_c220;
    g_vkernel_bin = g_vkernel_c220_bin;
    g_vkernel_bin_len = g_vkernel_c220_bin_len;
    l0c_size_ = 128 * 1024;
    local_mem_size_ = 192 * 1024;
  } else if (arch_ == kAiCore_C310) {
    bt_size_ = 4096;
    ub_workspace_size_ = 256; // avoid access overflow: vlds etc.
    g_access_func_offset_ = g_access_func_offset_c310;
    g_simd_func_offset_ = g_simd_func_offset_c310;
    g_visit_func_offset_ = g_visit_func_offset_c310;
    g_vkernel_bin = g_vkernel_c310_bin;
    g_vkernel_bin_len = g_vkernel_c310_bin_len;
    l0c_size_ = 256 * 1024;
    local_mem_size_ = 256 * 1024;
  }
  int32_t device_id = 0;
  if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
    code_launch_ = CodeLaunchNone;
    return;
  }
  GetAclCoreCount(device_id, &cube_core_num_, &vector_core_num_);
  inited_ = true;
#ifdef VK_SIM_MODEL
  auto set_device_ret = aclrtSetDevice(0);
  EXCEPTION_IF(set_device_ret != ACL_SUCCESS, "aclrtSetDevice failed");
  std::atexit([]() {
    (void)aclrtResetDeviceForce(0);
    (void)aclFinalize();
  });
#endif
  auto acl_handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
#ifndef VK_SIM_MODEL
  auto ret = MsprofRegisterCallback(0, ProfCommandHandler);
  EXCEPTION_IF(ret != MSPROF_ERROR_NONE, "MsprofRegisterCallBack failed.");
  bool try_reg_rt = true;
  if (acl_handle) {
    typedef aclError (*GetVersionFunc)(aclCANNPackageName, aclCANNPackageVersion *);
    if (auto get_version = reinterpret_cast<GetVersionFunc>(dlsym(acl_handle, "aclsysGetCANNVersion"))) {
      aclCANNPackageVersion ver;
      if (get_version(ACL_PKG_NAME_CANN, &ver) == ACL_SUCCESS) {
        int major = std::stoi(ver.majorVersion);
        int minor = std::stoi(ver.minorVersion);
        if (major > 9 || (major == 9 && minor >= 1)) {  // CANN 9.1+ use acl only
          try_reg_rt = false;
        }
      }
    }
  }
  if (try_reg_rt && arch_ == kAiCore_C220) {
    rt_handle_ = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
    if (rt_handle_) {
      kernel_launch_func_ = dlsym(rt_handle_, "rtKernelLaunch");
      get_ffts_addr_func_ = dlsym(rt_handle_, "rtGetC2cCtrlAddr");
      if (kernel_launch_func_ && get_ffts_addr_func_) {
        auto reg_binary = reinterpret_cast<RtDevBinaryRegisterFunc>(dlsym(rt_handle_, "rtDevBinaryRegister"));
        auto reg_function = reinterpret_cast<RtFunctionRegisterFunc>(dlsym(rt_handle_, "rtFunctionRegister"));
        if (reg_binary && reg_function) {
          RegKernelWithRT(reg_binary, reg_function, g_vkernel_bin, g_vkernel_bin_len, func_handles_);
          code_launch_ = CodeLaunchRT;
        }
        dlclose(acl_handle);
        return;
      }
      dlclose(rt_handle_);
      rt_handle_ = nullptr;
    }
  }
#endif
  rt_handle_ = acl_handle;
  typedef aclError (*LoadBinaryFunc)(const void *data, size_t len, const aclrtBinaryLoadOptions *opt,
                                     aclrtBinHandle *handle);
  typedef aclError (*GetFunctionFunc)(aclrtBinHandle handle, const char *name, void **funchandle);
  auto load_binary = reinterpret_cast<LoadBinaryFunc>(dlsym(rt_handle_, "aclrtBinaryLoadFromData"));
  auto get_function = reinterpret_cast<GetFunctionFunc>(dlsym(rt_handle_, "aclrtBinaryGetFunction"));
  kernel_launch_func_ = dlsym(rt_handle_, "aclrtLaunchKernelWithHostArgs");
  get_ffts_addr_func_ = dlsym(rt_handle_, "aclrtGetHardwareSyncAddr");
  EXCEPTION_IF(!(load_binary && get_function && kernel_launch_func_ && get_ffts_addr_func_), "dlsym load failed");
  const uint64_t *g_mix_symbols;
  unsigned int g_mix_symbol_len;
  if (arch_ == kAiCore_C220) {
    g_mix_symbols = g_mix_symbols_c220;
    g_mix_symbol_len = g_mix_symbol_len_c220;
  } else {
    g_mix_symbols = g_mix_symbols_c310;
    g_mix_symbol_len = g_mix_symbol_len_c310;
  }
  renamed_bin_ = std::malloc(g_vkernel_bin_len);
  std::memcpy(renamed_bin_, g_vkernel_bin, g_vkernel_bin_len);
  for (uint32_t pos = 0; pos < g_mix_symbol_len; ++pos) {
    *(static_cast<char *>(renamed_bin_) + g_mix_symbols[pos]) = 'a';  // mix -> aix
  }
  aclrtBinaryLoadOption opt_data[2];
  opt_data[0].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  opt_data[1].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
  opt_data[1].value.isLazyLoad = 1;
  aclrtBinaryLoadOptions bin_opt;
  bin_opt.numOpt = 2;
  bin_opt.options = opt_data;
  aclrtBinHandle bin_handle;

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
  auto err = load_binary(renamed_bin_, g_vkernel_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "dvm_aix_aiv", &func_handles_[Code::kTargetVec]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg vec failed");
  if (arch_ == kAiCore_C220) {
    func_handles_[Code::kTargetSimtVec] = func_handles_[Code::kTargetVec];
  } else {
    simt_bin_ = std::malloc(g_vkernel_bin_len);
    std::memcpy(simt_bin_, g_vkernel_bin, g_vkernel_bin_len);
    for (uint32_t pos = 0; pos < g_mix_symbol_len; ++pos) {
      *(static_cast<char *>(simt_bin_) + g_mix_symbols[pos]) = 's';  // mix -> six
    }
    *reinterpret_cast<uint32_t *>(static_cast<char *>(simt_bin_) + g_meta_aiv_type_value_offset_c310) = 4;
    err = load_binary(simt_bin_, g_vkernel_bin_len, &bin_opt, &bin_handle);
    err |= get_function(bin_handle, "dvm_six_aiv", &func_handles_[Code::kTargetSimtVec]);
    EXCEPTION_IF(err != ACL_SUCCESS, "reg simt vec failed");
  }

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE;
  err = load_binary(renamed_bin_, g_vkernel_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "dvm_aix_aic", &func_handles_[Code::kTargetCube]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg cube failed");

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
  err = load_binary(g_vkernel_bin, g_vkernel_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "dvm", &func_handles_[Code::kTargetMix]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg mix failed");
  code_launch_ = arch_ == kAiCore_C220 ? CodeLaunchACL_C220 : CodeLaunchACL_C310;
  pass_opt_ = pass::CreateOptimizer(arch_);
}

System::~System() {
  delete pass_opt_;
  if (online_tuner_) {
    delete online_tuner_;
  }
  if (lazy_tuner_) {
    delete lazy_tuner_;
  }
  if (rt_handle_) {
    dlclose(rt_handle_);
  }
  if (renamed_bin_) {
    std::free(renamed_bin_);
  }
  if (simt_bin_) {
    std::free(simt_bin_);
  }
  for (auto b : bins_) {
    std::free(b);
  }
}

void System::RegCustom(const std::string &nspace, const std::string &so_path, const std::string &bin_path,
                       const std::vector<std::pair<std::string, uint64_t>> &func_table) {
  const std::string nspace_prefix = nspace + "/";
  void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  auto op_def = handle == nullptr ? nullptr : reinterpret_cast<CustomDef *>(dlsym(handle, "__ALL_OPS__"));
  EXCEPTION_IF(handle == nullptr || op_def == nullptr, "reg_custom op failed");
  while (op_def->name != nullptr) {
    custom_def_[nspace_prefix + op_def->name] = op_def->create_func;
    ++op_def;
  }

  RegCustom(nspace, bin_path, func_table);
}

void System::RegCustom(const std::string &nspace, const std::string &bin_path,
                       const std::vector<std::pair<std::string, uint64_t>> &func_table) {
  const std::string nspace_prefix = nspace + "/";

  std::ifstream fin(bin_path, std::ios::binary | std::ios::ate);
  EXCEPTION_IF(!fin.is_open(), "RegCustomBin: open bin file failed");
  const size_t bin_size = static_cast<size_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);
  void *bin_data = std::malloc(bin_size);
  EXCEPTION_IF(bin_data == nullptr, "RegCustomBin: allocate bin buffer failed");
  fin.read(static_cast<char *>(bin_data), bin_size);
  fin.close();
  bins_.push_back(bin_data);

  aclrtBinaryLoadOption opt_data[2];
  opt_data[0].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
  opt_data[1].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
  opt_data[1].value.isLazyLoad = 0;
  aclrtBinaryLoadOptions bin_opt;
  bin_opt.numOpt = 2;
  bin_opt.options = opt_data;
  aclrtBinHandle bin_handle;
  auto err = aclrtBinaryLoadFromData(bin_data, bin_size, &bin_opt, &bin_handle);
  EXCEPTION_IF(err != ACL_SUCCESS, "RegisterCustom: aclrtBinaryLoadFromData failed");
  void *func_handle = nullptr;
  const std::string func_name = "dvm_custom_" + nspace;
  err = aclrtBinaryGetFunction(bin_handle, func_name.c_str(), &func_handle);
  EXCEPTION_IF(err != ACL_SUCCESS, "RegisterCustom: aclrtBinaryGetFunction failed");
  void *aic_addr = nullptr;
  void *aiv_addr = nullptr;
  err = aclrtGetFunctionAddr(func_handle, &aic_addr, &aiv_addr);
  EXCEPTION_IF(err != ACL_SUCCESS, "RegisterCustom: aclrtGetFunctionAddr failed");
  for (const auto &entry : func_table) {
    custom_funcs_[nspace_prefix + entry.first] = reinterpret_cast<uint64_t>(aiv_addr) + entry.second;
  }
}

uint64_t System::GetCustomFunc(const std::string &full_name) const {
  auto it = custom_funcs_.find(full_name);
  EXCEPTION_IF(it == custom_funcs_.end(), "custom func not exist");
  return it->second;
}

NDObject *System::CreateCustom(const std::string &op_name, const std::vector<NDObject *> &inputs,
                               const std::vector<ScalarRef> &attrs) const {
  auto it = custom_def_.find(op_name);
  EXCEPTION_IF(it == custom_def_.end(), "op_name not exist");
  return (it->second)(inputs, attrs);
}

std::string System::GetCustomFuncName(uint64_t func_id) const {
  for (const auto &pair : custom_funcs_) {
    if (pair.second == func_id) {
      return pair.first;
    }
  }
  return "null";
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

Config &System::SetVfFusion() {
  vf_fusion_ = true;
  return *this;
}

Config &System::UnsetVfFusion() {
  vf_fusion_ = false;
  return *this;
}

System g_system;
}  // namespace dvm
