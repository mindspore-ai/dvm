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

#include <dlfcn.h>
#include <stdexcept>
#include <sstream>
#ifndef VK_SIM_MODEL
#include "acl/acl_rt.h"
#endif
#include "system.h"
#include "code.h"

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

extern const unsigned char g_vkernel_c220_bin[];
extern unsigned int g_vkernel_c220_bin_len;

namespace dvm {

const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t)};
const char *DTYPE_NAMES[dvm::kTypeEnd] = {"bool", "float16", "bfloat16", "float32", "int32"};

static std::string GetSocName() {
  std::string res;
  const char *soc_name = getenv("DVM_SOC_NAME");
  if (soc_name == nullptr) {
#ifdef VK_SIM_MODEL
    DvmException("simulator must set environment variable DVM_SOC_NAME");
#else
    soc_name = aclrtGetSocName();
#endif
  }
  if (soc_name == nullptr) {
    return res;
  }
  res = soc_name;
  return res;
}

void DvmException(const char *error_str) {
  std::ostringstream oss;
  oss << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}

struct SocConfig {
  const char *name;
  SocType type;
  uint64_t aicore_num;
  uint64_t l2_size;
};

constexpr uint64_t MB = 1024 * 1024;
const SocConfig soc_configs[] = {
  {"Ascend910B1", kAscend910B1, 25, 192 * MB},       {"Ascend910B2", kAscend910B2, 24, 192 * MB},
  {"Ascend910B2C", kAscend910B2, 24, 192 * MB},      {"Ascend910B3", kAscend910B3, 20, 192 * MB},
  {"Ascend910B4", kAscend910B4, 20, 96 * MB},        {"Ascend910B4-1", kAscend910B4, 20, 96 * MB},
  {"Ascend910_9391", kAscend910_9391, 25, 192 * MB}, {"Ascend910_9392", kAscend910_9392, 25, 192 * MB},
  {"Ascend910_9381", kAscend910_9381, 24, 192 * MB}, {"Ascend910_9382", kAscend910_9382, 24, 192 * MB},
  {"Ascend910_9372", kAscend910_9372, 20, 192 * MB}, {"Ascend910_9361", kAscend910_9361, 20, 96 * MB},
};

System::System() {
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

  arch_ = kAiCore_C220;
  local_mem_size_ = 192 * 1024;
  event_num_ = 8;
  cube_core_num_ = config->aicore_num;
  vector_core_num_ = cube_core_num_ * 2;
  l2_size_ = config->l2_size;
  l1_size_ = 512 * 1024;
  l0c_size_ = 128 * 1024;
  ub_workspace_size_ = 512;
#ifdef VK_SIM_MODEL
  auto rt_binary_register = rtDevBinaryRegister;
  auto rt_function_register = rtFunctionRegister;
  launch_func_ = rtKernelLaunch;
#else
  void *handle = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(handle == nullptr, "Load libruntime.so failed");
  auto rt_binary_register =
    reinterpret_cast<rtError_t (*)(const rtDevBinary_t *, void **)>(dlsym(handle, "rtDevBinaryRegister"));
  EXCEPTION_IF(rt_binary_register == nullptr, "load rt_binary_register symbol failed");
  auto rt_function_register =
    reinterpret_cast<rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t)>(
      dlsym(handle, "rtFunctionRegister"));
  EXCEPTION_IF(rt_function_register == nullptr, "load rt_function_register symbol failed");
  launch_func_ = reinterpret_cast<rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t)>(
    dlsym(handle, "rtKernelLaunch"));
  EXCEPTION_IF(launch_func_ == nullptr, "load rt_kernel_launch symbol failed");
#endif
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  dev_bin.data = g_vkernel_c220_bin;
  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  dev_bin.length = g_vkernel_c220_bin_len;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
  uint8_t *stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetVec;
  err = rt_function_register(module, stub_func, "vmain_mix_aiv", "vmain_mix_aiv", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetCube;
  err = rt_function_register(module, stub_func, "vmain_mix_aic", "vmain_mix_aic", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
  err = rt_binary_register(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetMix;
  err = rt_function_register(module, stub_func, "vmain", "vmain", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");
#ifdef VK_SIM_MODEL
  get_c2c_addr_func_ = rtGetC2cCtrlAddr;
#else
  get_c2c_addr_func_ = reinterpret_cast<rtError_t (*)(uint64_t *, uint32_t *)>(dlsym(handle, "rtGetC2cCtrlAddr"));
#endif
}
}  // namespace dvm
