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
#ifndef VK_SIM_MODEL
#include "acl/acl_rt.h"
#endif
#include "system.h"
#include "code.h"
#include "tuning.h"

#ifdef VK_SIM_MODEL
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

#endif // VK_SIM_MODEL

extern const unsigned char g_vkernel_c220_bin[];
extern unsigned int g_vkernel_c220_bin_len;

extern const uint64_t g_mix_symbols_c220[];
extern const unsigned int g_mix_symbol_len_c220;

namespace dvm {
// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t), sizeof(int64_t)};
const char *DTYPE_NAMES[dvm::kTypeEnd] = {"bool", "float16", "bfloat16", "float32", "int32", "int64"};
const uint64_t ITEM_SIMD_WIDTH_MAX[kTypeEnd] = {128, 128, 128, 64, 64};

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
  const std::string soc_name = GetSocName();
  if (soc_name.empty() || soc_name == "MS_DRY_RUN") {
    return;
  }
  for (const SocConfig &c : soc_configs) {
    if (soc_name == c.name) {
      config = &c;
      break;
    }
  }
  EXCEPTION_IF(config == nullptr, "Unrecognized SoC Version.");
  soc_name_ = config->type;

  arch_ = kAiCore_C220;
  ub_workspace_size_ = 512;
  local_mem_size_ = 192 * 1024 - ub_workspace_size_;
  event_num_ = 8;
  cube_core_num_ = config->aicore_num;
  vector_core_num_ = cube_core_num_ * 2;
  l2_size_ = config->l2_size;
  l1_size_ = 512 * 1024;
  l0c_size_ = 128 * 1024;
#ifdef VK_SIM_MODEL
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  dev_bin.data = g_vkernel_c220_bin;
  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  dev_bin.length = g_vkernel_c220_bin_len;

  err = rtDevBinaryRegister(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
  uint8_t *stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetVec;
  err = rtFunctionRegister(module, stub_func, "vmain_mix_aiv", "vmain_mix_aiv", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  err = rtDevBinaryRegister(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetCube;
  err = rtFunctionRegister(module, stub_func, "vmain_mix_aic", "vmain_mix_aic", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
  err = rtDevBinaryRegister(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
  stub_func = reinterpret_cast<uint8_t *>(this) + Code::kTargetMix;
  err = rtFunctionRegister(module, stub_func, "vmain", "vmain", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");

  rt_get_c2c_addr_ = ::rtGetC2cCtrlAddr;
#else
  renamed_bin_ = std::malloc(g_vkernel_c220_bin_len);
  memcpy_s(renamed_bin_, g_vkernel_c220_bin_len, g_vkernel_c220_bin, g_vkernel_c220_bin_len);
  for (uint32_t pos = 0; pos < g_mix_symbol_len_c220; ++pos) {
    *(static_cast<char *>(renamed_bin_) + g_mix_symbols_c220[pos]) = 'a'; // mix -> aix
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
  auto err = aclrtBinaryLoadFromData(renamed_bin_, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= aclrtBinaryGetFunction(bin_handle, "vmain_aix_aiv", &func_handles_[Code::kTargetVec]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg vec failed");

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE;
  err = aclrtBinaryLoadFromData(renamed_bin_, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= aclrtBinaryGetFunction(bin_handle, "vmain_aix_aic", &func_handles_[Code::kTargetCube]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg cube failed");

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
  err = aclrtBinaryLoadFromData(g_vkernel_c220_bin, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= aclrtBinaryGetFunction(bin_handle, "vmain", &func_handles_[Code::kTargetMix]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg mix failed");
#endif
}

System::~System() {
  if (online_tuner_) {
    delete online_tuner_;
  }
  if (lazy_tuner_) {
    delete lazy_tuner_;
  }
  if (renamed_bin_) {
    std::free(renamed_bin_);
  }
}
}  // namespace dvm
