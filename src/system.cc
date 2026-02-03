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


static void RegKernelWithRT(void *reg_binary_func, void *reg_function_func, System *sys) {
  auto reg_binary = reinterpret_cast<rtError_t (*)(const rtDevBinary_t *, void **)>(reg_binary_func);
  auto reg_function = reinterpret_cast<rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t)>(reg_function_func);
  auto &func_handles = sys->func_handles_;
  func_handles[Code::kTargetVec] = reinterpret_cast<uint8_t *>(sys) + Code::kTargetVec;
  func_handles[Code::kTargetCube] = reinterpret_cast<uint8_t *>(sys) + Code::kTargetCube;
  func_handles[Code::kTargetMix] = reinterpret_cast<uint8_t *>(sys) + Code::kTargetMix;
  rtError_t err;
  void *module = nullptr;
  rtDevBinary_t dev_bin;
  dev_bin.version = 0;
  dev_bin.data = g_vkernel_c220_bin;
  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  dev_bin.length = g_vkernel_c220_bin_len;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec binary failed");
  err = reg_function(module, func_handles[Code::kTargetVec], "vmain_mix_aiv", "vmain_mix_aiv", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg vec function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore binary failed");
  err = reg_function(module, func_handles[Code::kTargetCube], "vmain_mix_aic", "vmain_mix_aic", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg aicore function failed");

  dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF;
  err = reg_binary(&dev_bin, &module);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix binary failed");
  err = reg_function(module, func_handles[Code::kTargetMix], "vmain", "vmain", 0);
  EXCEPTION_IF(err != RT_ERROR_NONE, "reg mix function failed");
}

int System::CodeLaunchRT(const System &self, const Code *code, void *extern_ws, void *stream) {
  typedef rtError_t (*GetFftsFunc)(uint64_t *addr, uint32_t *len);
  typedef rtError_t (*LaunchKernelFunc)(const void *func, uint32_t blockdim, void *args, uint32_t argssize,
                                          rtSmDesc_t *, rtStream_t);
  if (code->target_ == Code::kTargetMix) {
    auto get_ffts = reinterpret_cast<GetFftsFunc>(self.get_ffts_addr_func_);
    uint32_t len = 0;
    auto err = get_ffts(reinterpret_cast<uint64_t *>(code->data_), &len);
    if (err != 0) return err;
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

int System::CodeLaunchACL(const System &self, const Code *code, void *extern_ws, void *stream) {
  typedef int (*GetFftsFunc)(void **addr);
  typedef int (*LaunchHostArgFunc)(const void *func_handle, uint32_t blockdim, void *stream, void *cfg, void *hostargs,
                                   size_t argssize, void *placeHolder, size_t placehoderNum);
  if (code->target_ == Code::kTargetMix) {
    auto get_ffts = reinterpret_cast<GetFftsFunc>(self.get_ffts_addr_func_);
    auto err = get_ffts(reinterpret_cast<void **>(code->data_));
    if (err != 0) return err;
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
  RegKernelWithRT(rtDevBinaryRegister, rtFunctionRegister, this);
  code_launch_ = CodeLaunchRT;
  kernel_launch_func_ = ::rtKernelLaunch;
  get_ffts_addr_func_ = ::rtGetC2cCtrlAddr;
  return;
#endif
  rt_handle_ = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  if (rt_handle_) {
    kernel_launch_func_ = dlsym(rt_handle_, "rtKernelLaunch");
    get_ffts_addr_func_ = dlsym(rt_handle_, "rtGetC2cCtrlAddr");
    if (kernel_launch_func_ && get_ffts_addr_func_) {
      auto reg_binary = dlsym(rt_handle_, "rtDevBinaryRegister");
      auto reg_function = dlsym(rt_handle_, "rtFunctionRegister");
      EXCEPTION_IF(reg_binary == nullptr || reg_function == nullptr, "load rt_binary_register symbol failed");
      RegKernelWithRT(reg_binary, reg_function, this);
      code_launch_ = CodeLaunchRT;
      return;
    }
    dlclose(rt_handle_);
    rt_handle_ = nullptr;
  }
#ifdef __CANN_85__
  rt_handle_ = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
  EXCEPTION_IF(rt_handle_ == nullptr, "dlopen libascendcl failed");
  typedef aclError (*LoadBinaryFunc)(const void *data, size_t len, const aclrtBinaryLoadOptions *opt,
                                     aclrtBinHandle *handle);
  typedef aclError (*GetFunctionFunc)(aclrtBinHandle handle, const char *name, void **funchandle);
  auto load_binary = reinterpret_cast<LoadBinaryFunc>(dlsym(rt_handle_, "aclrtBinaryLoadFromData"));
  auto get_function = reinterpret_cast<GetFunctionFunc>(dlsym(rt_handle_, "aclrtBinaryGetFunction"));
  kernel_launch_func_ = dlsym(rt_handle_, "aclrtLaunchKernelWithHostArgs");
  get_ffts_addr_func_ = dlsym(rt_handle_, "aclrtGetHardwareSyncAddr");
  EXCEPTION_IF(!(load_binary && get_function && kernel_launch_func_ && get_ffts_addr_func_),
               "dlsym load failed");
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
  auto err = load_binary(renamed_bin_, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "vmain_aix_aiv", &func_handles_[Code::kTargetVec]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg vec failed");

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE;
  err = load_binary(renamed_bin_, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "vmain_aix_aic", &func_handles_[Code::kTargetCube]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg cube failed");

  opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
  err = load_binary(g_vkernel_c220_bin, g_vkernel_c220_bin_len, &bin_opt, &bin_handle);
  err |= get_function(bin_handle, "vmain", &func_handles_[Code::kTargetMix]);
  EXCEPTION_IF(err != ACL_SUCCESS, "reg mix failed");
  code_launch_ = CodeLaunchACL;
#else
  EXCEPTION_IF(rt_handle_ == nullptr, "dlopen libruntime failed");
#endif
}

System::~System() {
  if (online_tuner_) {
    delete online_tuner_;
  }
  if (lazy_tuner_) {
    delete lazy_tuner_;
  }
#ifndef __CANN_85__
  if (rt_handle_) {
    dlclose(rt_handle_);
  }
#endif
  if (renamed_bin_) {
    std::free(renamed_bin_);
  }
}
}  // namespace dvm
