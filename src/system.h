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

#ifndef _DVM_SYSTEM_H_
#define _DVM_SYSTEM_H_
#include <iostream>
#include "dvm.h"

// rts_runtime
typedef int32_t rtError_t;
typedef char char_t;
const int32_t RT_ERROR_NONE = 0;  // success
typedef void *rtStream_t;
struct tagRtSmCtrl;
typedef struct tagRtSmCtrl rtSmDesc_t;
struct MsprofApi;
typedef void *VOID_PTR;

namespace dvm {
#ifdef DEBUG
#define ASSERT(cond)                                                                                            \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      std::cerr << "[ASSERT ERROR]" << __FILE__ << ":" << __LINE__ << ": ASSERT(" << #cond << ")" << std::endl; \
      DvmException("assert");                                                                                   \
    }                                                                                                           \
  } while (0)

#define MESS(var, init) \
  do {                  \
    var = init;         \
  } while (0)
#else
#define ASSERT(cond)
#define MESS(var, init)
#endif

#define EXCEPTION_IF(cond, error_str)       \
  do {                                      \
    if (cond) dvm::DvmException(error_str); \
  } while (0)
void DvmException(const char *error_str);

#define ERROR_CHECK(func)  \
  do {                     \
    if ((func) != 0) {     \
      DvmException(#func); \
    }                      \
  } while (0)

enum AiCoreArch {
  kAiCore_C220,
  kAiCore_C310,
};

enum class CoreType {
  kAIV,
  kAIC,
};

enum SocType {
  // C220(B)
  kAscend910B1,
  kAscend910B2,
  kAscend910B3,
  kAscend910B4,
  // C220(C)
  kAscend910_9391,
  kAscend910_9392,
  kAscend910_9381,
  kAscend910_9382,
  kAscend910_9372,
  kAscend910_9361,
  // C310(PR)
  kAscend950PR_9579,
  kAscend950PR_9589,
  kAscend950PR_9599,
  kAscend950PR_958b,
  kAscend950PR_957b,
  kAscend950PR_957c,
  kAscend950PR_957d,
  kAscend950PR_950z,
  // C310(DT)
  kAscend950DT_950x,
  kAscend950DT_950y,
  kAscend950DT_9571,
  kAscend950DT_9572,
  kAscend950DT_9573,
  kAscend950DT_9574,
  kAscend950DT_9575,
  kAscend950DT_9576,
  kAscend950DT_9577,
  kAscend950DT_9578,
  kAscend950DT_9581,
  kAscend950DT_9582,
  kAscend950DT_9583,
  kAscend950DT_9584,
  kAscend950DT_9585,
  kAscend950DT_9586,
  kAscend950DT_9587,
  kAscend950DT_9588,
  kAscend950DT_9591,
  kAscend950DT_9592,
  kAscend950DT_9595,
  kAscend950DT_9596,
  kAscend950DT_95A1,
  kAscend950DT_95A2,
  kSocUnknow,
};

const char *AiCoreArchName(AiCoreArch arch);
const char *SocTypeName(SocType type);

enum CubeStoreType {
  kCubeStoreGM = 0,
  kCubeStoreUB,
  kCubeStoreUBOnce,
};

enum ProfilerLevel {
  Level0 = 0,
  Level1,
  Level2,
};


class CubeTuner;
class Code;
class System : public Config {
 public:
  System() = default;
  ~System();
  void Init() {
    if (!inited_) {
      DoInit();
    }
  }

  Config &SetDeterm() override;
  Config &UnsetDeterm() override;
  Config &SetOnlineTuner() override;
  Config &UnsetOnlineTuner() override;
  Config &SetLazyTuner() override;
  Config &UnsetLazyTuner() override;

  // hardware config
  AiCoreArch Arch() const { return arch_; }
  void SetLocalMemSize(uint64_t size) { local_mem_size_ = size; }
  uint64_t LocalMemSize() const { return local_mem_size_; }
  uint64_t UbWorkspaceSize() const { return ub_workspace_size_; }
  uint64_t L2Size() const { return l2_size_; }
  uint64_t L1Size() const { return l1_size_; }
  uint64_t L0CSize() const { return l0c_size_; }
  uint64_t CoreNum(CoreType core_type = CoreType::kAIV) const {
    return core_type == CoreType::kAIV ? vector_core_num_ : cube_core_num_;
  }
  uint64_t EventNum() const { return event_num_; }
  SocType SocName() const { return soc_name_; }
  uint32_t BtSize() const { return bt_size_; }
  void SetCubeStoreType(CubeStoreType type) { cube_store_type_ = type; }
  CubeStoreType GetCubeStoreType() { return Arch() == kAiCore_C310 ? cube_store_type_ : kCubeStoreGM; }

  void *CommStream() {
    if (comm_stream_ == nullptr) {
      comm_stream_ = CreateStream();
    }
    return comm_stream_;
  }

  // features config

  bool deterministic_{false};
  bool enable_profile_{false};
  ProfilerLevel profiler_level_{Level0};
  CubeTuner *online_tuner_{nullptr};
  CubeTuner *lazy_tuner_{nullptr};

  // runtime api
  int (*code_launch_)(const System &self, const Code *code, void *extern_ws, void *stream){nullptr};
  void *func_handles_[3];
  void *get_ffts_addr_func_{nullptr};
  void *kernel_launch_func_{nullptr};

  void *CreateStream();
  const uint64_t *g_simd_func_offset_;
  const uint64_t *g_access_func_offset_;
  const uint64_t *g_visit_func_offset_;

 private:
  void DoInit();
  void GetSocConfig();

  uint64_t local_mem_size_;
  uint64_t ub_workspace_size_;
  uint64_t l2_size_;
  uint64_t l1_size_;
  uint64_t l0c_size_;
  uint64_t event_num_;
  uint64_t vector_core_num_;
  uint64_t cube_core_num_;
  AiCoreArch arch_;
  uint32_t bt_size_;
  SocType soc_name_{kSocUnknow};
  bool inited_{false};
  CubeStoreType cube_store_type_{kCubeStoreGM};
  void *comm_stream_{nullptr};
  void *renamed_bin_{nullptr};
  void *rt_handle_{nullptr};

  template <AiCoreArch arch>
  static int CodeLaunchRT(const System &self, const Code *code, void *extern_ws, void *stream);
  template <AiCoreArch arch>
  static int CodeLaunchACL(const System &self, const Code *code, void *extern_ws, void *stream);
};

extern System g_system;

inline constexpr uint64_t SIMD_BLOCK_SIZE = 32;
inline constexpr uint64_t SIMD_REPEAT_SIZE = 256;
inline constexpr uint64_t PARAM_TABLE_LIMIT = 4096;
inline constexpr uint32_t MATMUL_ALIGN_MAX = 1024;

extern const uint64_t ITEM_SIZE[dvm::kDataTypeEnd];
extern const char *DTYPE_NAMES[dvm::kDataTypeEnd];
extern const uint64_t ITEM_SIMD_WIDTH_MAX[kDataTypeEnd];
}  // namespace dvm
#endif  // _DVM_SYSTEM_H_
