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

#ifndef _DVM_SYSTEM_H_
#define _DVM_SYSTEM_H_
#include <iostream>
#include "dvm.h"

// rts_runtime
typedef int32_t rtError_t;
typedef char char_t;
const int32_t RT_ERROR_NONE = 0; // success
typedef void *rtStream_t;
struct tagRtSmCtrl;
typedef struct tagRtSmCtrl rtSmDesc_t;

namespace dvm {
#ifdef DEBUG
#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cout << "[ASSERT ERROR]" << __FILE__ << ":" << __LINE__             \
	        << ": ASSERT(" << #cond << ")" << std::endl;                   \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)
#else
#define ASSERT(cond)
#endif

#define EXCEPTION_IF(cond, error_str)   do { if (cond) DvmException(error_str); } while (0)
void DvmException(const char* error_str);

enum AiCoreArch {
  kAiCore_C220,
};

enum CoreType {
  kVector,
  kCube,
};

enum SocType {
  kAscend910B1,
  kAscend910B2,
  kAscend910B3,
  kAscend910B4,
  kAscend910_9391,
  kAscend910_9392,
  kAscend910_9381,
  kAscend910_9382,
  kAscend910_9372,
  kAscend910_9361,
  kSocUnknow,
};

class System {
 public:
  static System &Instance() {
    static System obj;
    return obj;
  }

  AiCoreArch Arch() const { return arch_; }
  uint64_t LocalMemSize() const { return local_mem_size_; }
  uint64_t UbWorkspaceSize() const { return ub_workspace_size_; }
  uint64_t L2Size() const { return l2_size_; }
  uint64_t L1Size() const { return l1_size_; }
  uint64_t L0CSize() const { return l0c_size_; }
  uint64_t CoreNum(CoreType core_type = kVector) const {
    return core_type == kVector ? vector_core_num_ : cube_core_num_;
  }
  uint64_t EventNum() const { return event_num_; }
  SocType SocName() { return soc_name_; }

  rtError_t (*launch_func_)(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                              rtSmDesc_t *smDesc, rtStream_t stm);
  rtError_t(*get_c2c_addr_func_)(uint64_t*, uint32_t*){nullptr};

  uint8_t *StubFunc(int target) { return reinterpret_cast<uint8_t*>(this) + target; }
  bool deterministic_{false};
  bool online_tuning_{false};

 private:
  System();
  AiCoreArch arch_;
  uint64_t local_mem_size_;
  uint64_t ub_workspace_size_;
  uint64_t l2_size_;
  uint64_t l1_size_;
  uint64_t l0c_size_;
  uint64_t event_num_;
  uint64_t vector_core_num_;
  uint64_t cube_core_num_;
  SocType soc_name_{kSocUnknow};
};

const uint64_t SIMD_BLOCK_SIZE  = 32;
const uint64_t SIMD_REPEAT_SIZE = 256;
const uint64_t PARAM_TABLE_LIMIT = 4096;

// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t)};

} // namespace dvm 
#endif // _DVM_SYSTEM_H_
