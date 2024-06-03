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

#ifndef _DVM_CODE_H_
#define _DVM_CODE_H_
#include <iostream>
#include <sstream>
#include "dvm.h"
#include "isa.h"

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
  kAiCore_C100,
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
  kSocUnknow,
};

class DeviceInfo {
 public:
  static DeviceInfo &Instance() {
    static DeviceInfo obj;
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

 private:
  DeviceInfo();
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

// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t)};

struct Code {
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  Code() = default;
  Code(const Code&obj) = delete;
  Code&operator=(const Code&) = delete;
  virtual ~Code() {
    if (data_)
      std::free(data_);
  }
  void Alloc(size_t s) {
    if (data_) {
      data_ = static_cast<unsigned char *>(std::realloc(data_, s));
    } else {
      data_ = static_cast<unsigned char *>(std::malloc(s));
    }
  }
  void UpdateHead(uint64_t tile_num, uint64_t simd_width, uint64_t flags) {
    uint64_t *head = reinterpret_cast<uint64_t*>(data_);
    head[0] = 0;
    head[1] = tile_num << V_ENTRY_TILE_NUM_OFFSET | simd_width << V_ENTRY_SIMD_WIDTH_OFFSET |
             (static_cast<uint64_t>(data_size_) / sizeof(uint64_t) - 2) << V_ENTRY_CODE_SIZE_OFFSET | flags;
  }
  void UpdateParallelHead() {
    UpdateHead(block_dim_, 0, V_ENTRY_FLAG_PARALLEL);
  }

  uint64_t HeadSize() const { return sizeof(uint64_t) * 2; } // ffts + entry

  int Launch(void* stream) {
    if (target_ == kTargetMix) {
      uint32_t ffts_len;
      auto ret = DeviceInfo::Instance().get_c2c_addr_func_(reinterpret_cast<uint64_t*>(data_), &ffts_len);
      if (ret != RT_ERROR_NONE) return ret;
    }
    auto launch_func = DeviceInfo::Instance().launch_func_;
    uint8_t* stub_func = DeviceInfo::Instance().StubFunc(target_);
    if (!atomic_clean_.empty()) {
      for (auto a : atomic_clean_) {
        uint8_t* a_stub = DeviceInfo::Instance().StubFunc(a->target_);
        auto ret = launch_func(a_stub, a->block_dim_, a->data_, a->data_size_, nullptr, stream);
        if (ret != RT_ERROR_NONE) return ret;
      }
    }
    return launch_func(stub_func, block_dim_, data_, data_size_, nullptr, stream);
  }

  void DisAssemble(std::ostringstream &oss);

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
  std::vector<Code*> atomic_clean_;

  uint64_t simd_width_{0};
};
} // namespace dvm 
#endif // _DVM_CODE_H_
