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
#include <sstream>
#include "isa.h"
#include "system.h"

namespace dvm {
class NDAccess;
class Code {
 public:
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  Code() = default;
  Code(const Code&obj) = delete;
  Code&operator=(const Code&) = delete;
  ~Code();
  void Clear() {
    atomic_clean_.clear();
    reloc_reuse_.clear();
    reloc_workspaces_.clear();
  }
  void Alloc(size_t size);
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

  uint64_t ReserveWorkspace(uint64_t workspace_size) {
    if (data_size_ <= PARAM_TABLE_LIMIT) {
      extern_code_ = -1;
      return workspace_size;
    }
    uint64_t *head = reinterpret_cast<uint64_t*>(data_);
    head[1] |= V_ENTRY_FLAG_EXTERN_CODE;
    const uint64_t align = 512;
    extern_code_ = (workspace_size + align) & ~(align- 1);
    return extern_code_ + data_size_;
  }

  int Launch(void *workspace, void* stream) {
    if (workspace) {
      for (auto &r : reloc_workspaces_) {
        *(r.first) = reinterpret_cast<uint64_t>(static_cast<char*>(workspace) + r.second);
      }
    }
    if (!reloc_reuse_.empty()) {
      for (auto &r : reloc_reuse_) {
        *(r.first) = *(r.second);
      }
    }
    if (target_ == kTargetMix) {
      uint32_t ffts_len;
      auto ret = System::Instance().get_c2c_addr_func_(reinterpret_cast<uint64_t*>(data_), &ffts_len);
      if (ret != RT_ERROR_NONE) return ret;
    }
    if (!atomic_clean_.empty()) {
      auto ret = LaunchAtomicClean(stream);
      if (ret != RT_ERROR_NONE) return ret;
    }
    if (extern_code_ >= 0) {
      return LaunchEx(workspace, stream);
    }
    uint8_t* stub_func = System::Instance().StubFunc(target_);
    return System::Instance().launch_func_(stub_func, block_dim_, data_, data_size_, nullptr, stream);
  }

  void LinkBody(uint64_t offset, const Code &code, const std::vector<NDAccess*> &ios, uint64_t ws_base);
  void DisAssemble(std::ostringstream &oss);

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
  int extern_code_{-1};
  std::vector<Code*> atomic_clean_;
  std::vector<std::pair<uint64_t*, uint64_t>> reloc_workspaces_;
  std::vector<std::pair<uint64_t*, uint64_t*>> reloc_reuse_;

 private:
  int LaunchAtomicClean(void* stream);
  int LaunchEx(void *workspace, void* stream);
  size_t mem_size_{0};
};
} // namespace dvm 
#endif // _DVM_CODE_H_
