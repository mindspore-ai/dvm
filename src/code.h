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

class DeviceInfo {
 public:
  static DeviceInfo &Instance() {
    static DeviceInfo obj;
    return obj;
  }

  AiCoreArch Arch() const { return arch_; }
  uint64_t LocalMemSize() const { return local_mem_size_; }
  uint64_t UbWorkspaceSize() const { return ub_workspace_size_; }
  uint64_t CoreNum() const { return core_num_; }
  uint64_t EventNum() const { return event_num_; }

 private:
  DeviceInfo();
  AiCoreArch arch_;
  uint64_t local_mem_size_;
  uint64_t ub_workspace_size_;
  uint64_t event_num_;
  uint64_t core_num_;
};

const uint64_t SIMD_BLOCK_SIZE  = 32;
const uint64_t SIMD_REPEAT_SIZE = 256;

// {sizeof(int8_t), sizeof(float16), sizeof(bfloat16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, 2, sizeof(float), sizeof(int32_t)};

struct CodeBase {
  CodeBase() = default;
  CodeBase(const CodeBase &obj) = delete;
  CodeBase &operator=(const CodeBase &) = delete;
  virtual ~CodeBase() {
    if (data_)
      std::free(data_);
  }
  bool IsParallel() const {
    uint64_t* head = reinterpret_cast<uint64_t*>(data_);
    return (*head) >> 63;
  }
  void Alloc(size_t s) {
    if (data_) {
      data_ = static_cast<unsigned char *>(std::realloc(data_, s));
    } else {
      data_ = static_cast<unsigned char *>(std::malloc(s));
    }
  };
  virtual void DisAssemble(std::ostringstream &oss) = 0;
  unsigned char *data_{nullptr};
  size_t data_size_{0};
  uint64_t block_dim_{0};
  std::vector<CodeBase*> atomic_clean_{nullptr};
};

struct Code : public CodeBase {
  void Reset() {
    insn_num_ = 0;
    simd_width_ = 0;
    tile_num_ = 0;
    atomic_clean_.clear();
  }

  void FillHead() {
    uint64_t *ptr = reinterpret_cast<uint64_t*>(data_);
    *ptr = (tile_num_ - 1) << 40 | simd_width_ << 32 | insn_num_ << 16 | (data_size_ - sizeof(uint64_t) + 31) / 32;
  }
  uint64_t HeadSize() const { return sizeof(uint64_t); }

  void UpdateBlockDim(uint64_t core_num) {
    auto tile_per_block = (tile_num_ + core_num - 1) / core_num;
    block_dim_ = (tile_num_ + tile_per_block - 1) / tile_per_block;
  }

  void ApplyTileLimit(uint64_t core_tile_least) {
    while (block_dim_ > 1 && ((tile_num_ - 1) / block_dim_ + 1 < core_tile_least)) block_dim_--; // TODO: optimize me
  }
  void DisAssemble(std::ostringstream &oss) override;
  uint64_t insn_num_{0};
  uint64_t simd_width_{0};
  uint64_t tile_num_{0};
};

struct CodeP : public CodeBase {
 void LinkAll(std::vector<uint64_t> &offsets);
 void DisAssemble(std::ostringstream &oss) override;
 std::vector<Code*> children_;
};

} // namespace dvm 
#endif // _DVM_CODE_H_
