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
#include <string>
#include "dvm.h"
#include "isa.h"

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cout << "[ASSERT ERROR]" << __FILE__ << ":" << __LINE__             \
	        << ": ASSERT(" << #cond << ")" << std::endl;                   \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

namespace dvm {
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

// {sizeof(int8_t). sizeof(float16), sizeof(float32), sizeof(int32_t)}
const uint64_t ITEM_SIZE[dvm::kTypeEnd] = {sizeof(int8_t), 2, sizeof(float), sizeof(int32_t)};

struct Code {
  Code() = default;
  Code(const Code &obj) = delete;
  Code &operator=(const Code &) = delete;
  ~Code() {
    if (data)
      std::free(data);
  }
  void Reset() {
    insn_num = 0;
    simd_width = 0;
    tile_num = 0;
    tile_mem = 0;
  }
  bool isGen() { return data != nullptr; }
  void Alloc(size_t s) {
    if (data) {
      data = static_cast<unsigned char *>(std::realloc(data, s));
    } else {
      data = static_cast<unsigned char *>(std::malloc(s));
    }
  }
  void FillHead() {
    uint64_t *ptr = reinterpret_cast<uint64_t*>(data);
    *ptr = (tile_num - 1) << 32 | simd_width << 24 | insn_num << 16 | (size - sizeof(uint64_t) + 31) / 32;
  }
  uint64_t HeadSize() const { return sizeof(uint64_t); }

  uint64_t BlockDim() const {
    static auto device_core_num = DeviceInfo::Instance().CoreNum();
    auto tile_per_block = std::max((tile_num + device_core_num - 1) / device_core_num, core_tile_least_);
    return (tile_num + tile_per_block - 1) / tile_per_block;
  }

  void DisAssemble(std::ostringstream &oss);

  unsigned char *data{nullptr};
  size_t size{0};
  uint64_t insn_num{0};
  uint64_t simd_width{0};
  uint64_t tile_num{0};
  uint64_t tile_mem{0};
  uint64_t core_tile_least_{1};
};

} // namespace dvm 
#endif // _DVM_CODE_H_
