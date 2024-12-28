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
#include <atomic>
#include "isa.h"
#include "system.h"
#include "ops.h"

namespace dvm {
class NDAccess;
class Code {
 public:
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  Code() = default;
  Code(const Code &obj) = delete;
  Code &operator=(const Code &) = delete;
  Code &operator=(Code &&other);
  ~Code();
  void Clear() {
    sub_codes_.clear();
    unique_ids_.clear();
    bind_wss_ = nullptr;
    bind_ops_ = nullptr;
  }
  void ResetEager(int target) {
    target_ = target;
    Clear();
  }
  void Alloc(size_t size);
  void MoveCode(Code &other);

  void UpdateHead(uint64_t tile_num, uint64_t simd_width, uint64_t flags) {
    uint64_t *head = reinterpret_cast<uint64_t *>(data_);
    head[0] = 0;
    head[1] = tile_num << V_ENTRY_TILE_NUM_OFFSET | simd_width << V_ENTRY_SIMD_WIDTH_OFFSET |
              (static_cast<uint64_t>(data_size_) / sizeof(uint64_t) - 2) << V_ENTRY_CODE_SIZE_OFFSET | flags;
  }
  void UpdateParallelHead() { UpdateHead(block_dim_, 0, V_ENTRY_FLAG_PARALLEL); }

  uint64_t HeadSize() const { return sizeof(uint64_t) * 2; }  // ffts + entry

  uint64_t ReserveWorkspace(uint64_t workspace_size) {
    if (data_size_ <= PARAM_TABLE_LIMIT) {
      extern_code_ = -1;
      return workspace_size;
    }
    uint64_t *head = reinterpret_cast<uint64_t *>(data_);
    head[1] |= V_ENTRY_FLAG_EXTERN_CODE;
    const uint64_t align = 512;
    extern_code_ = (workspace_size + align) & ~(align - 1);
    return extern_code_ + data_size_;
  }

  int Launch(void *workspace, void *stream) {
    if (!sub_codes_.empty()) {
      for (auto a : sub_codes_) {
        auto ret = a->DoLaunch(workspace, stream);
        if (ret != RT_ERROR_NONE) return ret;
      }
    }
    // std::ostringstream oss;
    // this->DisAssemble(oss);
    // std::cout << oss.str() << std::endl;
    return DoLaunch(workspace, stream);
  }

  void LinkBody(uint64_t offset, const Code &code, const std::vector<NDAccess *> &ios, uint64_t ws_base);
  void DisAssemble(std::ostringstream &oss);

  void RelocBinds(void *workspace) {
    for (auto op = bind_wss_; op != nullptr; op = op->bind_list_) {
      op->Reloc(static_cast<char *>(workspace) + op->addr_.ws);
    }
    for (auto op = bind_ops_; op != nullptr; op = op->bind_list_) {
      op->Reloc(reinterpret_cast<void *>(*op->addr_.op->reloc_addr_));
    }
  }
  void BindWorkspace(NDAccess *op, uint64_t offset) {
    op->addr_.ws = offset;
    InsertBind(bind_wss_, op);
  }
  void BindOpFast(NDAccess *op, NDAccess *target) {
    op->addr_.op = target;
    InsertBind(bind_ops_, op);
  }
  void BindOp(NDAccess *op, NDAccess *target);
  void CombineBinds(const Code &code, uint64_t ws_base);

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
  int extern_code_{-1};
  NDAccess *bind_wss_{nullptr};
  NDAccess *bind_ops_{nullptr};
  std::vector<Code *> sub_codes_;
  std::vector<uint32_t *> unique_ids_;  // used to ensure softsync work, not affected by last kernel
  size_t mem_size_{0};

 private:
  int DoLaunch(void *workspace, void *stream) {
    if (!unique_ids_.empty()) {
      uint32_t cur_id = ++unique_id_;
      for (auto id : unique_ids_) {
        *id = cur_id;
      }
    }
    if (extern_code_ >= 0) {
      return LaunchEx(workspace, stream);
    }
    if (target_ == kTargetMix) {
      uint32_t ffts_len;
      auto ret = System::Instance().rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(data_), &ffts_len);
      if (ret != RT_ERROR_NONE) return ret;
    }
    uint8_t *stub_func = System::Instance().StubFunc(target_);
    return System::Instance().rtKernelLaunch(stub_func, block_dim_, data_, data_size_, stream);
  }
  int LaunchEx(void *workspace, void *stream);

  void InsertBind(NDAccess* &pos, NDAccess *op) {
#ifdef DEBUG
    for (auto x = pos; x != nullptr; x = x->bind_list_) {
      ASSERT(x != op);
    }
#endif
    op->bind_list_ = pos;
    pos = op;
  }

  static std::atomic<uint32_t> unique_id_;  // each kernel has a unique id
};
}  // namespace dvm
#endif  // _DVM_CODE_H_
