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
  void Alloc(size_t size);
  void MoveCode(Code &other);

  void UpdateHead(uint64_t data, uint64_t simd_width, uint64_t flags, uint64_t ktype) {
    uint64_t *head = reinterpret_cast<uint64_t *>(data_);
    head[0] = 0;
    head[1] = data | simd_width << V_ENTRY_SIMD_WIDTH_OFFSET | flags | ktype |
              (static_cast<uint64_t>(data_size_) / sizeof(uint64_t) - 2) << V_ENTRY_CODE_SIZE_OFFSET;
  }

  void UpdateV(uint64_t tile_num, uint64_t simd_width) {
    target_ = kTargetVec;
    uint64_t block_dim = static_cast<uint64_t>(block_dim_);
    uint64_t block_tile = CeilDiv<uint64_t>(tile_num, block_dim);
    uint64_t block_tail = block_dim * block_tile - tile_num;
    ASSERT(block_tail < block_dim);
    uint64_t data = block_tile << V_ENTRY_V_TILE_BODY_OFFSET | block_tail << V_ENTRY_V_TILE_TAIL_OFFSET |
                    block_dim << V_ENTRY_V_BLOCK_NUM_OFFSET;
    UpdateHead(data, simd_width, 0, V_ENTRY_TYPE_V);
  }

  void UpdateVP() {
    target_ = kTargetVec;
    UpdateHead(static_cast<uint64_t>(block_dim_) << V_ENTRY_VP_BLOCK_SUM_OFFSET, 0, 0, V_ENTRY_TYPE_VP);
  }

  void UpdateC(uint64_t group_num) {
    target_ = kTargetCube;
    UpdateHead(group_num << V_ENTRY_M_GROUP_NUM_OFFSET, 0, 0, V_ENTRY_TYPE_C);
  }

  void UpdateMix(uint64_t group_num, uint64_t simd_width, uint64_t flags) {
    target_ = kTargetMix;
    UpdateHead(group_num << V_ENTRY_M_GROUP_NUM_OFFSET, simd_width, flags, V_ENTRY_TYPE_MIX);
  }

  static constexpr uint64_t HeadSize() { return sizeof(uint64_t) * 2; }  // ffts + entry

  uint64_t ReserveWorkspace(uint64_t workspace_size) {
    return data_size_ <= PARAM_TABLE_LIMIT ? workspace_size : ReserveCodeSpace(workspace_size);
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

  void Combine(const Code &code, uint64_t ws_base);
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

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
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
    if (unlikely(data_size_ > PARAM_TABLE_LIMIT)) {
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
  uint64_t ReserveCodeSpace(uint64_t workspace_size);

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
