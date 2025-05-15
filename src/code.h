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

#ifndef _DVM_CODE_H_
#define _DVM_CODE_H_
#include <sstream>
#include <cstring>
#include "isa.h"
#include "system.h"

extern const uint64_t g_visit_func_offset[];

namespace dvm {

class VisitCoder {
 public:
  void AddReloc(uint64_t *pc, uint64_t offset) { rel_relocs_.emplace_back(pc, offset); }
  void Clear() { rel_relocs_.clear(); }
  std::vector<std::pair<uint64_t *, uint64_t>> rel_relocs_;

 protected:
  ~VisitCoder() = default;
};

struct RedVisitCoder : public VisitCoder {
 public:
  uint32_t block_num_;
  uint32_t ws_size_;
  uint32_t visit_id_;
  uint32_t code_size_;
  uint64_t *code_{nullptr};
};

class MixVisitCoder : public VisitCoder {};

struct RelocAddr {
  void Reloc(void *dst) {
    if (reloc_) {
      *reloc_ = reinterpret_cast<uint64_t>(dst);
    }
  }
  void Update(uint64_t *insn) { reloc_ = insn; }
  void Update(const RelocAddr &share) { reloc_ = share.reloc_; }

  union {  // NOTICE: bind after codegen
    void *gm;
    uint64_t ws;
    const RelocAddr *op;
    uint64_t data;
  };
  uint64_t *reloc_{nullptr};
  RelocAddr *bind_list_{nullptr};
};

class Code;
class CodeWrap {
 public:
  CodeWrap() = default;
  virtual ~CodeWrap() {}
  virtual int LaunchWrap(void *workspace, void *stream);
  virtual void CombineWrap(Code *to, uint64_t ws_base);
  virtual bool DasWrap(std::ostringstream &oss);
  CodeWrap *next_{nullptr};
};

class Code : public CodeWrap {
 public:
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  Code() = default;
  Code(const Code &obj) = delete;
  Code &operator=(const Code &) = delete;
  Code &operator=(Code &&other);
  ~Code();
  void Clear() {
    bind_wss_ = nullptr;
    bind_ops_ = nullptr;
    wrap_ = nullptr;
  }
  void Alloc(size_t size);
  void MoveCode(Code &other);

  void UpdateHead(uint64_t data, uint64_t flags, uint64_t ktype) {
    uint64_t *head = reinterpret_cast<uint64_t *>(data_);
    head[0] = 0;
    head[1] =
      data | flags | ktype | (static_cast<uint64_t>(data_size_) / sizeof(uint64_t) - 2) << V_ENTRY_CODE_SIZE_OFFSET;
  }

  void UpdateV(uint64_t tile_num) {
    target_ = kTargetVec;
    uint64_t block_dim = static_cast<uint64_t>(block_dim_);
    uint64_t block_tile = CeilDiv<uint64_t>(tile_num, block_dim);
    uint64_t block_tail = block_dim * block_tile - tile_num;
    ASSERT(block_tail < block_dim);
    uint64_t data = block_tile << V_ENTRY_V_TILE_BODY_OFFSET | block_tail << V_ENTRY_V_TILE_TAIL_OFFSET |
                    block_dim << V_ENTRY_V_BLOCK_NUM_OFFSET;
    UpdateHead(data, 0, V_ENTRY_TYPE_V);
  }

  void UpdateVE(const RedVisitCoder *visit) {
    target_ = kTargetMix;
    uint64_t *visit_code = reinterpret_cast<uint64_t *>(data_ + data_size_);
    data_size_ += visit->code_size_;
    std::memcpy(visit_code, visit->code_, visit->code_size_);
    for (auto &r : visit->rel_relocs_) {
      *r.first |= static_cast<uint64_t>(visit_code - r.first) << V_HEAD_EXT_OFFSET;
    }
    uint64_t offset = visit_code - reinterpret_cast<uint64_t *>(data_) - 2;
    uint64_t data =
      g_visit_func_offset[visit->visit_id_] << V_ENTRY_VE_VISIT_ID_OFFSET | offset << V_ENTRY_VE_VISIT_OFFSET_OFFSET;
    UpdateHead(data, 0, V_ENTRY_TYPE_VE);
  }

  void UpdateVP() {
    target_ = kTargetVec;
    UpdateHead(static_cast<uint64_t>(block_dim_) << V_ENTRY_VP_BLOCK_SUM_OFFSET, 0, V_ENTRY_TYPE_VP);
  }

  void UpdateC() {
    target_ = kTargetCube;
    UpdateHead(0, V_ENTRY_FLAG_CUBE_MIX, V_ENTRY_TYPE_C);
  }

  void UpdateMix(const MixVisitCoder *visit, uint64_t slice[], uint64_t tail[], uint64_t stride[], uint64_t subtile0,
                 uint64_t subtile1) {
    target_ = kTargetMix;
    uint64_t *visit_code = reinterpret_cast<uint64_t *>(data_ + data_size_);
    data_size_ += vVisitMix::Encode(visit_code, slice, tail, stride, subtile0, subtile1) * sizeof(uint64_t);
    for (auto &r : visit->rel_relocs_) {
      *(r.first + r.second) |= static_cast<uint64_t>(visit_code - r.first);
    }
    uint64_t offset =
      visit_code - reinterpret_cast<uint64_t *>(data_) - (HeadSize() + sizeof(vCubeOp)) / sizeof(uint64_t);
    uint64_t data =
      g_visit_func_offset[V_VISIT_MIX] << V_ENTRY_VE_VISIT_ID_OFFSET | offset << V_ENTRY_VE_VISIT_OFFSET_OFFSET;
    UpdateHead(data, V_ENTRY_FLAG_CUBE_MIX, V_ENTRY_TYPE_VE);
  }

  static constexpr uint64_t HeadSize() { return sizeof(uint64_t) * 2; }  // ffts + entry

  uint64_t ReserveWorkspace(uint64_t workspace_size) {
    return data_size_ <= PARAM_TABLE_LIMIT ? workspace_size : ReserveCodeSpace(workspace_size);
  }

  int Launch(void *workspace, void *stream) {
    if (!wrap_) {
      return DoLaunch(workspace, stream);
    }
    return wrap_->LaunchWrap(workspace, stream);
  }

  void CombineBind(const Code &code, uint64_t ws_base);
  void Combine(const Code &code, uint64_t ws_base);
  void DisAssemble(std::ostringstream &oss);
  bool DasWrap(std::ostringstream &oss) override;
  int LaunchWrap(void *workspace, void *stream) override;
  void CombineWrap(Code *to, uint64_t ws_base) override;

  void InsertWrap(CodeWrap *wrap) {
    wrap->next_ = wrap_ ? wrap_ : this;
    wrap_ = wrap;
  }

  void RelocBinds(void *workspace) {
    for (auto op = bind_wss_; op != nullptr; op = op->bind_list_) {
      op->Reloc(static_cast<char *>(workspace) + op->ws);
    }
    for (auto op = bind_ops_; op != nullptr; op = op->bind_list_) {
      op->Reloc(reinterpret_cast<void *>(*op->op->reloc_));
    }
  }
  void BindWorkspace(RelocAddr &op, uint64_t offset) {
    op.ws = offset;
    InsertBind(bind_wss_, op);
  }
  void BindOpFast(RelocAddr &op, const RelocAddr &target) {
    op.op = &target;
    InsertBind(bind_ops_, op);
  }
  void BindOp(RelocAddr &op, const RelocAddr &target);

  unsigned char *data_{nullptr};
  uint32_t data_size_{0};
  uint32_t block_dim_{0};
  int target_{0};
  RelocAddr *bind_wss_{nullptr};
  RelocAddr *bind_ops_{nullptr};
  CodeWrap *wrap_{nullptr};
  size_t mem_size_{0};

 private:
  int DoLaunch(void *workspace, void *stream) {
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

  void InsertBind(RelocAddr *&pos, RelocAddr &op) {
#ifdef DEBUG
    for (auto x = pos; x != nullptr; x = x->bind_list_) {
      ASSERT(x != &op);
    }
#endif
    op.bind_list_ = pos;
    pos = &op;
  }
};
}  // namespace dvm
#endif  // _DVM_CODE_H_
