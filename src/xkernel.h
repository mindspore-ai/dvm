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

#ifndef _DVM_X_KERNEL_H_
#define _DVM_X_KERNEL_H_

#include <string>
#include <vector>
#include "kernel.h"

namespace dvm {
class StagesKernel;
class MixKernel : public VKernel {
 public:
  MixKernel(uint32_t flags = 0) : VKernel(KernelType::kMix, flags), tuner_(g_system.online_tuner_) {}
  ~MixKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void SetTuner(CubeTuner *tuner) { tuner_ = tuner; }

 protected:
  void EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node);
  virtual void Release();
  uint64_t SplitKCodeGen();
  uint64_t UnAlignCodeGen();
  uint64_t AlignCodeGen();
  uint64_t BiasBF16CodeGen();

  VKernelS *post_fusion_{nullptr};
  CubeOp *cube_op_{nullptr};
  NDAccess *sload_{nullptr};

  StagesKernel *stage_kernel_{nullptr};
  CubeTuner *tuner_;
  RelocAddr gm_pos_;
  std::vector<std::pair<NDAccess *, NDAccess *>> reloads_;
};

class DynMixKernel : public MixKernel {
 public:
  DynMixKernel() : MixKernel(KernelFlag::kDynamic) { tuner_ = nullptr; }
  uint64_t CodeGen() override;

 protected:
  void Release() override;
  void Record();
  GraphTracker tracker_;
};

class StageCodeWrap : public CodeWrap {
 public:
  explicit StageCodeWrap(StagesKernel *kernel) : kernel_(kernel) {}
  int LaunchWrap(void *workspace, void *stream) override;
  void DasWrap(std::ostringstream &oss) override;

 private:
  StagesKernel *kernel_;
};

class StagesKernel : public VKernel {
 public:
  StagesKernel(uint32_t flags = 0) : VKernel(KernelType::kSequence, flags), builder_(this), code_wrap_(this) {}
  ~StagesKernel() override;

  VKernel *Current() const { return stages_.back()->kernel; }
  VKernel *KernelAt(size_t idx) const { return stages_[idx]->kernel; }

  void AddStage(VKernel *k) {
    SetStageIndex(k, stages_.size());
    stages_.push_back(new Stage(k));
  }
  void StageStore(VKernel *k, NDAccess *store) {
    stages_[GetStageIndex(k)]->StageStore(store);
  }
  void StageLoad(VKernel *k, NDAccess *load, NDAccess *store) {
    stages_[GetStageIndex(k)]->StageLoad(load, store);
  }
  void StageLoadRecord(NDAccess *load, NDAccess *store) { sloads_.emplace_back(load, store); }

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;

  class _Builder : public KernelBuilder {
   public:
    _Builder(StagesKernel *impl) : KernelBuilder(impl) {}
    void StageSwitch(KernelType type);
    NDObject *StageLoad(NDObject *stage_store);
    NDObject *StageStore(NDObject *input);
    NDObject *StagePadStore(NDObject *input, int64_t pad_size);
    StagesKernel *_StagesKernel() const { return static_cast<StagesKernel *>(kernel_); }
  };
  _Builder builder_;

 protected:
  static void SetWorkspace(NDAccess *op, int64_t offset) { op->addr_.ws = offset; }
  static int64_t GetWorkspace(NDAccess *op) { return op->addr_.ws; }
  static void SetOutputReuse(NDAccess *op, NDAccess *store) { op->addr_.gm = static_cast<void *>(store); }
  static NDAccess *GetOutputReuse(NDAccess *op) { return static_cast<NDAccess *>(op->addr_.gm); }
  static void SetStageStore(NDAccess *op, NDAccess *store) { op->addr_.gm = static_cast<void *>(store); }
  static NDAccess *GetStageStore(NDAccess *op) { return static_cast<NDAccess *>(op->addr_.gm); }
  static void SetStageIndex(VKernel *k, int idx) { k->code_.target_ = idx; }
  static int GetStageIndex(VKernel *k) { return k->code_.target_; }

  uint64_t AllocWorkspace();

  struct Stage {
    explicit Stage(VKernel *k) : kernel(k) {}
    VKernel *kernel;
    int64_t ws_size{-1};
    int64_t ws_offset{-1};
    std::vector<NDAccess *> ios;

    void StageStore(NDAccess *store) {
      store->SetFlag(OBJ_FLAG_STAGE_IO);
      ios.push_back(store);
    }
    void StageLoad(NDAccess *load, NDAccess *store) {
      load->SetFlag(OBJ_FLAG_STAGE_IO);
      SetStageStore(load, store);
      ios.push_back(load);
    }
  };
  std::vector<Stage *> stages_;
  StageCodeWrap code_wrap_;
  friend StageCodeWrap;
  std::vector<std::pair<NDAccess *, NDAccess *>> sloads_;
};

class SequenceKernel : public StagesKernel {
 public:
  SequenceKernel(uint32_t flags) : StagesKernel(flags) {}
  void Append(NDObject *obj) override;
 protected:
  static int GetStage(NDObject *obj) { return obj->prop_id_; }
  static void SetStage(NDObject *obj, int area_id) { obj->prop_id_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }
};

class SplitContext {
 public:
  enum { MEM_BLOCK_SIZE = 32 };
  struct MemNode {
    size_t size;
    void *addr;
    MemNode *next;
  };

  SplitContext();
  ~SplitContext();

  MemNode *Alloc(size_t size) { return mem_head_ && size <= mem_tail_->size ? DoAlloc(size) : nullptr; }
  MemNode *DoAlloc(size_t size);
  void Free(void *addr, size_t size);
  void MemReset() {
    mem_head_ = nullptr;
    mem_pos_ = 0;
  }

  std::vector<NDObject *> app_;
  std::vector<NDAccess *> gen_;
  std::vector<std::pair<NDAccess *, size_t>> kill_;
  std::vector<NDObject *> build_;

  using SlotWorkspace = std::vector<std::pair<size_t, RelocAddr *>>;
  SlotWorkspace slot_ws_;

 protected:
  std::vector<MemNode *> mem_blocks_;
  MemNode *mem_head_;
  MemNode *mem_tail_;
  uint64_t mem_pos_;
};

class EagerVector;
class EagerArea;
class _SplitKernel : public VKernel {
 public:
  _SplitKernel(KernelType type, uint32_t flags);
  ~_SplitKernel() override;

  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  std::string &DisAssemble() override;
  virtual void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc);

  NDObject *AppendCube(CubeOp *mm);

  const std::vector<EagerVector *> &GetKernels(int &begin, int &end) {
    begin = kernel_begin_;
    end = kernel_used_;
    return kernels_;
  }

  void Launch(int kernel_idx, void *stream) {
    auto &code = reinterpret_cast<VKernel *>(kernels_[kernel_idx])->code_;
    if (code.target_ == Code::kTargetCube && g_system.lazy_tuner_) {
      TunerLaunch(kernels_[kernel_idx], stream);
    } else {
      code.Launch(extern_code_, stream);
    }
  }

  void Launch(void *stream) {
    for (int i = kernel_begin_; i < kernel_used_; ++i) {
      Launch(i, stream);
    }
  }

  void SetSlotWorkspace();
  void SlotCodeGen(const RelocEntry *relocs, size_t reloc_size);

  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }
  static void SetStoreInplace(NDObject *store, int flag) { store->reuse_dep_ = flag; }
  static CubeOp *GetCubeOp(EagerVector *kernel);

 protected:
  static int GetArea(NDObject *obj) { return obj->prop_id_; }
  static void SetArea(NDObject *obj, int area_id) { obj->prop_id_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static int GetStoreInplace(NDObject *store) { return store->reuse_dep_; }
  static void SetStoreSize(NDObject *store, uint64_t size) { store->xbuf_ = size; }
  static uint64_t GetStoreSize(NDObject *store) { return store->xbuf_; }
  static NDAccess *GetRecentLoad(NDAccess *store) { return reinterpret_cast<NDAccess *>(store->addr_.reloc_); }
  static void SetRecentLoad(NDAccess *store, NDAccess *load) { store->addr_.reloc_ = reinterpret_cast<uint64_t *>(load); }

  void Reset() {
    objects_.clear();
    area_used_ = 0;
    kernel_used_ = 0;
    pv_black_mask_ = 0;
  }

  void InitObjInfo(NDObject *obj, int area = -1) {
    SetArea(obj, area);
    SetStore(obj, nullptr);
  }

  void InitStoreInfo(NDObject *store, int area = -1) {
    InitObjInfo(store, area);
    SetStoreSize(store, store->Size());
    SetStoreInplace(store, 0);
  }

  void *AllocWS(NDAccess *store, WsAllocator *alloc) {
    auto size = GetStoreSize(store);
    if (auto mem = ctx_->Alloc(size); mem != nullptr) {
      store->addr_.gm = mem->addr;
      SetStoreSize(store, mem->size);
    } else {
      store->addr_.gm = alloc->Alloc(size);
    }
    return store->addr_.gm;
  }

  void Split(NDObject *root);
  NDObject *SplitPush(EagerArea *area, NDObject *input);
  NDObject *Exchange(NDObject *input, int to_aid);
  void BuildKernel(EagerVector *kernel, const EagerArea *area, WsAllocator *alloc);
  void RelocBinds();
  void TunerLaunch(EagerVector *kernel, void *stream);

  std::vector<std::pair<EagerArea *, EagerArea *>> areas_;
  std::vector<EagerVector *> kernels_;
  union {
    int area_used_{0};
    int kernel_begin_;
  };
  int kernel_used_{0};
  union {
    uint64_t pv_black_mask_{0};
    void *extern_code_;
  };
  std::vector<NDObject *> objects_;
  SplitContext *__restrict__ ctx_;
  friend EagerArea;
};

class VKernelE : public _SplitKernel {
 public:
  VKernelE();
  ~VKernelE() override;

  void Clear() {
    for (auto op : objects_) {
      op->~NDObject();
      NDObject::mem_pool_.Put(op);
    }
    Reset();
  }
};

class _SplitGraph : public _SplitKernel {
 public:
  _SplitGraph(uint32_t flags);
  ~_SplitGraph() override;
  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  virtual void Infer();

 protected:
  std::vector<NDObject *> build_ops_;
};

class SplitGraphD : public _SplitGraph {
 public:
  SplitGraphD() : _SplitGraph(KernelFlag::kDynamic) {}
  void Append(NDObject *obj) override;
  void Infer() override;
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;

 protected:
  GraphTracker tracker_;
};

class SplitGraphS : public _SplitGraph {
 public:
  SplitGraphS(bool single_ws) : _SplitGraph(0), single_ws_(single_ws) {}
  void Infer() override;
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;

 protected:
  SplitContext::SlotWorkspace slot_ws_;
  bool single_ws_;
};

class SplitEagerW : public VKernelE {
 public:
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
};

class SplitGraphDW : public SplitGraphD {
 public:
  void CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) override;
};
}  // namespace dvm
#endif  // _DVM_X_KERNEL_H_
