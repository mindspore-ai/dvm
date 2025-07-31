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
#include <map>
#include "kernel.h"

namespace dvm {
class MixKernel : public VKernel {
 public:
  MixKernel() : VKernel(KernelType::kStaticMix), tuner_(System::Instance().online_tuner_) {}
  ~MixKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void SetTuner(CubeTuner *tuner) { tuner_ = tuner; }

 protected:
  void EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node);
  void Release();
  uint64_t SplitKCodeGen();
  uint64_t UnAlignCodeGen();
  uint64_t AlignCodeGen();
  uint64_t BiasBF16CodeGen();

  VectorKernel *post_fusion_{nullptr};
  CubeOp *cube_op_{nullptr};
  NDAccess *sload_{nullptr};

  Kernel *stage_kernel_{nullptr};
  CubeTuner *tuner_;
  RelocAddr gm_pos_;
};

class DynMixKernel : public MixKernel {
 public:
  DynMixKernel();

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
};

class StagesKernel;
class StageCodeWrap : public CodeWrap {
 public:
  explicit StageCodeWrap(StagesKernel *kernel) : kernel_(kernel) {}
  int LaunchWrap(void *workspace, void *stream) override;
  bool DasWrap(std::ostringstream &oss) override;

 private:
  StagesKernel *kernel_;
};

class StagesKernel : public VKernel {
 public:
  StagesKernel() : VKernel(KernelType::kStaticStages), code_wrap_(this) {}
  ~StagesKernel() override;

  void StageSwitch(KernelType type) {
    VKernel *kernel = nullptr;
    if (type == KernelType::kStaticShape) {
      kernel = new VKernelS();
    } else if (type == KernelType::kStaticMix) {
      kernel = new MixKernel();
    } else if (type == KernelType::kStaticParallel) {
      kernel = new VKernelP();
    } else if (type == KernelType::kDynMix) {
      kernel = new DynMixKernel();
    } else {
      ASSERT(0);
    }
    stages_.push_back(new Stage(kernel));
  }

  void ParallelSwitch() {
    auto current = stages_.back()->kernel;
    ASSERT(current->KType() != KernelType::kStaticParallel);
    static_cast<VKernelP *>(current)->AppendNext();
  }

  void StageStore(NDAccess *store) {
    store->SetFlag(OBJ_FLAG_STAGE_IO);
    stages_.back()->kernel->Append(store);
    stages_.back()->ios.push_back(store);
  }

  void StageLoad(NDAccess *load, NDAccess *store) {
    load->SetFlag(OBJ_FLAG_STAGE_IO);
    SetStageStore(load, store);
    stages_.back()->kernel->Append(load);
    stages_.back()->ios.push_back(load);
  }

  VKernel *Current() const { return stages_.back()->kernel; }
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;

 protected:
  static void SetWorkspace(NDAccess *op, int64_t offset) { op->addr_.ws = offset; }
  static int64_t GetWorkspace(NDAccess *op) { return op->addr_.ws; }
  static void SetOutputReuse(NDAccess *op, NDAccess *store) { op->addr_.gm = static_cast<void *>(store); }
  static NDAccess *GetOutputReuse(NDAccess *op) { return static_cast<NDAccess *>(op->addr_.gm); }
  static void SetStageStore(NDAccess *op, NDAccess *store) { op->addr_.gm = static_cast<void *>(store); }
  static NDAccess *GetStageStore(NDAccess *op) { return static_cast<NDAccess *>(op->addr_.gm); }

  uint64_t AllocWorkspace();

  struct Stage {
    explicit Stage(VKernel *k) : kernel(k) {}
    VKernel *kernel;
    int64_t ws_size{-1};
    int64_t ws_offset{-1};
    std::vector<NDAccess *> ios;
  };
  std::vector<Stage *> stages_;
  StageCodeWrap code_wrap_;
  friend StageCodeWrap;
};

class EagerVector;
class EagerArea;
class VKernelE : public VKernel {
 public:
  VKernelE(WsAllocFunc func, void *user_data);
  ~VKernelE() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  std::string &DisAssemble() override;
  NDObject *AppendCube(CubeOp *mm);

  const std::vector<EagerVector *> &GetKernels(int &begin, int &end) {
    begin = kernel_begin_;
    end = kernel_used_;
    return kernels_;
  }

  void Launch(int kernel_idx, void *stream) {
    auto &code = reinterpret_cast<VKernel *>(kernels_[kernel_idx])->code_;
    if (code.target_ == Code::kTargetCube && System::Instance().lazy_tuner_) {
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

  void Clear() {
    for (auto op : objects_) {
      op->~NDObject();
      NDObject::mem_pool_.Put(op);
    }
    objects_.clear();
    area_used_ = 0;
    kernel_used_ = 0;
    pv_black_mask_ = 0;
  }

  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }
  static void SetStoreInplace(NDObject *store, int flag) { store->reuse_dep_ = flag; }

 protected:
  static int GetArea(NDObject *obj) { return obj->reserved_; }
  static void SetArea(NDObject *obj, int area_id) { obj->reserved_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static int GetStoreInplace(NDObject *store) { return store->reuse_dep_; }
  static void SetStoreSize(NDObject *store, uint64_t size) { store->xbuf_ = size; }
  static uint64_t GetStoreSize(NDObject *store) { return store->xbuf_; }

  void InitObjInfo(NDObject *obj, int area = -1) {
    SetArea(obj, area);
    SetStore(obj, nullptr);
  }

  void InitStoreInfo(NDObject *store, int area = -1) {
    InitObjInfo(store, area);
    SetStoreSize(store, store->Size());
    SetStoreInplace(store, 0);
  }

  void *AllocWS(NDAccess *store) {
    auto size = GetStoreSize(store);
    if (auto it = wss_.upper_bound(size - 1); it != wss_.end()) {
      store->addr_.gm = it->second;
      SetStoreSize(store, it->first);
      wss_.erase(it);
    } else {
      store->addr_.gm = ws_alloc_(size, user_data_);
    }
    return store->addr_.gm;
  }

  void AllocVectorWSS(VectorKernel *kernel, size_t obj_size) {
    for (size_t i = obj_size; i < objects_.size(); ++i) {
      NDAccess *io = static_cast<NDAccess *>(objects_[i]);
      auto store = GetStore(io);
      if (store->addr_.gm == nullptr) {
        if (auto is =
              kernel->FindInplaceStore(io, [](NDAccess *op) -> bool { return VKernelE::GetStoreInplace(op) == 0; })) {
          store->addr_.gm = is->addr_.gm;
          SetStoreInplace(is, 1);
        } else {
          AllocWS(store);
        }
      }
      io->addr_.gm = store->addr_.gm;
    }
    objects_.resize(obj_size);
  }

  void Split(NDObject *root);
  NDObject *SplitPush(EagerArea *area, NDObject *input);
  NDObject *Exchange(EagerArea *area, NDObject *input);
  bool AppendOp(EagerVector *kernel, NDObject *op);
  void AppendOps(EagerVector *kernel, const std::vector<NDObject *> &objects) {
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
      AppendOp(kernel, *it);
    }
  }
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
  std::vector<NDObject *> temp_ops_;
  std::multimap<uint64_t, void *> wss_;
  WsAllocFunc ws_alloc_;
  void *user_data_;
  friend EagerArea;
};
}  // namespace dvm
#endif  // _DVM_X_KERNEL_H_
