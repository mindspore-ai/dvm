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

#ifndef _DVM_X_KERNEL_H_
#define _DVM_X_KERNEL_H_

#include <string>
#include <vector>
#include <map>
#include "kernel.h"

namespace dvm {
class MixKernel : public VKernel {
 public:
  MixKernel() : VKernel(KernelType::kStaticMix) {}
  ~MixKernel() override;

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;

 protected:
  void EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node);
  uint64_t SplitKCodeGen();
  uint64_t UnAlignCodeGen();
  uint64_t AlignCodeGen();
  uint64_t BiasBF16CodeGen();

  VKernelS *post_fusion_{nullptr};
  CubeOp *cube_op_{nullptr};
  NDAccess *sload_{nullptr};

  Kernel *stage_kernel_{nullptr};
};

class StagesKernel : public VKernel {
 public:
  StagesKernel() : VKernel(KernelType::kStaticStages) {}
  ~StagesKernel() override;

  void StageSwitch(KernelType type) {
    VKernel *kernel = nullptr;
    if (type == KernelType::kStaticShape) {
      kernel = new VKernelS();
    } else if (type == KernelType::kStaticMix) {
      kernel = new MixKernel();
    } else if (type == KernelType::kStaticParallel) {
      kernel = new VKernelP();
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
    store->is_stage_ = true;
    stages_.back()->kernel->Append(store);
    stages_.back()->ios.push_back(store);
  }

  void StageLoad(NDAccess *load, NDAccess *store) {
    load->is_stage_ = true;
    SetStageStore(load, store);
    stages_.back()->kernel->Append(load);
    stages_.back()->ios.push_back(load);
  }

  VKernel *Current() const { return stages_.back()->kernel; }

  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;

 protected:
  static void SetWorkspace(NDAccess *op, int64_t offset) { op->gm_ = reinterpret_cast<uint8_t *>(offset); }
  static int64_t GetWorkspace(NDAccess *op) { return reinterpret_cast<int64_t>(op->gm_); }
  static void SetOutputReuse(NDAccess *op, NDAccess *store) { op->gm_ = reinterpret_cast<uint8_t *>(store); }
  static NDAccess *GetOutputReuse(NDAccess *op) { return reinterpret_cast<NDAccess *>(op->gm_); }
  static void SetStageStore(NDAccess *op, NDAccess *store) { op->gm_ = reinterpret_cast<uint8_t *>(store); }
  static NDAccess *GetStageStore(NDAccess *op) { return reinterpret_cast<NDAccess *>(op->gm_); }

  uint64_t AllocWorkspace();

  struct Stage {
    Stage(VKernel *k) : kernel(k) {}
    VKernel *kernel;
    int64_t ws_size{-1};
    int64_t ws_offset{-1};
    std::vector<NDAccess *> ios;
  };
  std::vector<Stage *> stages_;
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

  const std::vector<EagerVector *> &GetKernels(int &num) {
    num = kernel_used_;
    return kernels_;
  }

  void Launch(void *stream) {
    for (int i = 0; i < kernel_used_; ++i) {
      reinterpret_cast<VKernel *>(kernels_[i])->code_.Launch(nullptr, stream);
    }
  }

  void Clear() {
    for (auto op : objects_) {
      op->~NDObject();
      NDObject::mem_pool_.Put(op);
    }
    objects_.clear();
    kernel_used_ = 0;
  }

  static NDAccess *GetStore(NDObject *obj) { return reinterpret_cast<NDAccess *>(obj->insn_); }

 protected:
  static int GetArea(NDObject *obj) { return obj->lead_dim_; }
  static void SetArea(NDObject *obj, int area_id) { obj->lead_dim_ = area_id; }
  static void SetStore(NDObject *obj, NDObject *store) { obj->insn_ = reinterpret_cast<uint64_t *>(store); }
  static void SetStoreInplace(NDObject *store, int flag) { store->reuse_dep_ = flag; }
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

  void Split(NDObject *root);
  NDObject *Exchange(EagerArea *area, NDObject *input);
  void CodeGenMix(EagerArea *area, EagerVector *kernel);

  std::vector<std::pair<EagerArea *, EagerArea *>> areas_;
  std::vector<EagerVector *> kernels_;
  int area_used_{0};
  int kernel_used_{0};
  std::vector<NDObject *> objects_;
  std::vector<NDObject *> temp_ops_;
  std::multimap<uint64_t, void *> wss_;
  WsAllocFunc ws_alloc_;
  void *user_data_;
  friend EagerArea;
};
}  // namespace dvm
#endif  // _DVM_X_KERNEL_H_
