/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef _DVM_SCHEDULE_H_
#define _DVM_SCHEDULE_H_

#include "kernel.h"

namespace dvm {
class VectorSchedule {
 public:
  VectorSchedule(VectorKernel *kernel) : kernel_(kernel) {}
  virtual ~VectorSchedule() = default;
  void SpaceInit();
  void SpaceSplit(int dim, int64_t npart, int64_t nfactor);
  void SpaceTrans(int dim1, int dim2);
  void SaveSpace();
  void ApplySubSpace(const DimArray &size) {
    for (auto &info : space_records_) {
      if (info.bcast_mask == SpaceRecord::OP_MASK) {
        info.change_op->DimChanged();
      } else {
        info.ndd->Reset();
        auto &dims = info.ndd->dims;
        for (size_t i = 0; i < dims.size(); ++i) {
          dims[i] = (info.bcast_mask >> i) & 1ul ?  1 : size[i];
        }
      }
    }
  }

  struct SpaceRecord {
    uint32_t bcast_mask;
    union {
      NDSpaceData *ndd;
      NDObject *change_op;
    };
    static constexpr uint32_t OP_MASK = 0xffffffffu;
  };
  std::vector<SpaceRecord> space_records_;
  VectorKernel *kernel_;
};

class SchGenHelper : public VectorSchedule {
 public:
  SchGenHelper(VectorKernel *kernel) : VectorSchedule(kernel) {}
  ~SchGenHelper() override;
  virtual int64_t CodeGen();
  RelocAddr *ReserveReloc(size_t size) {
    if (size > reloc_size_) {
      delete []reloc_array_;
      reloc_array_ = new RelocAddr[size];
      reloc_size_ = size;
    }
    return reloc_array_;
  }
 protected:
  void AllocStride(NDAccess *acc);
  void ResetStrides() {
    for (size_t i = 0; i < ext_stride_used_; ++i) {
      ext_strides_[i].acc->stride_ = nullptr;
    }
    ext_stride_used_ = 0;
  }

  struct ExtStride {
    NDAccess *acc;
    DimArray stride;
  };
  size_t ext_stride_used_{0};
  std::vector<ExtStride> ext_strides_;
  size_t reloc_size_{0};
  RelocAddr *reloc_array_{nullptr};
};

class FractalSchGen : public SchGenHelper {
 public:
  FractalSchGen(VectorKernel *kernel);
  int64_t CodeGen() override;
};

class ConcatSchGen : public SchGenHelper {
 public:
  ConcatSchGen(VectorKernel *kernel, ConcatOp *concat, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;
 protected:
  ConcatOp *concat_;
  struct SliceIO {
    std::vector<NDObject *> ios;
    std::vector<NDObject *> deads;
    size_t load_num{0};
    uint64_t slice_mask{0};
    uint64_t bcast_mask{0};
  };
  std::vector<SliceIO> slice_ios_;
};

class SplitSchGen : public SchGenHelper {
 public:
  SplitSchGen(VectorKernel *kernel, SplitOpM *split, const std::vector<NDObject *> &objects);
  int64_t CodeGen() override;
 protected:
  SplitOpM *split_;
  struct SliceIO {
    std::vector<NDObject *> ios;
    std::vector<NDObject *> deads;
    size_t load_num{0};
    uint64_t slice_mask{0};
    uint64_t bcast_mask{0};
  };
  std::vector<SliceIO> slice_ios_;
};

class DupTilingSchGen : public SchGenHelper {
 public:
  explicit DupTilingSchGen(VectorKernel *kernel) : SchGenHelper(kernel) {}
  int64_t DupCodeGen(int split_dim, int64_t truck_size);
};
}  // namespace dvm
#endif  // _DVM_SCHEDULE_H_