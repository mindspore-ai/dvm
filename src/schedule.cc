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

#include <algorithm>
#include "schedule.h"

namespace dvm {
void VectorSchedule::SpaceInit() {
  space_records_.clear();
  for (auto op : kernel_->objects_) {
    if (auto ndd = op->Ndd()) {
      space_records_.push_back({0, ndd});
    }
    if (op->IsSimd() && NDObject::meta_.dim_changed[op->obj_id_]) {
      auto &r = space_records_.emplace_back();
      r.bcast_mask = SpaceRecord::OP_MASK;
      r.change_op = op;
    }
  }
}

void VectorSchedule::SpaceSplit(int dim, int64_t npart, int64_t nfactor) {
  ASSERT(npart > 0 && nfactor > 0);
  auto &space = kernel_->DimSpace();
  size_t space_size = space.size();
  ASSERT(dim >= 0 && static_cast<size_t>(dim) < space_size);
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      size_t n = dims.size();
      int64_t orig_dim = dims[dim];
      dims.resize(n + 1);
      for (size_t i = n; i > static_cast<size_t>(dim) + 1; --i) {
        dims[i] = dims[i - 1];
      }
      if (orig_dim != 1) {
        dims[dim] = nfactor;
        dims[dim + 1] = npart;
      } else {
        dims[dim] = 1;
        dims[dim + 1] = 1;
      }
    }
  }
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < space_size) stride.resize(space_size, 0);
    int64_t orig_stride = stride[dim];
    size_t n = stride.size();
    stride.resize(n + 1);
    for (size_t j = n; j > static_cast<size_t>(dim) + 1; --j) {
      stride[j] = stride[j - 1];
    }
    stride[dim] = orig_stride;
    stride[dim + 1] = orig_stride * nfactor;
  }
}

void VectorSchedule::SpaceTrans(int dim1, int dim2) {
  auto &space = kernel_->DimSpace();
  ASSERT(dim1 >= 0 && static_cast<size_t>(dim1) < space.size());
  ASSERT(dim2 >= 0 && static_cast<size_t>(dim2) < space.size());
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      std::swap(r.ndd->dims[dim1], r.ndd->dims[dim2]);
    }
  }
  size_t nd_size = space.size();
  for (auto op : kernel_->static_ops_) {
    DimArray &stride = *static_cast<NDAccess *>(op)->stride_;
    if (stride.size() < nd_size) stride.resize(nd_size, 0);
    std::swap(stride[dim1], stride[dim2]);
  }
}

void VectorSchedule::SaveSpace() {
  for (auto &r : space_records_) {
    if (!r.bcast_mask) {
      auto &dims = r.ndd->dims;
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 1) {
          r.bcast_mask |= (1u << i);
        }
      }
    }
  }
}

class VectorDupHelper {
 public:
  VectorDupHelper(VectorKernel *kernel, int dup_num, RelocAddr *relocs, uint64_t total_quota)
      : kernel_(kernel), dup_num_(dup_num), dup_idx_(0), block_begin_(0), relocs_(relocs),
        remain_quota_(total_quota) {
    uint64_t dup = static_cast<uint64_t>(dup_num);
    // Reserve 1 core per segment up front (free_core_ = freely allocatable
    // cores); each segment gets its reserved core back via the +1 in DoAppend.
    uint64_t total_core = RoundUp(dup, g_system.CoreNum());
    free_core_ = total_core - dup;
    uint64_t code_reserve = dup_num * kernel_->ReserveCodeSize();
    encoder_.Reset(&kernel_->code_, Code::kTargetVec, dup_num, code_reserve, total_core);
  }

  void Reset() {
    int op_index = 0;
    for (auto op : kernel_->objects_) {
      op->Clear(op_index++);
    }
  }

  void DoAppend(uint64_t quota, uint64_t cap_core = 0) {
    ASSERT(dup_idx_ < dup_num_);
    ASSERT(remain_quota_ >= quota && quota > 0);
    dup_idx_++;
    kernel_->PrepareTiling();
    ASSERT(kernel_->tile_size_);
    auto code_begin = encoder_.ProgData();
    uint64_t code_reserve = kernel_->ReserveCodeSize();
    uint64_t core_limit = CeilDiv(quota * free_core_, remain_quota_) + 1;
    if (cap_core != 0 && core_limit > cap_core) {
      core_limit = cap_core;
    }
    remain_quota_ -= quota;
    auto code_end = kernel_->DoCodeGen(core_limit, code_begin, code_reserve);
    uint64_t code_size = code_end - code_begin;
    uint64_t block_dim = kernel_->CompactBlockDim(core_limit);
    ASSERT(kernel_->visit_ == nullptr);
    auto prog = encoder_.Append(Code::GenEntryV(kernel_->tile_num_, block_dim, code_size), code_size);
    encoder_.AssignAiv(block_begin_, block_dim, prog);
    block_begin_ += block_dim;
    for (auto op : kernel_->static_ops_) {
      if (!(op->flags_ & OBJ_FLAG_DEAD)) {
        auto &r = static_cast<NDAccess *>(op)->addr_;
        auto bind = relocs_++;
        bind->Update(r.reloc_);
        kernel_->code_.BindOpFast(*bind, r);
      }
    }
    free_core_ -= block_dim - 1;
  }

  void Append(uint64_t quota, uint64_t cap_core = 0) {
    Reset();
    DoAppend(quota, cap_core);
  }

  void Submit() {
    ASSERT(dup_idx_ == dup_num_);
    encoder_.Submit(block_begin_);
  }

 protected:
  VectorKernel *kernel_;
  PCodeEncoder encoder_;
  int dup_num_;
  int dup_idx_;
  uint64_t block_begin_;
  RelocAddr *relocs_;
  uint64_t free_core_;
  uint64_t remain_quota_;
};

SchGenHelper::~SchGenHelper() { delete []reloc_array_; }
int64_t SchGenHelper::CodeGen() { return -1; }

void SchGenHelper::AllocStride(NDAccess *acc) {
  if (ext_strides_.empty()) {
    ext_strides_.resize(kernel_->static_ops_.size());
  }
  auto &ext = ext_strides_[ext_stride_used_++];
  ext.acc = acc;
  acc->stride_ = &ext.stride;
  auto &dims = acc->nd_.dims();
  auto n = dims.size();
  ext.stride.resize(n);
  int64_t cur_stride = ITEM_SIZE[acc->type_id_];
  for (size_t i = 0; i < n; ++i) {
    ext.stride[i] = cur_stride;
    cur_stride *= dims[i];
  }
}

FractalSchGen::FractalSchGen(VectorKernel *kernel) : SchGenHelper(kernel) {
  for (auto op : kernel->static_ops_) {
    EXCEPTION_IF(!static_cast<NDAccess *>(op)->IsSupportView(), "unsupport view op");
  }
}

int64_t FractalSchGen::CodeGen() {
  constexpr int64_t min_dim_limit = 6;
  int h_idx = -1;
  int64_t item_size = 0;
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto load = static_cast<NDAccess *>(kernel_->static_ops_[i]);
    if (load->stride_ == nullptr) {
      continue;
    }
    auto &stride = *load->stride_;
    auto &dims = load->Ndd()->dims;
    int64_t type_size = ITEM_SIZE[load->type_id_];
    if (stride[0] != type_size && dims[0] >= min_dim_limit) {
      if (h_idx == -1) {
        if (type_size != 2 && type_size != 4) {
          continue;
        }
        for (size_t j = 1; j < stride.size(); ++j) {
          if (stride[j] == type_size) {
            if (dims[j] >= min_dim_limit) {
              h_idx = static_cast<int>(j);
              item_size = type_size;
              load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
            }
            break;
          }
        }
      } else if (static_cast<size_t>(h_idx) < stride.size() && type_size == item_size && stride[h_idx] == type_size && dims[h_idx] >= min_dim_limit) {
        load->SetFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL);
      }
    }
  }
  if (h_idx < 0) {
    return -1;
  }

  class FractalDunGen {
   public:
    FractalDunGen(SchGenHelper &gen, int h_idx, uint64_t item_size)
        : gen_(gen), space_(gen.kernel_->DimSpace()), h_idx_(h_idx) {
      w_fractal_ = g_system.Arch() == AiCoreArch::kAiCore_C310 ? 128 : 16;
      h_fractal_ = item_size == 2 ? 16 : 8;
      int64_t w_size = space_[0];
      w_body_ = w_size / w_fractal_;
      w_tail_ = w_size - w_body_ * w_fractal_;
      int64_t h_size = space_[h_idx];
      h_body_ = h_size / h_fractal_;
      h_tail_ = h_size - h_body_ * h_fractal_;
    }
    void CodeGen() {
      int64_t w_npart = w_body_;
      int64_t h_npart = h_body_;
      int dup_num;
      if (h_tail_ && w_tail_) {
        dup_num = h_body_ && w_body_ ? 4 : (h_body_ || w_body_ ? 2 : 1);
        w_npart++;
        h_npart++;
      } else if (h_tail_) {
        dup_num = h_body_ ? 2 : 1;
        h_npart++;
      } else if (w_tail_) {
        dup_num = w_body_ ? 2 : 1;
        w_npart++;
      } else {
        dup_num = 1;
      }
      gen_.SpaceInit();
      gen_.SpaceSplit(0, w_npart, w_fractal_);
      h_idx_ += 1;
      gen_.SpaceTrans(1, h_idx_);
      gen_.SpaceSplit(1, h_npart, h_fractal_);
      gen_.SaveSpace();
      part_base_ = 1;
      size_t w_part_dim = h_idx_ + 1;
      size_.resize(space_.size());
      size_[2] = space_[2];
      for (size_t i = 3; i < space_.size(); ++i) {
        size_[i] = space_[i];
        if (i != w_part_dim) {
          part_base_ *= space_[i];
        }
      }
      VectorDupHelper helper(gen_.kernel_, dup_num, gen_.ReserveReloc(gen_.kernel_->static_ops_.size() * 4),
                             h_npart * w_npart);
      if (w_tail_ && h_body_) {
        GenDup(helper, 0, w_body_, w_tail_, h_fractal_, h_body_, 1);
      }
      if (h_tail_ && w_body_) {
        GenDup(helper, h_body_, 0, w_fractal_, h_tail_, 1, w_body_);
      }
      if (w_tail_ && h_tail_) {
        GenDup(helper, h_body_, w_body_, w_tail_, h_tail_, 1, 1);
      }
      if (w_body_ && h_body_) {
        GenDup(helper, 0, 0, w_fractal_, h_fractal_, h_body_, w_body_);
      }
      helper.Submit();
    }

    void GenDup(VectorDupHelper &helper, int64_t h_part_off, int64_t w_part_off, int64_t w_fac_size, int64_t h_fac_size,
                int64_t h_part_size, int64_t w_part_size) {
      constexpr int w_factor_dim = 0;
      constexpr int h_factor_dim = 1;
      constexpr int h_part_dim = 2;
      int w_part_dim = h_idx_ + 1;
      size_[w_factor_dim] = w_fac_size;
      size_[h_factor_dim] = h_fac_size;
      size_[h_part_dim] = h_part_size;
      size_[w_part_dim] = w_part_size;
      gen_.ApplySubSpace(size_);
      // TODO: consider broadcast
      for (auto op : gen_.kernel_->static_ops_) {
        auto acc = static_cast<NDAccess *>(op);
        auto &stride = *acc->stride_;
        acc->ViewUpdate(stride[h_part_dim] * h_part_off + stride[w_part_dim] * w_part_off);
      }
      uint64_t part_num = h_part_size * w_part_size;
      helper.Append(part_num, part_base_ * part_num);
    };

    SchGenHelper &gen_;
    const DimArray &space_;
    int64_t h_fractal_;
    int64_t w_fractal_;
    int64_t w_body_;
    int64_t w_tail_;
    int64_t h_body_;
    int64_t h_tail_;
    DimArray size_;
    uint64_t part_base_;
    int h_idx_;
  };

  int64_t result = 0;
  for (auto op : kernel_->objects_) {
    if ((op->obj_id_ == ObjectType::kBroadcastTo || op->obj_id_ == ObjectType::kReduce) &&
        op->lhs_->nd_[0] != op->nd_[0]) {
      result = -1;
      break;
    }
  }
  if (!result) {
    for (auto op : kernel_->static_ops_) {
      auto acc = static_cast<NDAccess *>(op);
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
    FractalDunGen gen(*this, h_idx, item_size);
    gen.CodeGen();
    ResetStrides();
  }
  for (size_t i = 0; i < kernel_->load_num_; ++i) {
    auto op = kernel_->static_ops_[i];
    if (op->CheckFlag(OBJ_FLAG_VIEW_LOAD_FRACTAL)) {
      op->flags_ &= ~OBJ_FLAG_VIEW_LOAD_FRACTAL;
    }
  }
  return result;
}

namespace {
template <typename T>
void TravelInput(std::vector<NDObject *> &stack, NDObject *root, const T &func) {
  stack.push_back(root);
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (func(top)) {
      top->ForInput([&stack](NDObject *in) { stack.push_back(in); });
    }
  }
}

uint64_t AddPengLoad(std::vector<NDObject *> &ios, const std::vector<NDObject *> &pend_load) {
  uint64_t num = 0;
  for (auto op : pend_load) {
    if (std::find(ios.begin(), ios.end(), op) == ios.end()) {
        ios.push_back(op);
        num++;
    }
  }
  return num;
}
}

ConcatSchGen::ConcatSchGen(VectorKernel *kernel, ConcatOp *concat, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), concat_(concat) {
  constexpr int kInvalidDomain = -1;
  constexpr int kConcatDomain = -2;
  for (auto op : objects) {
    op->reuse_dep_ = kInvalidDomain;
  }
  concat->reuse_dep_ = kConcatDomain;
  std::vector<NDObject *> stack;
  int slice_cnt = concat->slices_.size();
  slice_ios_.resize(slice_cnt);
  for (int i = 0; i < slice_cnt; ++i) {
    auto &s = slice_ios_[i];
    TravelInput(stack, concat->slices_[i].input, [&s, i](NDObject *in) {
      if (in->IsLoad()) {
        s.ios.push_back(in);
        s.load_num++;
      }
      in->reuse_dep_ = i;
      return true;
    });
  }
  std::vector<NDObject *> pend_load;
  for (size_t k = 0; k < kernel->load_num_; ++k) {
    kernel->static_ops_[k]->xbuf_ = k;
  }
  for (size_t k = kernel->load_num_; k < kernel->static_ops_.size(); ++k) {
    auto store = kernel->static_ops_[k];
    store->xbuf_ = k;
    int joined = kInvalidDomain;
    TravelInput(stack, store, [&joined, &pend_load, kInvalidDomain](NDObject *in) -> bool {
      if (in->IsLoad()) {
        pend_load.push_back(in);
      }
      if (auto dom = in->reuse_dep_; dom != kInvalidDomain) {
        if (joined == kInvalidDomain) {
          joined = dom;
          return false;
        }
        EXCEPTION_IF(joined != dom && !in->IsLoad(), "concat multi domain depend");
      }
      return true;
    });
    store->reuse_dep_ = joined;
    if (joined == kConcatDomain) {
      for (auto &s : slice_ios_) {
        s.ios.push_back(store);
      }
    } else if (joined >= 0) {
      slice_ios_[joined].ios.push_back(store);
      for (size_t i = 0; i < slice_ios_.size(); ++i) {
        if (i != static_cast<size_t>(joined)) {
          slice_ios_[i].deads.push_back(store);
        }
      }
    } else {
      DvmException("store is not joined");  // TODO: add seperate path if joined is invalid
    }
    if (!pend_load.empty()) {
      for (auto op : pend_load) {
        EXCEPTION_IF(
          (op->reuse_dep_ >= 0 && joined == kConcatDomain) || (op->reuse_dep_ == kConcatDomain && joined >= 0),
          "concat multi domain depend");
        op->reuse_dep_ = joined;
      }
      if (joined == kConcatDomain) {
        for (auto &s : slice_ios_) {
          s.load_num += AddPengLoad(s.ios, pend_load);;
        }
      } else {
        slice_ios_[joined].load_num += AddPengLoad(slice_ios_[joined].ios, pend_load);;
      }
      pend_load.clear();
    }
  }
  for (auto &s : slice_ios_) {
    std::sort(s.ios.begin(), s.ios.end(), [](NDObject *a, NDObject *b) { return a->xbuf_ < b->xbuf_; });
    for (size_t i = 0; i < s.ios.size(); ++i) {
      if (s.ios[i]->reuse_dep_ != kConcatDomain) {
        s.slice_mask |= 1ull << i;
      }
    }
  }
}

int64_t ConcatSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  DimArray size = concat_->nd_.dims();
  int cat_dim = concat_->CatDim();
  for (auto &s : slice_ios_) {
    s.bcast_mask = 0;
    for (size_t j = 0; j < s.ios.size(); ++j) {
      if ((s.slice_mask >> j) & 1) continue;  // concat input partition: skip
      auto acc = static_cast<NDAccess *>(s.ios[j]);
      if (acc->nd_[cat_dim] == 1) {
        s.bcast_mask |= (1ULL << j);
      }
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int dup_num = concat_->slices_.size();
  auto &static_ops = kernel_->static_ops_;
  VectorDupHelper helper(kernel_, dup_num, ReserveReloc(static_ops.size() * dup_num), size[cat_dim]);
  size_t cat_offset = 0;
  auto ctx = concat_->PartialInit();
  for (int cat_idx = 0; cat_idx < dup_num; ++cat_idx) {
    auto &slice = concat_->slices_[cat_idx];
    auto cat_size = slice.size;
    size[cat_dim] = cat_size;
    concat_->PartialSet(slice.input);
    ApplySubSpace(size);
    helper.Reset();
    auto &entry = slice_ios_[cat_idx];
    for (size_t j = 0; j < entry.ios.size(); ++j) {
      auto acc = static_cast<NDAccess *>(entry.ios[j]);
      if (!((entry.slice_mask >> j) & 1)) {
        auto bcast = (entry.bcast_mask >> j) & 1;
        acc->ViewUpdate(bcast ? 0 : (*acc->stride_)[cat_dim] * static_cast<uint64_t>(cat_offset));
      }
    }
    for (auto op : entry.deads) {
      op->flags_ |= OBJ_FLAG_DEAD;
    }
    std::swap(static_ops, entry.ios);
    kernel_->load_num_ = entry.load_num;
    cat_offset += cat_size;
    helper.DoAppend(cat_size);
    std::swap(static_ops, entry.ios);
  }
  concat_->PartialRecover(ctx);
  helper.Submit();
  ResetStrides();
  return 0;
}

SplitSchGen::SplitSchGen(VectorKernel *kernel, SplitOpM *split, const std::vector<NDObject *> &objects)
    : SchGenHelper(kernel), split_(split) {
  ASSERT(split_->slice_idx_ == 0);
  constexpr int kInvalidDomain = -1;
  constexpr int kSplitDomain = -2;
  constexpr int kSliceDomain = -3;
  for (auto op : objects) {
    op->reuse_dep_ = kInvalidDomain;
  }
  std::vector<NDObject *> stack;
  std::vector<NDObject *> pend_load;
  TravelInput(stack, split->lhs_, [&pend_load, kSplitDomain](NDObject *in) {
    in->reuse_dep_ = kSplitDomain;
    if (in->IsLoad()) {
      pend_load.push_back(in);
    }
    return true;
  });
  slice_ios_.resize(split->siblings_.size());
  for (int i = 0; i < static_cast<int>(split->siblings_.size()); ++i) {
    split->siblings_[i]->reuse_dep_ = i;
    slice_ios_[i].ios = pend_load;
    slice_ios_[i].load_num = pend_load.size();
  }
  pend_load.clear();
  for (uint64_t i = kernel->load_num_; i < kernel->static_ops_.size(); ++i) {
    auto store = kernel->static_ops_[i];
    int joined = kInvalidDomain;
    TravelInput(stack, store, [&](NDObject *in) -> bool {
      auto dom = in->reuse_dep_;
      if (in->IsLoad() && dom != kSplitDomain) {
        pend_load.push_back(in);
      }
      if (dom != kInvalidDomain && dom != kSliceDomain) {
        if (joined == kInvalidDomain) {
          joined = dom;
          return false;
        }
        EXCEPTION_IF(joined != dom, "split multi domain depend");
      }
      return true;
    });
    store->reuse_dep_ = joined;
    if (joined == kSplitDomain) {
      for (auto &s : slice_ios_) {
        s.ios.push_back(store);
      }
    } else if (joined >= 0) {
      slice_ios_[joined].ios.push_back(store);
      for (size_t i = 0; i < slice_ios_.size(); ++i) {
        if (i != static_cast<size_t>(joined)) {
          slice_ios_[i].deads.push_back(store);
        }
      }
    } else {
      DvmException("store is not joined");  // TODO: add seperate path if joined is invalid
    }
    if (!pend_load.empty()) {
      for (auto op : pend_load) {
        op->reuse_dep_ = kSliceDomain;
      }
      if (joined == kSplitDomain) {
        for (auto &s : slice_ios_) {
          s.load_num += AddPengLoad(s.ios, pend_load);
        }
      } else {
        slice_ios_[joined].load_num += AddPengLoad(slice_ios_[joined].ios, pend_load);
      }
      pend_load.clear();
    }
  }
  for (uint64_t i = 0; i < kernel->static_ops_.size(); ++i) {
    kernel->static_ops_[i]->xbuf_ = i;
  }
  for (auto &s : slice_ios_) {
    std::sort(s.ios.begin(), s.ios.end(), [](NDObject *a, NDObject *b) { return a->xbuf_ < b->xbuf_; });
    for (size_t i = 0; i < s.ios.size(); ++i) {
      if (s.ios[i]->reuse_dep_ != kSplitDomain) {
        s.slice_mask |= 1ull << i;
      }
    }
    EXCEPTION_IF(s.slice_mask >> s.load_num == 0, "split has no slice store");
  }
}

int64_t SplitSchGen::CodeGen() {
  SpaceInit();
  SaveSpace();
  int split_dim = split_->split_dim_;
  for (auto &s : slice_ios_) {
    s.bcast_mask = 0;
    for (size_t j = 0; j < s.ios.size(); ++j) {
      if ((s.slice_mask >> j) & 1) continue;
      auto acc = static_cast<NDAccess *>(s.ios[j]);
      if (acc->nd_[split_dim] == 1) {
        s.bcast_mask |= 1ull << j;
      }
      if (acc->stride_ == nullptr) {
        AllocStride(acc);
      }
    }
  }
  int slice_num = static_cast<int>(split_->siblings_.size());
  int64_t split_dim_size = split_->lhs_->nd_[split_dim];
  auto &static_ops = kernel_->static_ops_;
  VectorDupHelper helper(kernel_, slice_num, ReserveReloc(static_ops.size() * slice_num), split_dim_size);
  int64_t split_offset = 0;
  DimArray size = split_->nd_.dims();
  for (int slice_idx = 0; slice_idx < slice_num; ++slice_idx) {
    int64_t split_size =
      slice_idx < slice_num - 1 ? split_->split_size_ : split_dim_size - split_->split_size_ * (slice_num - 1);
    size[split_dim] = split_size;
    ApplySubSpace(size);
    helper.Reset();
    auto &entry = slice_ios_[slice_idx];
    for (size_t j = 0; j < entry.ios.size(); ++j) {
      auto acc = static_cast<NDAccess *>(entry.ios[j]);
      if (!((entry.slice_mask >> j) & 1)) {
        auto bcast = (entry.bcast_mask >> j) & 1;
        acc->ViewUpdate(bcast ? 0 : (*acc->stride_)[split_dim] * static_cast<uint64_t>(split_offset));
      }
    }
    for (auto op : entry.deads) {
      op->flags_ |= OBJ_FLAG_DEAD;
    }
    std::swap(static_ops, entry.ios);
    kernel_->load_num_ = entry.load_num;
    split_offset += split_size;
    helper.DoAppend(split_size);
    std::swap(static_ops, entry.ios);
  }
  helper.Submit();
  ResetStrides();
  return 0;
}

int64_t DupTilingSchGen::DupCodeGen(int split_dim, int64_t truck_size) {
  DimArray dim_space = kernel_->DimSpace();
  int64_t split_size = dim_space[split_dim];
  int64_t body_size = split_size / truck_size * truck_size;
  int64_t tail_size = split_size - body_size;
  if (tail_size == 1) { // avoid x view
    body_size = split_size / 2;
    tail_size = split_size  - body_size;
  }
  ASSERT(body_size > 1 && tail_size > 1);
  auto &static_ops = kernel_->static_ops_;
  uint64_t bcast_mask = 0;
  for (size_t i = 0; i < static_ops.size(); ++i) {
    auto acc = static_cast<NDAccess *>(static_ops[i]);
    if (acc->nd_[split_dim] == 1) {
      bcast_mask |= 1ull << i;
    }
    if (acc->stride_ == nullptr) {
      AllocStride(acc);
    }
  }
  auto reloc_ios = [&static_ops, split_dim, bcast_mask](int64_t offset) {
    for (size_t i = 0; i < static_ops.size(); ++i) {
      auto acc = static_cast<NDAccess *>(static_ops[i]);
      acc->ViewUpdate(bcast_mask >> i & 1ull ? 0 : (*acc->stride_)[split_dim] * offset);
    }
  };
  SpaceInit();
  SaveSpace();
  VectorDupHelper helper(kernel_, 2, ReserveReloc(static_ops.size() * 2), split_size);
  dim_space[split_dim] = tail_size;
  ApplySubSpace(dim_space);
  reloc_ios(body_size);
  helper.Append(tail_size);
  dim_space[split_dim] = body_size;
  ApplySubSpace(dim_space);
  reloc_ios(0);
  helper.Append(body_size);
  helper.Submit();
  ResetStrides();
  return 0;
}
}  // namespace dvm