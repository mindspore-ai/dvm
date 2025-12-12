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

#include <cstdlib>
#include <queue>
#include "xkernel.h"
#include "comm.h"
#include "tuning.h"

namespace dvm {
namespace {
void ReplaceOperands(NDObject *replaced, NDObject *replacing, NDObject *op) {
  auto InplaceOp = [replaced, replacing](NDObject *&op) {
    if (op && op == replaced) {
      op = replacing;
    }
  };
  InplaceOp(op->lhs_);
  InplaceOp(op->rhs_);
  if (op->flags_ & OBJ_FLAG_XHS) {
    InplaceOp(static_cast<FlexOp *>(op)->xhs_);
  }
}
}  // namespace

MixKernel::~MixKernel() {
  if (cube_op_) {
    delete cube_op_->lhs_;
    delete cube_op_->rhs_;
    delete cube_op_->output_;
    delete cube_op_->bias_;
    if (cube_op_->GetObjectType() == kGmmOp) {
      delete static_cast<GmmOp *>(cube_op_)->group_list_;
    }
    delete cube_op_;
  }
  delete post_fusion_;
  delete stage_kernel_;
}

void MixKernel::Append(NDObject *obj) {
  constexpr uint32_t LOAD_VEC_USED = 0;
  constexpr uint32_t LOAD_PENDING = 1;
  constexpr uint32_t LOAD_CUBE_USED = 2;
  if (obj->IsLoad()) {
    obj->xbuf_ = LOAD_PENDING;
  } else if (obj->IsCube()) {
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    cube_op_ = static_cast<CubeOp *>(obj);
    auto reload_check = [this](NDObject *&input) {
      auto load = static_cast<NDAccess *>(input);
      if (load->xbuf_ == LOAD_PENDING) {
        load->xbuf_ = LOAD_CUBE_USED;
      } else {
        auto new_load = new NDLoad(nullptr, load->shape_ref_, load->type_id_);
        reloads_.emplace_back(new_load, load);
        input = new_load;
      }
    };
    reload_check(cube_op_->lhs_);
    reload_check(cube_op_->rhs_);
    if (cube_op_->bias_) {
      reload_check(cube_op_->bias_);
    }
    std::vector<NDObject *> empty_run_ops;
    obj->lhs_->Normalize(empty_run_ops);
    obj->rhs_->Normalize(empty_run_ops);
    cube_op_->NormalizeCube();
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    obj->nd_ = cube_op_->nd_;
    cube_op_->output_ = static_cast<NDAccess *>(obj);
  } else {
    if (post_fusion_ == nullptr) {
      post_fusion_ = new VKernelS();
    }
    auto WorkLoad = [this](NDObject *&op) {
      if (op == cube_op_) {
        if (sload_ == nullptr) {
          sload_ = new NDLoad(nullptr, cube_op_->shape_ref_, cube_op_->type_id_);
          sload_->flags_ |= OBJ_FLAG_LOAD_FROM_CUBE;
          post_fusion_->build_ops_.emplace_back(sload_);
        }
        op = sload_;
      } else if (op->IsLoad() && op->xbuf_ != LOAD_VEC_USED) {
        if (op->xbuf_ == LOAD_PENDING) {
          op->xbuf_ = LOAD_VEC_USED;
        } else { // LOAD_CUBE_USED
          auto new_load = new NDLoad(nullptr, op->shape_ref_, op->type_id_);
          reloads_.emplace_back(new_load, static_cast<NDAccess *>(op));
          op = new_load;
        }
        post_fusion_->build_ops_.emplace_back(op);
      }
    };
    if (obj->lhs_) {
      WorkLoad(obj->lhs_);
      if (obj->rhs_) {
        WorkLoad(obj->rhs_);
        if (obj->flags_ & OBJ_FLAG_XHS) {
          WorkLoad(static_cast<FlexOp *>(obj)->xhs_);
        }
      }
    }
    post_fusion_->build_ops_.emplace_back(obj);
  }
}

void MixKernel::EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node) {
  for (auto &op : post_fusion_->build_ops_) {
    if (op != replaced_node) {
      ReplaceOperands(replaced_node, replacing_node, op);
      stage_kernel_->GetImpl()->Append(op);
    }
  }
}

void MixKernel::Release() {
  if (!stage_kernel_) {
    return;
  }
  auto current_kernel = static_cast<MixKernel *>(static_cast<StagesKernel *>(stage_kernel_->GetImpl())->Current());
  VectorKernel *target_post_fusion = current_kernel->post_fusion_;
  if (post_fusion_) {
    std::unordered_set<NDObject *> post_ops_set(post_fusion_->build_ops_.begin(), post_fusion_->build_ops_.end());
    for (auto &obj : target_post_fusion->build_ops_) {
      if (post_ops_set.find(obj) != post_ops_set.end()) {
        obj = nullptr;
      }
    }
    for (auto &op : post_fusion_->build_ops_) {
      ReplaceOperands(current_kernel->sload_, sload_, op);
    }
  }

  auto cube_op = current_kernel->cube_op_;
  if (cube_op->GetObjectType() == kGmmOp) {
    static_cast<GmmOp *>(cube_op)->group_list_ = nullptr;
  }
}

uint64_t MixKernel::UnAlignCodeGen() {
  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  int64_t pad_size[2] = {cube_op_->tactics_.lhs_pad_size, cube_op_->tactics_.rhs_pad_size};
  NDObject *inputs[2], *pad_inputs[2];
  NDObject *src_inputs[2] = {cube_op_->lhs_, cube_op_->rhs_};
  for (size_t i = 0; i < 2; i++) {
    if (pad_size[i]) {
      stage_kernel_->StageSwitch(dvm::KernelType::kStaticShape);
      pad_inputs[i] = stage_kernel_->Load(nullptr, src_inputs[i]->shape_ref_, src_inputs[i]->type_id_);
      auto load = stage_kernel_->Copy(pad_inputs[i]);
      inputs[i] = stage_kernel_->StagePadStore(load, pad_size[i]);
    }
  }
  stage_kernel_->StageSwitch(KernelType::kStaticMix);
  for (size_t i = 0; i < 2; i++) {
    if (pad_size[i]) {
      inputs[i] = stage_kernel_->StageLoad(inputs[i]);
    } else {
      inputs[i] = stage_kernel_->Load(nullptr, src_inputs[i]->shape_ref_, src_inputs[i]->type_id_);
      pad_inputs[i] = inputs[i];
    }
  }
  auto src_bias = cube_op_->bias_;
  NDObject *bias = src_bias ? stage_kernel_->Load(nullptr, src_bias->shape_ref_, src_bias->type_id_) : nullptr;
  NDObject *matmul_op;
  if (cube_op_->GetObjectType() == kCubeOp) {
    matmul_op = stage_kernel_->MatMul(inputs[0], inputs[1], cube_op_->trans_a_, cube_op_->trans_b_, bias);
  } else {
    matmul_op = stage_kernel_->GroupedMatMul(inputs[0], inputs[1], cube_op_->trans_a_, cube_op_->trans_b_,
                                             bias, static_cast<GmmOp *>(cube_op_)->group_list_,
                                             static_cast<GmmOp *>(cube_op_)->group_type_);
  }
  static_cast<CubeOp *>(matmul_op)->SetRealShape(cube_op_->m_real_, cube_op_->n_real_, cube_op_->k_real_, 0, 0);
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_op);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(stage_kernel_->Store(nullptr, matmul_op));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size;
  code_ = std::move(stage_kernel_->GetImpl()->code_);

  auto src_lhs = static_cast<NDAccess *>(cube_op_->lhs_);
  auto src_rhs = static_cast<NDAccess *>(cube_op_->rhs_);
  src_lhs->addr_.Update(static_cast<NDAccess *>(pad_inputs[0])->addr_);
  src_rhs->addr_.Update(static_cast<NDAccess *>(pad_inputs[1])->addr_);
  if (bias) {
    static_cast<NDAccess *>(cube_op_->bias_)->addr_.Update(static_cast<NDAccess *>(bias)->addr_);
  }
  if (real_out) {
    cube_op_->output_->addr_.Update(real_out->addr_);
  }
  return workspace_size;
}

uint64_t MixKernel::SplitKCodeGen() {
  size_t k_stride = cube_op_->tactics_.k_stride;
  auto split_num = CeilDiv(static_cast<size_t>(cube_op_->k_real_), k_stride);
  size_t k_tail = cube_op_->k_real_ % k_stride ? cube_op_->k_real_ % k_stride : k_stride;

  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  std::vector<NDAccess *> split_lhs;
  std::vector<NDAccess *> split_rhs;
  std::vector<NDAccess *> split_out;
  NDObject *bias_op = nullptr;
  for (size_t i = 0; i < split_num; i++) {
    size_t offset_a = i * k_stride;
    size_t offset_b = i * k_stride;
    if (cube_op_->trans_a_) offset_a *= cube_op_->m_align_;
    if (!cube_op_->trans_b_) offset_b *= cube_op_->n_align_;
    stage_kernel_->StageSwitch(KernelType::kStaticMix);
    auto x = stage_kernel_->Load(nullptr, cube_op_->lhs_->shape_ref_, cube_op_->lhs_->type_id_);
    auto y = stage_kernel_->Load(nullptr, cube_op_->rhs_->shape_ref_, cube_op_->rhs_->type_id_);
    (void)split_lhs.emplace_back(static_cast<NDAccess *>(x));
    (void)split_rhs.emplace_back(static_cast<NDAccess *>(y));
    NDObject *matmul_op;
    if (i + 1 == split_num && cube_op_->bias_) {
      bias_op = stage_kernel_->Load(nullptr, cube_op_->bias_->shape_ref_, cube_op_->bias_->type_id_);
    }
    if (cube_op_->GetObjectType() == kCubeOp) {
      matmul_op = stage_kernel_->MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_op);
    } else {
      matmul_op = stage_kernel_->GroupedMatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_op,
                                               static_cast<GmmOp *>(cube_op_)->group_list_,
                                               static_cast<GmmOp *>(cube_op_)->group_type_);
    }
    static_cast<CubeOp *>(matmul_op)->SetRealShape(cube_op_->m_real_, cube_op_->n_real_,
                                                   i + 1 == split_num ? k_tail : k_stride, offset_a, offset_b);
    static_cast<CubeOp *>(matmul_op)->SetOutFp32(i != 0);
    (void)split_out.emplace_back(static_cast<NDAccess *>(stage_kernel_->Store(nullptr, matmul_op)));
  }
  auto matmul_fp32 = split_out.back()->lhs_;
  auto matmul_fp16 = stage_kernel_->Cast(matmul_fp32, cube_op_->lhs_->type_id_);
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_fp16);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(stage_kernel_->Store(nullptr, matmul_fp16));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size + split_out.back()->Size();
  code_ = std::move(stage_kernel_->GetImpl()->code_);

  auto src_lhs = static_cast<NDAccess *>(cube_op_->lhs_);
  auto src_rhs = static_cast<NDAccess *>(cube_op_->rhs_);
  src_lhs->addr_.Update(split_lhs[0]->addr_);
  src_rhs->addr_.Update(split_rhs[0]->addr_);
  code_.BindWorkspace(split_out[0]->addr_, stage_workspace_size);
  for (size_t i = 1; i < split_num; i++) {
    code_.BindOpFast(split_lhs[i]->addr_, src_lhs->addr_);
    code_.BindOpFast(split_rhs[i]->addr_, src_rhs->addr_);
    code_.BindWorkspace(split_out[i]->addr_, stage_workspace_size);
  }
  if (bias_op) {
    static_cast<NDAccess *>(cube_op_->bias_)->addr_.Update(static_cast<NDAccess *>(bias_op)->addr_);
  }
  if (real_out) {
    cube_op_->output_->addr_.Update(real_out->addr_);
  }
  return workspace_size;
}

uint64_t MixKernel::BiasBF16CodeGen() {
  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  stage_kernel_->StageSwitch(dvm::KernelType::kStaticShape);
  auto bias_bf16 = stage_kernel_->Load(nullptr, cube_op_->bias_->shape_ref_, cube_op_->bias_->type_id_);
  auto bias_fp32 = stage_kernel_->StageStore(stage_kernel_->Cast(bias_bf16, kFloat32));

  stage_kernel_->StageSwitch(KernelType::kStaticMix);
  auto x = stage_kernel_->Load(nullptr, cube_op_->lhs_->shape_ref_, cube_op_->lhs_->type_id_);
  auto y = stage_kernel_->Load(nullptr, cube_op_->rhs_->shape_ref_, cube_op_->rhs_->type_id_);
  NDObject *matmul_op;
  if (cube_op_->GetObjectType() == kCubeOp) {
    matmul_op =
      stage_kernel_->MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, stage_kernel_->StageLoad(bias_fp32));
  } else {
    matmul_op = stage_kernel_->GroupedMatMul(
      x, y, cube_op_->trans_a_, cube_op_->trans_b_, stage_kernel_->StageLoad(bias_fp32),
      static_cast<GmmOp *>(cube_op_)->group_list_, static_cast<GmmOp *>(cube_op_)->group_type_);
  }
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_op);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(stage_kernel_->Store(nullptr, matmul_op));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size;
  code_ = std::move(stage_kernel_->GetImpl()->code_);

  auto src_lhs = static_cast<NDAccess *>(cube_op_->lhs_);
  auto src_rhs = static_cast<NDAccess *>(cube_op_->rhs_);
  src_lhs->addr_.Update(static_cast<NDAccess *>(x)->addr_);
  src_rhs->addr_.Update(static_cast<NDAccess *>(y)->addr_);
  static_cast<NDAccess *>(cube_op_->bias_)->addr_.Update(static_cast<NDAccess *>(bias_bf16)->addr_);
  if (real_out) {
    cube_op_->output_->addr_.Update(real_out->addr_);
  }
  return workspace_size;
}

uint64_t MixKernel::AlignCodeGen() {
  size_t head_reserve = code_.HeadSize() + sizeof(vCubeOp);
  size_t post_reserve = 0;
  if (post_fusion_) {
    post_fusion_->Normalize();
    post_reserve = post_fusion_->ReserveCodeSize();
    if (cube_op_->batch_fold_) {
      auto fold_check = [this]() -> bool {
        NDSpaceData &cube_nd = cube_op_->ndd_;
        for (auto op : post_fusion_->objects_) {
          if (auto ndd = op->Ndd(); ndd != nullptr) {
            if (ndd->dims.size() != cube_nd.size()) return false;
            for (size_t i = 1; i < cube_nd.size(); ++i) {
              if (ndd->dims[i] != cube_nd[i]) return false;
            }
          }
        }
        return true;
      };
      if (!fold_check()) {
        cube_op_->batch_fold_ = false;
      }
    }
  }
  code_.Alloc(head_reserve + post_reserve);
  vCubeOp *cube_code = reinterpret_cast<vCubeOp *>(code_.data_ + code_.HeadSize());
  cube_op_->CodeGen(cube_code, tuner_);
  static_cast<NDAccess *>(cube_op_->lhs_)->addr_.Update(&cube_code->gm_a);
  static_cast<NDAccess *>(cube_op_->rhs_)->addr_.Update(&cube_code->gm_b);
  if (cube_code->flags & V_CUBE_FLAG_WITH_BIAS) {
    static_cast<NDAccess *>(cube_op_->bias_)->addr_.Update(&cube_code->gm_bias);
  }
  if (cube_code->flags & V_CUBE_FLAG_GROUPED_LIST) {
    static_cast<NDAccess *>(static_cast<GmmOp *>(cube_op_)->group_list_)->addr_.Update(&cube_code->gm_group_list);
  }
  cube_op_->output_->addr_.Update(&cube_code->gm_c);
  code_.block_dim_ = cube_op_->block_dim_;
  if (!post_fusion_) {
    code_.data_size_ = head_reserve;
    code_.UpdateC();
    return 0;
  }
  if (cube_op_->batch_fold_) {
    for (auto op : post_fusion_->objects_) {
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        ASSERT(ndd->dims.prod() == sload_->nd_.dims().prod());
        ndd->dims[1] = cube_op_->m_real_;
        ndd->dims.resize(2);
      }
      if (op->obj_id_ == kStore) {
        static_cast<NDStore *>(op)->UpdateDimMask();
      }
    }
  }
  if (auto comm = post_fusion_->comm_op_) {
    comm->mix_ = true;
    if (comm->lhs_ == sload_) {
      comm->SetCubeOp(cube_op_);
    }
  }
  post_fusion_->Optimize();
  uint64_t ws_size = sizeof(vMixGroupMsg) * code_.block_dim_;
  gm_pos_.reloc_ = &cube_code->gm_pos;
  code_.BindWorkspace(gm_pos_, 0);
  if (auto comm = post_fusion_->comm_op_; comm != nullptr && comm->lhs_ == sload_) {
    cube_op_->pingpong_store_ = true;
    static_cast<NDLoad *>(sload_)->flags_ |= OBJ_FLAG_LOAD_PINGPONG;
    cube_code->rank_size = comm->comm_->GetRankSize();
    cube_code->flags |= V_CUBE_FLAG_PEER_STORE;
    cube_code->flags |= V_CUBE_FLAG_PINGPONG_STORE;
    comm->id_wrap_.ids_.push_back(&cube_code->unique_id);  // record vCubeOp's unique_id address
    cube_code->gm_c = reinterpret_cast<uint64_t>(comm->comm_->GetPeerMemPtr(comm->comm_->GetRankId()));
    code_.BindOpFast(sload_->addr_, cube_op_->output_->addr_);
  } else if (cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    if (auto inplace_store = post_fusion_->FindInplaceStore(sload_, nullptr)) {
      code_.BindOpFast(cube_op_->output_->addr_, inplace_store->addr_);
      code_.BindOpFast(sload_->addr_, inplace_store->addr_);
    } else {
      cube_op_->pingpong_store_ = true;
      static_cast<NDLoad *>(sload_)->flags_ |= OBJ_FLAG_LOAD_PINGPONG;
      cube_code->flags |= V_CUBE_FLAG_PINGPONG_STORE;
      code_.BindWorkspace(cube_op_->output_->addr_, ws_size);
      code_.BindWorkspace(sload_->addr_, ws_size);
      ws_size += cube_op_->PostFusionWorkSpace();
    }
  } else {
    code_.BindOpFast(sload_->addr_, cube_op_->output_->addr_);
  }
  ShardParam shard;
  shard.base = 0;
  shard.dom = &cube_op_->ndd_.dims;
  shard.tile[0] = cube_op_->n0_;
  shard.tail[0] = cube_op_->n_real_ % cube_op_->n0_;
  shard.tile[1] = cube_op_->m0_;
  shard.tail[1] = cube_op_->m_real_ % cube_op_->m0_;
  shard.stride[0] = cube_op_->n_real_;
  shard.stride[1] = cube_op_->n_real_ * cube_op_->m_real_;
  post_fusion_->root_dom_.Shard(shard);
  post_fusion_->root_dom_.PrepareTiling(post_fusion_);
  MixVisitCoder visit;
  post_fusion_->AddVisitor(&visit);
  auto code_end = post_fusion_->DoCodeGen(2, code_.data_ + head_reserve, post_reserve);
  uint64_t subtile_0 = (post_fusion_->tile_num_ + 1) / 2;
  uint64_t subtile_1 = post_fusion_->tile_num_ - subtile_0;
  cube_code->flags |= V_CUBE_FLAG_GROUP_SET;
  code_.data_size_ = code_end - code_.data_;
  code_.UpdateMix(&visit, shard.tile, shard.tail, shard.stride, subtile_0, subtile_1);
  code_.Combine(post_fusion_->code_, 0);
  return ws_size;
}

uint64_t MixKernel::CodeGen() {
  if (cube_op_->output_ == nullptr) {
    auto output = new NDStore(nullptr, cube_op_);
    output->SetFlag(OBJ_FLAG_STAGE_IO);
    cube_op_->output_ = output;
  }
  cube_op_->InferCubeConfig();
  uint64_t workspace_size = 0;
  if (cube_op_->tactics_.enable_bias_cast) {
    workspace_size = BiasBF16CodeGen();
  } else if (cube_op_->tactics_.enable_pad) {
    workspace_size = UnAlignCodeGen();
  } else if (cube_op_->tactics_.enable_splitk) {
    workspace_size = SplitKCodeGen();
  } else {
    workspace_size = AlignCodeGen();
  }
  if (!reloads_.empty()) {
    for (auto &r : reloads_) {
      code_.BindOpFast(r.first->addr_, r.second->addr_);
    }
  }
  Release();
  return workspace_size;
}

void MixKernel::Dump(std::ostringstream &oss, const std::string &indent) {
  oss << indent << "vgraph.mix(tile_num=" << cube_op_->core_loop_ << ") {\n";
  std::string body_indent = indent + "  ";
  oss << body_indent << "// cube" << std::endl;
  oss << body_indent << "%" << cube_op_->index_ << cube_op_->nd_;
  oss << " = MatMul(%" << cube_op_->lhs_->index_ << cube_op_->lhs_->nd_;
  oss << ", %" << cube_op_->rhs_->index_ << cube_op_->rhs_->nd_;
  if (cube_op_->bias_) {
    oss << ", %" << cube_op_->bias_->index_ << cube_op_->bias_->nd_;
  }
  oss << ")\n";
  if (post_fusion_) {
    oss << body_indent << "// post_fusion" << std::endl;
    post_fusion_->Dump(oss, body_indent);
    oss << std::endl;
  }
  oss << indent << "}";
}

DynMixKernel::DynMixKernel() : MixKernel() {
  ktype_ = kDynMix;
  tuner_ = nullptr;
}

uint64_t DynMixKernel::CodeGen() {
  delete stage_kernel_;
  stage_kernel_ = nullptr;
  code_.~Code();
  code_.data_ = nullptr;
  code_ = std::move(Code());
  if (cube_op_->output_ == nullptr) {
    auto output = new NDStore(nullptr, cube_op_);
    output->SetFlag(OBJ_FLAG_STAGE_IO);
    cube_op_->output_ = output;
  }
  std::vector<NDObject *> empty_run_ops;
  cube_op_->lhs_->Normalize(empty_run_ops);
  cube_op_->rhs_->Normalize(empty_run_ops);
  cube_op_->NormalizeCube();
  cube_op_->output_->Normalize(empty_run_ops);
  cube_op_->InferCubeConfig();
  uint64_t workspace_size = 0;
  if (cube_op_->tactics_.enable_bias_cast) {
    workspace_size = BiasBF16CodeGen();
  } else if (cube_op_->tactics_.enable_pad) {
    workspace_size = UnAlignCodeGen();
  } else if (cube_op_->tactics_.enable_splitk) {
    workspace_size = SplitKCodeGen();
  } else {
    workspace_size = AlignCodeGen();
  }
  if (!reloads_.empty()) {
    for (auto &r : reloads_) {
      code_.BindOpFast(r.first->addr_, r.second->addr_);
    }
  }
  Release();
  return workspace_size;
}

void DynMixKernel::Append(NDObject *obj) {
  constexpr uint32_t LOAD_VEC_USED = 0;
  constexpr uint32_t LOAD_PENDING = 1;
  constexpr uint32_t LOAD_CUBE_USED = 2;
  if (obj->IsLoad()) {
    obj->xbuf_ = LOAD_PENDING;
  } else if (obj->IsCube()) {
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    cube_op_ = static_cast<CubeOp *>(obj);
    auto reload_check = [this](NDObject *&input) {
      auto load = static_cast<NDAccess *>(input);
      if (load->xbuf_ == LOAD_PENDING) {
        load->xbuf_ = LOAD_CUBE_USED;
      } else {
        auto new_load = new NDLoad(nullptr, load->shape_ref_, load->type_id_);
        reloads_.emplace_back(new_load, load);
        input = new_load;
      }
    };
    reload_check(cube_op_->lhs_);
    reload_check(cube_op_->rhs_);
    if (cube_op_->bias_) {
      reload_check(cube_op_->bias_);
    }
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    cube_op_->output_ = static_cast<NDAccess *>(obj);
  } else {
    if (post_fusion_ == nullptr) {
      post_fusion_ = new VKernelD();
    }
    auto WorkLoad = [this](NDObject *&op) {
      if (op == cube_op_) {
        if (sload_ == nullptr) {
          sload_ = new NDLoad(nullptr, cube_op_->shape_ref_, cube_op_->type_id_);
          sload_->flags_ |= OBJ_FLAG_LOAD_FROM_CUBE;
          post_fusion_->Append(sload_);
        }
        op = sload_;
      } else if (op->IsLoad() && op->xbuf_ != LOAD_VEC_USED) {
        if (op->xbuf_ == LOAD_PENDING) {
          op->xbuf_ = LOAD_VEC_USED;
        } else { // LOAD_CUBE_USED
          auto new_load = new NDLoad(nullptr, op->shape_ref_, op->type_id_);
          reloads_.emplace_back(new_load, static_cast<NDAccess *>(op));
          op = new_load;
        }
        post_fusion_->Append(op);
      }
    };
    if (obj->lhs_) {
      WorkLoad(obj->lhs_);
      if (obj->rhs_) {
        WorkLoad(obj->rhs_);
        if (obj->flags_ & OBJ_FLAG_XHS) {
          WorkLoad(static_cast<FlexOp *>(obj)->xhs_);
        }
      }
    }
    post_fusion_->Append(obj);
  }
}

int StageCodeWrap::LaunchWrap(void *workspace, void *stream) {
  for (auto s : kernel_->stages_) {
    s->kernel->code_.Launch(workspace, stream);
  }
  return 0;
}

bool StageCodeWrap::DasWrap(std::ostringstream &oss) {
  for (auto s : kernel_->stages_) {
    s->kernel->code_.DisAssemble(oss);
    oss << std::endl;
  }
  return false;
}

StagesKernel::~StagesKernel() {
  for (auto s : stages_) {
    delete s->kernel;
    delete s;
  }
}

void StagesKernel::Append(NDObject *obj) {
  stages_.back()->kernel->Append(obj);
  if (!obj->IsSimd()) {
    stages_.back()->ios.push_back(static_cast<NDAccess *>(obj));
  }
}

#define STAGE_FLAG_WORKSPACE 1
#define STAGE_FLAG_REUSE 2

uint64_t StagesKernel::CodeGen() {
  for (auto s : stages_) {
    auto k = s->kernel;
    s->ws_size = k->code_.ReserveWorkspace(k->CodeGen());
  }
  uint64_t ws_size = AllocWorkspace();
  for (auto s : stages_) {
    auto &code = s->kernel->code_;
    code_.CombineBind(code, s->ws_offset);
    for (auto op : s->ios) {
      if (!op->CheckFlag(OBJ_FLAG_STAGE_IO)) continue;
      if (op->IsStore()) {
        if (op->xbuf_ == STAGE_FLAG_REUSE) {
          code_.BindOp(op->addr_, GetOutputReuse(op)->addr_);
        } else {
          code_.BindWorkspace(op->addr_, GetWorkspace(op));
        }
      } else {
        code_.BindOp(op->addr_, GetStageStore(op)->addr_);
      }
    }
  }
  code_.InsertWrap(&code_wrap_);
  return ws_size;
}

uint64_t StagesKernel::AllocWorkspace() {
  struct Group {
    Group(uint64_t s, bool l) : size(s), live(l) {}
    std::vector<NDAccess *> ops;
    std::vector<Stage *> wss;
    uint64_t size;
    bool live;
  };
  std::vector<Group> groups;
  groups.reserve(stages_.size() * 8);
  std::unordered_map<NDAccess *, int> lives;  // >= 0: group_idx. -1: reused
  auto select_group = [&groups](uint64_t size) -> int {
    int up = -1, down = -1;
    for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
      auto &g = groups[i];
      if (g.live) continue;
      if (g.size >= size) {
        if (up == -1 || g.size < groups[up].size) {
          up = i;
        }
      } else if (up == -1 && (down == -1 || g.size > groups[down].size)) {
        down = i;
      }
    }
    if (up >= 0) {
      return up;
    } else if (down >= 0) {
      groups[down].size = size;
      return down;
    }
    return -1;
  };
  for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) {
    auto stage = *it;
    // stage buffer gen
    for (auto io : stage->ios) {
      if (io->IsLoad() && io->CheckFlag(OBJ_FLAG_STAGE_IO)) {
        auto store = GetStageStore(io);
        if (!store->CheckFlag(OBJ_FLAG_STAGE_IO) || lives.find(store) != lives.end()) continue;
        if (stage->kernel->KType() == kStaticShape) {  // TODO: parallel fusion
          NDAccess *inplace_stage = nullptr;
          auto inplace_out = static_cast<VectorKernel *>(stage->kernel)
                               ->FindInplaceStore(io, [&lives, &inplace_stage](NDAccess *op) -> bool {
                                 if (!op->CheckFlag(OBJ_FLAG_STAGE_IO) || op->xbuf_ == STAGE_FLAG_REUSE) {
                                   return true;
                                 }
                                 if (inplace_stage == nullptr && lives[op] >= 0) {
                                   inplace_stage = op;
                                 }
                                 return false;
                               });
          if (inplace_out) {
            store->xbuf_ = STAGE_FLAG_REUSE;
            SetOutputReuse(store,
                           inplace_out->CheckFlag(OBJ_FLAG_STAGE_IO) ? GetOutputReuse(inplace_out) : inplace_out);
            lives[store] = -1;
            continue;
          }
          if (inplace_stage) {
            auto inplace_it = lives.find(inplace_stage);
            groups[inplace_it->second].ops.push_back(store);
            lives[store] = inplace_it->second;
            inplace_it->second = -1;
            continue;
          }
        }
        auto size = store->Size();
        int index = select_group(size);
        if (index == -1) {
          index = groups.size();
          groups.emplace_back(size, true);
        }
        groups[index].ops.emplace_back(store);
        lives[store] = index;
      }
    }
    if (stage->ws_size > 0) {
      auto index = select_group(stage->ws_size);
      if (index == -1) {
        index = groups.size();
        groups.emplace_back(stage->ws_size, false);
      }
      groups[index].wss.push_back(stage);
    }
    // stage buffer kill
    for (auto io : stage->ios) {
      if (io->IsStore() && io->CheckFlag(OBJ_FLAG_STAGE_IO)) {
        auto live_it = lives.find(io);
        if (live_it->second >= 0) {
          groups[live_it->second].live = false;
        }
        lives.erase(live_it);
      }
    }
  }
  uint64_t workspace_size = 0;
  for (auto &g : groups) {
    for (auto op : g.ops) {
      op->xbuf_ = STAGE_FLAG_WORKSPACE;
      SetWorkspace(op, workspace_size);
    }
    for (auto stage : g.wss) {
      stage->ws_offset = workspace_size;
    }
    workspace_size += g.size;
  }
  return workspace_size;
}

void StagesKernel::Dump(std::ostringstream &oss, const std::string &indent) {
  oss << indent << "vgraph.stages() {\n";
  int stage_idx = 0;
  std::string body_indent = indent + "  ";
  for (auto &s : stages_) {
    oss << body_indent << "// stage " << stage_idx << std::endl;
    stage_idx++;
    s->kernel->Dump(oss, body_indent);
    oss << std::endl;
  }
  oss << indent << "}";
}

class CubeOptimizer {
 public:
  explicit CubeOptimizer(CubeOp *dom) : dom_(dom) { dom_->InferCubeConfig(); }

  bool AlignA(std::vector<NDObject *> &ops) {
    if (dom_->tactics_.lhs_pad_size == 0) {
      return false;
    }
    AlignInput(dom_->lhs_, dom_->tactics_.lhs_pad_size, ops);
    return true;
  }

  bool AlignB(std::vector<NDObject *> &ops) {
    if (dom_->tactics_.rhs_pad_size == 0) {
      return false;
    }
    AlignInput(dom_->rhs_, dom_->tactics_.rhs_pad_size, ops);
    return true;
  }

  bool CastBias(std::vector<NDObject *> &ops) {
    if (dom_->bias_ == nullptr || dom_->bias_->type_id_ != kBFloat16) {
      return false;
    }
    auto bias = dom_->bias_;
    auto cast = new CastOp(bias, kFloat32);
    auto store = new NDStore(cast);
    auto load = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
    cast->Normalize(ops);
    store->Normalize(ops);
    load->Normalize(ops);
    ops.push_back(cast);
    ops.push_back(store);
    ops.push_back(load);
    dom_->bias_ = load;
    return true;
  }

  bool SplitK(std::vector<NDObject *> &ops) {
    if (!dom_->tactics_.enable_splitk) {
      return false;
    }
    auto load = new NDLoad(nullptr, dom_->shape_ref_, kFloat32);
    auto cast = new CastOp(load, dom_->type_id_);
    size_t k_stride = dom_->tactics_.k_stride;
    auto split_num = CeilDiv(static_cast<size_t>(dom_->k_real_), k_stride);
    size_t k_tail = dom_->k_real_ % k_stride ? dom_->k_real_ % k_stride : k_stride;
    size_t offset_a = 0;
    size_t offset_b = 0;
    dom_->output_->type_id_ = DType::kFloat32;
    for (size_t i = 0; i < split_num - 1; ++i) {
      auto op = new CubeOp(dom_->lhs_, dom_->rhs_, dom_->trans_a_, dom_->trans_b_, i == 0 ? dom_->bias_ : nullptr);
      op->NormalizeCube();
      op->SetRealShape(dom_->m_real_, dom_->n_real_, k_stride, offset_a, offset_b);
      op->SetOutFp32(i > 0);
      ops.push_back(op);
      offset_a += dom_->trans_a_ ? dom_->m_align_ * k_stride : k_stride;
      offset_b += dom_->trans_b_ ? k_stride : dom_->n_align_ * k_stride;
      op->output_ = dom_->output_;
      op->batch_fold_ = dom_->batch_fold_;
    }
    dom_->SetRealShape(dom_->m_real_, dom_->n_real_, k_tail, offset_a, offset_b);
    dom_->SetOutFp32(true);
    load->Normalize(ops);
    cast->Normalize(ops);
    ops.push_back(load);
    ops.push_back(cast);
    return true;
  }

 protected:
  void AlignInput(NDObject *&input, int64_t align_shape, std::vector<NDObject *> &ops) {
    NDObject *op = input;
    if (op->IsLoad()) {
      op = new CopyOp(op);
      op->Normalize(ops);
      ops.push_back(op);
    }
    auto pad = new NDPadStore(nullptr, op, align_shape);
    pad->Normalize(ops);
    ops.push_back(pad);
    auto load = new NDLoad(nullptr, pad->shape_ref_, pad->type_id_);
    load->Normalize(ops);
    ops.push_back(load);
    input = load;
    auto m = dom_->m_real_;
    auto n = dom_->n_real_;
    auto k = dom_->k_real_;
    dom_->NormalizeCube();
    dom_->SetRealShape(m, n, k, 0, 0);
  }

  CubeOp *dom_;
};

class EagerVector : public VectorKernel {
 public:
  EagerVector() : VectorKernel(KernelType::kEager) {
    objects_.reserve(64);
    static_ops_.reserve(16);
  }

  void Append(NDObject *obj) override {
    obj->index_ = objects_.size();
    objects_.push_back(obj);
    obj->pd_next_ = next_;
    next_ = obj;
    int type = obj->type_id_;
    if (type > max_type_) {
      max_type_ = type;
    } else if (type < min_type_) {
      min_type_ = type;
    }
    if (obj->IsLoad()) {
      static_ops_.push_back(obj);
    } else if (obj->IsStore()) {
      static_ops_.push_back(obj->lhs_);
    }
  }

  void Optimize() override {
    ASSERT(0);
    return;
  }

  uint64_t CodeGen() override {
    ASSERT(0);
    return 0;
  }

  void Reset(NDObject *dom) {
    max_type_ = dom->type_id_;
    min_type_ = dom->type_id_;
    next_ = nullptr;
    objects_.clear();
    code_.Clear();
    static_ops_.clear();
  }

  void CodeGenCube(CubeOp *mm) {
    size_t size = code_.HeadSize() + sizeof(vCubeOp);
    code_.Alloc(size);
    vCubeOp *body = reinterpret_cast<vCubeOp *>(code_.data_ + code_.HeadSize());
    if (unlikely(!mm->m_align_ || !mm->n_align_ || !mm->k_align_)) {
      static_ops_.clear();
      if (mm->m_align_ && mm->n_align_) {
        static_ops_.push_back(mm->output_);
      }
      UpdateIdle(static_ops_);
      return;
    }
    mm->CodeGen(body, System::Instance().lazy_tuner_);
    code_.block_dim_ = mm->block_dim_;
    code_.data_size_ = size;
    code_.UpdateC();
    next_ = mm;
  }

  vCubeOp *CodeGenMix(CubeOp *mm) {
    size_t head_reserve = code_.HeadSize() + sizeof(vCubeOp);
    size_t post_reserve = ReserveCodeSize();  // TODO: visit size
    code_.Alloc(head_reserve + post_reserve);
    vCubeOp *cube_code = reinterpret_cast<vCubeOp *>(code_.data_ + code_.HeadSize());
    if (unlikely(!mm->m_align_ || !mm->n_align_ || !mm->k_align_)) {
      static_ops_.clear();
      if (mm->m_align_ && mm->n_align_) {
        static_ops_.push_back(mm->output_);
      }
      CollectIdle(static_ops_);
      UpdateIdle(static_ops_);
      return cube_code;
    }
    mm->CodeGen(cube_code, System::Instance().lazy_tuner_);
    ASSERT(!mm->batch_fold_);
    root_dom_.SetHead(next_);
    NormalizeDomain();
    ShardParam shard;
    shard.base = 0;
    shard.dom = &mm->ndd_.dims;
    shard.tile[0] = mm->n0_;
    shard.tail[0] = mm->n_real_ % mm->n0_;
    shard.tile[1] = mm->m0_;
    shard.tail[1] = mm->m_real_ % mm->m0_;
    shard.stride[0] = mm->n_real_;
    shard.stride[1] = mm->n_real_ * mm->m_real_;
    root_dom_.Shard(shard);
    root_dom_.PrepareTiling(this);
    MixVisitCoder visit;
    AddVisitor(&visit);
    auto code_end = DoCodeGen(2, code_.data_ + head_reserve, post_reserve);
    root_dom_.ClearShard();
    uint64_t subtile_0 = (tile_num_ + 1) / 2;
    uint64_t subtile_1 = tile_num_ - subtile_0;
    cube_code->flags |= V_CUBE_FLAG_GROUP_SET;
    code_.data_size_ = code_end - code_.data_;
    code_.block_dim_ = mm->block_dim_;
    code_.UpdateMix(&visit, shard.tile, shard.tail, shard.stride, subtile_0, subtile_1);
    next_ = mm;
    return cube_code;
  }

  enum { kMaxPvNum = 8 };
  uint64_t CodeGenV(EagerVector **others, int other_num) {
    uint64_t core_total = System::Instance().CoreNum();
    root_dom_.SetHead(next_);
    NormalizeDomain();
    if (other_num == 0) {
      return DoCodeGen(core_total);
    }
    root_dom_.PrepareTiling(this);
    auto GetWorkLoad = [](EagerVector *k) -> uint64_t { return k->root_dom_.TileSize() * k->objects_.size(); };
    uint64_t child_offset = Code::HeadSize() + RoundUp(core_total * sizeof(uint64_t), 32ul);
    uint64_t code_reserve = child_offset + RoundUp(ReserveCodeSize(), 32ul);
    uint64_t prim_load = GetWorkLoad(this);
    uint64_t total_load = prim_load;
    uint64_t other_loads[EagerVector::kMaxPvNum];
    uint64_t child_num = prim_load ? 1 : 0;
    for (int i = 0; i < other_num; ++i) {
      auto k = others[i];
      k->root_dom_.SetHead(k->next_);
      k->NormalizeDomain();
      k->root_dom_.PrepareTiling(k);
      other_loads[i] = GetWorkLoad(k);
      if (other_loads[i]) {
        total_load += other_loads[i];
        code_reserve += RoundUp(k->ReserveCodeSize(), 32ul);
        child_num++;
      }
    }
    if (unlikely(child_num == 0)) {
      static_ops_.clear();
      CollectIdle(static_ops_);
      for (int i = 0; i < other_num; ++i) {
        others[i]->CollectIdle(static_ops_);
      }
      UpdateIdle(static_ops_);
      return 0;
    }
    code_.Alloc(code_reserve);
    uint64_t *summaries = reinterpret_cast<uint64_t *>(code_.data_ + Code::HeadSize());
    uint64_t core_free = core_total;
    auto GenChild = [this, &core_free, &child_offset, &total_load, &summaries](uint64_t workload, EagerVector *k) {
      uint64_t core_limit = std::max(core_free * workload / total_load, 1ul);
      auto code_begin = code_.data_ + child_offset;
      uint64_t code_size = k->DoCodeGen(core_limit, code_begin, k->ReserveCodeSize()) - code_begin;
      ASSERT(k->visit_ == nullptr);
      uint64_t block_dim = k->CompactBlockDim(core_limit);
      total_load -= workload;
      core_free -= block_dim;
      k->code_.block_dim_ = block_dim;
      child_offset = VKernelP::UpdateSummary(k, child_offset, code_size, summaries);
    };
    if (prim_load) {
      GenChild(prim_load, this);
    }
    for (int i = 0; i < other_num; ++i) {
      if (other_loads[i]) {
        auto k = others[i];
        GenChild(other_loads[i], k);
        code_.Combine(k->code_, 0);
      }
    }
    code_.data_size_ = child_offset;
    code_.block_dim_ = core_total - core_free;
    code_.UpdateVP();
    return 0;
  }

  void Dump(std::ostringstream &oss, const std::string &indent) {
    if (next_ && next_->obj_id_ == ObjectType::kCubeOp) {
      auto mm = static_cast<CubeOp *>(next_);
      oss << indent << "vgraph_cube() {" << std::endl;
      auto body_indent = indent + "  ";
      oss << body_indent << "%0" << mm->nd_ << " = ";
      mm->Dump(true, oss);
      oss << "(%1" << mm->lhs_->nd_;
      oss << ", %2" << mm->rhs_->nd_;
      if (mm->bias_) {
        oss << ", %3" << mm->bias_->nd_;
      }
      oss << ")" << std::endl << indent << "}";
      if (objects_.empty()) return;
      oss << std::endl;
    }
    VectorKernel::Dump(oss, indent);
  }

  NDObject *next_;
};

class EagerArea {
 public:
  enum { kFree = 0, kPending, kSubmitted };
  EagerArea() {
    objects_.reserve(64);
    fused_.reserve(8);
  }
  ~EagerArea() = default;

  void Reset(NDObject *dom) {
    if (dom->obj_id_ != kReduce) {
      dom_ = dom;
      state_ = kPending;
    } else {
      dom_ = dom->lhs_;
      state_ = kSubmitted;
    }
    objects_.clear();
    fused_.clear();
  }

  void ResetMix(NDObject *dom, int state) {
    dom_ = dom;
    state_ = state;
    objects_.clear();
    fused_.clear();
  }

  bool FuseCheck(EagerArea *a) {
    ASSERT(dom_ != nullptr);
    const auto &dom_nd = dom_->nd_;
    const auto &op_nd = a->dom_->nd_;
    if (dom_nd.size() != op_nd.size()) {
      return false;
    }
    for (size_t i = 0; i < op_nd.size(); ++i) {
      if (dom_nd[i] != op_nd[i]) return false;
    }
    return true;
  }

  static EagerArea *Assign(VKernelE *k, size_t aid) {
    EagerArea *area;
    auto &pool = k->areas_;
    if (aid == pool.size()) {
      area = new EagerArea();
      pool.emplace_back(area, area);
    } else {
      area = pool[aid].first;
      pool[aid].second = area;
    }
    k->kernel_used_++;
    area->area_id_ = aid;
    area->depend_mask_ = 1ul << aid;
    return area;
  }

  int state_;
  int area_id_;
  NDObject *dom_;
  std::vector<NDObject *> objects_;
  std::vector<EagerArea *> fused_;
  uint64_t depend_mask_;
};

static int g_eager_pv_width = -1;
static bool g_eager_cv_enable = false;
VKernelE::VKernelE(WsAllocFunc func, void *user_data)
    : VKernel(KernelType::kEager), ws_alloc_(func), user_data_(user_data) {
  objects_.reserve(128);
  temp_ops_.reserve(64);
  areas_.reserve(16);
  kernels_.reserve(8);
  if (g_eager_pv_width == -1) {
    const char *width = getenv("DVM_EAGER_PV_WIDTH");
    g_eager_pv_width = width != nullptr ? std::stoi(width) : 0;
    ASSERT(g_eager_pv_width <= EagerVector::kMaxPvNum);
    if (auto en = getenv("DVM_EAGER_CV_FUSION"); en != nullptr && std::stoi(en) != 0) {
      g_eager_cv_enable = true;
    }
  }
}

VKernelE::~VKernelE() {
  Clear();
  for (auto k : kernels_) {
    delete k;
  }
  for (auto &a : areas_) {
    delete a.first;
  }
}

NDObject *VKernelE::Exchange(EagerArea *area, NDObject *input) {
  ASSERT(!input->IsLoad());
  NDAccess *store = GetStore(input);
  if (store == nullptr) {
    store = new NDStore(nullptr, input);
    store->Normalize(temp_ops_);
    InitStoreInfo(store, GetArea(input));
    SetStore(input, store);
    objects_.push_back(store);
  }
  auto load = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
  SetStore(load, store);
  load->Normalize(area->objects_);
  objects_.push_back(load);
  return load;
}

NDObject *VKernelE::SplitPush(EagerArea *area, NDObject *input) {
  if (auto idx = GetArea(input); idx >= 0) {
    auto a = areas_[idx].second;
    if (a == area) return input;
    if (a->state_ == EagerArea::kPending) {
      if (area->FuseCheck(a)) {
        if (a->dom_->obj_id_ == ObjectType::kCubeOp) {
          if (input == a->dom_) {
            input = Exchange(area, input);
            SetArea(input, area->area_id_);
            temp_ops_.push_back(input);
          }
          area->dom_ = a->dom_;
          pv_black_mask_ |= 1ul << area->area_id_;
        }
        area->fused_.push_back(a);
        areas_[a->area_id_].second = area;
        if (!a->fused_.empty()) {
          for (auto x : a->fused_) {
            area->fused_.push_back(x);
            areas_[x->area_id_].second = area;
          }
        }
        kernel_used_--;
        pv_black_mask_ |= 1ul << a->area_id_;
        area->depend_mask_ |= a->depend_mask_;
        return input;
      }
      if (!input->IsLoad()) {
        a->state_ = EagerArea::kSubmitted;
      }
    }
    if (input->IsLoad()) {
      auto ac = static_cast<NDAccess *>(input);
      auto load = new NDLoad(ac->addr_.gm, ac->shape_ref_, ac->type_id_);
      SetStore(load, input);
      load->Normalize(a->objects_);
      objects_.push_back(load);
      input = load;
    } else {
      area->depend_mask_ |= a->depend_mask_;
      input = Exchange(a, input);
    }
  } else if (input->IsLoad()) {
    if (auto store = GetStore(input); store != nullptr && store->IsStore()) {
      auto a = areas_[GetArea(store)].second;
      area->depend_mask_ |= a->depend_mask_;
      a->state_ = EagerArea::kSubmitted;
    }
  }
  SetArea(input, area->area_id_);
  temp_ops_.push_back(input);
  return input;
}

void VKernelE::Split(NDObject *root) {
  int kidx = area_used_++;
  EagerArea *area = EagerArea::Assign(this, kidx);
  area->Reset(root);
  SetArea(root, kidx);
  temp_ops_.push_back(root);
  while (!temp_ops_.empty()) {
    auto op = temp_ops_.back();
    temp_ops_.pop_back();
    area->objects_.push_back(op);
    if (auto lhs = op->lhs_) {
      op->lhs_ = SplitPush(area, lhs);
      if (auto rhs = op->rhs_) {
        op->rhs_ = SplitPush(area, rhs);
      }
    }
  }
}

void VKernelE::Append(NDObject *obj) {
  obj->flags_ |= OBJ_FLAG_EAGER;  // TODO: add eager param for Normalize
  if (obj->IsStore()) {
    obj->Normalize(objects_);
    auto src = obj->lhs_;
    auto area_id = GetArea(src);
    if (area_id == -1) {
      Split(src);
      area_id = GetArea(src);
    }
    SetStore(src, obj);
    InitStoreInfo(obj, area_id);
  } else {
    auto exchange_input = [this](NDObject *input) -> NDObject * {
      if (GetArea(input) == -1) {
        Split(input);
      }
      auto load = Exchange(areas_[GetArea(input)].second, input);
      SetArea(load, -1);
      return load;
    };
    if (auto lhs = obj->lhs_) {
      if (lhs->obj_id_ == ObjectType::kReduce) {
        obj->lhs_ = exchange_input(lhs);
      }
      if (auto rhs = obj->rhs_; rhs != nullptr && rhs->obj_id_ == ObjectType::kReduce) {
        obj->rhs_ = exchange_input(rhs);
      }
    }
    InitObjInfo(obj);
    auto add_idx = objects_.size();
    obj->Normalize(objects_);
    if (add_idx < objects_.size()) {
      while (add_idx < objects_.size()) {
        InitObjInfo(objects_[add_idx++]);
      }
    }
  }
  objects_.push_back(obj);
}

NDObject *VKernelE::AppendCube(CubeOp *mm) {
  mm->flags_ |= OBJ_FLAG_EAGER;
  mm->NormalizeCube();
  CubeOptimizer opt(mm);
  uint64_t dep_mask = 0;
  auto prepare_input = [this, &dep_mask](bool is_stuff, NDObject *&input) {
    if (is_stuff) {
      for (auto op : temp_ops_) {
        InitObjInfo(op);
        objects_.push_back(op);
      }
      auto store = temp_ops_[temp_ops_.size() - 2];
      temp_ops_.clear();
      InitStoreInfo(store);
      Split(store);
      SetStore(input, store);
      auto aid = GetArea(store);
      areas_[aid].second->state_ = EagerArea::kSubmitted;
      dep_mask |= 1ul << aid;
    } else if (input->IsSimd()) {
      auto aid = GetArea(input);
      if (aid == -1) {
        Split(input);
        aid = GetArea(input);
      }
      auto area = areas_[aid].second;
      input = Exchange(area, input);
      area->state_ = EagerArea::kSubmitted;
      dep_mask |= 1ul << aid;
    }
  };
  prepare_input(opt.AlignA(temp_ops_), mm->lhs_);
  prepare_input(opt.AlignB(temp_ops_), mm->rhs_);
  if (mm->bias_) {
    prepare_input(opt.CastBias(temp_ops_), mm->bias_);
  }
  auto output = new NDStore(nullptr, mm);
  output->Normalize(temp_ops_);
  objects_.push_back(output);
  mm->output_ = output;
  NDObject *ret = mm;
  if (opt.SplitK(temp_ops_)) {
    for (auto op : temp_ops_) {
      if (op->obj_id_ == kCubeOp) {
        int aid = area_used_++;
        auto area = EagerArea::Assign(this, aid);
        area->ResetMix(op, EagerArea::kSubmitted);
        area->depend_mask_ |= dep_mask;
        dep_mask |= 1ul << aid;
        pv_black_mask_ |= 1ul << aid;
      } else {
        InitObjInfo(op);
      }
      objects_.push_back(op);
    }
    SetStore(temp_ops_[temp_ops_.size() - 2], output);
    ret = temp_ops_.back();
    temp_ops_.clear();
  }
  int aid = area_used_++;
  auto area = EagerArea::Assign(this, aid);
  area->ResetMix(mm, g_eager_cv_enable && !mm->batch_fold_? EagerArea::kPending : EagerArea::kSubmitted);
  area->depend_mask_ |= dep_mask;
  SetArea(mm, aid);
  SetStore(mm, output);
  InitStoreInfo(output, aid);
  objects_.push_back(mm);
  pv_black_mask_ |= 1ul << aid;
  return ret;
}

bool VKernelE::AppendOp(EagerVector *kernel, NDObject *op) {
  constexpr int kSubmitArea = -1;
  constexpr int kPendArea = -2;
  auto pend_check = [this, kernel](NDObject *obj) -> bool {
    return GetArea(obj) >= 0 || (GetArea(obj) == kPendArea && !AppendOp(kernel, obj));
  };
  if (op->lhs_ && (pend_check(op->lhs_) || (op->rhs_ && pend_check(op->rhs_)))) {
    SetArea(op, kPendArea);
    return false;
  }
  if (NDObject::attrs_[op->obj_id_].share_ndd) {
    op->nd_.data = op->lhs_->nd_.data;
  }
  kernel->EagerVector::Append(op);
  if (op->IsLoad()) {
    if (!(op->flags_ & OBJ_FLAG_EAGER)) {
      objects_.push_back(op);
    }
  } else if (auto store = GetStore(op); store && store->addr_.gm) {
    ASSERT(NDObject::attrs_[store->obj_id_].share_ndd);
    store->nd_.data = op->nd_.data;
    kernel->EagerVector::Append(store);
    temp_ops_.push_back(store);
  }
  SetArea(op, kSubmitArea);
  return true;
}

uint64_t VKernelE::CodeGen() {
  size_t obj_size = objects_.size();
  int kidx = kernel_used_;
  if (int ksize = static_cast<int>(kernels_.size()); kidx > ksize) {
    for (int i = ksize; i < kidx; ++i) {
      kernels_.push_back(new EagerVector());
    }
  }
  uint64_t extern_code_size = 0;
  int kernel_begin = 0;
  while (area_used_ > 0 && kidx > kernel_begin) {
    auto area = areas_[--area_used_].second;
    if (area->state_ == EagerArea::kFree) continue;
    area->state_ = EagerArea::kFree;
    auto kernel = kernels_[--kidx];
    kernel->Reset(area->dom_);
    if (!area->fused_.empty()) {
      for (auto it = area->fused_.rbegin(); it != area->fused_.rend(); ++it) {
        AppendOps(kernel, (*it)->objects_);
      }
    }
    AppendOps(kernel, area->objects_);
    if (area->dom_->obj_id_ == ObjectType::kCubeOp) {
      auto mm = static_cast<CubeOp *>(area->dom_);
      if (auto io = static_cast<NDAccess *>(mm->lhs_); io->addr_.gm == nullptr) {
        io->addr_.gm = AllocWS(GetStore(io));
      }
      if (auto io = static_cast<NDAccess *>(mm->rhs_); io->addr_.gm == nullptr) {
        io->addr_.gm = AllocWS(GetStore(io));
      }
      if (auto io = static_cast<NDAccess *>(mm->bias_); io && io->addr_.gm == nullptr) {
        io->addr_.gm = AllocWS(GetStore(io));
      }
      if (kernel->objects_.empty()) {
        kernel->CodeGenCube(mm);
      } else {
        AllocVectorWSS(kernel, obj_size);
        auto cube_code = kernel->CodeGenMix(mm);
        uint64_t pos_size = sizeof(vMixGroupMsg) * mm->block_dim_;
        void *pos_mem;
        if (auto it = wss_.upper_bound(pos_size - 1); it != wss_.end()) {
          pos_mem = it->second;
        } else {
          pos_mem = ws_alloc_(pos_size, user_data_);
          wss_.insert({pos_size, pos_mem});
        }
        cube_code->gm_pos = reinterpret_cast<uint64_t>(pos_mem);
      }
      if (auto output = mm->output_; !mm->atomic_add_ && !(output->flags_ & OBJ_FLAG_EAGER)) {
        wss_.insert({GetStoreSize(output), output->addr_.gm});
      }
    } else {
      EagerVector **pv_kernels = nullptr;
      int pv_num = 0;
      if (uint64_t pv_mask = ~(area->depend_mask_ | pv_black_mask_) & ((1ul << area_used_) - 1); pv_mask > 0) {
        pv_kernels = kernels_.data() + kernel_begin;
        for (; pv_mask && pv_num < g_eager_pv_width; ++pv_num) {
          int area_id = 63 - __builtin_clzl(pv_mask);
          pv_mask &= ~(1ul << area_id);
          pv_black_mask_ |= 1ul << area_id;
          auto a = areas_[area_id].second;
          a->state_ = EagerArea::kFree;
          auto k = kernels_[kernel_begin++];
          k->Reset(a->dom_);
          if (!a->fused_.empty()) {
            for (auto it = a->fused_.rbegin(); it != a->fused_.rend(); ++it) {
              AppendOps(k, (*it)->objects_);
            }
          }
          AppendOps(k, a->objects_);
        }
      }
      AllocVectorWSS(kernel, obj_size);
      if (uint64_t ws_size = kernel->EagerVector::CodeGenV(pv_kernels, pv_num); ws_size > 0) {
        void *ws_mem;
        if (auto it = wss_.upper_bound(ws_size - 1); it != wss_.end()) {
          ws_mem = it->second;
        } else {
          ws_mem = ws_alloc_(ws_size, user_data_);
          wss_.insert({ws_size, ws_mem});
        }
        for (auto op = kernel->code_.bind_wss_; op != nullptr; op = op->bind_list_) {
          op->Reloc(static_cast<char *>(ws_mem) + op->ws);
        }
        kernel->code_.bind_wss_ = nullptr;
      }
    }
    if (uint64_t code_size = kernel->code_.ReserveWorkspace(0); code_size > extern_code_size) {
      extern_code_size = code_size;
    }
    if (kidx > kernel_begin + 1) {
      for (auto op : temp_ops_) {
        if (auto store = static_cast<NDAccess *>(op); !GetStoreInplace(store)) {
          wss_.insert({GetStoreSize(store), store->addr_.gm});
        }
      }
    }
    temp_ops_.clear();
  }
  kernel_begin_ = kernel_begin;
  wss_.clear();
  if (extern_code_size) {
    extern_code_ = ws_alloc_(extern_code_size, user_data_);
  }
  return 0;
}

class EagerDumpRef : public DumpRefHelper {
 public:
  explicit EagerDumpRef(std::ostringstream &oss) : DumpRefHelper(oss) {}
  NDObject *GetInput(NDObject *input) override {
    while (input && idx_map_.count(input) == 0) {
      if (input->IsLoad()) {
        auto acc = VKernelE::GetStore(input);
        if (acc) {
          if (acc->IsLoad()) {
            return acc;
          }
          input = acc->lhs_;
          continue;
        }
      }
      input = input->lhs_;
    }
    return input;
  }
};

void VKernelE::Dump(std::ostringstream &oss, const std::string &indent) {
  if (kernel_used_ == 0) return;
  if (kernel_begin_ < kernel_used_) {
    oss << "vgraph.eager() {" << std::endl;
    std::string body_indent = indent + "  ";
    for (int i = 0; i < kernel_begin_; ++i) {
      auto k = kernels_[i];
      oss << body_indent << "// pv_merged " << i << std::endl;
      k->Dump(oss, body_indent);
      oss << std::endl;
    }
    for (int i = kernel_begin_; i < kernel_used_; ++i) {
      auto k = kernels_[i];
      oss << body_indent << "// eager " << i << std::endl;
      k->Dump(oss, body_indent);
      oss << std::endl;
    }
  } else {
    EagerDumpRef helper(oss);
    oss << "rgraph.eager() {" << std::endl;
    std::string body_indent = indent + "  ";
    for (auto op : objects_) {
      if (!op->IsStore() && (op->flags_ & OBJ_FLAG_EAGER)) {
        oss << body_indent;
        helper.Dump(op);
        oss << std::endl;
        if (auto store = GetStore(op); store && store->flags_ & OBJ_FLAG_EAGER) {
          oss << body_indent;
          helper.Dump(store);
          oss << std::endl;
        }
      }
    }
  }
  oss << "}";
}

std::string &VKernelE::DisAssemble() {
  std::ostringstream oss;
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    oss << "// eager " << i << std::endl;
    kernels_[i]->code_.DisAssemble(oss);
    oss << std::endl;
  }
  dump_str_ = oss.str();
  return dump_str_;
}

void VKernelE::TunerLaunch(EagerVector *kernel, void *stream) {
  auto tuner = static_cast<LazyCubeTuner *>(System::Instance().lazy_tuner_);
  tuner->Launch(static_cast<CubeOp *>(kernel->next_), kernel->code_, stream);
}
}  // namespace dvm
