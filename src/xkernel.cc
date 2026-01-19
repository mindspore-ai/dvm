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
#include "msprof.h"

namespace dvm {
namespace {
void ReplaceOperands(NDObject *replaced, NDObject *replacing, NDObject *obj) {
  obj->ForInput([replaced, replacing](NDObject *&op) {
    if (op && op == replaced) {
      op = replacing;
    }
  });
}

void TryBatchFold(CubeOp *mm, int dim_size, const std::vector<NDObject *> &objects) {
  PropRange range;
  range.base = dim_size - 1;
  range.depth = range.base;
  range.affine = PropRange::ELEMWISE;
  for (auto op : objects) {
    op->FoldProp(range);
  }
  if (range.depth < range.base) {
    mm->batch_fold_ = false;
  } else {
    for (auto op : objects) {
      if (auto ndd = op->Ndd(); ndd != nullptr) {
        auto &dims = ndd->dims;
        if (dims.size() <= 2) continue;
        for (size_t i = 2; i < dims.size(); ++i) {
          dims[1] *= dims[i];
        }
        dims.resize(2);
      }
      op->DimChanged();
    }
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
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    cube_op_->output_ = static_cast<NDAccess *>(obj);
  } else {
    if (post_fusion_ == nullptr) {
      if (IsDynamic()) {
        post_fusion_ = new VKernelD();
      } else {
        post_fusion_ = new VKernelS();
      }
    }
    obj->ForInput([this](NDObject *&op) {
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
    });
    post_fusion_->Append(obj);
  }
}

void MixKernel::EmplacePostFusion(NDObject *replaced_node, NDObject *replacing_node) {
  for (auto &op : post_fusion_->build_ops_) {
    if (op != replaced_node) {
      ReplaceOperands(replaced_node, replacing_node, op);
      stage_kernel_->Append(op);
    }
  }
}

void MixKernel::Release() {
  cube_op_->Recover();
  if (!stage_kernel_) {
    return;
  }
  auto current_kernel = static_cast<MixKernel *>(stage_kernel_->Current());
  auto target_post_fusion = current_kernel->post_fusion_;
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
  stage_kernel_ = new StagesKernel(flags_);
  auto &builder = stage_kernel_->builder_;
  int64_t pad_size[2] = {cube_op_->tactics_.lhs_pad_size, cube_op_->tactics_.rhs_pad_size};
  NDObject *inputs[2], *pad_inputs[2];
  NDObject *src_inputs[2] = {cube_op_->lhs_, cube_op_->rhs_};
  for (size_t i = 0; i < 2; i++) {
    if (pad_size[i]) {
      builder.StageSwitch(KernelType::kVector);
      pad_inputs[i] = builder.Load(nullptr, src_inputs[i]->shape_ref_, src_inputs[i]->type_id_);
      auto load = builder.Copy(pad_inputs[i]);
      inputs[i] = builder.StagePadStore(load, pad_size[i]);
    }
  }
  builder.StageSwitch(ktype_);
  for (size_t i = 0; i < 2; i++) {
    if (pad_size[i]) {
      inputs[i] = builder.StageLoad(inputs[i]);
    } else {
      inputs[i] = builder.Load(nullptr, src_inputs[i]->shape_ref_, src_inputs[i]->type_id_);
      pad_inputs[i] = inputs[i];
    }
  }
  auto src_bias = cube_op_->bias_;
  NDObject *bias = src_bias ? builder.Load(nullptr, src_bias->shape_ref_, src_bias->type_id_) : nullptr;
  NDObject *matmul_op;
  if (cube_op_->GetObjectType() == kCubeOp) {
    matmul_op = builder.MatMul(inputs[0], inputs[1], cube_op_->trans_a_, cube_op_->trans_b_, bias);
  } else {
    auto gmm_op = static_cast<GmmOp *>(cube_op_);
    matmul_op = builder.GroupedMatMul(inputs[0], inputs[1], cube_op_->trans_a_, cube_op_->trans_b_, bias,
                                      gmm_op->group_list_, gmm_op->group_type_, gmm_op->group_list_type_);
  }
  static_cast<CubeOp *>(matmul_op)->SetRealShape(cube_op_->m_real_, cube_op_->n_real_, cube_op_->k_real_, 0, 0);
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_op);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(builder.Store(nullptr, matmul_op));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size;
  code_ = std::move(stage_kernel_->code_);

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

  stage_kernel_ = new StagesKernel(flags_);
  auto &builder = stage_kernel_->builder_;
  std::vector<NDAccess *> split_lhs;
  std::vector<NDAccess *> split_rhs;
  std::vector<NDAccess *> split_out;
  NDObject *bias_op = nullptr;
  for (size_t i = 0; i < split_num; i++) {
    size_t offset_a = i * k_stride;
    size_t offset_b = i * k_stride;
    if (cube_op_->trans_a_) offset_a *= cube_op_->m_align_;
    if (!cube_op_->trans_b_) offset_b *= cube_op_->n_align_;
    builder.StageSwitch(ktype_);
    auto x = builder.Load(nullptr, cube_op_->lhs_->shape_ref_, cube_op_->lhs_->type_id_);
    auto y = builder.Load(nullptr, cube_op_->rhs_->shape_ref_, cube_op_->rhs_->type_id_);
    (void)split_lhs.emplace_back(static_cast<NDAccess *>(x));
    (void)split_rhs.emplace_back(static_cast<NDAccess *>(y));
    NDObject *matmul_op;
    if (i + 1 == split_num && cube_op_->bias_) {
      bias_op = builder.Load(nullptr, cube_op_->bias_->shape_ref_, cube_op_->bias_->type_id_);
    }
    if (cube_op_->GetObjectType() == kCubeOp) {
      matmul_op = builder.MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_op);
    } else {
      auto gmm_op = static_cast<GmmOp *>(cube_op_);
      matmul_op = builder.GroupedMatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_op, gmm_op->group_list_,
                                        gmm_op->group_type_, gmm_op->group_list_type_);
    }
    static_cast<CubeOp *>(matmul_op)->SetRealShape(cube_op_->m_real_, cube_op_->n_real_,
                                                   i + 1 == split_num ? k_tail : k_stride, offset_a, offset_b);
    static_cast<CubeOp *>(matmul_op)->SetOutFp32(i != 0);
    (void)split_out.emplace_back(static_cast<NDAccess *>(builder.Store(nullptr, matmul_op)));
  }
  auto matmul_fp32 = split_out.back()->lhs_;
  auto matmul_fp16 = builder.Cast(matmul_fp32, cube_op_->lhs_->type_id_);
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_fp16);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(builder.Store(nullptr, matmul_fp16));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size + split_out.back()->Size();
  code_ = std::move(stage_kernel_->code_);

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
  stage_kernel_ = new StagesKernel(flags_);
  auto &builder = stage_kernel_->builder_;
  builder.StageSwitch(KernelType::kVector);
  auto bias_bf16 = builder.Load(nullptr, cube_op_->bias_->shape_ref_, cube_op_->bias_->type_id_);
  auto bias_fp32 = builder.StageStore(builder.Cast(bias_bf16, kFloat32));

  builder.StageSwitch(ktype_);
  auto x = builder.Load(nullptr, cube_op_->lhs_->shape_ref_, cube_op_->lhs_->type_id_);
  auto y = builder.Load(nullptr, cube_op_->rhs_->shape_ref_, cube_op_->rhs_->type_id_);
  NDObject *matmul_op;
  auto bias_obj = builder.StageLoad(bias_fp32);
  if (cube_op_->GetObjectType() == kCubeOp) {
    matmul_op = builder.MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_obj);
  } else {
    auto gmm_op = static_cast<GmmOp *>(cube_op_);
    matmul_op = builder.GroupedMatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, bias_obj, gmm_op->group_list_,
                                      gmm_op->group_type_, gmm_op->group_list_type_);
  }
  if (post_fusion_) {
    EmplacePostFusion(sload_, matmul_op);
  }
  NDAccess *real_out{nullptr};
  if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
    real_out = static_cast<NDAccess *>(builder.Store(nullptr, matmul_op));
  }
  auto stage_workspace_size = stage_kernel_->CodeGen();
  auto workspace_size = stage_workspace_size;
  code_ = std::move(stage_kernel_->code_);

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
    if (auto comm = post_fusion_->comm_op_) {
      comm->mix_ = true;
      if (comm->lhs_ == sload_) {
        comm->SetCubeOp(cube_op_);
      }
    }
    if (!post_fusion_->NormBuild()) DvmException("MixKernel broker affine failed");
    if (cube_op_->batch_fold_) {
      TryBatchFold(cube_op_, sload_->nd_.size(), post_fusion_->objects_);
    }
    post_reserve = post_fusion_->ReserveCodeSize();
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
    CubeStoreType sync_type = (cube_code->flags & V_CUBE_FLAG_OUT_FP32) ? kCubeStoreGM : g_system.GetCubeStoreType();
    if (sync_type == kCubeStoreUBOnce) {
      static_cast<NDLoad *>(sload_)->flags_ |= OBJ_FLAG_LOAD_FROM_CC_ONCE;
      cube_code->flags |= V_CUBE_FLAG_STORE_UB_ONCE;
    } else if (sync_type == kCubeStoreUB) {
      static_cast<NDLoad *>(sload_)->flags_ |= OBJ_FLAG_LOAD_FROM_CC;
      cube_code->flags |= V_CUBE_FLAG_STORE_UB;
    } else if (auto inplace_store = post_fusion_->FindInplaceStore(sload_, nullptr)) {
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
  shard.sink = true;
  shard.dom = &cube_op_->ndd_.dims;
  shard.tile[0] = cube_op_->n0_;
  shard.tail[0] = cube_op_->n_real_ % cube_op_->n0_;
  shard.tile[1] = cube_op_->m0_;
  shard.tail[1] = cube_op_->m_real_ % cube_op_->m0_;
  shard.stride[0] = cube_op_->n_real_;
  shard.stride[1] = cube_op_->n_real_ * cube_op_->m_real_;
  post_fusion_->Shard(shard);
  post_fusion_->PrepareTiling();
  MixVisitCoder visit;
  post_fusion_->AddVisitor(&visit);
  uint8_t *code_end;
  if (cube_code->flags & V_CUBE_FLAG_STORE_UB_ONCE) {
    g_system.SetLocalMemSize(g_system.LocalMemSize() - cube_op_->BaseSize());
    code_end = post_fusion_->DoCodeGen(2, code_.data_ + head_reserve, post_reserve);
    g_system.SetLocalMemSize(g_system.LocalMemSize() + cube_op_->BaseSize());
  } else {
    code_end = post_fusion_->DoCodeGen(2, code_.data_ + head_reserve, post_reserve);
  }
  uint64_t subtile0 = (post_fusion_->tile_num_ + 1) / 2;
  uint64_t subtile1 = post_fusion_->tile_num_ - subtile0;
  if (cube_code->flags & V_CUBE_FLAG_STORE_UB) {
    if (CeilDiv(cube_op_->m0_, post_fusion_->tile_num_) % CubeOp::BLOCK_SIZE) {
      cube_code->flags |= V_CUBE_FLAG_DIS_UNITFLAG;
    }
    cube_code->ub_c = vCubeOp::UbInfoEncode(sload_->xbuf_, subtile0, subtile1);
  } else if (cube_code->flags & V_CUBE_FLAG_STORE_UB_ONCE) {
    cube_code->ub_c = vCubeOp::UbInfoEncode(g_system.LocalMemSize() - cube_op_->BaseSize(), subtile0, subtile1);
  } else {
    cube_code->flags |= V_CUBE_FLAG_GROUP_SET;
  }
  code_.data_size_ = code_end - code_.data_;
  code_.UpdateMix(&visit, shard.tile, shard.tail, shard.stride, subtile0, subtile1);
  code_.Combine(post_fusion_->code_, 0);
  return ws_size;
}

uint64_t MixKernel::CodeGen() {
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

void MixKernel::Clone(VKernel *base, CloneHelper &helper) {
  auto k = static_cast<MixKernel *>(base);
  auto cube = k->cube_op_;
  cube->lhs_->CloneUpdate(helper);
  cube->rhs_->CloneUpdate(helper);
  if (cube->bias_) {
    cube->bias_->CloneUpdate(helper);
  }
  if (cube->obj_id_ == ObjectType::kGmmOp) {
    static_cast<GmmOp *>(cube)->group_list_->CloneUpdate(helper);
  }
  cube_op_ = static_cast<CubeOp *>(cube->CloneUpdate(helper));
  if (k->post_fusion_) {
    post_fusion_ = IsDynamic() ? new VKernelD() : new VKernelS();
    post_fusion_->Clone(k->post_fusion_, helper);
    sload_ = static_cast<NDAccess *>(helper.GetClone(k->sload_));
  }
  for (auto &[load1, load2] : k->reloads_) {
    auto clone1 = static_cast<NDAccess *>(helper.GetClone(load1));
    auto clone2 = static_cast<NDAccess *>(helper.GetClone(load2));
    reloads_.emplace_back(clone1, clone2);
  }
}

void DynMixKernel::Record() {
  if (post_fusion_) {
    static_cast<VKernelD *>(post_fusion_)->Recover();
    for (auto op : post_fusion_->build_ops_) {
      op->ForInput([this](NDObject *&in) { tracker_.Record(&in); });
    }
  }
}

void DynMixKernel::Release() {
  tracker_.RecoverClear();
  MixKernel::Release();
}

uint64_t DynMixKernel::CodeGen() {
  delete stage_kernel_;
  stage_kernel_ = nullptr;
  code_.~Code();
  code_.data_ = nullptr;
  code_ = std::move(Code());
  Record();
  auto ws_size = MixKernel::CodeGen();
  Release();
  return ws_size;
}

int StageCodeWrap::LaunchWrap(void *workspace, void *stream) {
  for (auto s : kernel_->stages_) {
    s->kernel->code_.Launch(workspace, stream);
  }
  return 0;
}

void StageCodeWrap::DasWrap(std::ostringstream &oss) {
  for (auto s : kernel_->stages_) {
    s->kernel->code_.DisAssemble(oss);
    oss << std::endl;
  }
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
    stages_.back()->ios.emplace_back(static_cast<NDAccess *>(obj), nullptr);
  }
}

#define STAGE_FLAG_WORKSPACE 1
#define STAGE_FLAG_REUSE 2

uint64_t StagesKernel::CodeGen() {
  if (IsDynamic()) {
    code_.Clear();
  }
  for (auto s : stages_) {
    auto k = s->kernel;
    s->ws_size = k->code_.ReserveWorkspace(k->CodeGen());
  }
  uint64_t ws_size = AllocWorkspace();
  for (auto s : stages_) {
    auto &code = s->kernel->code_;
    code_.CombineBind(code, s->ws_offset);
    for (auto &[op, prod] : s->ios) {
      if (!op->CheckFlag(OBJ_FLAG_STAGE_IO)) continue;
      if (op->IsStore()) {
        if (op->xbuf_ == STAGE_FLAG_REUSE) {
          code_.BindOp(op->addr_, GetOutputReuse(op)->addr_);
        } else {
          code_.BindWorkspace(op->addr_, GetWorkspace(op));
        }
      } else {
        code_.BindOp(op->addr_, prod->addr_);
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
    for (auto &[io, store] : stage->ios) {
      if (store != nullptr) {
        if (!store->CheckFlag(OBJ_FLAG_STAGE_IO) || lives.find(store) != lives.end()) continue;
        if (stage->kernel->KType() == KernelType::kVector && !stage->kernel->IsDynamic()) {  // TODO: parallel fusion
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
    for (auto &p : stage->ios) {
      auto io = p.first;
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

void StagesKernel::Clone(VKernel *base, CloneHelper &helper) {
  auto k = static_cast<StagesKernel *>(base);
  _Builder b(this);
  for (auto stage : k->stages_) {
    b.StageSwitch(stage->kernel->KType());
    auto to_stage = stages_.back();
    to_stage->kernel->Clone(stage->kernel, helper);
    for (auto &[op, prod] : stage->ios) {
      auto clone = static_cast<NDAccess *>(helper.GetClone(op));
      to_stage->ios.emplace_back(clone, prod ? static_cast<NDAccess *>(helper.GetClone(prod)) : nullptr);
      if (op->CheckFlag(OBJ_FLAG_STAGE_IO)) {
        clone->flags_ |= OBJ_FLAG_STAGE_IO;
      }
    }
  }
}

void StagesKernel::_Builder::StageSwitch(KernelType type) {
  VKernel *kernel = nullptr;
  auto is_dyn = kernel_->IsDynamic();
  if (type == KernelType::kVector) {
    kernel = is_dyn ? new VKernelD() : new VKernelS();
  } else if (type == KernelType::kMix) {
    kernel = is_dyn ? new DynMixKernel() : new MixKernel();
  } else if (type == KernelType::kParallel) {
    ASSERT(!is_dyn);
    kernel = new VKernelP();
  } else {
    ASSERT(0);
  }
  _StagesKernel()->AddStage(kernel);
}

NDObject *StagesKernel::_Builder::StageLoad(NDObject *stage_store) {
  auto store = static_cast<NDAccess *>(stage_store);
  auto op = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
  auto current = _StagesKernel()->Current();
  current->Append(op);
  _StagesKernel()->StageLoad(current, op, store);
  return op;
}

NDObject *StagesKernel::_Builder::StageStore(NDObject *input) {
  auto op = new NDStore(nullptr, input);
  auto current = _StagesKernel()->Current();
  current->Append(op);
  _StagesKernel()->StageStore(current, op);
  return op;
}

NDObject *StagesKernel::_Builder::StagePadStore(NDObject *input, int64_t pad_size) {
  auto op = new NDPadStore(nullptr, input, pad_size);
  auto current = _StagesKernel()->Current();
  current->Append(op);
  _StagesKernel()->StageStore(current, op);
  return op;
}

void SequenceKernel::Append(NDObject *obj) {
  int cur_sid = stages_.size() - 1;
  SetStore(obj, nullptr);
  SetStage(obj, cur_sid);
  if (obj->obj_id_ == ObjectType::kStore) {
    SetStore(obj->lhs_, obj);
  }
  obj->ForInput([this, cur_sid](NDObject *&in) {
    int sid = GetStage(in);
    if (sid == cur_sid) {
      return;
    }
    ASSERT(!in->IsLoad());
    NDAccess *store;
    if (in->obj_id_ == ObjectType::kStore) {
      store = static_cast<NDAccess *>(in);
    } else {
      store = GetStore(in);
      if (store == nullptr) {
        store = new NDStore(in);
        auto stage = stages_[sid];
        stage->kernel->Append(store);
        stage->StageStore(store);
        SetStore(in, store);
      }
    }
    NDAccess *load = nullptr;
    auto stage = stages_[cur_sid];
    for (auto &[op, prod]: stage->ios) {
      if (prod == store) {
        load = op;
        break;
      }
    }
    if (load == nullptr) {
      load = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
      stage->kernel->Append(load);
      stage->StageLoad(load, store);
    }
    in = load;
  });
  StagesKernel::Append(obj);
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
    dom_->output_->type_id_ = DataType::kFloat32;
    for (size_t i = 0; i < split_num - 1; ++i) {
      CubeOp *op;
      if (dom_->obj_id_ == kCubeOp) {
        op = new CubeOp(dom_->lhs_, dom_->rhs_, dom_->trans_a_, dom_->trans_b_, i == 0 ? dom_->bias_ : nullptr);
      } else {
        auto gmm_op = static_cast<GmmOp *>(dom_);
        op = new GmmOp(dom_->lhs_, dom_->rhs_, dom_->trans_a_, dom_->trans_b_, (i == 0 ? dom_->bias_ : nullptr),
                       gmm_op->group_list_, gmm_op->group_type_, gmm_op->group_list_type_);
      }
      op->SetRealShape(dom_->m_real_, dom_->n_real_, k_stride, offset_a, offset_b);
      op->NormalizeCube();
      op->SetOutFp32(i > 0);
      ops.push_back(op);
      offset_a += dom_->trans_a_ ? dom_->m_align_ * k_stride : k_stride;
      offset_b += dom_->trans_b_ ? k_stride : dom_->n_align_ * k_stride;
      op->output_ = dom_->output_;
      op->batch_fold_ = dom_->batch_fold_;
    }
    dom_->SetRealShape(dom_->m_real_, dom_->n_real_, k_tail, offset_a, offset_b);
    dom_->SetOutFp32(true);
    dom_->NormalizeOutput();
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
    dom_->SetRealShape(m, n, k, 0, 0);
    dom_->NormalizeCube();
  }

  CubeOp *dom_;
};

class EagerVector : public VectorKernel {
 public:
  EagerVector() : VectorKernel(KernelType::kEager, 0) {
    objects_.reserve(64);
    static_ops_.reserve(16);
  }

  void Append(NDObject *obj) override {
    obj->index_ = objects_.size();
    objects_.push_back(obj);
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

  void Reset(NDObject *dom) {
    max_type_ = dom->type_id_;
    min_type_ = dom->type_id_;
    mm_ = nullptr;
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
    mm->CodeGen(body, g_system.lazy_tuner_);
    code_.block_dim_ = mm->block_dim_;
    code_.data_size_ = size;
    code_.UpdateC();
    mm_ = mm;
  }

  vCubeOp *CodeGenMix(CubeOp *mm) {
    size_t head_reserve = code_.HeadSize() + sizeof(vCubeOp);
    size_t post_reserve = ReserveCodeSize();  // TODO: visit size
    code_.Alloc(head_reserve + post_reserve);
    if (mm->batch_fold_) {
      TryBatchFold(mm, mm->nd_.size(), objects_);
    }
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
    mm->CodeGen(cube_code, g_system.lazy_tuner_);
    BuildDomain();
    ShardParam shard;
    shard.base = 0;
    shard.sink = true;
    shard.dom = &mm->ndd_.dims;
    shard.tile[0] = mm->n0_;
    shard.tail[0] = mm->n_real_ % mm->n0_;
    shard.tile[1] = mm->m0_;
    shard.tail[1] = mm->m_real_ % mm->m0_;
    shard.stride[0] = mm->n_real_;
    shard.stride[1] = mm->n_real_ * mm->m_real_;
    Shard(shard);
    PrepareTiling();
    MixVisitCoder visit;
    AddVisitor(&visit);
    auto code_end = DoCodeGen(2, code_.data_ + head_reserve, post_reserve);
    ClearShard();
    uint64_t subtile0 = (tile_num_ + 1) / 2;
    uint64_t subtile1 = tile_num_ - subtile0;
    cube_code->flags |= V_CUBE_FLAG_GROUP_SET;
    code_.data_size_ = code_end - code_.data_;
    code_.block_dim_ = mm->block_dim_;
    code_.UpdateMix(&visit, shard.tile, shard.tail, shard.stride, subtile0, subtile1);
    mm_ = mm;
    return cube_code;
  }

  enum { kMaxPvNum = 8 };
  uint64_t CodeGenV(EagerVector **others, int other_num) {
    uint64_t core_total = g_system.CoreNum();
    BuildDomain();
    if (other_num == 0) {
      return DoCodeGen(core_total);
    }
    PrepareTiling();
    std::pair<EagerVector *, uint64_t> children[EagerVector::kMaxPvNum + 1];
    uint64_t total_load = 0;
    uint64_t code_reserve = 0;
    int child_num = 0;
    int max_idx = 0;
    for (int i = 0; i < other_num + 1; ++i) {
      EagerVector *kernel;
      if (i < other_num) {
        kernel = others[i];
        kernel->BuildDomain();
        kernel->PrepareTiling();
      } else {
        kernel = this;
      }
      uint64_t load = kernel->tile_size_ * kernel->objects_.size();
      if (load > 0) {
        auto &node = children[child_num];
        node.first = kernel;
        node.second = load;
        if (max_idx > 0 && load >= children[max_idx].second) {
          max_idx = child_num;
        }
        total_load += load;
        child_num++;
        code_reserve += kernel->ReserveCodeSize();
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
    if (max_idx != child_num - 1) {
      std::swap(children[max_idx], children[child_num - 1]); // avoid blockdim overflow
    }
    PCodeEncoder encoder;
    encoder.Reset(&code_, Code::kTargetVec, child_num, code_reserve);
    uint64_t block_begin = 0;
    for (int i = 0; i < child_num; ++i) {
      auto k = children[i].first;
      uint64_t workload = children[i].second;
      uint64_t core_limit = std::max((core_total - block_begin) * workload / total_load, 1ul);
      auto code_begin = encoder.ProgData();
      uint64_t code_size = k->DoCodeGen(core_limit, code_begin, k->ReserveCodeSize()) - code_begin;
      ASSERT(k->visit_ == nullptr);
      uint64_t block_dim = k->CompactBlockDim(core_limit);
      total_load -= workload;
      auto prog = encoder.Append(Code::GenEntryV(k->tile_num_, block_dim, code_size), code_size);
      encoder.AssignAiv(block_begin, block_dim, prog);
      block_begin += block_dim;
      if (k != this) {
        code_.Combine(k->code_, 0);
      }
    }
    encoder.Submit(block_begin);
    ASSERT(block_begin <= core_total);
    return 0;
  }

  void Dump(std::ostringstream &oss, const std::string &indent) {
    if (mm_) {
      oss << indent << "vgraph_cube() {" << std::endl;
      auto body_indent = indent + "  ";
      oss << body_indent << "%0" << mm_->nd_ << " = ";
      mm_->Dump(true, oss);
      oss << "(%1" << mm_->lhs_->nd_;
      oss << ", %2" << mm_->rhs_->nd_;
      if (mm_->bias_) {
        oss << ", %3" << mm_->bias_->nd_;
      }
      oss << ")" << std::endl << indent << "}";
      if (objects_.empty()) return;
      oss << std::endl;
    }
    VectorKernel::Dump(oss, indent);
  }

  CubeOp *mm_;
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

  static EagerArea *Assign(_SplitKernel *k, size_t aid) {
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

SplitContext::SplitContext() {
  app_.reserve(24);
  gen_.reserve(12);
  kill_.reserve(12);
  build_.reserve(64);
  mem_blocks_.push_back(new MemNode[MEM_BLOCK_SIZE]);
}

SplitContext::~SplitContext() {
  for (auto block : mem_blocks_) {
    delete []block;
  }
}

SplitContext::MemNode *SplitContext::DoAlloc(size_t size) {
  if (size <= mem_head_->size) {
    auto node = mem_head_;
    mem_head_ = mem_head_->next;
    return node;
  }
  auto last = mem_head_;
  for (auto node = last->next; node != nullptr; node = node->next) {
    if (size <= node->size) {
      last->next = node->next;
      if (node == mem_tail_) {
        mem_tail_ = last;
      }
      return node;
    }
    last = node;
  }
  ASSERT(0);
  return nullptr;
}

void SplitContext::Free(void *addr, size_t size) {
  MemNode *node;
  uint64_t block_idx = mem_pos_ >> 32;
  uint64_t node_idx = mem_pos_ & 0xfffffffful;
  if (node_idx < MEM_BLOCK_SIZE) {
    node = &mem_blocks_[block_idx][node_idx];
    mem_pos_++;
  } else {
    block_idx++;
    if (block_idx == mem_blocks_.size()) {
      mem_blocks_.push_back(new MemNode[MEM_BLOCK_SIZE]);
    }
    node = &mem_blocks_[block_idx][0];
    mem_pos_ = block_idx << 32 | 1ul;
  }
  node->size = size;
  node->addr = addr;
  if (mem_head_ == nullptr) {
    mem_head_ = mem_tail_ = node;
    node->next = nullptr;
  } else if (size <= mem_head_->size) {
    node->next = mem_head_;
    mem_head_ = node;
  } else if (size >= mem_tail_->size) {
    mem_tail_->next = node;
    node->next = nullptr;
    mem_tail_ = node;
  } else {
    auto last = mem_head_;
    for (auto x = last->next; x != nullptr; x = x->next) {
      if (size <= x->size) {
        last->next = node;
        node->next = x;
        break;
      }
      last = x;
    }
  }
}

static int g_eager_pv_width = -1;
_SplitKernel::_SplitKernel(KernelType type, uint32_t flags) : VKernel(type, flags) {
  if (g_eager_pv_width == -1) {
    if (const char *width = getenv("DVM_EAGER_PV_WIDTH"); width != nullptr) {
      g_eager_pv_width = std::stoi(width);
    } else {
      g_eager_pv_width = 0;
    }
    ASSERT(g_eager_pv_width <= EagerVector::kMaxPvNum);
  }
}

_SplitKernel::~_SplitKernel() {
  for (auto k : kernels_) {
    delete k;
  }
  for (auto &a : areas_) {
    delete a.first;
  }
}

NDObject *_SplitKernel::Exchange(NDObject *input, int to_aid) {
  ASSERT(!input->IsLoad());
  NDAccess *store = GetStore(input);
  if (store == nullptr) {
    store = new NDStore(nullptr, input);
    store->Normalize(ctx_->app_);
    InitStoreInfo(store, GetArea(input));
    SetStore(input, store);
    objects_.push_back(store);
    ASSERT(GetRecentLoad(store) == nullptr);  // construct to nullptr
  } else if (auto recent = GetRecentLoad(store); recent != nullptr && GetArea(recent) == to_aid) {
    return recent;
  }
  auto load = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
  SetRecentLoad(store, load);
  SetStore(load, store);
  load->Normalize(objects_);
  objects_.push_back(load);
  return load;
}

NDObject *_SplitKernel::SplitPush(EagerArea *area, NDObject *input) {
  auto cube_check = [this, &input](EagerArea *c, EagerArea *v) -> bool {
    if (v->state_ == EagerArea::kSubmitted) { // reduce
      return false;
    }
    if (input == c->dom_) {
      input = Exchange(input, v->area_id_);
      SetArea(input, v->area_id_);
      ctx_->app_.push_back(input);
    }
    v->dom_ = c->dom_;
    pv_black_mask_ |= 1ul << v->area_id_;
    return true;
  };
  if (auto idx = GetArea(input); idx >= 0) {
    auto a = areas_[idx].second;
    if (a == area) return input;
    if (a->state_ == EagerArea::kPending) {
      if (area->FuseCheck(a) && (!a->dom_->IsCube() || cube_check(a, area))) {
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
      if (input->CheckFlag(OBJ_FLAG_EAGER)) {
        SetStore(load, input);  // TRICK: force load entry wss alloc as swap load to update its gm
      } else {
        auto store = GetStore(input);
        ASSERT(store != nullptr && store->IsStore());
        SetStore(load, store);
        auto a = areas_[GetArea(store)].second;
        area->depend_mask_ |= a->depend_mask_;
        a->state_ = EagerArea::kSubmitted;
      }
      load->Normalize(a->objects_);
      objects_.push_back(load);
      input = load;
    } else {
      area->depend_mask_ |= a->depend_mask_;
      input = Exchange(input, area->area_id_);
    }
  } else if (input->IsLoad()) {
    if (auto store = GetStore(input); store != nullptr && store->IsStore()) {
      auto a = areas_[GetArea(store)].second;
      area->depend_mask_ |= a->depend_mask_;
      a->state_ = EagerArea::kSubmitted;
    }
  }
  SetArea(input, area->area_id_);
  ctx_->app_.push_back(input);
  return input;
}

void _SplitKernel::Split(NDObject *root) {
  int kidx = area_used_++;
  EagerArea *area = EagerArea::Assign(this, kidx);
  area->Reset(root);
  SetArea(root, kidx);
  auto &stack = ctx_->app_;
  stack.push_back(root);
  while (!stack.empty()) {
    auto op = stack.back();
    stack.pop_back();
    area->objects_.push_back(op);
    op->ForInput([this, area](NDObject *&in) { in = SplitPush(area, in); });
  }
}

void _SplitKernel::Append(NDObject *obj) {
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
    obj->ForInput([this](NDObject *&op) {
      if (op->obj_id_ == ObjectType::kReduce) {
        if (GetArea(op) == -1) {
          Split(op);
        }
        op = Exchange(op, -1);
        SetArea(op, -1);
      }
    });
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

NDObject *_SplitKernel::AppendCube(CubeOp *mm) {
  mm->flags_ |= OBJ_FLAG_EAGER;
  mm->NormalizeCube();
  CubeOptimizer opt(mm);
  auto &temp_ops = ctx_->app_;
  uint64_t dep_mask = 0;
  auto prepare_input = [this, &dep_mask, &temp_ops](bool is_stuff, NDObject *&input) {
    if (is_stuff) {
      for (auto op : temp_ops) {
        InitObjInfo(op);
        objects_.push_back(op);
      }
      auto store = temp_ops[temp_ops.size() - 2];
      temp_ops.clear();
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
      input = Exchange(input, area_used_);
      area->state_ = EagerArea::kSubmitted;
      dep_mask |= 1ul << aid;
    }
    SetArea(input, area_used_);
  };
  prepare_input(opt.AlignA(temp_ops), mm->lhs_);
  prepare_input(opt.AlignB(temp_ops), mm->rhs_);
  if (mm->bias_) {
    prepare_input(opt.CastBias(temp_ops), mm->bias_);
  }
  auto output = new NDStore(nullptr, mm);
  output->Normalize(temp_ops);
  objects_.push_back(output);
  mm->output_ = output;
  NDObject *ret = mm;
  if (opt.SplitK(temp_ops)) {
    for (auto op : temp_ops) {
      if (op->IsCube()) {
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
    SetStore(temp_ops[temp_ops.size() - 2], output);
    ret = temp_ops.back();
    temp_ops.clear();
  }
  int aid = area_used_++;
  auto area = EagerArea::Assign(this, aid);
  area->ResetMix(mm, EagerArea::kPending);
  area->depend_mask_ |= dep_mask;
  SetArea(mm, aid);
  SetStore(mm, output);
  InitStoreInfo(output, aid);
  objects_.push_back(mm);
  pv_black_mask_ |= 1ul << aid;
  return ret;
}

void _SplitKernel::BuildKernel(EagerVector *kernel, const EagerArea *area, WsAllocator *ws_alloc) {
  auto push_input = [](std::vector<NDObject *> &stack, NDObject *op) {
    bool pend = false;
    op->ForInput([&pend, &stack](NDObject *in) {
      if (GetArea(in) >= 0) {
        stack.push_back(in);
        pend = true;
      }
    });
    return pend;
  };
  auto build_op = [this, kernel](NDObject *op) {
    if (op->SharedNdd()) {
      op->nd_.data = op->lhs_->nd_.data;
    }
    kernel->EagerVector::Append(op);
    if (op->IsLoad()) {
      if (!(op->flags_ & OBJ_FLAG_EAGER)) {
        ctx_->gen_.push_back(static_cast<NDAccess *>(op));
      }
    } else if (auto store = GetStore(op); store && store->addr_.gm) {
      ASSERT(store->SharedNdd());
      store->nd_.data = op->nd_.data;
      kernel->EagerVector::Append(store);
      if (!GetStoreInplace(store)) {
        ctx_->kill_.emplace_back(store, GetStoreSize(store));
      }
    }
    SetArea(op, -1);
  };
  kernel->Reset(area->dom_);
  size_t kill_begin = ctx_->kill_.size();
  size_t fused_size = area->fused_.size();
  auto &stack = ctx_->build_;
  const EagerArea *cur = nullptr;
  do {
    cur = fused_size ? area->fused_[--fused_size] : area;
    for (auto it = cur->objects_.rbegin(); it != cur->objects_.rend(); ++it) {
      auto op = *it;
      if (GetArea(op) < 0) continue;
      if (push_input(stack, op)) {
        while (!stack.empty()) {
          auto top = stack.back();
          if (!push_input(stack, top)) {
            stack.pop_back();
            if (GetArea(top) >= 0) {
              build_op(top);
            }
          }
        }
      }
      build_op(op);
    }
  } while (cur != area);
  for (auto gen : ctx_->gen_) {
    auto store = GetStore(gen);
    if (store->addr_.gm == nullptr) {
      bool inplaced = false;
      auto store_size = GetStoreSize(store);
      for (auto it = ctx_->kill_.begin() + kill_begin; it != ctx_->kill_.end(); ++it) {
        if (it->first != nullptr && store_size == it->second && gen->nd_.dims() == it->first->nd_.dims()) {
          store->addr_.gm = it->first->addr_.gm;
          it->first = nullptr;
          inplaced = true;
          break;
        }
      }
      if (!inplaced) {
        AllocWS(store, ws_alloc);
      }
    }
    gen->addr_.gm = store->addr_.gm;
  }
  ctx_->gen_.clear();
}

void _SplitKernel::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  for (auto reloc = relocs; reloc < relocs + reloc_size; ++reloc) {
    static_cast<NDAccess *>(reloc->io)->addr_.gm = reloc->addr;
  }
  int kidx = kernel_used_;
  if (int ksize = static_cast<int>(kernels_.size()); kidx > ksize) {
    for (int i = ksize; i < kidx; ++i) {
      kernels_.push_back(new EagerVector());
    }
  }
  ctx_->MemReset();
  uint64_t extern_code_size = 0;
  int kernel_begin = 0;
  while (area_used_ > 0 && kidx > kernel_begin) {
    auto area = areas_[--area_used_].second;
    if (area->state_ == EagerArea::kFree) continue;
    area->state_ = EagerArea::kFree;
    auto kernel = kernels_[--kidx];
    BuildKernel(kernel, area, ws_alloc);
    if (area->dom_->IsCube()) {
      auto mm = static_cast<CubeOp *>(area->dom_);
      auto cube_gen = [this, ws_alloc](NDObject *in) {
        if (auto io = static_cast<NDAccess *>(in); io->addr_.gm == nullptr) {
          auto store = GetStore(io);
          io->addr_.gm = store->addr_.gm ? store->addr_.gm : AllocWS(store, ws_alloc);
        }
      };
      cube_gen(mm->lhs_);
      cube_gen(mm->rhs_);
      if (mm->bias_) {
        cube_gen(mm->bias_);
      }
      if (kernel->objects_.empty()) {
        kernel->CodeGenCube(mm);
      } else {
        auto cube_code = kernel->CodeGenMix(mm);
        uint64_t pos_size = sizeof(vMixGroupMsg) * mm->block_dim_;
        void *pos_mem;
        if (auto mem = ctx_->Alloc(pos_size)) {
          pos_mem = mem->addr;
          pos_size = mem->size;
        } else {
          pos_mem = ws_alloc->Alloc(pos_size);
        }
        ctx_->Free(pos_mem, pos_size);
        cube_code->gm_pos = reinterpret_cast<uint64_t>(pos_mem);
      }
      if (auto output = mm->output_; !mm->atomic_add_ && !(output->flags_ & OBJ_FLAG_EAGER)) {
        ctx_->Free(output->addr_.gm, GetStoreSize(output));
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
          BuildKernel(k, a, ws_alloc);
        }
      }
      if (uint64_t ws_size = kernel->EagerVector::CodeGenV(pv_kernels, pv_num); ws_size > 0) {
        void *ws_mem;
        if (auto mem = ctx_->Alloc(ws_size)) {
          ws_mem = mem->addr;
        } else {
          ws_mem = ws_alloc->Alloc(ws_size);
          ctx_->Free(ws_mem, ws_size);
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
      for (auto &kill : ctx_->kill_) {
        if (kill.first) {
          ctx_->Free(kill.first->addr_.gm, kill.second);
        }
      }
    }
    ctx_->kill_.clear();
  }
  kernel_begin_ = kernel_begin;
  if (extern_code_size) {
    extern_code_ = ws_alloc->Alloc(extern_code_size);
  }
}

CubeOp *_SplitKernel::GetCubeOp(EagerVector *kernel) { return kernel->mm_; }

class EagerDumpRef : public DumpRefHelper {
 public:
  explicit EagerDumpRef(std::ostringstream &oss) : DumpRefHelper(oss) {}
  NDObject *GetInput(NDObject *input) override {
    while (input && idx_map_.count(input) == 0) {
      if (input->IsLoad()) {
        auto acc = _SplitKernel::GetStore(input);
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

void _SplitKernel::Dump(std::ostringstream &oss, const std::string &indent) {
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

std::string &_SplitKernel::DisAssemble() {
  std::ostringstream oss;
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    oss << "// eager " << i << std::endl;
    kernels_[i]->code_.DisAssemble(oss);
    oss << std::endl;
  }
  dump_str_ = oss.str();
  return dump_str_;
}

int _SplitKernel::Launch(void *stream) {
  auto child_launch = [this, stream](EagerVector *kernel) {
    auto &code = kernel->code_;
    if (code.target_ == Code::kTargetCube && g_system.lazy_tuner_) {
      auto tuner = static_cast<LazyCubeTuner *>(g_system.lazy_tuner_);
      tuner->Launch(kernel->mm_, code, stream);
    } else {
      code.Launch(extern_code_, stream);
    }
  };
  if (likely(!g_system.enable_profile_)) {
    for (int i = kernel_begin_; i < kernel_used_; ++i) {
      child_launch(kernels_[i]);
    }
    return 0;
  }
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    MsprofHelper msprof_helper;
    auto &info = msprof_helper.info_;
    auto vector_kernel = kernels_[i];
    info.block_dim = vector_kernel->code_.block_dim_;
    auto target = vector_kernel->code_.target_;
    std::ostringstream oss;
    oss << "Dvm";
    std::vector<NDObject *> inputs;
    std::vector<NDObject *> outputs;
    if (target > Code::kTargetVec) {
      auto cube_op = GetCubeOp(kernels_[i]);
      if (cube_op) {
        cube_op->Dump(false, oss);
        inputs.emplace_back(cube_op->lhs_);
        inputs.emplace_back(cube_op->rhs_);
        if (cube_op->bias_) {
          inputs.emplace_back(cube_op->bias_);
        }
        if (cube_op->output_->IsStore()) {
          outputs.emplace_back(cube_op->output_);
        }
      }
    }
    for (auto op : vector_kernel->objects_) {
      if (op->IsLoad()) {
        inputs.emplace_back(op);
      } else if (op->IsStore()) {
        outputs.emplace_back(op);
      }
      if (op->GetObjectType() != kStore && op->GetObjectType() != kLoad) {
        op->Dump(false, oss);
      }
    }
    for (auto op : inputs) {
      if (op->flags_ & OBJ_FLAG_EAGER) {
        info.AppendInput(op);
      }
    }
    for (auto op : outputs) {
      if (op->flags_ & OBJ_FLAG_EAGER) {
        info.AppendOutput(op);
      }
    }
    auto prof_name = oss.str();
    info.op_name = prof_name.c_str();
    info.op_fullname = info.op_name;
    msprof_helper.InitReportNode();
    msprof_helper.Update(target);
    child_launch(vector_kernel);
    msprof_helper.ReportTask();
  }
  return 0;
}

class SlotWsAllocator : public WsAllocator {
 public:
  SlotWsAllocator(SplitContext::SlotWorkspace &ws) : ws_(ws) {}
  void *Alloc(uint64_t size) override {
    if (size > 0) {
      ws_.emplace_back(size, nullptr);
    }
    return reinterpret_cast<void *>(ws_.size());
  }
  SplitContext::SlotWorkspace &ws_;
};

void _SplitKernel::SlotCodeGen(const RelocEntry *relocs, size_t reloc_size) {
  auto add_reloc = [this](uint64_t &insn, uint64_t ws_size) {
    auto op = new NDLoadDummy(kFloat32);
    objects_.push_back(op);
    op->addr_.data = insn;
    op->addr_.Update(&insn);
  };
  SlotWsAllocator slot_alloc(ctx_->slot_ws_);
  _SplitKernel::CodeGenR(relocs, reloc_size, &slot_alloc);
  uint64_t ws_size = ctx_->slot_ws_.size();
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    if (auto mm = kernels_[i]->mm_) {
      auto code = reinterpret_cast<vCubeOp *>(kernels_[i]->code_.data_ + Code::HeadSize());
      if (mm->atomic_add_) {
        add_reloc(code->gm_a, ws_size);
        add_reloc(code->gm_b, ws_size);
        if (mm->bias_) {
          add_reloc(code->gm_bias, ws_size);
        }
        add_reloc(code->gm_c, ws_size);
        if (mm->obj_id_ == ObjectType::kGmmOp) {
          add_reloc(code->gm_group_list, ws_size);
        }
      } else {
        static_cast<NDAccess *>(mm->lhs_)->addr_.Update(&code->gm_a);
        static_cast<NDAccess *>(mm->rhs_)->addr_.Update(&code->gm_b);
        if (mm->bias_) {
          static_cast<NDAccess *>(mm->bias_)->addr_.Update(&code->gm_bias);
        }
        static_cast<NDAccess *>(mm->output_)->addr_.Update(&code->gm_c);
        if (mm->obj_id_ == ObjectType::kGmmOp) {
          auto gmm = static_cast<GmmOp *>(mm);
          static_cast<NDAccess *>(gmm->group_list_)->addr_.Update(&code->gm_group_list);
        }
      }
      if (code->flags & V_CUBE_FLAG_GROUP_SET) {
        add_reloc(code->gm_pos, ws_size);
      }
    }
  }
  auto &ws = ctx_->slot_ws_;
  if (!ws.empty()) {
    for (auto op : objects_) {
      if (op->IsSimd()) continue;
      auto &addr = static_cast<NDAccess *>(op)->addr_;
      if (addr.data > 0 && addr.data <= ws_size) {
        auto &slot = ws[addr.data - 1];
        if (slot.second == nullptr) {
          slot.second = &addr;
        } else {
          code_.BindOpFast(addr, *slot.second);
        }
      }
    }
    ASSERT(std::find_if(ws.begin(), ws.end(), [](std::pair<size_t, RelocAddr *> &slot)
                        { return slot.second == nullptr; }) == ws.end());
  }
}

void _SplitKernel::RelocBinds() {
  code_.RelocBinds(0);
  code_.bind_ops_ = nullptr;
  code_.bind_wss_ = nullptr;
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    auto &code = kernels_[i]->code_;
    code.RelocBinds(0);
    code.bind_ops_ = nullptr;
    code.bind_wss_ = nullptr;
  }
}

VKernelE::VKernelE() : _SplitKernel(KernelType::kEager, 0) {
  static SplitContext ctx;
  ctx_ = &ctx;
  objects_.reserve(128);
  areas_.reserve(16);
  kernels_.reserve(8);
}

VKernelE::~VKernelE() { Clear(); }

void VKernelE::Normalize() {}

_SplitGraph::_SplitGraph(uint32_t flags) : _SplitKernel(KernelType::kSplit, flags) {
  static SplitContext ctx;
  ctx_ = &ctx;
}

_SplitGraph:: ~_SplitGraph() {
  auto &del_ops = objects_.empty() ? build_ops_ : objects_;
  for (auto op : del_ops) {
    delete op;
  }
}

void _SplitGraph::Append(NDObject *obj) {
  if (obj->IsStore() && obj->lhs_ != build_ops_.back()) {
    build_ops_.push_back(build_ops_.back());
    for (size_t i = build_ops_.size() - 2; i > 0; --i) {
      if (build_ops_[i - 1] == obj->lhs_) {
        build_ops_[i] = obj;
        break;
      }
      build_ops_[i] = build_ops_[i - 1];
    }
  } else {
    build_ops_.push_back(obj);
  }
}

void _SplitGraph::Normalize() {
  for (auto op : build_ops_) {
    if (op->IsCube()) {
      auto mm = static_cast<CubeOp *>(op);
      // TODO: cube support reuse
      mm->atomic_add_ = false;
      mm->batch_fold_ = false;
      mm->set_real_ = false;
      auto real_out = AppendCube(mm);
      mm->block_dim_ = reinterpret_cast<uint64_t>(real_out);
    } else {
      op->ForInput([op](NDObject *&input) {
        if (input->IsCube()) {
          auto mm = static_cast<CubeOp *>(input);
          auto real_out = reinterpret_cast<NDObject *>(mm->block_dim_);
          if (real_out != mm) {
            input = real_out;
          } else if (op->IsStore()) {
            mm->output_ = static_cast<NDAccess *>(op);
          }
        }
      });
      _SplitKernel::Append(op);
    }
  }
}

void _SplitGraph::Dump(std::ostringstream &oss, const std::string &indent) {
  if (!objects_.empty()) {
    _SplitKernel::Dump(oss, indent);
    return;
  }
  DumpRefHelper helper(oss);
  oss << indent << "rgraph.split() {" << std::endl;
  std::string body_indent = indent + "  ";
  for (auto op : build_ops_) {
    oss << body_indent;
    helper.Dump(op);
    oss << std::endl;
  }
  oss << indent << "}";
}

void SplitGraphD::Append(NDObject *op) {
  if (op->lhs_) {
    tracker_.Record(&op->lhs_);
    if (op->rhs_) {
      tracker_.Record(&op->rhs_);
      if (op->flags_ & OBJ_FLAG_XHS) {
        tracker_.Record(&static_cast<FlexOp *>(op)->xhs_);
      } else if (op->IsCube()) {
        tracker_.Record(&static_cast<CubeOp *>(op)->bias_);
      }
    }
  }
  _SplitGraph::Append(op);
}

void SplitGraphD::Normalize() {
  Reset();
  tracker_.Recover();
  for (auto op : objects_) {
    if (!(op->flags_ & OBJ_FLAG_EAGER)) {
      op->~NDObject();
      NDObject::mem_pool_.Put(op);
    }
  }
  _SplitGraph::Normalize();
}

void SplitGraphD::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  _SplitGraph::CodeGenR(relocs, reloc_size, ws_alloc);
  RelocBinds();
}

void SplitGraphD::Clone(VKernel *base, CloneHelper &helper) {
  auto k = static_cast<SplitGraphD *>(base);
  k->tracker_.Recover();
  for (auto op : k->build_ops_) {
    Append(op->CloneUpdate(helper));
  }
}

void SplitGraphS::Normalize() {
  if (!objects_.empty()) {
    return;
  }
  std::unordered_set<void *> ios;
  for (auto op : build_ops_) {
    if (!op->IsSimd()) {
      static_cast<NDAccess *>(op)->addr_.gm = op;
      ios.insert(op);
    }
  }
  _SplitGraph::Normalize();
  SlotCodeGen(nullptr, 0);
  slot_ws_ = std::move(ctx_->slot_ws_);
  size_t slot_size = slot_ws_.size();
  for (auto op : objects_) {
    if (!(op->IsSimd() || (op->flags_ & OBJ_FLAG_EAGER))) {
      auto &addr = static_cast<NDAccess *>(op)->addr_;
      if (addr.data > slot_size && ios.count(addr.gm)) {
        auto bind_op = reinterpret_cast<NDAccess *>(addr.gm);
        code_.BindOpFast(addr, bind_op->addr_);
      }
    }
  }
}

static void CombineAlloc(SplitContext::SlotWorkspace &ws, WsAllocator *alloc) {
  uint64_t total_size = 0;
  for (auto &slot : ws) {
    slot.first = RoundUp(slot.first, 512ul);
    total_size += slot.first;
  }
  uint8_t *ws_mem = static_cast<uint8_t *>(alloc->Alloc(total_size));
  for (auto &slot : ws) {
    slot.second->Reloc(ws_mem);
    ws_mem += slot.first;
  }
}

void SplitGraphS::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  for (size_t i = 0; i < reloc_size; ++i, ++relocs) {
    static_cast<NDAccess *>(relocs->io)->addr_.Reloc(relocs->addr);
  }
  if (single_ws_) {
    CombineAlloc(slot_ws_, ws_alloc);
  } else {
    for (auto &slot : slot_ws_) {
      slot.second->Reloc(ws_alloc->Alloc(slot.first));
    }
  }
  code_.RelocBinds(0);
  for (int i = kernel_begin_; i < kernel_used_; ++i) {
    kernels_[i]->code_.RelocBinds(0);
  }
}

void SplitEagerW::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  SlotCodeGen(relocs, reloc_size);
  CombineAlloc(ctx_->slot_ws_, ws_alloc);
  ctx_->slot_ws_.clear();
  RelocBinds();
}

void SplitGraphDW::CodeGenR(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc) {
  SlotCodeGen(relocs, reloc_size);
  CombineAlloc(ctx_->slot_ws_, ws_alloc);
  ctx_->slot_ws_.clear();
  RelocBinds();
}
}  // namespace dvm
