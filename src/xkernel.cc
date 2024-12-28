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

#include <cstdlib>
#include <cstring>
#include "xkernel.h"
#include "comm.h"
#include "tuning.h"

namespace dvm {
constexpr int64_t MAX_K = 1 << 15;

MixKernel::~MixKernel() {
  if (cube_op_) {
    // TODO: remove check
    if (auto lhs = cube_op_->lhs_; lhs->IsLoad()) {
      delete lhs;
    }
    if (auto rhs = cube_op_->rhs_; rhs->IsLoad()) {
      delete rhs;
    }
    if (auto out = cube_op_->output_; out->IsStore()) {
      delete out;
    }
    if (auto bias = cube_op_->bias_; bias != nullptr) {
      delete bias;
    }
    delete cube_op_;
  }
  if (post_fusion_) delete post_fusion_;
  if (stage_kernel_) delete stage_kernel_;
}

void MixKernel::Append(NDObject *obj) {
  const uint32_t LOAD_PENDING = 1;
  if (obj->IsLoad()) {
    obj->xbuf_ = LOAD_PENDING;
  } else if (obj->obj_id_ == kCubeOp) {
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    std::vector<NDObject *> empty_run_ops;
    obj->lhs_->Normalize(empty_run_ops);
    obj->rhs_->Normalize(empty_run_ops);
    cube_op_ = static_cast<CubeOp *>(obj);
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
          sload_ = new NDSLoad(nullptr, cube_op_->shape_ref_, cube_op_->type_id_);
          post_fusion_->build_ops_.emplace_back(sload_);
        }
        op = sload_;
      } else if (op->IsLoad() && op->xbuf_ == LOAD_PENDING) {
        op->xbuf_ = 0;
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
  auto InplaceOp = [replaced_node, replacing_node](NDObject *&op) {
    if (op && op == replaced_node) {
      op = replacing_node;
    }
  };
  for (auto &op : post_fusion_->build_ops_) {
    if (op != replaced_node) {
      InplaceOp(op->lhs_);
      InplaceOp(op->rhs_);
      if (op->flags_ & OBJ_FLAG_XHS) {
        InplaceOp(static_cast<FlexOp *>(op)->xhs_);
      }
      stage_kernel_->GetImpl()->Append(op);
    }
  }
  post_fusion_->build_ops_.clear();
  if (replaced_node) {
    post_fusion_->Append(replaced_node);
  }
}

uint64_t MixKernel::UnAlignCodeGen() {
  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  int64_t pad_size[2] = {cube_op_->pad_a_, cube_op_->pad_b_};
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
  stage_kernel_->StageSwitch(dvm::KernelType::kStaticMix);
  for (size_t i = 0; i < 2; i++) {
    if (pad_size[i]) {
      inputs[i] = stage_kernel_->StageLoad(inputs[i]);
    } else {
      inputs[i] = stage_kernel_->Load(nullptr, src_inputs[i]->shape_ref_, src_inputs[i]->type_id_);
      pad_inputs[i] = inputs[i];
    }
  }
  auto matmul_op = stage_kernel_->MatMul(inputs[0], inputs[1], cube_op_->trans_a_, cube_op_->trans_b_, cube_op_->bias_);
  cube_op_->bias_ = nullptr;
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
  src_lhs->reloc_addr_ = static_cast<NDAccess *>(pad_inputs[0])->reloc_addr_;
  src_rhs->reloc_addr_ = static_cast<NDAccess *>(pad_inputs[1])->reloc_addr_;
  if (real_out) {
    cube_op_->output_->reloc_addr_ = real_out->reloc_addr_;
  }
  return workspace_size;
}

uint64_t MixKernel::SplitKCodeGen() {
  size_t k_stride = MAX_K >> 1;
  auto split_num = CeilDiv(static_cast<size_t>(cube_op_->k_real_), k_stride);
  size_t k_tail = cube_op_->k_real_ % k_stride ? cube_op_->k_real_ % k_stride : k_stride;

  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  std::vector<NDAccess *> split_lhs;
  std::vector<NDAccess *> split_rhs;
  std::vector<NDAccess *> split_out;
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
    auto output =
      stage_kernel_->MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, i == 0 ? cube_op_->bias_ : nullptr);
    static_cast<CubeOp *>(output)->SetRealShape(cube_op_->m_real_, cube_op_->n_real_,
                                                i + 1 == split_num ? k_tail : k_stride, offset_a, offset_b);
    static_cast<CubeOp *>(output)->SetOutFp32(i != 0);
    (void)split_out.emplace_back(static_cast<NDAccess *>(stage_kernel_->Store(nullptr, output)));
  }
  cube_op_->bias_ = nullptr;
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
  src_lhs->reloc_addr_ = split_lhs[0]->reloc_addr_;
  src_rhs->reloc_addr_ = split_rhs[0]->reloc_addr_;
  code_.BindWorkspace(split_out[0], stage_workspace_size);
  for (size_t i = 1; i < split_num; i++) {
    code_.BindOpFast(split_lhs[i], src_lhs);
    code_.BindOpFast(split_rhs[i], src_rhs);
    code_.BindWorkspace(split_out[i], stage_workspace_size);
  }
  if (real_out) {
    cube_op_->output_->reloc_addr_ = real_out->reloc_addr_;
  }
  return workspace_size;
}

uint64_t MixKernel::BiasBF16CodeGen() {
  stage_kernel_ = new Kernel();
  stage_kernel_->Reset(KernelType::kStaticStages);
  stage_kernel_->StageSwitch(dvm::KernelType::kStaticShape);
  auto bias_bf16 = stage_kernel_->Load(nullptr, cube_op_->bias_->shape_ref_, cube_op_->bias_->type_id_);
  auto bias_fp32 = stage_kernel_->StageStore(stage_kernel_->Cast(bias_bf16, kFloat32));

  stage_kernel_->StageSwitch(dvm::KernelType::kStaticMix);
  auto x = stage_kernel_->Load(nullptr, cube_op_->lhs_->shape_ref_, cube_op_->lhs_->type_id_);
  auto y = stage_kernel_->Load(nullptr, cube_op_->rhs_->shape_ref_, cube_op_->rhs_->type_id_);
  auto matmul_op =
    stage_kernel_->MatMul(x, y, cube_op_->trans_a_, cube_op_->trans_b_, stage_kernel_->StageLoad(bias_fp32));
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
  src_lhs->reloc_addr_ = static_cast<NDAccess *>(x)->reloc_addr_;
  src_rhs->reloc_addr_ = static_cast<NDAccess *>(y)->reloc_addr_;
  static_cast<NDAccess *>(cube_op_->bias_)->reloc_addr_ = static_cast<NDAccess *>(bias_bf16)->reloc_addr_;
  if (real_out) {
    cube_op_->output_->reloc_addr_ = real_out->reloc_addr_;
  }
  return workspace_size;
}

uint64_t MixKernel::AlignCodeGen() {
  size_t size = code_.HeadSize() + sizeof(vCubeOp);
  vCubeOp cube_code;
  cube_op_->CodeGen(&cube_code, tuner_);
  code_.block_dim_ = cube_op_->block_dim_;
  cube_code.subtilenum = 0;
  uint64_t head_flags = V_ENTRY_FLAG_MIX;
  uint64_t head_simd = 0;
  NDAccess *inplace_store = nullptr;
  bool peer_store = false;
  if (post_fusion_) {
    post_fusion_->Normalize();
    if (cube_op_->batch_fold_) {  // Todo: Support BatchMatMul Broadcast
      for (auto op : post_fusion_->objects_) {
        ASSERT(op->nd_.prod() == sload_->nd_.prod());
        op->nd_[1] = cube_op_->m_real_;
        op->nd_.resize(2);
      }
    }
    post_fusion_->Optimize();
    post_fusion_->BuildDomain(post_fusion_->objects_);
    post_fusion_->NormalizeDomain();
    if (auto comm = post_fusion_->comm_op_; comm != nullptr && comm->lhs_ == sload_) {
      cube_op_->pingpong_store_ = true;
      static_cast<NDSLoad *>(sload_)->pingpong_load_ = true;
      cube_code.rank_size = comm->comm_->GetRankSize();
      cube_code.flags |= V_CUBE_FLAG_PEER_STORE;
      cube_code.flags |= V_CUBE_FLAG_PINGPONG_STORE;
      peer_store = true;
    } else if (cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
      inplace_store = post_fusion_->FindInplaceStore(sload_, nullptr);
      if (inplace_store == nullptr) {
        cube_op_->pingpong_store_ = true;
        static_cast<NDSLoad *>(sload_)->pingpong_load_ = true;
        cube_code.flags |= V_CUBE_FLAG_PINGPONG_STORE;
      }
    }
    for (auto op : post_fusion_->objects_) {
      if (op->IsLoad()) {
        static_cast<NDSLoad *>(op)->SetCubeOp(cube_op_);
      } else if (op->IsStore()) {
        static_cast<NDSStore *>(op)->SetCubeOp(cube_op_);
      } else if (op->IsComm()) {
        auto comm = static_cast<CommOp *>(op);
        comm->mix_ = true;
        if (peer_store) {
          static_cast<CommOp *>(op)->SetCubeOp(cube_op_);
        }
      }
    }
    auto m = sload_->nd_[1];
    auto n = sload_->nd_[0];
    post_fusion_->root_dom_.GroupTile(1, m, cube_code.m0);
    post_fusion_->root_dom_.GroupTile(0, n, cube_code.n0);
    for (size_t i = 2; i < sload_->nd_.size(); i++) {
      post_fusion_->root_dom_.GroupTile(i, sload_->nd_[i], 1);
    }
    post_fusion_->NormalizeDomain();
    post_fusion_->DoCodeGen(2);
    size += post_fusion_->code_.data_size_;
    uint64_t subtile_0 = (post_fusion_->tile_num_ + 1) / 2;
    uint64_t subtile_1 = post_fusion_->tile_num_ - subtile_0;
    cube_code.subtilenum = subtile_1 << 32 | subtile_0;
    cube_code.flags |= V_CUBE_FLAG_GROUP_SET;
    head_flags |= V_ENTRY_FLAG_PRE_WAIT;
    head_simd = post_fusion_->simd_width_;
  }
  code_.target_ = post_fusion_ ? Code::kTargetMix : Code::kTargetCube;
  code_.data_size_ = size;
  code_.Alloc(size);
  code_.UpdateHead(cube_op_->core_loop_, head_simd, head_flags);
  std::memcpy(code_.data_ + code_.HeadSize(), &cube_code, sizeof(vCubeOp));
  // only support post fusion
  vCubeOp *link_cube = reinterpret_cast<vCubeOp *>(code_.data_ + code_.HeadSize());
  static_cast<NDAccess *>(cube_op_->lhs_)->reloc_addr_ = &link_cube->gm_a;
  static_cast<NDAccess *>(cube_op_->rhs_)->reloc_addr_ = &link_cube->gm_b;
  if (cube_op_->bias_) {
    static_cast<NDAccess *>(cube_op_->bias_)->reloc_addr_ = &link_cube->gm_bias;
  }
  cube_op_->output_->reloc_addr_ = &link_cube->gm_c;
  if (post_fusion_) {
    std::vector<NDAccess *> ios;
    for (auto op : post_fusion_->objects_) {
      if (!op->IsSimd()) ios.push_back(static_cast<NDAccess *>(op));
    }
    code_.LinkBody(code_.HeadSize() + sizeof(vCubeOp), post_fusion_->code_, ios, 0);
    if (inplace_store) {
      code_.BindOpFast(cube_op_->output_, inplace_store);
      code_.BindOpFast(sload_, inplace_store);
    } else if (!cube_op_->output_->CheckFlag(OBJ_FLAG_STAGE_IO)) {
      code_.BindOpFast(sload_, cube_op_->output_);
    } else if (peer_store) {
      // record vCubeOp's unique_id address
      auto distance = reinterpret_cast<uint8_t *>(&cube_code.unique_id) - reinterpret_cast<uint8_t *>(&cube_code);
      code_.unique_ids_.push_back(reinterpret_cast<uint32_t *>(code_.data_ + code_.HeadSize() + distance));
      const auto comm = post_fusion_->comm_op_->comm_;
      link_cube->gm_c = reinterpret_cast<uint64_t>(comm->GetPeerMemPtr(comm->GetRankId()));
      code_.BindOpFast(sload_, cube_op_->output_);
    } else {
      code_.BindWorkspace(cube_op_->output_, 0);
      code_.BindWorkspace(sload_, 0);
      return cube_op_->PostFusionWorkSpace();
    }
  }
  return 0;
}

uint64_t MixKernel::CodeGen() {
  if (cube_op_->output_ == nullptr) {
    auto output = new NDStore(nullptr, cube_op_);
    output->SetFlag(OBJ_FLAG_STAGE_IO);
    cube_op_->output_ = output;
  }
  cube_op_->InitPadShape();
  if (cube_op_->bias_ && cube_op_->bias_->type_id_ == kBFloat16) {
    return BiasBF16CodeGen();
  }
  if (cube_op_->pad_a_ || cube_op_->pad_b_) {
    return UnAlignCodeGen();
  }
  if (cube_op_->k_real_ > MAX_K) {
    return SplitKCodeGen();
  }
  return AlignCodeGen();
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
    code_.CombineBinds(code, s->ws_offset);
    for (auto op : s->ios) {
      if (!op->CheckFlag(OBJ_FLAG_STAGE_IO)) continue;
      if (op->IsStore()) {
        if (op->xbuf_ == STAGE_FLAG_REUSE) {
          code_.BindOp(op, GetOutputReuse(op));
        } else {
          code_.BindWorkspace(op, GetWorkspace(op));
        }
      } else {
        code_.BindOp(op, GetStageStore(op));
      }
    }
    if (!code.sub_codes_.empty()) {
      for (auto a : code.sub_codes_) {
        code_.sub_codes_.push_back(a);
      }
      code.sub_codes_.clear();
    }
    if (s != stages_.back()) {
      code_.sub_codes_.push_back(&code);
    } else {
      code_.MoveCode(code);
    }
  }
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
    return up >= 0 ? up : down;
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
            SetOutputReuse(store, inplace_out->CheckFlag(OBJ_FLAG_STAGE_IO) ? GetOutputReuse(inplace_out) : inplace_out);
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
  CubeOptimizer(CubeOp *dom) : dom_(dom) { dom_->InitPadShape(); }

  bool AlignA(std::vector<NDObject *> &ops) {
    if (dom_->pad_a_ == 0) {
      return false;
    }
    AlignInput(dom_->lhs_, dom_->pad_a_, ops);
    return true;
  }

  bool AlignB(std::vector<NDObject *> &ops) {
    if (dom_->pad_b_ == 0) {
      return false;
    }
    AlignInput(dom_->rhs_, dom_->pad_b_, ops);
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
    if (dom_->k_real_ <= MAX_K) {
      return false;
    }
    auto load = new NDLoad(nullptr, dom_->shape_ref_, kFloat32);
    auto cast = new CastOp(load, dom_->type_id_);
    size_t k_stride = MAX_K >> 1;
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
    if (obj->obj_id_ == kCast || obj->IsLoad()) {
      int type = obj->type_id_;
      if (type > max_type_) {
        max_type_ = type;
      } else if (type < min_type_) {
        min_type_ = type;
      }
    }
    if (obj->IsLoad()) {
      static_ops_.push_back(obj);
    } else if (obj->IsStore()) {
      static_ops_.push_back(obj->lhs_);
    }
  }

  uint64_t CodeGen() override {
    root_dom_.SetHead(next_);
    NormalizeDomain();
    DoCodeGen(System::Instance().CoreNum());
    return 0;
  }

  void Reset(NDObject *dom) {
    max_type_ = dom->type_id_;
    min_type_ = dom->type_id_;
    next_ = nullptr;
    objects_.clear();
    code_.ResetEager(Code::kTargetVec);
    static_ops_.clear();
  }

  void CodeGenMix(CubeOp *mm) {
    objects_.clear();
    static_ops_.clear();
    code_.ResetEager(Code::kTargetCube);
    size_t size = code_.HeadSize() + sizeof(vCubeOp);
    code_.Alloc(size);
    vCubeOp *body = reinterpret_cast<vCubeOp *>(code_.data_ + code_.HeadSize());
    mm->CodeGen(body, System::Instance().lazy_tuner_);
    body->subtilenum = 0;
    code_.block_dim_ = mm->block_dim_;
    code_.data_size_ = size;
    code_.UpdateHead(mm->core_loop_, 0, V_ENTRY_FLAG_MIX);
    next_ = mm;
  }

  void Dump(std::ostringstream &oss, const std::string &indent) {
    if (next_ && next_->obj_id_ == ObjectType::kCubeOp) {
      auto mm = static_cast<CubeOp *>(next_);
      oss << indent << "vgraph() {" << std::endl;
      auto body_indent = indent + "  ";
      oss << body_indent << "%0" << mm->nd_ << " = ";
      mm->Dump(true, oss);
      oss << "(%1" << mm->lhs_->nd_;
      oss << ", %2" << mm->rhs_->nd_;
      if (mm->bias_) {
        oss << ", %3" << mm->bias_->nd_;
      }
      oss << ")" << std::endl << indent << "}";
      return;
    }
    return VectorKernel::Dump(oss, indent);
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

  void Reset(NDObject *dom, int area_id) {
    dom_ = dom->obj_id_ != kReduce ? dom : dom->lhs_;
    area_id_ = area_id;
    state_ = kPending;
    objects_.clear();
    fused_.clear();
  }

  void ResetMix(NDObject *dom, int area_id) {
    dom_ = dom;
    area_id_ = area_id;
    state_ = kSubmitted;
  }

  bool TryFuse(EagerArea *a, std::vector<std::pair<EagerArea *, EagerArea *>> &areas) {
    ASSERT(dom_ != nullptr);
    const auto &dom_nd = dom_->nd_;
    const auto &op_nd = a->dom_->nd_;
    if (dom_nd.size() != op_nd.size()) {
      return false;
    }
    for (size_t i = 0; i < op_nd.size(); ++i) {
      if (dom_nd[i] != op_nd[i]) return false;
    }
    fused_.push_back(a);
    areas[a->area_id_].second = this;
    if (!a->fused_.empty()) {
      for (auto x : a->fused_) {
        fused_.push_back(x);
        areas[x->area_id_].second = this;
      }
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
    return area;
  }

  int state_;
  int area_id_;
  NDObject *dom_;
  std::vector<NDObject *> objects_;
  std::vector<EagerArea *> fused_;
};

VKernelE::VKernelE(WsAllocFunc func, void *user_data)
    : VKernel(KernelType::kEager), ws_alloc_(func), user_data_(user_data) {
  objects_.reserve(128);
  temp_ops_.reserve(64);
  areas_.reserve(16);
  kernels_.reserve(8);
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
  NDAccess *load;
  if (input->IsLoad()) {
    auto ac = static_cast<NDAccess *>(input);
    load = new NDLoad(ac->addr_.gm, ac->shape_ref_, ac->type_id_);
    SetStore(load, input);
    load->Normalize(area->objects_);
  } else {
    NDAccess *store = GetStore(input);
    if (store == nullptr) {
      store = new NDStore(nullptr, input);
      store->Normalize(temp_ops_);
      InitStoreInfo(store, GetArea(input));
      SetStore(input, store);
      objects_.push_back(store);
    }
    load = new NDLoad(nullptr, store->shape_ref_, store->type_id_);
    SetStore(load, store);
    load->Normalize(area->objects_);
  }
  objects_.push_back(load);
  return load;
}

void VKernelE::Split(NDObject *root) {
  int kidx = area_used_++;
  EagerArea *area = EagerArea::Assign(this, kidx);
  area->Reset(root, kidx);
  auto push_input = [area, this](NDObject *input, NDObject *&update) {
    if (auto idx = GetArea(input); idx >= 0) {
      if (auto a = areas_[idx].second; a != area) {
        if (a->state_ == EagerArea::kPending) {
          if (area->TryFuse(a, areas_)) {
            kernel_used_--;
            return;
          }
          if (!input->IsLoad()) {
            a->state_ = EagerArea::kSubmitted;
          }
        }
        update = input = Exchange(a, input);
      }
    } else if (input->IsLoad()) {
      if (auto store = GetStore(input); store != nullptr && store->IsStore()) {
        areas_[GetArea(store)].second->state_ = EagerArea::kSubmitted;
      }
    }
    temp_ops_.push_back(input);
  };
  temp_ops_.push_back(root);
  while (!temp_ops_.empty()) {
    auto op = temp_ops_.back();
    temp_ops_.pop_back();
    SetArea(op, kidx);
    area->objects_.push_back(op);
    if (auto lhs = op->lhs_) {
      push_input(lhs, op->lhs_);
      if (auto rhs = op->rhs_) {
        push_input(rhs, op->rhs_);
      }
    }
  }
}

void VKernelE::Append(NDObject *obj) {
  obj->flags_ |= OBJ_FLAG_EAGER;  // TODO: add eager param for Normalize
  if (obj->IsStore()) {
    obj->Normalize(objects_);
    InitStoreInfo(obj);
    auto src = obj->lhs_;
    SetStore(src, obj);
    if (GetArea(src) == -1) {
      Split(src);
    }
  } else {
    auto fuse_break = [](NDObject *op) -> bool { return op->obj_id_ == ObjectType::kReduce; };
    if (auto lhs = obj->lhs_) {
      if (fuse_break(lhs) && GetArea(lhs) == -1) {
        Split(lhs);
        obj->lhs_ = Exchange(areas_[GetArea(lhs)].second, lhs);
        SetArea(obj->lhs_, -1);
      }
      if (auto rhs = obj->rhs_; rhs != nullptr && fuse_break(rhs) && GetArea(rhs) == -1) {
        Split(rhs);
        obj->rhs_ = Exchange(areas_[GetArea(rhs)].second, rhs);
        SetArea(obj->rhs_, -1);
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
  auto prepare_input = [this](bool is_stuff, NDObject *&input) {
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
      areas_[GetArea(store)].second->state_ = EagerArea::kSubmitted;
    } else if (input->IsSimd()) {
      auto aid = GetArea(input);
      if (aid == -1) {
        Split(input);
        aid = GetArea(input);
      }
      auto area = areas_[aid].second;
      input = Exchange(area, input);
      area->state_ = EagerArea::kSubmitted;
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
        area->ResetMix(op, aid);
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
  area->ResetMix(mm, aid);
  SetArea(mm, aid);
  SetStore(mm, output);
  InitStoreInfo(output, aid);
  objects_.push_back(mm);
  return ret;
}

void VKernelE::CodeGenMix(EagerArea *area, EagerVector *kernel) {
  auto mm = static_cast<CubeOp *>(area->dom_);
  auto alloc_input = [this](NDAccess *input) {
    auto store = GetStore(input);
    if (store->addr_.gm == nullptr) {  // TODO: abstract common func
      auto size = GetStoreSize(store);
      if (auto it = wss_.upper_bound(size - 1); it != wss_.end()) {
        store->addr_.gm = it->second;
        wss_.erase(it);
        SetStoreSize(store, it->first);
      } else {
        store->addr_.gm = ws_alloc_(size, user_data_);
      }
    }
    input->addr_.gm = store->addr_.gm;
  };
  if (auto io = static_cast<NDAccess *>(mm->lhs_); io->addr_.gm == nullptr) {
    alloc_input(io);
  }
  if (auto io = static_cast<NDAccess *>(mm->rhs_); io->addr_.gm == nullptr) {
    alloc_input(io);
  }
  if (auto io = static_cast<NDAccess *>(mm->bias_); io && io->addr_.gm == nullptr) {
    alloc_input(io);
  }
  if (!mm->atomic_add_) {
    wss_.insert({GetStoreSize(mm->output_), mm->output_->addr_.gm});
  }
  kernel->CodeGenMix(mm);
}

uint64_t VKernelE::CodeGen() {
  auto append_ops = [this](EagerVector *kernel, const std::vector<NDObject *> &objects) {
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
      auto op = *it;
      if (GetArea(op) == -1) continue;
      SetArea(op, -1);
      kernel->EagerVector::Append(op);
      if (op->IsLoad()) {
        if (!(op->flags_ & OBJ_FLAG_EAGER)) {
          objects_.push_back(op);
        }
      } else if (auto store = GetStore(op)) {
        kernel->EagerVector::Append(store);
        temp_ops_.push_back(store);
      }
    }
  };
  size_t obj_size = objects_.size();
  int kidx = kernel_used_;
  if (static_cast<size_t>(kidx) > kernels_.size()) {
    for (int i = kernels_.size(); i < kidx; ++i) {
      kernels_.push_back(new EagerVector());
    }
  }
  uint64_t extern_code_size = 0;
  while (area_used_ > 0 && kidx > 0) {
    auto area = areas_[--area_used_].second;
    if (area->state_ == EagerArea::kFree) continue;
    area->state_ = EagerArea::kFree;
    auto kernel = kernels_[--kidx];
    if (area->dom_->obj_id_ == kCubeOp) {
      CodeGenMix(area, kernel);
      continue;
    }
    kernel->Reset(area->dom_);
    if (!area->fused_.empty()) {
      for (auto it = area->fused_.rbegin(); it != area->fused_.rend(); ++it) {
        append_ops(kernel, (*it)->objects_);
      }
    }
    append_ops(kernel, area->objects_);
    for (size_t i = obj_size; i < objects_.size(); ++i) {
      NDAccess *io = static_cast<NDAccess *>(objects_[i]);
      auto store = GetStore(io);
      if (store->addr_.gm == nullptr) {
        if (auto is =
              kernel->FindInplaceStore(io, [](NDAccess *op) -> bool { return VKernelE::GetStoreInplace(op) == 0; })) {
          store->addr_.gm = is->addr_.gm;
          SetStoreInplace(is, 1);
        } else {
          auto size = GetStoreSize(store);
          if (auto it = wss_.upper_bound(size - 1); it != wss_.end()) {
            store->addr_.gm = it->second;
            wss_.erase(it);
            SetStoreSize(store, it->first);
          } else {
            store->addr_.gm = ws_alloc_(size, user_data_);
          }
        }
      }
      io->addr_.gm = store->addr_.gm;
    }
    if (kidx > 1) {  // kidx 1 is last wss user
      for (auto op : temp_ops_) {
        if (auto store = static_cast<NDAccess *>(op); !GetStoreInplace(store)) {
          wss_.insert({GetStoreSize(store), store->addr_.gm});
        }
      }
    }
    temp_ops_.clear();
    objects_.resize(obj_size);
    (void)kernel->EagerVector::CodeGen();
    if (uint64_t code_size = kernel->code_.ReserveWorkspace(0); code_size > extern_code_size) {
      extern_code_size = code_size;
    }
  }
  wss_.clear();
  area_used_ = 0;
  if (extern_code_size) {
    extern_code_ = ws_alloc_(extern_code_size, user_data_);
  }
  return 0;
}

class EagerDumpRef : public DumpRefHelper {
 public:
  EagerDumpRef(std::ostringstream &oss) : DumpRefHelper(oss) {}
  virtual NDObject *GetInput(NDObject *input) {
    while (input && idx_map_.count(input) == 0) {
      if (input->IsLoad()) {
        auto acc = VKernelE::GetStore(input);
        if (acc) {
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
  if (area_used_ == 0) {
    oss << "vgraph.eager() {" << std::endl;
    std::string body_indent = indent + "  ";
    for (int i = 0; i < kernel_used_; ++i) {
      oss << body_indent << "// eager " << i << std::endl;
      kernels_[i]->Dump(oss, body_indent);
      oss << std::endl;
    }
    oss << "}";
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
    oss << "}";
  }
}

std::string &VKernelE::DisAssemble() {
  std::ostringstream oss;
  for (int i = 0; i < kernel_used_; ++i) {
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
