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

#include <vector>
#include "tile_builder.h"
#include "ops.h"  // DimArray

namespace dvm {
namespace {
struct InsnIdTable {
  const char *name;
  vSimdInsnID ids[DataType::kDataTypeEnd];
};

const InsnIdTable unary_id_list[kUnaryTypeEnd] = {
  {"Sqrt", {V_NONE, V_SQRT_FP16, V_NONE, V_SQRT, V_NONE, V_NONE}},
  {"Abs", {V_NONE, V_ABS_FP16, V_NONE, V_ABS, V_ABS_INT32, V_NONE}},
  {"Log", {V_NONE, V_LOG_FP16, V_NONE, V_LOG, V_NONE, V_NONE}},
  {"Exp", {V_NONE, V_EXP_FP16, V_NONE, V_EXP, V_NONE, V_NONE}},
  {"Reciprocal", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"IsFinite", {V_NONE, V_ISFINITE_FP16, V_ISFINITE_BF16, V_ISFINITE, V_NONE, V_NONE}},
  {"LogicalNot", {V_LOGICAL_NOT_BOOL, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},
  {"Round", {V_NONE, V_NONE, V_NONE, V_ROUND, V_NONE, V_NONE}},
  {"Floor", {V_NONE, V_NONE, V_NONE, V_FLOOR, V_NONE, V_NONE}},
  {"Ceil", {V_NONE, V_NONE, V_NONE, V_CEIL, V_NONE, V_NONE}},
  {"Trunc", {V_NONE, V_NONE, V_NONE, V_TRUNC, V_NONE, V_NONE}}};

static const InsnIdTable binary_id_list[] = {
  // must keep consistent order with BinaryType
  {"Equal", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"NotEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Greater", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"GreaterEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Less", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"LessEqual", {V_CMP_BOOL, V_CMP_FP16, V_CMP_BF16, V_CMP, V_CMP_INT32, V_CMP_INT64}},
  {"Add", {V_NONE, V_ADD_FP16, V_ADD_BF16, V_ADD, V_ADD_INT32, V_ADD_INT64}},
  {"Sub", {V_NONE, V_SUB_FP16, V_SUB_BF16, V_SUB, V_SUB_INT32, V_SUB_INT64}},
  {"Mul", {V_NONE, V_MUL_FP16, V_MUL_BF16, V_MUL, V_MUL_INT32, V_NONE}},
  {"Div", {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE, V_NONE}},
  {"Pow", {V_NONE, V_NONE, V_NONE, V_NONE, V_NONE, V_NONE}},  // power: individual implement
  {"Maximum", {V_NONE, V_MAX_FP16, V_MAX_BF16, V_MAX, V_MAX_INT32, V_NONE}},
  {"Minimum", {V_NONE, V_MIN_FP16, V_MIN_BF16, V_MIN, V_MIN_INT32, V_NONE}},
  {"LogicalAnd", {V_LOGICAL_AND_BOOL, V_MIN_FP16, V_MAX_BF16, V_MIN, V_MIN_INT32, V_NONE}},
  {"LogicalOr", {V_LOGICAL_OR_BOOL, V_MAX_FP16, V_MIN_BF16, V_MAX, V_MAX_INT32, V_NONE}}};

}  // namespace

enum InsnId {
  kLoadInsn = 0,
  kStoreInsn,
  kUnaryInsn,
  kBinaryInsn,
};

class TObject {
 public:
  explicit TObject(DataType dtype, InsnId id, TObject *lhs, TObject *rhs)
      : dtype_(dtype), id_(id), lhs_(lhs), rhs_(rhs) {}
  virtual ~TObject() {}

  virtual uint64_t Emit(uint64_t *insn) = 0;
  virtual void Dump(bool verbose, std::ostringstream &oss) = 0;

  bool IsLoad() const { return id_ == kLoadInsn; }
  bool IsStore() const { return id_ == kStoreInsn; }
  bool IsSimd() const { return id_ > kStoreInsn; }

  DataType dtype_;
  InsnId id_;
  NDSpace nd_;
  uint64_t xbuf_;
  TObject *lhs_;
  TObject *rhs_;
  uint64_t sync_mask_{0};
  int index_;
};

class TAccess : public TObject {
 public:
  TAccess(DataType dtype, InsnId id, GmRef *gm, TileRef *tile, IntArrayRef *stride, TObject *input = nullptr)
      : TObject(dtype, id, input, nullptr), gm_ref_(gm), tile_(tile), stride_(stride) {}

  GmRef *gm_ref_;
  TileRef *tile_;
  IntArrayRef *stride_;
  RelocAddr addr_;
};

class TLoad : public TAccess {
 public:
  TLoad(DataType type, GmRef *gm, IntArrayRef *shape, TileRef *tile, IntArrayRef *stride)
      : TAccess(type, kLoadInsn, gm, tile, stride), shape_(shape) {
    nd_.data = &ndd_;
  }
  uint64_t Emit(uint64_t *insn) override {
    ndd_.dims.resize(shape_->size);
    for (size_t i = 0; i < shape_->size; ++i) {
      ndd_.dims[i] = shape_->data[shape_->size - i - 1];
    }
    ndd_.UpdateStride(SIMD_BLOCK_SIZE);
    int64_t lead_align = nd_.lead_stride();
    int64_t lead_dim = nd_.lead_dim();
    uint64_t src_tile_stride = ndd_.stride_back() / lead_align * lead_dim;
    uint64_t item_size = ITEM_SIZE[dtype_];
    uint64_t tail_size = tile_->GetTail();
    vLoad op;
    op.from = *gm_ref_;
    op.xn = xbuf_;
    op.tile_stride = src_tile_stride * item_size;
    op.body_iter = ndd_.stride_back() / lead_align;
    op.iter_size = lead_dim * item_size;
    op.pad_size = lead_align * item_size - op.iter_size;
    if (op.body_iter == 1) {
      op.tail_iter = tail_size == 0 ? op.iter_size : tail_size * item_size;
    } else {
      op.tail_iter = tail_size == 0 ? op.body_iter : op.body_iter / ndd_.dims.back() * tail_size;
      if (op.pad_size == 0) {
        op.tail_iter *= op.iter_size;
        op.iter_size *= op.body_iter;
        op.body_iter = 1;
      }
    }
    op.round_rank = 0;
    addr_.Update(insn + vLoad::RELOC_OFFSET);
    return vLoad::Encode(insn, vAccInsnID::V_LOAD, op, nullptr);
  }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "Load"; }
  NDSpaceData ndd_;
  IntArrayRef *shape_;
};

class TStore : public TAccess {
 public:
  TStore(TObject *input, GmRef *gm, TileRef *tile, IntArrayRef *stride)
      : TAccess(input->dtype_, kStoreInsn, gm, tile, stride, input) {
    nd_ = input->nd_;
  }
  uint64_t Emit(uint64_t *insn) override {
    int64_t lead_align = nd_.lead_stride();
    int64_t lead_dim = nd_.lead_dim();
    uint64_t dst_tile_stride = nd_.stride_back() / lead_align * lead_dim;
    uint64_t item_size = ITEM_SIZE[dtype_];
    uint64_t tail_size = tile_->GetTail();
    vStore op;
    uint64_t iter_size = lead_dim * item_size;
    uint64_t pad_size = (lead_align - lead_dim) * item_size;
    uint64_t body_iter = nd_.stride_back() / lead_align;
    uint64_t tail_iter;
    if (body_iter == 1) {
      tail_iter = tail_size == 0 ? iter_size : tail_size * item_size;
    } else {
      tail_iter = tail_size == 0 ? body_iter : body_iter / nd_.back() * tail_size;
      if (pad_size == 0) {
        tail_iter *= iter_size;
        iter_size *= body_iter;
        body_iter = 1;
      }
    }
    op.xn = lhs_->xbuf_;
    op.to = reinterpret_cast<uint64_t>(*gm_ref_);
    op.tile_stride = dst_tile_stride * item_size;
    op.iter_num = body_iter;
    op.iter_tail = tail_iter;
    op.iter_size = iter_size;
    op.pad_size = pad_size;
    op.round_rank = 0;
    addr_.Update(insn + vStore::RELOC_OFFSET);
    return vStore::Encode(insn, vAccInsnID::V_STORE, op, nullptr);
  }
  void Dump(bool verbose, std::ostringstream &oss) override { oss << "Store"; }
};

class TUnary : public TObject {
 public:
  TUnary(int op_type, TObject *input) : TObject(input->dtype_, kUnaryInsn, input, nullptr) {
    ASSERT(size_t(op_type) < sizeof(unary_id_list) / sizeof(InsnIdTable));
    simd_id_ = unary_id_list[op_type].ids[dtype_];
    ASSERT(simd_id_ != V_NONE);
    nd_ = input->nd_;
  }
  uint64_t Emit(uint64_t *insn) override {
    vUnary op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.count = nd_.stride_back();
    return vUnary::Encode(insn, simd_id_, op);
  }
  void Dump(bool verbose, std::ostringstream &oss) override {
    oss << "Unary";
    if (verbose) {
      oss << '<' << simd_id_ << '<';
    }
  }
  vSimdInsnID simd_id_;
};

class TBinary : public TObject {
 public:
  TBinary(int op_type, TObject *lhs, TObject *rhs) : TObject(lhs->dtype_, kBinaryInsn, lhs, rhs) {
    ASSERT(size_t(op_type) < sizeof(binary_id_list) / sizeof(InsnIdTable));
    simd_id_ = binary_id_list[op_type].ids[dtype_];
    ASSERT(simd_id_ != V_NONE);
    nd_ = lhs->nd_;
  }
  uint64_t Emit(uint64_t *insn) override {
    vBinary op;
    op.xd = xbuf_;
    op.xn = lhs_->xbuf_;
    op.xm = rhs_->xbuf_;
    op.count = nd_.stride_back();
    return vBinary::Encode(insn, simd_id_, op);
  }
  void Dump(bool verbose, std::ostringstream &oss) override {
    oss << "Binary";
    if (verbose) {
      oss << '<' << simd_id_ << '<';
    }
  }
  vSimdInsnID simd_id_;
};

class TBuilder {
 public:
  virtual ~TBuilder() {}
  virtual TObject *Append(TObject *obj) { return nullptr; }
  virtual void Reloc() {}
  virtual void Dump(std::ostringstream &oss, const std::string &indent) = 0;

  std::string &DumpGraph() {
    std::ostringstream oss;
    Dump(oss, "");
    dump_str_ = oss.str();
    return dump_str_;
  }
  std::string &DisAssemble() {
    std::ostringstream oss;
    code_.DisAssemble(oss);
    dump_str_ = oss.str();
    return dump_str_;
  }

  Code code_;
  std::string dump_str_;
};

class VectorBuilder : public TBuilder {
 public:
  ~VectorBuilder() override;
  TObject *Append(TObject *obj) override;
  void Reloc() override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void Init();
  void CodeGen(int64_t tile_num, int64_t block_num);
  std::vector<TObject *> objects_;
  std::vector<TAccess *> access_;
  int64_t max_tile_size_{0};
};

namespace {
class SlotInitializer {
 public:
  static constexpr uint64_t INV_SLOT = -1;
  SlotInitializer(std::vector<TObject *> &insns) : objects_(insns) {}
  int64_t Run(Code &code) {
    InitSlot();
    ForwardSync();
    BackwardSync();
    size_t code_reserve = (SIMD_BLOCK_SIZE + objects_.size() * V_INSN_SIZE_MAX + 511ul) & ~511ul;  // 512B align
    code.Alloc(code_reserve);
    uint64_t xbuf_size =
      ((g_system.LocalMemSize() - g_system.UbWorkspaceSize() - code_reserve) / slots_.size()) & ~511ul;
    for (auto obj : objects_) {
      obj->xbuf_ = obj->xbuf_ * xbuf_size + code_reserve;
    }
    return xbuf_size;
  }
  void InitSlot() {
    for (auto obj : objects_) {
      if (obj->IsLoad()) {
        obj->xbuf_ = slots_.size();
        slots_.emplace_back();
      } else if (obj->IsStore()) {
        uint64_t xbuf = slots_.size();
        obj->lhs_->xbuf_ = xbuf;
        obj->xbuf_ = xbuf;
        slots_.emplace_back();
      } else {
        obj->xbuf_ = INV_SLOT;
      }
    }
    uint64_t dyn_slot_begin = slots_.size();
    auto alloc_slot = [dyn_slot_begin, this]() -> uint64_t {
      for (auto i = dyn_slot_begin; i < slots_.size(); ++i) {
        if (!slots_[i].busy) {
          slots_[i].busy = true;
          return i;
        }
      }
      auto id = slots_.size();
      auto &back = slots_.emplace_back();
      back.busy = true;
      return id;
    };
    for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
      auto obj = *it;
      bool dyn_free = obj->xbuf_ >= dyn_slot_begin;
      if (auto lhs = obj->lhs_) {
        if (lhs->xbuf_ == INV_SLOT) {
          if (dyn_free) {
            lhs->xbuf_ = obj->xbuf_;
            dyn_free = false;
          } else {
            lhs->xbuf_ = alloc_slot();
          }
        }
        if (auto rhs = obj->rhs_; rhs && rhs->xbuf_ == INV_SLOT) {
          if (dyn_free) {
            rhs->xbuf_ = obj->xbuf_;
            dyn_free = false;
          } else {
            rhs->xbuf_ = alloc_slot();
          }
        }
      }
      if (dyn_free) {
        slots_[obj->xbuf_].busy = false;
      }
    }
  }
  void ForwardSync() {
    int load_sync_idx = -1;
    int load_sync_event = 0;
    int simd_barrer_idx = -1;
    auto simd_sync = [&load_sync_idx, &load_sync_event, &simd_barrer_idx](TObject *from, TObject *to) {
      if (from->IsSimd()) {
        if (from->index_ >= simd_barrer_idx) {
          to->sync_mask_ |= 0x1ul << V_HEAD_BAR_FLAG_OFFSET;
          simd_barrer_idx = to->index_;
        }
      } else if (from->index_ > load_sync_idx) {
        from->sync_mask_ |= 0x1ul << V_M_HEAD_SET_FLAG_OFFSET | load_sync_event << V_M_HEAD_SET_EVENT_OFFSET;
        to->sync_mask_ |= 0x1ul << V_HEAD_WAIT_FLAG_OFFSET | load_sync_event << V_HEAD_WAIT_EVENT_OFFSET;
        load_sync_event++;
        load_sync_idx = from->index_;
      }
    };
    int simd_sync_idx = -1;
    int simd_sync_event = 0;
    auto store_sync = [&simd_sync_idx, &simd_sync_event](TObject *from, TObject *to) {
      if (from->index_ > simd_sync_idx) {
        from->sync_mask_ |= 0x1ul << V_HEAD_SET_FLAG_OFFSET | simd_sync_event << V_HEAD_SET_EVENT_OFFSET;
        to->sync_mask_ |= 0x1ul << V_M_HEAD_WAIT_FLAG_OFFSET | simd_sync_event << V_M_HEAD_WAIT_EVENT_OFFSET;
        simd_sync_event++;
        simd_sync_idx = from->index_;
      }
    };
    for (auto obj : objects_) {
      if (obj->IsSimd()) {
        if (obj->rhs_ == nullptr) {
          ASSERT(obj->lhs_);
          simd_sync(slots_[obj->lhs_->xbuf_].prod, obj);
        } else {
          auto prod1 = slots_[obj->lhs_->xbuf_].prod;
          auto prod2 = slots_[obj->rhs_->xbuf_].prod;
          if (prod1->index_ > prod2->index_) {
            auto tmp = prod1;
            prod1 = prod2;
            prod2 = tmp;
          }
          simd_sync(prod2, obj);
          simd_sync(prod1, obj);
        }
      } else if (obj->IsStore()) {
        store_sync(slots_[obj->lhs_->xbuf_].prod, obj);
      }
      slots_[obj->xbuf_].prod = obj;
    }
  }
  void BackwardSync() {
    int b_store_sync_idx = objects_.size();
    int b_store_sync_event = 0;
    int b_simd_sync_idx = objects_.size();
    int b_simd_sync_event = 0;
    auto store_back_sync = [&b_store_sync_idx, &b_store_sync_event](TObject *from, TObject *to) {
      if (to->index_ < b_store_sync_idx) {
        from->sync_mask_ |= 1ul << V_M_HEAD_SET_FLAG_OFFSET | b_store_sync_event << V_M_HEAD_SET_EVENT_OFFSET;
        to->sync_mask_ |= 1ul << V_HEAD_BACK_WAIT_OFFSET | b_store_sync_event << V_HEAD_B_WAIT_EVENT_OFFSET;
        b_store_sync_idx = to->index_;
        b_store_sync_event++;
      }
    };
    auto simd_back_sync = [&b_simd_sync_idx, &b_simd_sync_event](TObject *from, TObject *to) {
      if (to->index_ < b_simd_sync_idx) {
        from->sync_mask_ |= 1ul << V_HEAD_BACK_SET_OFFSET | b_simd_sync_event << V_HEAD_B_SET_EVENT_OFFSET;
        to->sync_mask_ |= 1ul << V_M_HEAD_WAIT_FLAG_OFFSET | b_simd_sync_event << V_M_HEAD_WAIT_EVENT_OFFSET;
        b_simd_sync_idx = to->index_;
        b_simd_sync_event++;
      }
    };
    for (auto it = objects_.rbegin(); it != objects_.rend(); ++it) {
      auto obj = *it;
      if (obj->IsStore()) {
        store_back_sync(obj, obj->lhs_);
      } else if (obj->IsSimd()) {
        if (obj->rhs_ == nullptr) {
          if (obj->lhs_->IsLoad()) {
            simd_back_sync(obj, obj->lhs_);
          }
        } else {
          TObject *to = obj->lhs_->IsLoad() ? obj->lhs_ : nullptr;
          if (auto rhs = obj->rhs_; rhs->IsLoad() && (to == nullptr || rhs->index_ < to->index_)) {
            to = rhs;
          }
          if (to) {
            simd_back_sync(obj, to);
          }
        }
      }
    }
  }

  struct Slot {
    bool busy{false};
    TObject *prod{nullptr};
  };
  std::vector<Slot> slots_;
  std::vector<TObject *> &objects_;
};
}  // namespace

VectorBuilder::~VectorBuilder() {
  for (auto obj : objects_) {
    delete obj;
  }
}

TObject *VectorBuilder::Append(TObject *obj) {
  obj->index_ = objects_.size();
  objects_.push_back(obj);
  if (!obj->IsSimd()) {
    access_.push_back(static_cast<TAccess *>(obj));
  }
  return obj;
}

void VectorBuilder::Reloc() {
  for (auto acc : access_) {
    acc->addr_.Reloc(*acc->gm_ref_);
  }
}

void VectorBuilder::CodeGen(int64_t tile_num, int64_t block_num) {
  if (code_.data_ == nullptr) {
    Init();
  } else {
    code_.Clear();
  }
  uint64_t *insn = reinterpret_cast<uint64_t *>(code_.data_ + Code::HeadSize());
  for (auto obj : objects_) {
    auto size = obj->Emit(insn);
    if (obj->sync_mask_) {
      *insn |= obj->sync_mask_;
    }
    insn += size;
  }
  code_.data_size_ = reinterpret_cast<uint8_t *>(insn) - code_.data_;
  auto tile_per_block = (tile_num + block_num - 1) / block_num;
  code_.block_dim_ = (tile_num + tile_per_block - 1) / tile_per_block;
  code_.UpdateV(tile_num, false);
}

void VectorBuilder::Init() {
  SlotInitializer initializer(objects_);
  max_tile_size_ = initializer.Run(code_);
}

void VectorBuilder::Dump(std::ostringstream &oss, const std::string &indent) {
  auto dump_tensor = [&oss](TObject *t) {
    oss << "%" << t->index_ << t->nd_ << "<" << DTYPE_NAMES[t->dtype_] << ">";
  };
  oss << indent << "vector() {" << std::endl;
  std::string body_indent = indent + "  ";
  for (auto obj : objects_) {
    oss << body_indent;
    dump_tensor(obj);
    oss << " = ";
    obj->Dump(true, oss);
    oss << "(";
    if (obj->lhs_) {
      dump_tensor(obj->lhs_);
      if (obj->rhs_) {
        oss << ", ";
        dump_tensor(obj->rhs_);
      }
    }
    oss << ") // stride=" << obj->nd_.data->strides << std::endl;
  }
  oss << indent << "}";
}

TileBuilder::TileBuilder() {
  impl_ = new VectorBuilder();
}

TileBuilder::~TileBuilder() { delete impl_; }

TObject *TileBuilder::Load(DataType type, GmRef *gm, IntArrayRef *tile_shape, TileRef *tile_space, IntArrayRef *stride) {
  return impl_->Append(new TLoad(type, gm, tile_shape, tile_space, stride));
}

TObject *TileBuilder::Store(TObject *input, GmRef *gm, TileRef *tile_space, IntArrayRef *stride) {
  return impl_->Append(new TStore(input, gm, tile_space, stride));
}

template <UnaryType op_type>
TObject *TileBuilder::Unary(TObject *x) {
  return impl_->Append(new TUnary(op_type, x));
}

#define DEF_UNARY(op) template TObject *TileBuilder::Unary<op>(TObject *)
DEF_UNARY(UnaryType::kSqrt);
DEF_UNARY(UnaryType::kAbs);
DEF_UNARY(UnaryType::kLog);
DEF_UNARY(UnaryType::kExp);
DEF_UNARY(UnaryType::kReciprocal);
DEF_UNARY(UnaryType::kIsFinite);
DEF_UNARY(UnaryType::kLogicalNot);
DEF_UNARY(UnaryType::kRound);
DEF_UNARY(UnaryType::kFloor);
DEF_UNARY(UnaryType::kCeil);
DEF_UNARY(UnaryType::kTrunc);

template <BinaryType op_type>
TObject *TileBuilder::Binary(TObject *lhs, TObject *rhs) {
  return impl_->Append(new TBinary(op_type, lhs, rhs));
}

#define DEF_BINARY(op) template TObject *TileBuilder::Binary<op>(TObject *, TObject *)
DEF_BINARY(BinaryType::kEqual);
DEF_BINARY(BinaryType::kNotEqual);
DEF_BINARY(BinaryType::kGreater);
DEF_BINARY(BinaryType::kGreaterEqual);
DEF_BINARY(BinaryType::kLess);
DEF_BINARY(BinaryType::kLessEqual);
DEF_BINARY(BinaryType::kAdd);
DEF_BINARY(BinaryType::kSub);
DEF_BINARY(BinaryType::kMul);
DEF_BINARY(BinaryType::kDiv);
DEF_BINARY(BinaryType::kPow);
DEF_BINARY(BinaryType::kMaximum);
DEF_BINARY(BinaryType::kMinimum);
DEF_BINARY(BinaryType::kLogicalAnd);
DEF_BINARY(BinaryType::kLogicalOr);

void TileBuilder::CodeGen(int64_t tile_num, int64_t block_num) {
  g_system.Init();
  if (block_num == 0) {
    block_num = g_system.CoreNum();
  }
  static_cast<VectorBuilder *>(impl_)->CodeGen(tile_num, block_num);
}

int TileBuilder::Launch(bool reloc, void *stream) {
  if (reloc) {
    impl_->Reloc();
  }
  return impl_->code_.Launch(nullptr, stream);
}

int64_t TileBuilder::MaxTileSize() const { return static_cast<VectorBuilder *>(impl_)->max_tile_size_; }

const char *TileBuilder::Dump() const { return impl_->DumpGraph().c_str(); }

const char *TileBuilder::Das() const { return impl_->DisAssemble().c_str(); }

void TileBuilderTest() {
  void *gm_a = reinterpret_cast<void *>(0x10000);
  void *gm_b = reinterpret_cast<void *>(0x20000);
  void *gm_c = reinterpret_cast<void *>(0x30000);
  std::vector<int64_t> shape = {2, 2048};
  IntArrayRef shape_ref(shape);
  TileRef tile_ref;
  tile_ref.dim_size = 1;
  tile_ref.dims[0] = 512;

  TileBuilder coder;
  auto x0 = coder.Load(kFloat32, &gm_a, &shape_ref, &tile_ref);
  auto x1 = coder.Load(kFloat32, &gm_b, &shape_ref, &tile_ref);
  auto x2 = coder.Binary<kAdd>(x0, x1);
  auto x3 = coder.Unary<kSqrt>(x2);
  auto x4 = coder.Binary<kSub>(x0, x3);
  coder.Store(x4, &gm_c, &tile_ref);

  std::cout << "before:\n" << coder.Dump() << std::endl;
  coder.CodeGen(512);
  std::cout << "after:\n" << coder.Dump() << std::endl;
  std::cout << "***********\n" << coder.Das() << std::endl;
}
}  // namespace dvm