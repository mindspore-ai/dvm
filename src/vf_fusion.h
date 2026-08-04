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

#ifndef _DVM_VF_FUSION_H_
#define _DVM_VF_FUSION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ops.h"

namespace dvm {
namespace pass {
class BasicBlock;

// Keep the limits in one public-to-the-pass structure so a future overflow
// splitter can use exactly the same admission criteria as the first version.
struct VfFusionLimits {
  static constexpr size_t kMinOps = 2;
  static constexpr size_t kMaxIO = 16;
};

enum class VfOverflowPolicy {
  kSkip,
  // Reserved for a later implementation.  The C310 v1 pass deliberately
  // never invokes this policy yet.
  kSplit,
};

struct VfPartition {
  std::vector<NDObject *> nodes;
  std::vector<NDObject *> inputs;
  std::vector<NDObject *> outputs;
  size_t first_index{0};
  size_t last_index{0};
};

// Shared host/device V_CUSTOM payload ABI.  Keep the layout inspectable
// without exposing the source-emitter implementation.
struct VfPayloadLayout {
  VfPayloadLayout(size_t input_count, size_t output_count);
  explicit VfPayloadLayout(const VfPartition &partition);

  std::string DecodeField(size_t field) const;

  const size_t field_count;
  const size_t word_count;
  const size_t extra_code_bytes;
};

// Pure CCE source-generation seam for VF instruction lowering.  Host-side
// CustomOp construction and registration remain implementation details of
// VfFusion.
std::string BuildVfCceSource(const VfPartition &partition, const std::string &nspace);

namespace detail {

constexpr uint64_t kPayloadSlotBits = 20;
constexpr uint64_t kPayloadSlotMask = (1ull << kPayloadSlotBits) - 1;
constexpr size_t kPayloadSlotsPerWord = 3;
constexpr size_t kMaxPayloadWords =
  (VfFusionLimits::kMaxIO + 1 + kPayloadSlotsPerWord - 1) / kPayloadSlotsPerWord;
constexpr int kMaxVfExtraInputs = static_cast<int>(VfFusionLimits::kMaxIO - 3);
constexpr int kMaxVfExtraOutputs = static_cast<int>(VfFusionLimits::kMaxIO - 2);

struct VfInstruction {
  NDObject *node;
  std::vector<NDObject *> inputs;
};

struct VfProgram {
  VfProgram(const VfPartition &partition, std::string nspace);

  const char *BlockElementType() const;

  const VfPartition &partition;
  const std::string nspace;
  const VfPayloadLayout payload;
  const size_t max_type_size;
  const char *const block_element_type;
  const char *const mask_suffix;
  const std::string mask_name;
  std::vector<VfInstruction> instructions;
};

template <int N>
struct VfXOutN : CustomOp::XOut {
  VfXOutN(NDObject *input, size_t count) {
    ASSERT(count <= static_cast<size_t>(N));
    out_num = static_cast<int>(count);
    for (int i = 0; i < out_num; ++i) {
      data_ext[i] = new ExtOutOp(input);
    }
  }

  ExtOutOp *data_ext[N]{};
};

class VfFusionOp final : public CustomOp {
 public:
  VfFusionOp(const VfPartition &partition, uint64_t func_id);

  void Normalize(std::vector<NDObject *> &) override;
  void TileCollect(TileInfo &info) override;
  uint64_t EmitEx(uint64_t *payload) override;

 private:
  const size_t input_count_;
  const size_t output_count_;
  const VfPayloadLayout payload_;
  XhsN<kMaxVfExtraInputs> xhs_data_;
  VfXOutN<kMaxVfExtraOutputs> xout_data_;
  std::array<DataType, kMaxVfExtraOutputs> xout_types_{};
};

using VfValueNames = std::unordered_map<NDObject *, std::string>;

class VfCceInstructionEmitter {
 public:
  VfCceInstructionEmitter(std::ostringstream &source, const VfProgram &program, const VfValueNames &values);

  void Emit(size_t index, const VfInstruction &instruction) const;

 private:
  const std::string &Value(NDObject *node) const;
  void EmitUnary(const VfInstruction &instruction) const;
  void EmitBinary(const VfInstruction &instruction) const;
  void EmitBinaryScalar(size_t index, const VfInstruction &instruction) const;
  void EmitCast(size_t index, const VfInstruction &instruction) const;
  void EmitCompareResult(size_t index, const VfInstruction &instruction, const char *intrinsic,
                         const std::string &rhs) const;
  void EmitCompare(size_t index, const VfInstruction &instruction) const;
  void EmitCompareScalar(size_t index, const VfInstruction &instruction) const;
  void EmitSelect(size_t index, const VfInstruction &instruction) const;

  std::ostringstream &source_;
  const VfProgram &program_;
  const VfValueNames &values_;
};

class VfCceSourceEmitter {
 public:
  explicit VfCceSourceEmitter(const VfProgram &program);

  std::string Emit();

 private:
  void BuildProgramState();
  void EmitPreamble();
  void EmitEntryPoint();
  void EmitPayloadBindings();
  void EmitDeclarations();
  void EmitCompareConstants();
  void EmitChunkLoop();
  void EmitLoads();
  void EmitInstructions();
  void EmitStores();

  const VfProgram &program_;
  std::ostringstream source_;
  VfValueNames values_;
  std::array<bool, kDataTypeEnd> compare_types_{};
};

class VfFusionCompiler {
 public:
  void CompileAndRegister(const VfPartition &partition);
  static std::string NamespaceFor(const VfPartition &partition);

 private:
  static bool WriteFile(const std::string &path, const std::string &contents);
  static std::optional<uint64_t> FindFunctionOffset(const std::string &symbols, const std::string &name);
};

}  // namespace detail

// CapabilityBasedPartitioner-style partitioner for normalized BasicBlock
// graphs.  It is intentionally reusable by a future kSplit overflow policy.
class VfCapabilityPartitioner {
 public:
  explicit VfCapabilityPartitioner(BasicBlock &bb) : bb_(bb) {}

  std::vector<VfPartition> Propose();

 private:
  BasicBlock &bb_;
};

// Reserved split hook.  v1 uses kSkip and therefore returns no replacement
// partitions for an over-limit candidate.
std::vector<VfPartition> SplitOverLimit(const VfPartition &partition, const VfFusionLimits &limits);

// C310 static graph pass.  It is a no-op unless Config::SetVfFusion() was
// called before static graph codegen.
void VfFusion(BasicBlock &bb);

}  // namespace pass
}  // namespace dvm

#endif  // _DVM_VF_FUSION_H_
