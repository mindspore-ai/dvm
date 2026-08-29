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

#include "vf_fusion.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "pass.h"
#include "system.h"

namespace dvm::pass {
namespace detail {

constexpr VfOverflowPolicy kOverflowPolicy = VfOverflowPolicy::kSkip;
constexpr char kVfCodegenVersion[] = "c310-vf-v16";

using NodeSet = std::unordered_set<NDObject *>;
using NodeIndex = std::unordered_map<NDObject *, size_t>;

// BasicBlock::ToVector is intentionally defined in pass.cc and is only
// instantiated there.  Keep this pass independent of that implementation
// detail so vf_fusion.o never introduces a new exported template reference.
std::vector<NDObject *> CollectObjects(BasicBlock &bb) {
  std::vector<NDObject *> objects;
  objects.reserve(bb.size());
  for (NDObject *it = bb.Begin(); it != bb.End(); it = bb.Next(it)) {
    objects.push_back(it);
  }
  return objects;
}

std::vector<NDObject *> GetInputs(NDObject *obj) {
  std::vector<NDObject *> inputs;
  obj->ForInput([&inputs](NDObject *input) { inputs.push_back(input); });
  return inputs;
}

size_t CountInputOccurrences(NDObject *obj, NDObject *input) {
  size_t count = 0;
  obj->ForInput([&](NDObject *candidate) {
    if (candidate == input) {
      ++count;
    }
  });
  return count;
}

const char *UnaryIntrinsic(int op);
const char *BinaryIntrinsic(int op);
const char *BinaryScalarIntrinsic(int op);
const char *CompareScalarIntrinsic(int op);

bool IsSupportedUnary(const UnaryOp *op) { return *UnaryIntrinsic(op->GetOpType()) != '\0'; }

bool IsSupportedBinary(const BinaryOp *op) { return *BinaryIntrinsic(op->GetOpType()) != '\0'; }

bool IsSupportedBinaryScalar(const BinaryScalarOp *op) {
  return !op->IsScalarRef() && *BinaryScalarIntrinsic(op->GetOpType()) != '\0';
}

bool IsSupportedCompareScalar(const CompareScalarOp *op) {
  return !op->IsScalarRef() && *CompareScalarIntrinsic(op->GetCmpType()) != '\0';
}

bool IsCapabilityNode(NDObject *op) {
  if (op->type_id_ == kInt64) {
    return false;
  }
  switch (op->GetObjectType()) {
    case kUnary:
      return IsSupportedUnary(static_cast<UnaryOp *>(op));
    case kBinary:
      return IsSupportedBinary(static_cast<BinaryOp *>(op));
    case kBinaryS:
      return IsSupportedBinaryScalar(static_cast<BinaryScalarOp *>(op));
    case kCast:
      return true;
    case kCompare:
      return true;
    case kCompareS:
      return IsSupportedCompareScalar(static_cast<CompareScalarOp *>(op));
    case kSelect:
      return GetInputs(op).size() == 3;
    default:
      return false;
  }
}

// One Vec scope uses the widest element in the partition for its lane count.
size_t DataTypeByteSize(DataType type) {
  switch (type) {
    case kBool:
      return sizeof(uint8_t);
    case kFloat16:
    case kBFloat16:
      return sizeof(uint16_t);
    case kFloat32:
    case kInt32:
      return sizeof(uint32_t);
    default:
      return 0;
  }
}

size_t MaxDataTypeByteSize(const VfPartition &partition) {
  size_t max_size = 0;
  auto update = [&max_size](const std::vector<NDObject *> &nodes) {
    for (auto *node : nodes) {
      max_size = std::max(max_size, DataTypeByteSize(node->type_id_));
    }
  };
  update(partition.inputs);
  update(partition.nodes);
  update(partition.outputs);
  return max_size;
}

const char *LoadDistribution(DataType type, size_t max_type_size) {
  const size_t type_size = DataTypeByteSize(type);
  if (type_size == max_type_size) {
    return "NORM";
  }
  if (type_size == sizeof(uint8_t) && max_type_size == sizeof(uint16_t)) {
    return "UNPK_B8";
  }
  if (type_size == sizeof(uint8_t) && max_type_size == sizeof(uint32_t)) {
    return "UNPK4_B8";
  }
  if (type_size == sizeof(uint16_t) && max_type_size == sizeof(uint32_t)) {
    return "UNPK_B16";
  }
  return "NORM";
}

const char *StoreDistribution(DataType type, size_t max_type_size) {
  const size_t type_size = DataTypeByteSize(type);
  if (type_size == max_type_size) {
    return max_type_size == sizeof(uint8_t) ? "dist_b8" : max_type_size == sizeof(uint16_t) ? "dist_b16" : "dist_b32";
  }
  if (type_size == sizeof(uint8_t) && max_type_size == sizeof(uint16_t)) {
    return "PK_B16";
  }
  if (type_size == sizeof(uint8_t) && max_type_size == sizeof(uint32_t)) {
    return "PK4_B32";
  }
  if (type_size == sizeof(uint16_t) && max_type_size == sizeof(uint32_t)) {
    return "PK_B32";
  }
  return "NORM";
}

std::optional<std::vector<NDObject *>> StableKahnOrder(BasicBlock &bb) {
  const auto objects = CollectObjects(bb);
  NodeSet members(objects.begin(), objects.end());
  NodeIndex list_indices;
  for (size_t i = 0; i < objects.size(); ++i) {
    list_indices.emplace(objects[i], i);
  }
  std::unordered_map<NDObject *, size_t> indegree;
  auto later = [&list_indices](NDObject *lhs, NDObject *rhs) { return list_indices.at(lhs) > list_indices.at(rhs); };
  std::priority_queue<NDObject *, std::vector<NDObject *>, decltype(later)> ready(later);
  for (auto *obj : objects) {
    size_t degree = 0;
    obj->ForInput([&](NDObject *input) {
      if (members.count(input) != 0) {
        ++degree;
      }
    });
    indegree[obj] = degree;
    if (degree == 0) {
      ready.push(obj);
    }
  }

  std::vector<NDObject *> order;
  order.reserve(objects.size());
  while (!ready.empty()) {
    auto *obj = ready.top();
    ready.pop();
    order.push_back(obj);
    for (auto *user : bb.GetUsers(obj)) {
      if (members.count(user) == 0) {
        continue;
      }
      auto &degree = indegree[user];
      if (degree == 0) {
        return std::nullopt;
      }
      if (--degree == 0) {
        ready.push(user);
      }
    }
  }
  if (order.size() != objects.size()) {
    return std::nullopt;
  }
  return order;
}

bool IsAcyclic(BasicBlock &bb) { return StableKahnOrder(bb).has_value(); }

// A capability partition may be non-contiguous in the physical object list:
// Normalize can leave an unsupported helper between two supported nodes.  The
// Custom only depends on the partition's external inputs, so it can be placed
// anywhere after all of them and before the first external user of any
// partition output.  This keeps the list execution order valid without
// needlessly rejecting an otherwise acyclic capability partition.
NDObject *FindSafeInsertionPoint(BasicBlock &bb, const VfPartition &partition) {
  const auto objects = CollectObjects(bb);
  NodeIndex indices;
  for (size_t i = 0; i < objects.size(); ++i) {
    indices.emplace(objects[i], i);
  }

  bool has_input = false;
  size_t latest_input = 0;
  for (auto *input : partition.inputs) {
    const auto input_it = indices.find(input);
    if (input_it == indices.end()) {
      return nullptr;
    }
    has_input = true;
    latest_input = std::max(latest_input, input_it->second);
  }

  NodeSet members(partition.nodes.begin(), partition.nodes.end());
  std::optional<size_t> earliest_external_user;
  for (auto *output : partition.outputs) {
    for (auto *user : bb.GetUsers(output)) {
      if (members.count(user) != 0) {
        continue;
      }
      const auto user_it = indices.find(user);
      if (user_it == indices.end()) {
        return nullptr;
      }
      if (!earliest_external_user.has_value() || user_it->second < *earliest_external_user) {
        earliest_external_user = user_it->second;
      }
    }
  }
  if (!earliest_external_user.has_value() || (has_input && latest_input >= *earliest_external_user)) {
    return nullptr;
  }
  return objects[*earliest_external_user];
}

// Build a shadow graph that contracts every member of `partition` into a
// single node.  This catches exactly the cycles that a real Custom rewrite
// could introduce while keeping the original BasicBlock untouched.
bool IsShadowAcyclic(BasicBlock &bb, const VfPartition &partition) {
  if (partition.nodes.empty()) {
    return false;
  }
  const auto objects = CollectObjects(bb);
  NodeSet members(partition.nodes.begin(), partition.nodes.end());
  NDObject *contracted = partition.nodes.front();
  std::unordered_set<NDObject *> shadow_nodes;
  shadow_nodes.reserve(objects.size());
  for (auto *obj : objects) {
    shadow_nodes.insert(members.count(obj) ? contracted : obj);
  }

  std::unordered_map<NDObject *, size_t> indegree;
  std::unordered_map<NDObject *, std::vector<NDObject *>> users;
  for (auto *node : shadow_nodes) {
    indegree[node] = 0;
  }
  for (auto *obj : objects) {
    auto *to = members.count(obj) ? contracted : obj;
    obj->ForInput([&](NDObject *input) {
      auto *from = members.count(input) ? contracted : input;
      if (from == to || shadow_nodes.count(from) == 0) {
        return;
      }
      users[from].push_back(to);
      ++indegree[to];
    });
  }

  std::queue<NDObject *> ready;
  for (const auto &entry : indegree) {
    if (entry.second == 0) {
      ready.push(entry.first);
    }
  }
  size_t visited = 0;
  while (!ready.empty()) {
    auto *node = ready.front();
    ready.pop();
    ++visited;
    for (auto *user : users[node]) {
      auto &degree = indegree[user];
      if (degree == 0) {
        return false;
      }
      if (--degree == 0) {
        ready.push(user);
      }
    }
  }
  return visited == shadow_nodes.size();
}

struct WorkingPartition {
  bool alive{true};
  std::vector<NDObject *> nodes;
};

bool WouldCreateCycle(BasicBlock &bb, const NodeSet &merged_nodes,
                      const std::unordered_map<NDObject *, size_t> &assignment,
                      const std::vector<WorkingPartition> &partitions) {
  // This follows CapabilityBasedPartitioner's key rule: when the DFS reaches
  // an already assigned partition, traverse all of that partition's outgoing
  // edges as if the partition were already a contracted node.
  for (auto *root : merged_nodes) {
    for (auto *first_user : bb.GetUsers(root)) {
      if (merged_nodes.count(first_user) != 0) {
        continue;
      }
      std::vector<NDObject *> stack = {first_user};
      NodeSet visited;
      while (!stack.empty()) {
        auto *node = stack.back();
        stack.pop_back();
        if (!visited.insert(node).second) {
          continue;
        }
        if (merged_nodes.count(node) != 0) {
          return true;
        }
        auto assigned = assignment.find(node);
        if (assigned != assignment.end()) {
          const auto &partition = partitions[assigned->second];
          NodeSet partition_nodes(partition.nodes.begin(), partition.nodes.end());
          for (auto *partition_node : partition.nodes) {
            for (auto *user : bb.GetUsers(partition_node)) {
              if (partition_nodes.count(user) == 0) {
                stack.push_back(user);
              }
            }
          }
        } else {
          for (auto *user : bb.GetUsers(node)) {
            stack.push_back(user);
          }
        }
      }
    }
  }
  return false;
}

void SortUniqueByIndex(std::vector<NDObject *> &objects, const NodeIndex &indices) {
  std::sort(objects.begin(), objects.end(),
            [&](NDObject *lhs, NDObject *rhs) { return indices.at(lhs) < indices.at(rhs); });
  objects.erase(std::unique(objects.begin(), objects.end()), objects.end());
}

VfPartition BuildPartition(BasicBlock &bb, std::vector<NDObject *> nodes, const NodeIndex &indices) {
  SortUniqueByIndex(nodes, indices);
  VfPartition partition;
  partition.nodes = std::move(nodes);
  if (partition.nodes.empty()) {
    return partition;
  }
  partition.first_index = indices.at(partition.nodes.front());
  partition.last_index = indices.at(partition.nodes.back());

  NodeSet members(partition.nodes.begin(), partition.nodes.end());
  NodeSet seen_inputs;
  for (auto *node : partition.nodes) {
    node->ForInput([&](NDObject *input) {
      if (members.count(input) == 0 && seen_inputs.insert(input).second) {
        partition.inputs.push_back(input);
      }
    });
    bool has_external_user = false;
    for (auto *user : bb.GetUsers(node)) {
      if (members.count(user) == 0) {
        has_external_user = true;
        break;
      }
    }
    if (has_external_user) {
      partition.outputs.push_back(node);
    }
  }
  SortUniqueByIndex(partition.inputs, indices);
  SortUniqueByIndex(partition.outputs, indices);
  return partition;
}

bool SameShapeAndNdd(const VfPartition &partition) {
  if (partition.nodes.empty()) {
    return false;
  }
  if (partition.nodes.front()->nd_.data == nullptr) {
    return false;
  }
  const auto &dims = partition.nodes.front()->nd_.dims();
  const auto *shape = partition.nodes.front()->shape_ref_;
  if (shape == nullptr) {
    return false;
  }
  auto same = [&](NDObject *node) {
    if (node == nullptr || node->nd_.data == nullptr || node->shape_ref_ == nullptr || !(node->nd_.dims() == dims) ||
        node->shape_ref_->size != shape->size) {
      return false;
    }
    for (size_t i = 0; i < shape->size; ++i) {
      if (node->shape_ref_->data[i] != shape->data[i]) {
        return false;
      }
    }
    return true;
  };
  for (auto *node : partition.nodes) {
    if (!same(node)) {
      return false;
    }
  }
  for (auto *node : partition.inputs) {
    if (!same(node)) {
      return false;
    }
  }
  return true;
}

bool SameShapeAndNdd(const NDObject *lhs, const NDObject *rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs->nd_.data == nullptr || rhs->nd_.data == nullptr ||
      lhs->shape_ref_ == nullptr || rhs->shape_ref_ == nullptr || !(lhs->nd_.dims() == rhs->nd_.dims()) ||
      lhs->shape_ref_->size != rhs->shape_ref_->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->shape_ref_->size; ++i) {
    if (lhs->shape_ref_->data[i] != rhs->shape_ref_->data[i]) {
      return false;
    }
  }
  return true;
}

bool WithinLimits(BasicBlock &bb, const VfPartition &partition) {
  if (partition.nodes.size() < VfFusionLimits::kMinOps || partition.inputs.empty() || partition.outputs.empty() ||
      partition.inputs.size() + partition.outputs.size() > VfFusionLimits::kMaxIO ||
      (partition.inputs.size() + partition.outputs.size() + 1 + kPayloadSlotsPerWord - 1) / kPayloadSlotsPerWord >
        kMaxPayloadWords ||
      !SameShapeAndNdd(partition)) {
    return false;
  }

  auto topo_order = StableKahnOrder(bb);
  if (!topo_order.has_value()) {
    return false;
  }
  NodeIndex topo_indices;
  const auto list_order = CollectObjects(bb);
  for (size_t i = 0; i < topo_order->size(); ++i) {
    topo_indices.emplace((*topo_order)[i], i);
  }
  for (auto *node : partition.nodes) {
    if (topo_indices.count(node) == 0) {
      return false;
    }
    if (std::find(list_order.begin(), list_order.end(), node) == list_order.end()) {
      return false;
    }
  }
  return FindSafeInsertionPoint(bb, partition) != nullptr;
}

std::string DataTypeCpp(DataType type) {
  switch (type) {
    case kBool:
      return "int8_t";
    case kFloat16:
      return "half";
    case kBFloat16:
      return "bfloat16_t";
    case kFloat32:
      return "float";
    case kInt32:
      return "int32_t";
    default:
      return "void";
  }
}

std::string VectorTypeCpp(DataType type) {
  switch (type) {
    case kBool:
      return "vector_s8";
    case kFloat16:
      return "vector_f16";
    case kBFloat16:
      return "vector_bf16";
    case kFloat32:
      return "vector_f32";
    case kInt32:
      return "vector_s32";
    default:
      return "void";
  }
}

struct VfIntrinsic {
  int op_type;
  const char *name;
};

template <size_t N>
const char *FindIntrinsic(const std::array<VfIntrinsic, N> &intrinsics, int op_type) {
  const auto it = std::find_if(intrinsics.begin(), intrinsics.end(),
                               [op_type](const VfIntrinsic &intrinsic) { return intrinsic.op_type == op_type; });
  return it == intrinsics.end() ? "" : it->name;
}

// Keep normalized DVM op kinds next to their CCE spellings.  Adding another
// unary, binary, or compare instruction only needs one entry here plus its
// normal capability admission.
constexpr std::array<VfIntrinsic, 4> kUnaryIntrinsics = {{
  {kSqrt, "vsqrt"},
  {kAbs, "vabs"},
  {kLog, "vln"},
  {kExp, "vexp"},
}};
constexpr std::array<VfIntrinsic, 6> kBinaryIntrinsics = {{
  {kAdd, "vadd"},
  {kSub, "vsub"},
  {kMul, "vmul"},
  {kDiv, "vdiv"},
  {kMaximum, "vmax"},
  {kMinimum, "vmin"},
}};
constexpr std::array<VfIntrinsic, 6> kCompareIntrinsics = {{
  {kEqual, "vcmp_eq"},
  {kNotEqual, "vcmp_ne"},
  {kGreater, "vcmp_gt"},
  {kGreaterEqual, "vcmp_ge"},
  {kLess, "vcmp_lt"},
  {kLessEqual, "vcmp_le"},
}};
constexpr std::array<VfIntrinsic, 6> kBinaryScalarIntrinsics = {{
  {kAdds, "vadds"},
  {kMuls, "vmuls"},
  {kDivs, "vdiv"},
  {ksDiv, "vdiv"},
  {kMaximums, "vmaxs"},
  {kMinimums, "vmins"},
}};
constexpr std::array<VfIntrinsic, 6> kCompareScalarIntrinsics = {{
  {kEquals, "vcmps_eq"},
  {kNotEquals, "vcmps_ne"},
  {kGreaters, "vcmps_gt"},
  {kGreaterEquals, "vcmps_ge"},
  {kLesss, "vcmps_lt"},
  {kLessEquals, "vcmps_le"},
}};

const char *UnaryIntrinsic(int op) { return FindIntrinsic(kUnaryIntrinsics, op); }

const char *BinaryIntrinsic(int op) { return FindIntrinsic(kBinaryIntrinsics, op); }

const char *CompareIntrinsic(int op) { return FindIntrinsic(kCompareIntrinsics, op); }

const char *BinaryScalarIntrinsic(int op) { return FindIntrinsic(kBinaryScalarIntrinsics, op); }

const char *CompareScalarIntrinsic(int op) { return FindIntrinsic(kCompareScalarIntrinsics, op); }

bool IsScalarDivision(int op) { return op == kDivs || op == ksDiv; }

std::string FloatLiteral(float value) {
  std::ostringstream stream;
  stream << std::showpoint << std::setprecision(std::numeric_limits<float>::max_digits10) << value << 'f';
  return stream.str();
}

std::string ScalarLiteral(scode_t scalar, DataType type) {
  switch (type) {
    case kFloat16:
      return "static_cast<half>(" + FloatLiteral(static_cast<float>(Float16(static_cast<uint16_t>(scalar)))) + ")";
    case kBFloat16:
      return "static_cast<bfloat16_t>(" + FloatLiteral(static_cast<float>(BFloat16(static_cast<uint16_t>(scalar)))) +
             ")";
    case kFloat32: {
      float value;
      std::memcpy(&value, &scalar, sizeof(value));
      return FloatLiteral(value);
    }
    case kInt32:
      return "static_cast<int32_t>(" + std::to_string(static_cast<int32_t>(scalar)) + ")";
    case kBool:
      return "static_cast<int8_t>(" + std::to_string(scalar != 0) + ")";
    default:
      return "0";
  }
}

uint64_t Fnv1aAppend(uint64_t hash, const std::string &text) {
  for (unsigned char c : text) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

void AppendNodeAttributes(std::ostringstream &stream, NDObject *node) {
  // Every node property that affects CCE lowering belongs in this key.  That
  // keeps structurally different JIT kernels out of the same cache entry.
  if (node->GetObjectType() == kUnary) {
    stream << static_cast<UnaryOp *>(node)->GetOpType();
  } else if (node->GetObjectType() == kBinary) {
    stream << static_cast<BinaryOp *>(node)->GetOpType();
  } else if (node->GetObjectType() == kCompare) {
    stream << static_cast<CompareOp *>(node)->GetCmpType();
  } else if (node->GetObjectType() == kBinaryS) {
    const auto *binary = static_cast<BinaryScalarOp *>(node);
    stream << binary->GetOpType() << ':' << binary->GetScalar();
  } else if (node->GetObjectType() == kCompareS) {
    const auto *compare = static_cast<CompareScalarOp *>(node);
    stream << compare->GetCmpType() << ':' << compare->GetScalar();
  } else if (node->GetObjectType() == kCast) {
    stream << node->lhs_->type_id_ << '>' << node->type_id_;
  }
}

void AppendNodeDescriptor(std::ostringstream &stream, NDObject *node,
                          const std::unordered_map<NDObject *, size_t> &input_ids,
                          const std::unordered_map<NDObject *, size_t> &node_ids) {
  stream << node->GetObjectType() << ':' << node->type_id_ << ':';
  AppendNodeAttributes(stream, node);
  stream << '(';
  for (auto *input : GetInputs(node)) {
    if (auto it = input_ids.find(input); it != input_ids.end()) {
      stream << 'i' << it->second;
    } else {
      stream << 'n' << node_ids.at(input);
    }
    stream << ',';
  }
  stream << ")|";
}

std::string DescriptorKey(const VfPartition &partition) {
  std::unordered_map<NDObject *, size_t> input_ids;
  std::unordered_map<NDObject *, size_t> node_ids;
  for (size_t i = 0; i < partition.inputs.size(); ++i) {
    input_ids.emplace(partition.inputs[i], i);
  }
  for (size_t i = 0; i < partition.nodes.size(); ++i) {
    node_ids.emplace(partition.nodes[i], i);
  }

  std::ostringstream oss;
  // Bump this whenever emitted host/device source or its ABI changes so a
  // persistent DVM_VF_JIT_CACHE_DIR never reuses an incompatible artifact.
  oss << kVfCodegenVersion << '|';
  for (auto *node : partition.nodes) {
    AppendNodeDescriptor(oss, node, input_ids, node_ids);
  }
  oss << "out:";
  for (auto *output : partition.outputs) {
    oss << node_ids.at(output) << ':' << output->type_id_ << ',';
  }
  return oss.str();
}

std::string HashNamespace(const VfPartition &partition) {
  auto hash = Fnv1aAppend(1469598103934665603ull, DescriptorKey(partition));
  std::ostringstream oss;
  oss << "vf_" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

std::string EntryName(const std::string &nspace) { return nspace + "_entry"; }

struct VfCompileUnit {
  std::string nspace;
  const VfPartition *partition;
};

std::vector<VfCompileUnit> BuildCompileUnits(const std::vector<VfPartition> &partitions) {
  std::vector<VfCompileUnit> units;
  std::unordered_set<std::string> seen;
  units.reserve(partitions.size());
  seen.reserve(partitions.size());
  for (const auto &partition : partitions) {
    auto nspace = HashNamespace(partition);
    if (seen.insert(nspace).second) {
      units.push_back({std::move(nspace), &partition});
    }
  }
  std::sort(units.begin(), units.end(),
            [](const VfCompileUnit &lhs, const VfCompileUnit &rhs) { return lhs.nspace < rhs.nspace; });
  return units;
}

std::string HashBundleNamespace(const std::vector<VfCompileUnit> &units) {
  auto hash = Fnv1aAppend(1469598103934665603ull, std::string(kVfCodegenVersion) + "|bundle|");
  for (const auto &unit : units) {
    hash = Fnv1aAppend(hash, unit.nspace + '|');
  }
  std::ostringstream oss;
  oss << "vf_bundle_" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

std::string RunCommand(const std::vector<std::string> &command, const char *error_message) {
  std::string output;
  int fd[2];
  if (pipe(fd) != 0) {
    const std::string message = std::string(error_message) + ": " + std::strerror(errno);
    DvmException(message.c_str());
    return {};
  }
  const auto pid = fork();
  if (pid == 0) {
    close(fd[0]);
    dup2(fd[1], STDOUT_FILENO);
    dup2(fd[1], STDERR_FILENO);
    close(fd[1]);
    unsetenv("LD_PRELOAD");
    std::vector<char *> argv;
    argv.reserve(command.size() + 1);
    for (const auto &arg : command) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (pid < 0) {
    close(fd[0]);
    close(fd[1]);
    const std::string message = std::string(error_message) + ": " + std::strerror(errno);
    DvmException(message.c_str());
    return {};
  }
  close(fd[1]);
  std::array<char, 4096> buffer{};
  ssize_t size;
  while ((size = read(fd[0], buffer.data(), buffer.size())) > 0) {
    output.append(buffer.data(), static_cast<size_t>(size));
  }
  close(fd[0]);
  int status = 0;
  if (waitpid(pid, &status, 0) == -1) {
    const std::string message = std::string(error_message) + ": " + std::strerror(errno);
    DvmException(message.c_str());
    return {};
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string message = std::string(error_message) + ": " + output;
    DvmException(message.c_str());
    return {};
  }
  return output;
}

std::filesystem::path FindCacheRoot() {
  if (const char *cache = std::getenv("DVM_VF_JIT_CACHE_DIR"); cache != nullptr && *cache != '\0') {
    return cache;
  }
  return std::filesystem::current_path() / "vf_fusion";
}

std::string PayloadFieldExpr(size_t field) {
  std::ostringstream oss;
  oss << "((pc[" << field / kPayloadSlotsPerWord << "] >> "
      << (kPayloadSlotsPerWord - 1 - field % kPayloadSlotsPerWord) * kPayloadSlotBits << ") & 0x" << std::hex
      << kPayloadSlotMask << "u)";
  return oss.str();
}

// The device source emitter consumes this immutable program description.  The
// graph-specific discovery (input order, node operands, payload ABI and vector
// layout) is therefore kept out of the generated-source string builder.
VfProgram::VfProgram(const VfPartition &partition, std::string nspace)
    : partition(partition),
      nspace(std::move(nspace)),
      payload(partition),
      max_type_size(MaxDataTypeByteSize(partition)),
      block_element_type(max_type_size == sizeof(uint8_t)    ? "uint8_t"
                         : max_type_size == sizeof(uint16_t) ? "uint16_t"
                                                             : "uint32_t"),
      mask_suffix(max_type_size == sizeof(uint8_t)    ? "b8"
                  : max_type_size == sizeof(uint16_t) ? "b16"
                                                      : "b32"),
      mask_name(std::string("mask_") + mask_suffix) {
  instructions.reserve(partition.nodes.size());
  for (auto *node : partition.nodes) {
    instructions.push_back({node, GetInputs(node)});
  }
}

const char *VfProgram::BlockElementType() const { return block_element_type; }

VfFusionOp::VfFusionOp(const VfPartition &partition, uint64_t func_id)
    : CustomOp(partition.inputs.front(), partition.inputs.size() > 1 ? partition.inputs[1] : nullptr,
               partition.outputs.front()->type_id_),
      input_count_(partition.inputs.size()),
      output_count_(partition.outputs.size()),
      payload_(input_count_, output_count_),
      xout_data_(this, output_count_ - 1) {
  ASSERT(input_count_ > 0 && output_count_ > 0);
  ASSERT(input_count_ + output_count_ <= VfFusionLimits::kMaxIO);
  func_id_ = func_id;
  if (input_count_ > 2) {
    xhs_data_.in_num = static_cast<int>(input_count_ - 2);
    xhs_data_.free_mask = 0;
    for (size_t i = 0; i < input_count_ - 2; ++i) {
      xhs_data_.data[i] = partition.inputs[i + 2];
    }
    SetXhs(&xhs_data_);
  }
  for (size_t i = 1; i < output_count_; ++i) {
    const auto type = partition.outputs[i]->type_id_;
    xout_types_[i - 1] = type;
    xout_data_.data[i - 1]->type_id_ = type;
  }
  if (output_count_ > 1) {
    SetXOut(&xout_data_);
  }
}

void VfFusionOp::Normalize(std::vector<NDObject *> &) {
  ndd_.Reset(lhs_->nd_.dims());
  shape_ = *lhs_->shape_ref_;
  for (int i = 0; i < xout_data_.out_num; ++i) {
    auto *output = xout_data_.data[i];
    output->type_id_ = xout_types_[i];
    output->shape_ref_ = &shape_;
    output->nd_ = nd_;
  }
}

void VfFusionOp::TileCollect(TileInfo &info) { info.code_reserve += payload_.extra_code_bytes; }

uint64_t VfFusionOp::EmitEx(uint64_t *payload) {
  std::array<uint64_t, kMaxPayloadWords * kPayloadSlotsPerWord> values{};
  size_t field = 0;
  for (size_t i = 0; i < input_count_; ++i) {
    values[field++] = i == 0 ? lhs_->xbuf_ : i == 1 ? rhs_->xbuf_ : xhs_->data[i - 2]->xbuf_;
  }
  values[field++] = xbuf_;
  for (int i = 0; i < xout_data_.out_num; ++i) {
    values[field++] = xout_data_.data[i]->xbuf_;
  }
  values[field++] = nd_.stride_back();
  ASSERT(field == payload_.field_count);
  for (size_t i = 0; i < payload_.word_count; ++i) {
    payload[i] = 0;
  }
  for (size_t i = 0; i < field; ++i) {
    payload[i / kPayloadSlotsPerWord] |= (values[i] & kPayloadSlotMask)
                                         << ((kPayloadSlotsPerWord - 1 - i % kPayloadSlotsPerWord) * kPayloadSlotBits);
  }
  return payload_.word_count;
}

// This owns only node-to-intrinsic lowering.  Generic unary/binary/compare
// additions use the tables above; a special normalized node adds one focused
// Emit* method here without touching the vec-scope plumbing.
VfCceInstructionEmitter::VfCceInstructionEmitter(std::ostringstream &source, const VfProgram &program,
                                                 const VfValueNames &values)
    : source_(source), program_(program), values_(values) {}

void VfCceInstructionEmitter::Emit(size_t index, const VfInstruction &instruction) const {
  switch (instruction.node->GetObjectType()) {
    case kUnary:
      EmitUnary(instruction);
      break;
    case kBinary:
      EmitBinary(instruction);
      break;
    case kBinaryS:
      EmitBinaryScalar(index, instruction);
      break;
    case kCast:
      EmitCast(index, instruction);
      break;
    case kCompare:
      EmitCompare(index, instruction);
      break;
    case kCompareS:
      EmitCompareScalar(index, instruction);
      break;
    case kSelect:
      EmitSelect(index, instruction);
      break;
    default:
      break;
  }
}

const std::string &VfCceInstructionEmitter::Value(NDObject *node) const { return values_.at(node); }

void VfCceInstructionEmitter::EmitUnary(const VfInstruction &instruction) const {
  source_ << "      " << UnaryIntrinsic(static_cast<UnaryOp *>(instruction.node)->GetOpType()) << '('
          << Value(instruction.node) << ", " << Value(instruction.inputs[0]) << ", " << program_.mask_name
          << ", MODE_ZEROING);\n";
}

void VfCceInstructionEmitter::EmitBinary(const VfInstruction &instruction) const {
  source_ << "      " << BinaryIntrinsic(static_cast<BinaryOp *>(instruction.node)->GetOpType()) << '('
          << Value(instruction.node) << ", " << Value(instruction.inputs[0]) << ", " << Value(instruction.inputs[1])
          << ", " << program_.mask_name << ", MODE_ZEROING);\n";
}

void VfCceInstructionEmitter::EmitBinaryScalar(size_t index, const VfInstruction &instruction) const {
  const auto *binary = static_cast<BinaryScalarOp *>(instruction.node);
  const int op_type = binary->GetOpType();
  const std::string scalar = ScalarLiteral(binary->GetScalar(), instruction.node->type_id_);
  if (IsScalarDivision(op_type)) {
    source_ << "      vdiv(" << Value(instruction.node) << ", ";
    if (op_type == ksDiv) {
      source_ << "vscalar" << index << ", " << Value(instruction.inputs[0]);
    } else {
      source_ << Value(instruction.inputs[0]) << ", vscalar" << index;
    }
    source_ << ", " << program_.mask_name << ", MODE_ZEROING);\n";
    return;
  }
  source_ << "      " << BinaryScalarIntrinsic(op_type) << '(' << Value(instruction.node) << ", "
          << Value(instruction.inputs[0]) << ", " << scalar << ", " << program_.mask_name << ", MODE_ZEROING);\n";
}

void VfCceInstructionEmitter::EmitCast(size_t index, const VfInstruction &instruction) const {
  const auto source_type = instruction.inputs[0]->type_id_;
  const auto target_type = instruction.node->type_id_;
  if (source_type == kInt32 && target_type == kFloat16) {
    source_ << "      vcvt(cast_tmp" << index << ", " << Value(instruction.inputs[0]) << ", " << program_.mask_name
            << ", VfRound<ROUND::A>());\n";
    source_ << "      vcvt(" << Value(instruction.node) << ", cast_tmp" << index << ", " << program_.mask_name
            << ", VfRound<ROUND::A>(), RS_ENABLE, PART_EVEN);\n";
    return;
  }
  source_ << "      vcvt(" << Value(instruction.node) << ", " << Value(instruction.inputs[0]) << ", "
          << program_.mask_name;
  if (source_type == kFloat32 && target_type == kFloat16) {
    source_ << ", VfRound<ROUND::A>(), RS_ENABLE, PART_EVEN";
  } else if (source_type == kFloat32 && target_type == kBFloat16) {
    source_ << ", VfRound<ROUND::R>(), RS_ENABLE, PART_EVEN";
  } else if (source_type == kFloat32 && target_type == kInt32) {
    source_ << ", VfRound<ROUND::A>(), RS_ENABLE";
  } else if (source_type == kFloat16 && target_type == kBool) {
    source_ << ", VfRound<ROUND::A>(), RS_ENABLE, PART_EVEN";
  } else if (source_type == kFloat16 && target_type == kInt32) {
    source_ << ", VfRound<ROUND::Z>(), PART_EVEN";
  } else if (source_type == kBFloat16 && target_type == kInt32) {
    source_ << ", VfRound<ROUND::A>(), RS_ENABLE, PART_EVEN";
  } else if (source_type == kInt32 && target_type == kFloat32) {
    source_ << ", VfRound<ROUND::A>()";
  } else {
    source_ << ", PART_EVEN";
  }
  source_ << ");\n";
}

void VfCceInstructionEmitter::EmitCompareResult(size_t index, const VfInstruction &instruction, const char *intrinsic,
                                                const std::string &rhs) const {
  const int type = instruction.node->type_id_;
  source_ << "      " << intrinsic << "(pred" << index << ", " << Value(instruction.inputs[0]) << ", " << rhs << ", "
          << program_.mask_name << ");\n";
  source_ << "      vsel(" << Value(instruction.node) << ", one_" << type << ", zero_" << type << ", pred" << index
          << ");\n";
}

void VfCceInstructionEmitter::EmitCompare(size_t index, const VfInstruction &instruction) const {
  const auto *compare = static_cast<CompareOp *>(instruction.node);
  EmitCompareResult(index, instruction, CompareIntrinsic(compare->GetCmpType()), Value(instruction.inputs[1]));
}

void VfCceInstructionEmitter::EmitCompareScalar(size_t index, const VfInstruction &instruction) const {
  const auto *compare = static_cast<CompareScalarOp *>(instruction.node);
  EmitCompareResult(index, instruction, CompareScalarIntrinsic(compare->GetCmpType()),
                    ScalarLiteral(compare->GetScalar(), instruction.node->type_id_));
}

void VfCceInstructionEmitter::EmitSelect(size_t index, const VfInstruction &instruction) const {
  // ForInput order is lhs, rhs, then xhs condition.
  source_ << "      { vector_bool select_pred" << index << "; vcmps_ne(select_pred" << index << ", "
          << Value(instruction.inputs[2]) << ", (" << DataTypeCpp(instruction.inputs[2]->type_id_) << ")0, "
          << program_.mask_name << "); vsel(" << Value(instruction.node) << ", " << Value(instruction.inputs[0]) << ", "
          << Value(instruction.inputs[1]) << ", select_pred" << index << "); }\n";
}

VfCceSourceEmitter::VfCceSourceEmitter(const VfProgram &program, std::string registration_namespace)
    : program_(program), registration_namespace_(std::move(registration_namespace)) {
  BuildProgramState();
}

std::string VfCceSourceEmitter::Emit(bool include_preamble) {
  if (include_preamble) {
    EmitPreamble();
  }
  EmitEntryPoint();
  return source_.str();
}

void VfCceSourceEmitter::BuildProgramState() {
  for (size_t i = 0; i < program_.partition.inputs.size(); ++i) {
    values_.emplace(program_.partition.inputs[i], "vin" + std::to_string(i));
  }
  for (size_t i = 0; i < program_.instructions.size(); ++i) {
    const auto &instruction = program_.instructions[i];
    values_.emplace(instruction.node, "v" + std::to_string(i));
    if (instruction.node->GetObjectType() == kCompare || instruction.node->GetObjectType() == kCompareS) {
      compare_types_[instruction.node->type_id_] = true;
    }
  }
}

void VfCceSourceEmitter::EmitPreamble() {
  source_ << "#define __aicore_inline__ static[aicore] __attribute__((always_inline))\n";
  source_ << "#include <stdint.h>\n#include <type_traits>\n\n";
  // CCEC's standalone JIT preamble does not provide vm_aiv.h's helper.
  source_ << "#ifndef BlockNum\n#define BlockNum(x) (CCE_VL / sizeof(x))\n#endif\n\n";
  source_ << "template <ROUND R> __aicore_inline__ auto VfRound() { return std::integral_constant<ROUND, R>(); }\n\n";
  source_ << "extern \"C\" __global__ [aicore] void dvm_custom_" << registration_namespace_
          << "(uint64_t) { pipe_barrier(PIPE_ALL); }\n\n";
}

void VfCceSourceEmitter::EmitEntryPoint() {
  source_ << "extern \"C\" [aicore] void " << EntryName(program_.nspace)
          << "(__gm__ uint64_t *__restrict__ pc, uint64_t, uint64_t) {\n";
  EmitPayloadBindings();
  source_ << "  __VEC_SCOPE__ {\n";
  EmitDeclarations();
  EmitBroadcastConstants();
  EmitChunkLoop();
  source_ << "  }\n}\n";
}

void VfCceSourceEmitter::EmitPayloadBindings() {
  size_t field = 0;
  for (size_t i = 0; i < program_.partition.inputs.size(); ++i, ++field) {
    const auto type = program_.partition.inputs[i]->type_id_;
    source_ << "  __ubuf__ " << DataTypeCpp(type) << " *in" << i << " = (__ubuf__ " << DataTypeCpp(type) << " *)"
            << program_.payload.DecodeField(field) << ";\n";
  }
  for (size_t i = 0; i < program_.partition.outputs.size(); ++i, ++field) {
    const auto type = program_.partition.outputs[i]->type_id_;
    source_ << "  __ubuf__ " << DataTypeCpp(type) << " *out" << i << " = (__ubuf__ " << DataTypeCpp(type) << " *)"
            << program_.payload.DecodeField(field) << ";\n";
  }
  source_ << "  uint32_t count = static_cast<uint32_t>(" << program_.payload.DecodeField(field) << ");\n";
}

void VfCceSourceEmitter::EmitDeclarations() {
  for (size_t i = 0; i < program_.partition.inputs.size(); ++i) {
    source_ << "    " << VectorTypeCpp(program_.partition.inputs[i]->type_id_) << " vin" << i << ";\n";
  }
  for (size_t i = 0; i < program_.instructions.size(); ++i) {
    const auto &instruction = program_.instructions[i];
    source_ << "    " << VectorTypeCpp(instruction.node->type_id_) << " v" << i << ";\n";
    if (instruction.node->GetObjectType() == kBinaryS &&
        IsScalarDivision(static_cast<BinaryScalarOp *>(instruction.node)->GetOpType())) {
      source_ << "    " << VectorTypeCpp(instruction.node->type_id_) << " vscalar" << i << ";\n";
    }
    if (instruction.node->GetObjectType() == kCast && instruction.inputs[0]->type_id_ == kInt32 &&
        instruction.node->type_id_ == kFloat16) {
      source_ << "    vector_f32 cast_tmp" << i << ";\n";
    }
    if (instruction.node->GetObjectType() == kCompare || instruction.node->GetObjectType() == kCompareS) {
      source_ << "    vector_bool pred" << i << ";\n";
    }
  }
  for (int type = kBool; type < kDataTypeEnd; ++type) {
    if (compare_types_[type]) {
      const auto data_type = static_cast<DataType>(type);
      source_ << "    " << VectorTypeCpp(data_type) << " zero_" << type << ";\n";
      source_ << "    " << VectorTypeCpp(data_type) << " one_" << type << ";\n";
    }
  }
  const char *dist_suffix = program_.max_type_size == sizeof(uint8_t)    ? "B8"
                            : program_.max_type_size == sizeof(uint16_t) ? "B16"
                                                                         : "B32";
  source_ << "    constexpr auto dist_" << program_.mask_suffix
          << " = std::integral_constant<DistVST, DistVST::DIST_NORM_" << dist_suffix << ">();\n";
  source_ << "    constexpr uint32_t sregLower = BlockNum(" << program_.BlockElementType() << ");\n";
}

void VfCceSourceEmitter::EmitBroadcastConstants() {
  for (size_t i = 0; i < program_.instructions.size(); ++i) {
    const auto &instruction = program_.instructions[i];
    if (instruction.node->GetObjectType() != kBinaryS) {
      continue;
    }
    const auto *binary = static_cast<BinaryScalarOp *>(instruction.node);
    if (IsScalarDivision(binary->GetOpType())) {
      source_ << "    vbr(vscalar" << i << ", " << ScalarLiteral(binary->GetScalar(), instruction.node->type_id_)
              << ");\n";
    }
  }
  for (int type = kBool; type < kDataTypeEnd; ++type) {
    if (compare_types_[type]) {
      const auto data_type = static_cast<DataType>(type);
      source_ << "    vbr(zero_" << type << ", (" << DataTypeCpp(data_type) << ")0);\n";
      source_ << "    vbr(one_" << type << ", (" << DataTypeCpp(data_type) << ")1);\n";
    }
  }
}

void VfCceSourceEmitter::EmitChunkLoop() {
  source_ << "    uint32_t remaining = count;\n";
  source_ << "    uint16_t repeat = static_cast<uint16_t>((count + sregLower - 1) / sregLower);\n";
  source_ << "    for (uint16_t i = 0; i < repeat; ++i) {\n";
  source_ << "      vector_bool " << program_.mask_name << " = plt_" << program_.mask_suffix
          << "(remaining, POST_UPDATE);\n";
  EmitLoads();
  EmitInstructions();
  EmitStores();
  source_ << "    }\n";
}

void VfCceSourceEmitter::EmitLoads() {
  for (size_t i = 0; i < program_.partition.inputs.size(); ++i) {
    source_ << "      vlds(vin" << i << ", in" << i << ", i * sregLower, "
            << LoadDistribution(program_.partition.inputs[i]->type_id_, program_.max_type_size) << ");\n";
  }
}

void VfCceSourceEmitter::EmitInstructions() {
  VfCceInstructionEmitter instruction_emitter(source_, program_, values_);
  for (size_t i = 0; i < program_.instructions.size(); ++i) {
    instruction_emitter.Emit(i, program_.instructions[i]);
  }
}

void VfCceSourceEmitter::EmitStores() {
  for (size_t i = 0; i < program_.partition.outputs.size(); ++i) {
    auto *output = program_.partition.outputs[i];
    source_ << "      vsts(" << values_.at(output) << ", out" << i << ", i * sregLower"
            << ", " << StoreDistribution(output->type_id_, program_.max_type_size) << ", " << program_.mask_name
            << ");\n";
  }
}

std::string BuildVfBundleCceSource(const std::vector<VfCompileUnit> &units, const std::string &bundle_namespace) {
  std::string source;
  bool emit_preamble = true;
  for (const auto &unit : units) {
    const VfProgram program(*unit.partition, unit.nspace);
    source += VfCceSourceEmitter(program, bundle_namespace).Emit(emit_preamble);
    emit_preamble = false;
  }
  return source;
}

VfFusionCompiler::FunctionMap ResolveFunctions(const std::vector<VfCompileUnit> &units,
                                               const std::string &registration_namespace, bool is_bundle) {
  VfFusionCompiler::FunctionMap functions;
  for (const auto &unit : units) {
    const auto full_name = is_bundle ? registration_namespace + "/" + unit.nspace : unit.nspace + "/VfFusionOp";
    functions.emplace(unit.nspace, g_system.GetCustomFunc(full_name));
  }
  return functions;
}

void CompileVfSource(const std::string &source_path, const std::string &binary_path) {
  RunCommand(
    {
      "ccec",
      "-c",
      "-O2",
      "--std=c++17",
      "-Wno-int-to-pointer-cast",
      "--cce-aicore-only",
      "--cce-auto-sync=off",
      "--cce-simd-vf-fusion=true",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-or-combine=false",
      "-mllvm",
      "-instcombine-code-sinking=false",
      "-mllvm",
      "-cce-aicore-jump-expand=true",
      "-mllvm",
      "-cce-aicore-mask-opt=false",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "--cce-aicore-arch=dav-c310-vec",
      source_path,
      "-o",
      binary_path,
    },
    "device VF JIT compile failed");
}

VfFusionCompiler::FunctionMap VfFusionCompiler::CompileAndRegister(const std::vector<VfPartition> &partitions) {
  if (partitions.empty()) {
    return {};
  }

  const auto units = BuildCompileUnits(partitions);
  const bool is_bundle = partitions.size() > 1;
  const auto registration_namespace = is_bundle ? HashBundleNamespace(units) : units.front().nspace;
  const auto cache_root = FindCacheRoot();
  const std::string cache_key = registration_namespace + "|" + cache_root.string();
  static std::mutex mutex;
  static std::unordered_set<std::string> cache;
  std::lock_guard<std::mutex> lock(mutex);

  if (cache.count(cache_key) != 0) {
    return ResolveFunctions(units, registration_namespace, is_bundle);
  }

  std::error_code fs_error;
  std::filesystem::create_directories(cache_root, fs_error);
  if (fs_error) {
    DvmException("cannot create VF JIT cache directory");
  }

  const auto prefix = cache_root / registration_namespace;
  const auto cce_path = prefix.string() + ".cce";
  const auto bin_path = prefix.string() + ".o";
  const auto source = is_bundle ? BuildVfBundleCceSource(units, registration_namespace)
                                : BuildVfCceSource(*units.front().partition, units.front().nspace);
  if (!WriteFile(cce_path, source)) {
    DvmException("cannot write VF JIT source files");
  }
  CompileVfSource(cce_path, bin_path);
  g_system.RegCustom(registration_namespace, bin_path);
  cache.insert(cache_key);
  return ResolveFunctions(units, registration_namespace, is_bundle);
}

std::string VfFusionCompiler::NamespaceFor(const VfPartition &partition) { return HashNamespace(partition); }

bool VfFusionCompiler::WriteFile(const std::string &path, const std::string &contents) {
  std::ofstream stream(path, std::ios::trunc);
  if (!stream.is_open()) {
    return false;
  }
  stream << contents;
  return stream.good();
}

bool ValidateCustomMetadata(NDObject *custom, const VfPartition &partition) {
  if (custom == nullptr || custom->GetObjectType() != kCustom || partition.outputs.empty() ||
      custom->type_id_ != partition.outputs.front()->type_id_ || !SameShapeAndNdd(custom, partition.outputs.front())) {
    return false;
  }
  const auto custom_inputs = GetInputs(custom);
  if (custom_inputs.size() != partition.inputs.size() ||
      !std::equal(custom_inputs.begin(), custom_inputs.end(), partition.inputs.begin())) {
    return false;
  }
  auto *custom_op = static_cast<CustomOp *>(custom);
  const size_t expected_extra_outputs = partition.outputs.size() - 1;
  if (expected_extra_outputs == 0) {
    return custom_op->xout_ == nullptr;
  }
  if (custom_op->xout_ == nullptr || custom_op->xout_->out_num != static_cast<int>(expected_extra_outputs)) {
    return false;
  }
  for (size_t i = 0; i < expected_extra_outputs; ++i) {
    auto *ext_output = custom_op->xout_->data[i];
    auto *old_output = partition.outputs[i + 1];
    if (ext_output == nullptr || ext_output->GetObjectType() != kExtOut || ext_output->lhs_ != custom ||
        ext_output->type_id_ != old_output->type_id_ || !SameShapeAndNdd(ext_output, old_output)) {
      return false;
    }
  }
  return true;
}

void DeleteUnownedCustom(NDObject *custom) {
  if (custom == nullptr) {
    return;
  }
  if (custom->GetObjectType() == kCustom) {
    auto *custom_op = static_cast<CustomOp *>(custom);
    if (custom_op->xout_ != nullptr) {
      for (int i = 0; i < custom_op->xout_->out_num; ++i) {
        delete custom_op->xout_->data[i];
      }
    }
  }
  delete custom;
}

bool RewritePartition(BasicBlock &bb, const VfPartition &partition, uint64_t func_id) {
  if (!IsShadowAcyclic(bb, partition)) {
    return false;
  }

  NDObject *custom = nullptr;
  try {
    custom = new VfFusionOp(partition, func_id);
    std::vector<NDObject *> generated_ops;
    custom->Normalize(generated_ops);
    if (!generated_ops.empty() || !ValidateCustomMetadata(custom, partition)) {
      DeleteUnownedCustom(custom);
      return false;
    }
  } catch (const std::exception &) {
    DeleteUnownedCustom(custom);
    return false;
  }

  auto *custom_op = static_cast<CustomOp *>(custom);
  std::vector<NDObject *> replacements = {custom};
  if (custom_op->xout_ != nullptr) {
    for (int i = 0; i < custom_op->xout_->out_num; ++i) {
      replacements.push_back(custom_op->xout_->data[i]);
    }
  }
  if (replacements.size() != partition.outputs.size()) {
    DeleteUnownedCustom(custom);
    return false;
  }

  // All fallible metadata checks have completed.  From here mutation is a
  // single graph transaction whose shadow graph was already Kahn-validated.
  auto *insertion_point = FindSafeInsertionPoint(bb, partition);
  if (insertion_point == nullptr) {
    DeleteUnownedCustom(custom);
    return false;
  }
  NDObject *insert = bb.Insert(insertion_point, custom);
  for (size_t i = 1; i < replacements.size(); ++i) {
    NDObject *next = bb.Next(insert);
    insert = bb.Insert(next, replacements[i]);
  }

  NodeSet members(partition.nodes.begin(), partition.nodes.end());
  for (size_t output_index = 0; output_index < partition.outputs.size(); ++output_index) {
    auto *old = partition.outputs[output_index];
    auto *replacement = replacements[output_index];
    const auto users = bb.GetUsers(old);
    for (auto *user : users) {
      if (members.count(user) != 0) {
        continue;
      }
      const auto occurrences = CountInputOccurrences(user, old);
      for (size_t occurrence = 0; occurrence < occurrences; ++occurrence) {
        bb.UpdateInput(user, old, replacement);
        bb.AddUser(replacement, user);
      }
    }
  }
  for (auto it = partition.nodes.rbegin(); it != partition.nodes.rend(); ++it) {
    bb.Erase(*it);
  }
  return true;
}

}  // namespace detail

VfPayloadLayout::VfPayloadLayout(size_t input_count, size_t output_count)
    : field_count(input_count + output_count + 1),
      word_count((field_count + detail::kPayloadSlotsPerWord - 1) / detail::kPayloadSlotsPerWord),
      extra_code_bytes(word_count > 2 ? (word_count - 2) * sizeof(uint64_t) : 0) {}

VfPayloadLayout::VfPayloadLayout(const VfPartition &partition)
    : VfPayloadLayout(partition.inputs.size(), partition.outputs.size()) {}

std::string VfPayloadLayout::DecodeField(size_t field) const { return detail::PayloadFieldExpr(field); }

std::string BuildVfCceSource(const VfPartition &partition, const std::string &nspace) {
  const detail::VfProgram program(partition, nspace);
  return detail::VfCceSourceEmitter(program, nspace).Emit(true);
}

std::vector<VfPartition> VfCapabilityPartitioner::Propose() {
  auto topo_order = detail::StableKahnOrder(bb_);
  if (!topo_order.has_value()) {
    return {};
  }
  const auto &objects = *topo_order;
  detail::NodeIndex indices;
  for (size_t i = 0; i < objects.size(); ++i) {
    indices.emplace(objects[i], i);
  }

  std::unordered_map<NDObject *, size_t> assignment;
  std::vector<detail::WorkingPartition> partitions;
  for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
    auto *node = *it;
    if (!detail::IsCapabilityNode(node)) {
      continue;
    }
    const size_t base = partitions.size();
    partitions.push_back({true, {node}});
    assignment[node] = base;

    std::vector<size_t> merge_candidates;
    for (auto *user : bb_.GetUsers(node)) {
      auto user_partition = assignment.find(user);
      if (user_partition != assignment.end() && user_partition->second != base &&
          partitions[user_partition->second].alive) {
        merge_candidates.push_back(user_partition->second);
      }
    }
    std::sort(merge_candidates.begin(), merge_candidates.end(), [&](size_t lhs, size_t rhs) {
      return indices.at(partitions[lhs].nodes.front()) < indices.at(partitions[rhs].nodes.front());
    });
    merge_candidates.erase(std::unique(merge_candidates.begin(), merge_candidates.end()), merge_candidates.end());
    for (size_t other : merge_candidates) {
      if (!partitions[other].alive) {
        continue;
      }
      detail::NodeSet merged(partitions[base].nodes.begin(), partitions[base].nodes.end());
      merged.insert(partitions[other].nodes.begin(), partitions[other].nodes.end());
      if (detail::WouldCreateCycle(bb_, merged, assignment, partitions)) {
        continue;
      }
      partitions[base].nodes.insert(partitions[base].nodes.end(), partitions[other].nodes.begin(),
                                    partitions[other].nodes.end());
      for (auto *merged_node : partitions[other].nodes) {
        assignment[merged_node] = base;
      }
      partitions[other].alive = false;
    }
  }

  std::vector<VfPartition> result;
  for (auto &partition : partitions) {
    if (partition.alive) {
      result.push_back(detail::BuildPartition(bb_, std::move(partition.nodes), indices));
    }
  }
  std::sort(result.begin(), result.end(),
            [](const VfPartition &lhs, const VfPartition &rhs) { return lhs.first_index < rhs.first_index; });
  return result;
}

std::vector<VfPartition> SplitOverLimit(const VfPartition &, const VfFusionLimits &) {
  // The interface intentionally exists now, but OverflowPolicy::kSkip is the
  // only enabled behavior for this first implementation.
  static_assert(detail::kOverflowPolicy == VfOverflowPolicy::kSkip);
  return {};
}

void VfFusion(BasicBlock &bb) {
  if (!g_system.vf_fusion_ || !detail::IsAcyclic(bb)) {
    return;
  }

  VfCapabilityPartitioner partitioner(bb);
  auto partitions = partitioner.Propose();
  partitions.erase(std::remove_if(partitions.begin(), partitions.end(), [&](const VfPartition &partition) {
                     return !detail::WithinLimits(bb, partition) || !detail::IsShadowAcyclic(bb, partition);
                   }),
                   partitions.end());

  detail::VfFusionCompiler::FunctionMap functions;
  try {
    detail::VfFusionCompiler compiler;
    functions = compiler.CompileAndRegister(partitions);
  } catch (...) {
    return;
  }

  // Process consumers first.  This makes a later producer replacement update
  // the already-created downstream Custom input through BasicBlock edges.
  for (auto it = partitions.rbegin(); it != partitions.rend(); ++it) {
    const auto function = functions.find(detail::VfFusionCompiler::NamespaceFor(*it));
    if (function == functions.end()) {
      continue;
    }
    detail::RewritePartition(bb, *it, function->second);
  }
}

}  // namespace dvm::pass
