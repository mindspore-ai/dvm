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

#ifndef _DVM_G_KERNEL_H_
#define _DVM_G_KERNEL_H_

#include "xkernel.h"

namespace dvm {

struct GraphStage;
class GraphKernel : public StagesKernel {
 public:
  explicit GraphKernel(uint32_t flags = 0);
  void Append(NDObject *op) override;
  uint64_t CodeGen() override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  void Dump(std::ostringstream &oss, const std::string &indent, bool rgraph) override;

 protected:
  std::vector<NDObject *> build_ops_;
  friend class GraphSpliter;
};
}  // namespace dvm
#endif  // _DVM_G_KERNEL_H_