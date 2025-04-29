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

#include "profiling.h"
#include "opdev/bfloat16.h"
using namespace op;
using namespace dvm;

int main() {
  ASCEND_CALL(rtSetDevice(0));
  Tensor<bfloat16> a({1024, 1024}, 1.0f);
  Tensor<bfloat16> b({4, 1024, 1024}, 1);
  Tensor<int64_t> c({4}, {256, 512, 768, 1024});
  Tensor<bfloat16> o1({1024, 1024});
  Kernel kernel;
  kernel.Reset(kStaticMix);
  auto a_dvm = kernel.Load(a.dev_, a.shape_ref_.get(), DType::kBFloat16);
  auto b_dvm = kernel.Load(b.dev_, b.shape_ref_.get(), DType::kBFloat16);
  auto c_dvm = kernel.Load(c.dev_, c.shape_ref_.get(), DType::kInt64);

  auto res_dvm = kernel.GroupedMatMul(a_dvm, b_dvm, false, false, nullptr, c_dvm, kSplit_M);

  (void)kernel.Store(o1.dev_, res_dvm);
  auto workspace = PrepareWorkspace(kernel.CodeGen());
  kernel.Launch(workspace, nullptr);

  ASCEND_CALL(rtStreamSynchronize(nullptr));
  std::cout << kernel.Das() << '\n';
  o1.ToHost();
  if (workspace) {
    ASCEND_CALL(rtFree(workspace));
  }
  ASCEND_CALL(rtDeviceReset(0));
}
