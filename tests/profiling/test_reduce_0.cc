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
using namespace dvm;

int main() {
  ASCEND_CALL(rtSetDevice(0));
  Tensor<float> a({128, 32, 1}, 2.0f);
  Tensor<float> o1({128, 1, 1});
  Kernel kernel;
  kernel.Reset(kStaticShape);
  auto a_dvm = kernel.Load(a.dev_, a.shape_ref_.get(), DType::kFloat32);

  std::vector<int64_t> reduce_axis{1};
  auto reduce_axis_ref = std::make_shared<ShapeRef>(reduce_axis);
  auto b_dvm = kernel.Reduce(ReduceOpType::kSum, a_dvm, reduce_axis_ref.get(), true);

  (void)kernel.Store(o1.dev_, b_dvm);
  auto workspace = PrepareWorkspace(kernel.CodeGen());
  kernel.Launch(workspace, nullptr);

  ASCEND_CALL(rtStreamSynchronize(nullptr));
  std::cout << kernel.Das() << '\n';
  o1.ToHost();
  assert(allclose(o1.ToData(), 64.0f, o1.size() / sizeof(float), 1e-5f, 1e-5f));
  if (workspace) {
    ASCEND_CALL(rtFree(workspace));
  }
  ASCEND_CALL(rtDeviceReset(0));
}
