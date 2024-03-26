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
  Tensor<float> a({7, 116}, 2.0f);
  Tensor<float> b({7, 116}, 9.0f);
  Tensor<float> c({7, 1}, 3.0f);
  Tensor<float> o1({7, 116});
  Tensor<float> o2({7, 116});
  Kernel kernel;
  kernel.Reset(kStaticShape);
  auto a_dvm = kernel.Load(a.dev_, a.shape_ref_.get(), DType::kFloat32);
  auto b_dvm = kernel.Load(b.dev_, b.shape_ref_.get(), DType::kFloat32);
  auto c_dvm = kernel.Load(c.dev_, c.shape_ref_.get(), DType::kFloat32);

  auto d_dvm = kernel.Binary(BinaryOpType::kDiv, b_dvm, c_dvm);
  auto e_dvm = kernel.Binary(BinaryOpType::kAdd, d_dvm, 2);
  auto f_dvm = kernel.Unary(UnaryOpType::kAbs, e_dvm);
  auto g_dvm = kernel.Binary(BinaryOpType::kMul, f_dvm, a_dvm);

  (void)kernel.Store(o1.dev_, d_dvm);
  (void)kernel.Store(o2.dev_, g_dvm);
  kernel.CodeGen();
  kernel.Launch(nullptr);
  ASCEND_CALL(rtStreamSynchronize(nullptr));
  std::cout << kernel.Das() << '\n';
  ASCEND_CALL(rtDeviceReset(0));
}
