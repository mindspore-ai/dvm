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
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl_rt.h"
#include "dvm.h"

#define ACL_CHECK(func)                                   \
  do {                                                    \
    if ((func) != 0) {                                    \
      std::cerr << "call failed: " << #func << std::endl; \
      exit(-1);                                           \
    }                                                     \
  } while (0)

#define DEVICE_ID  0
#define DATA_SIZE 1024 

int main() {
  // set target ascend device
  ACL_CHECK(aclrtSetDevice(DEVICE_ID));

  // prepare input and output memory
  float *host_x = new float[DATA_SIZE];
  float *host_y = new float[DATA_SIZE];
  float *host_out = new float[DATA_SIZE];
  for (size_t i = 0;  i < DATA_SIZE; ++i) {
    host_x[i] = 2.0f;
    host_y[i] = 3.0f;
  }
  constexpr size_t MEM_SIZE = DATA_SIZE * sizeof(float);
  constexpr aclrtStream stream = nullptr;  // use default stream
  void *dev_x, *dev_y, *dev_out;
  ACL_CHECK(aclrtMalloc(&dev_x, MEM_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK(aclrtMalloc(&dev_y, MEM_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK(aclrtMalloc(&dev_out, MEM_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK(aclrtMemcpyAsync(dev_x, MEM_SIZE, host_x, MEM_SIZE, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK(aclrtMemcpyAsync(dev_y, MEM_SIZE, host_y, MEM_SIZE, ACL_MEMCPY_HOST_TO_DEVICE, stream));

  // define dvm kernel
  dvm::Kernel kernel;
  std::vector<int64_t> shape_data = {1024};
  dvm::IntArrayRef shape_ref(shape_data);
  kernel.Reset(dvm::kVector, 0);
  auto x = kernel.Load(dev_x, &shape_ref, dvm::kFloat32);
  auto y = kernel.Load(dev_y, &shape_ref, dvm::kFloat32);
  auto z = kernel.Binary<dvm::kMul>(x,  y);
  auto r = kernel.Binary<dvm::kAdd>(z,  0.5f);
  auto out = kernel.Store(dev_out, r);

  // codegen and run the kernel
  kernel.CodeGen();
  auto ret = kernel.Launch(nullptr, 0, 0, stream);
  if (ret != 0) {
    std::cerr << "dvm launch failed" << std::endl;
    exit(-1);
  }

  // get output data and check
  ACL_CHECK(aclrtMemcpyAsync(host_out, MEM_SIZE, dev_out, MEM_SIZE, ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
  constexpr float epsilon = 1e-5f;
  constexpr float expect = 6.5f;
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    if (std::abs(host_out[i] - expect) > 1e-5f) {
      std::cerr << "data check failed" << std::endl;
      break;
    }
  }

  // release memory
  ACL_CHECK(aclrtFree(dev_x));
  ACL_CHECK(aclrtFree(dev_y));
  ACL_CHECK(aclrtFree(dev_out));
  delete []host_x;
  delete []host_y;
  delete []host_out;
  return 0;
}