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

#include "acl_ext.h"
#include "dvm.h"
#include <assert.h>
#include <random>
#include <memory>
#include <cstring>
#include <iostream>
#include <functional>
#include <cstdio>
#include <algorithm>
using namespace dvm;

#define ASCEND_CALL(func)                                                                               \
  do {                                                                                                  \
    auto err = (func);                                                                                  \
    if (err != 0) {                                                                                     \
      std::cerr << "Ascend error in function " << #func << " : " << static_cast<int>(err) << std::endl; \
      exit(0);                                                                                          \
    }                                                                                                   \
  } while (0)
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size() - 1; i++) {
    os << vec[i] << ", ";
  }
  os << vec.back() << "]";
  return os;
}

void *PrepareWorkspace(uint64_t workspace_size) {
  void *workspace = nullptr;
  if (workspace_size > 0) {
    ASCEND_CALL(rtMalloc(&workspace, workspace_size, RT_MEMORY_HBM, 0));
  }
  return workspace;
}

template <typename T>
struct Tensor {
  void *host_{nullptr};
  void *dev_{nullptr};
  std::vector<int64_t> shape_;
  std::shared_ptr<dvm::ShapeRef> shape_ref_;

  Tensor(const std::vector<int64_t> shape) : shape_(shape) {
    int64_t size = this->size() + 512;
    host_ = malloc(size);
    std::memset(host_, 0, size);
    shape_ref_ = std::make_shared<dvm::ShapeRef>(shape_);
    ASCEND_CALL(rtMalloc(&dev_, size, RT_MEMORY_HBM, 0));
    ASCEND_CALL(rtMemcpy(dev_, size, host_, size, RT_MEMCPY_HOST_TO_DEVICE));
  }
  Tensor(const std::vector<int64_t> shape, T v) : Tensor(shape) {
    T *data = ToData();
    std::fill(data, data + (this->size() / sizeof(T)), v);
    ToDev();
  }

  void random(int l, int r) {
    std::random_device rd;
    std::mt19937 gen(rd());
    T *data = ToData();
    for (size_t i = 0; i < this->size(); i++) {
      std::uniform_real_distribution<T> distr(l, r);
      data[i] = distr(gen);
    }
  }

  void ToDev() {
    ASCEND_CALL(rtMalloc(&dev_, this->size(), RT_MEMORY_HBM, 0));
    ASCEND_CALL(rtMemcpy(dev_, this->size(), host_, this->size(), RT_MEMCPY_HOST_TO_DEVICE));
  }

  void ToHost() { ASCEND_CALL(rtMemcpy(host_, this->size(), dev_, this->size(), RT_MEMCPY_DEVICE_TO_HOST)); }
  T *ToData() { return (T *)host_; }

  size_t size() { return std::accumulate(shape_.begin(), shape_.end(), sizeof(T), std::multiplies<int64_t>()); }

  ~Tensor() {
    rtFree(dev_);
    free(host_);
  }
};

template <typename T>
bool allclose(const T *a, const T *b, size_t size, T rtol = T(1e-5), T atol = T(1e-8)) {
  for (size_t i = 0; i < size; ++i) {
    if (std::fabs(a[i] - b[i]) > (atol + rtol * std::fabs(b[i]))) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool allclose(const T *a, const T b, size_t size, T rtol = T(1e-5), T atol = T(1e-8)) {
  for (size_t i = 0; i < size; ++i) {
    if (std::fabs(a[i] - b) > (atol + rtol * std::fabs(b))) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &v) {
  T *val = (T *)v.host;
  auto shape = v.shape;
  std::vector<int64_t> suf_sum(shape.size(), 1);
  if (shape.size() >= 2) {
    for (int i = shape.size() - 2; i >= 0; i--) {
      suf_sum[i] = suf_sum[i + 1] * shape[i + 1];
    }
  }
  std::function<void(int, std::vector<int64_t>)> dfs = [&](uint64_t step, std::vector<int64_t> index) {
    if (step == shape.size()) {
      auto n = 0;
      for (size_t i = 0; i < index.size(); i++) {
        n += index[i] * suf_sum[i];
      }
      os << val[n] << ", ";
      return;
    }
    if (step != 0) os << "[";
    for (int i = 0; i < shape[step]; i++) {
      auto p = index;
      p[step] = i;
      dfs(step + 1, p);
    }
    if (step != 0) os << "]\n";
  };
  dfs(0, {0, 0, 0});
  return os;
}