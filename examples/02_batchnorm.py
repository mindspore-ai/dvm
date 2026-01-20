# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import dvm

def np_bn(X, gamma, beta):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(var)
    out = gamma * X_norm + beta
    return out

@dvm.kernel
def bn_kernel(k, x, gamma, beta, rec_batch):
    x = k.load(x, dvm.float32)
    mean_sum = k.sum(x, (0,), False)
    rec_batch = k.scalar(rec_batch)
    mean = k.mul(mean_sum, rec_batch)

    x_sub = k.sub(x, mean)
    var_mul = k.mul(x_sub, x_sub)
    var_sum = k.sum(var_mul, (0,), False)
    var = k.mul(var_sum, rec_batch)

    norm_sqrt = k.sqrt(var)
    x_norm = k.div(x_sub, norm_sqrt)

    gamma = k.load(gamma, dvm.float32)
    beta = k.load(beta, dvm.float32)
    out_mul = k.mul(gamma, x_norm)
    out_add = k.add(out_mul, beta)
    out = k.store(out_add)
    return out

def my_bn(x, gamma, beta, batch_size):
    return bn_kernel(x, gamma, beta, 1.0 / batch_size)

input_x = np.random.normal(0, 1, (32, 1000)).astype(np.float32)
gamma = np.random.normal(0, 1, [1000]).astype(np.float32)
beta = np.random.normal(0, 1, [1000]).astype(np.float32)
out = my_bn(input_x, gamma, beta, 32)
print("***** output *****")
print(out)
print("***** expect *****")
expect = np_bn(input_x, gamma, beta)
print(expect)
assert(np.allclose(out, expect , rtol=1e-3, atol=1e-3))
