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

def dvm_silu(k, x):
    neg = k.mul(x, -1.0)
    exp_neg = k.exp(neg)
    denom = k.add(exp_neg, 1.0)
    sigmoid = k.div(1.0, denom)
    return k.mul(x, sigmoid)

@dvm.kernel
def matmul_post_fusion(k, a, b, bias, scale, shift):
    a = k.load(a, dvm.float16)
    b = k.load(b, dvm.float16)
    bias = k.load(bias, dvm.float32)
    scale = k.scalar(scale)
    shift = k.scalar(shift)
    out = k.matmul(a, b, False, False)
    out = k.cast(out, dvm.float32)
    out = k.add(out, bias)
    out = k.mul(out, scale)
    out = k.add(out, shift)
    out = dvm_silu(k, out)
    out = k.store(out)
    return out

np.random.seed(1)
m, k_dim, n = 32, 64, 48
a = np.random.normal(0, 0.02, (m, k_dim)).astype(np.float16)
b = np.random.normal(0, 0.02, (k_dim, n)).astype(np.float16)
bias = np.random.normal(0, 0.02, (n,)).astype(np.float32)
scale = 0.5
shift = -0.1
out = matmul_post_fusion(a, b, bias, scale, shift)
print("***** output *****")
print(out)
print("***** expect *****")
expect = np.matmul(a.astype(np.float32), b.astype(np.float32))
expect = expect + bias
expect = expect * scale + shift
expect = expect / (1.0 + np.exp(-expect))
print(expect)
assert np.allclose(out, expect, rtol=1e-2, atol=1e-2)
