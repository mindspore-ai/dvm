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

import math
import numpy as np
import dvm

def np_attention(q, k_mat, v):
    dk = q.shape[-1]
    scale = 1.0 / math.sqrt(dk)
    scores = np.matmul(q.astype(np.float32), k_mat.astype(np.float32).T) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = np.matmul(probs, v.astype(np.float32))
    return out.astype(np.float16)

def dvm_softmax(k, x, axis):
    max_x = k.max(x, (axis,), True)
    x = k.sub(x, max_x)
    exp_x = k.exp(x)
    denom = k.sum(exp_x, (axis,), True)
    return k.div(exp_x, denom)

@dvm.kernel
def attention_kernel(k, q, k_mat, v, scale):
    q = k.load(q, "float16")
    k_mat = k.load(k_mat, "float16")
    v = k.load(v, "float16")
    scale = k.scalar(scale)
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    probs = dvm_softmax(k, scores, 1)
    out = k.matmul(probs, v, False, False)
    out = k.store(out)
    return out

np.random.seed(0)
q = np.random.normal(0, 0.02, (4, 8)).astype(np.float16)
k_mat = np.random.normal(0, 0.02, (6, 8)).astype(np.float16)
v = np.random.normal(0, 0.02, (6, 12)).astype(np.float16)
scale = 1.0 / math.sqrt(q.shape[-1])
out = attention_kernel(q, k_mat, v, scale)
print("***** output *****")
print(out)
print("***** expect *****")
expect = np_attention(q, k_mat, v)
print(expect)
assert np.allclose(out, expect, rtol=1e-2, atol=1e-2)
