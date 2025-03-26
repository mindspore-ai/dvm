# Copyright 2025 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from dvm.tester import Tester

import numpy as np


def np_gmm(x, w, b, group_list):
    """
    x: [M, K]，例如 [1024, 1024]
    w: [G, K, N]，例如 [2, 1024, 1024]
    group_list: 长度为 G，每个元素表示 x 在第 0 维要分出去的行块大小

    逻辑：
        out_0 = x[0 : group_list[0]]     @ w[0]  -> 形状 [group_list[0], N]
        out_1 = x[group_list[0]: ... ]  @ w[1]  -> 形状 [group_list[1] - group_list[0], N]
        ...
      将上述结果在 axis=0 上拼接 (concat)，返回整体 [M, N]。
    """
    outs = []
    l = 0
    for i, g in enumerate(group_list):
        # 取第 i 段的若干行做 matmul
        x_chunk = x[l:g].astype(np.float32)  # [g, K]
        w_chunk = w[i].astype(np.float32)              # [K, N]

        # 单段 matmul
        out_i = x_chunk @ w_chunk  # [g, N]
        if b is not None:
            out_i += b[i]
        outs.append(out_i)

        l = g

    # 沿着第 0 维（行）拼接
    out = np.concatenate(outs, axis=0).astype(
        x.dtype)  # 若要最终返回 float16，可 cast 回去
    return out


@pytest.mark.mix
@pytest.mark.parametrize('m, n, k, group_list', [[1024, 4096, 512, [512, 1024]], [2333, 1111, 2222, [777, 2333]], [4096, 4096, 4096,  [10, 256, 3000, 4096]]])
def test_gmm(m, n, k, group_list):
    b = len(group_list)
    x_shape = [m, k]
    w_shape = [b, k, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm(x, w, None, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, None, group_list_d)
    t.store_expect(res, expect)
    assert (t.run_check())


@pytest.mark.mix
@pytest.mark.parametrize('m, n, k, group_list', [[1024, 4096, 512, [512, 1024]], [4096, 4096, 4096,  [10,  1024, 2048,  4096]]])
def test_gmm_bias(m, n, k, group_list):
    b = len(group_list)
    x_shape = [m, k]
    w_shape = [b, k, n]
    bias_shape = [b, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    bias = np.random.normal(0, 0.01, bias_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm(x, w, bias, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    bias_d = t.load(bias)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, bias_d, group_list_d)
    t.store_expect(res, expect)
    assert (t.run_check())

def test_dyn_gmm():
    t = Tester('dyn_mix')
    x = t.load([-1, 256], "float16")
    w = t.load([4, 256, 2048], "float16")
    group_list = t.load([-1], "int64")
    c = t.gmm(x, w, None, group_list)
    out = t.store(c)
    iterations = [[[256, 256], [4, 256, 2048], [10, 100, 200, 256]], [
        [4096, 256], [4, 256, 2048], [1024, 1256, 2000, 4096]]]
    for x_shape, w_shape, group_list_d in iterations:
        x_data = np.random.normal(0, 0.01, x_shape).astype(np.float16)
        w_data = np.random.normal(0, 0.01, w_shape).astype(np.float16)
        group_list_data = np.array(group_list_d).astype(np.int64)
        expect = np_gmm(x_data, w_data, None, group_list_data)
        t.input(x, x_data)
        t.input(w, w_data)
        t.input(group_list, group_list_data)
        t.run()
        t.check(out, expect, 1e-3)
