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


def np_gmm_split_m(x, w, b, group_list):
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
        x_chunk = x[l:g].astype(np.float32)
        w_chunk = w[i].astype(np.float32)

        out_i = x_chunk @ w_chunk
        if b is not None:
            out_i += b[i]
        outs.append(out_i)

        l = g

    out = np.concatenate(outs, axis=0)
    return out


def np_gmm_split_k(x, w, b, group_list):
    """
    x: numpy 数组，形状为 [M, K]，例如 [1024, 1024]
    w: numpy 数组，形状为 [K, N]，例如 [1024, 1024]
    b: 偏置项，形状为 [B, N]（批次大小 B）
    group_list: 长度为 B，每个元素表示 x 和 w 在 K 维度上的分块大小

    返回:
    result: numpy 数组，形状为 [B, M, N]
    """
    M, K = x.shape
    K_, N = w.shape

    B = group_list.shape[0]
    out = np.zeros((B, M, N))
    l = 0

    for i, g in enumerate(group_list):
        x_block = x[:, l:g].astype(np.float32)
        w_block = w[l:g, :].astype(np.float32)
        out[i] = np.matmul(x_block, w_block)
        if b is not None:
            out[i] += b[i]
        l = g
    return out


@pytest.mark.mix
@pytest.mark.parametrize(
    "m, n, k, group_list",
    [
        [1024, 4096, 512, [512, 1024]],
        [2333, 1111, 2222, [777, 2333]],
        [4096, 4096, 4096, [10, 256, 256, 3000, 4096]],
    ],
)
@pytest.mark.parametrize(
    "trans", [[False, True], [True, True], [False, False], [True, False]]
)
def test_gmm(m, n, k, group_list, trans):
    b = len(group_list)
    x_shape = [k, m] if trans[0] else [m, k]
    w_shape = [b, n, k] if trans[1] else [b, k, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_m(x if not trans[0] else x.T, w if not trans[1] else w.transpose(
        0, 2, 1), None, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, trans[0], trans[1],  None, group_list_d, 0)
    t.store_expect(res, expect)
    assert t.run_check()


@pytest.mark.mix
@pytest.mark.parametrize(
    "m, n, k, group_list",
    [[1024, 4096, 512, [512, 1024]], [4096, 4096, 4096, [10, 1024, 2048, 4096]]],
)
def test_gmm_bias(m, n, k, group_list):
    b = len(group_list)
    x_shape = [m, k]
    w_shape = [b, k, n]
    bias_shape = [b, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    bias = np.random.normal(0, 0.01, bias_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_m(x, w, bias, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    bias_d = t.load(bias)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, False, False, bias_d, group_list_d, 0)
    t.store_expect(res, expect)
    assert t.run_check()


@pytest.mark.mix
@pytest.mark.parametrize("m, n, k, group_list", [[1024, 4096, 512, [512, 1024]]])
def test_gmm_bias_bf16(m, n, k, group_list):
    b = len(group_list)
    x_shape = [m, k]
    w_shape = [b, k, n]
    bias_shape = [b, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float32)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float32)
    bias = np.random.normal(0, 0.01, bias_shape).astype(np.float32)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_m(x, w, bias, group_list)
    t = Tester("mix")
    x_d = t.load(x, "bfloat16")
    w_d = t.load(w, "bfloat16")
    bias_d = t.load(bias, "bfloat16")
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, False, False, bias_d, group_list_d, 0)
    t.store_expect(res, expect, 3e-3)
    assert t.run_check()


def test_dyn_gmm_type0():
    t = Tester("dyn_mix")
    x = t.load([-1, 256], "float16")
    w = t.load([4, 256, 2048], "float16")
    group_list = t.load([-1], "int64")
    c = t.gmm(x, w, False, False, None, group_list, 0)
    out = t.store(c)
    iterations = [
        [[256, 256], [4, 256, 2048], [10, 100, 200, 256]],
        [[4096, 256], [4, 256, 2048], [1024, 1256, 2000, 4096]],
    ]
    for x_shape, w_shape, group_list_d in iterations:
        x_data = np.random.normal(0, 0.01, x_shape).astype(np.float16)
        w_data = np.random.normal(0, 0.01, w_shape).astype(np.float16)
        group_list_data = np.array(group_list_d).astype(np.int64)
        expect = np_gmm_split_m(x_data, w_data, None, group_list_data)
        t.input(x, x_data)
        t.input(w, w_data)
        t.input(group_list, group_list_data)
        t.run()
        assert(t.check(out, expect, 1e-3))


@pytest.mark.mix
@pytest.mark.parametrize(
    "m, n, k, group_list",
    [
        [1024, 4096, 4096, [256, 512]],
        [1024, 4096, 3333, [10, 128, 256, 1024, 2048, 3333]],
    ],
)
@pytest.mark.parametrize(
    "trans", [[False, True], [True, True], [False, False], [True, False]]
)
def test_gmm_type2(m, n, k, group_list, trans):
    x_shape = [k, m] if trans[0] else [m, k]
    w_shape = [n, k] if trans[1] else [k, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_k(
        x if not trans[0] else x.T, w if not trans[1] else w.T, None, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, trans[0], trans[1],  None, group_list_d, 2)
    t.store_expect(res, expect)
    assert t.run_check()


@pytest.mark.mix
@pytest.mark.parametrize(
    "m, n, k, group_list",
    [
        [1024, 512, 3333, [0, 0, 10, 128, 256, 256, 1024, 2048, 2048, 3333]],
    ],
)
def test_gmm_type2_zero(m, n, k, group_list):
    x_shape = [m, k]
    w_shape = [k, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_k(
        x, w, None, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, False, False, None, group_list_d, 2)
    s = t.store_expect(res, expect)
    t.clear_store_memory(s)
    assert t.run_check()


@pytest.mark.mix
@pytest.mark.parametrize(
    "m, n, k, group_list",
    [
        [1280, 2560, 1024, [256, 512, 800, 1024]],
        [4096, 4096, 4096, [10, 256, 3000, 4096]],
    ],
)
def test_gmm_type2_postfusion(m, n, k, group_list):
    b = len(group_list)
    x_shape = [m, k]
    w_shape = [k, n]
    d_shape = [b, m, n]
    x = np.random.normal(0, 0.01, x_shape).astype(np.float16)
    w = np.random.normal(0, 0.01, w_shape).astype(np.float16)
    d = np.random.normal(0, 0.01, d_shape).astype(np.float32)
    group_list = np.array(group_list).astype(np.int64)
    expect = np_gmm_split_k(
        x, w, None, group_list)
    t = Tester("mix")
    x_d = t.load(x)
    w_d = t.load(w)
    d_d = t.load(d)
    group_list_d = t.load(group_list)
    res = t.gmm(x_d, w_d, False, False,  None, group_list_d, 2)
    res = t.cast(res, "float32")
    res = t.binary("Add", d_d, res)
    t.store_expect(res, expect.astype(np.float16).astype(np.float32) + d, 1e-3)
    assert t.run_check()


def test_dyn_gmm_type2():
    t = Tester("dyn_mix")
    x = t.load([4096, -1], "float16")
    w = t.load([-1, 2048], "float16")
    group_list = t.load([-1], "int64")
    c = t.gmm(x, w, False, False, None, group_list, 2)
    c = t.cast(c, "float32")
    c = t.binary("Add", c, 1)
    out = t.store(c)
    iterations = [
        [[4096, 256], [256, 2048], [10, 100, 200, 256]],
        [[4096, 2000], [2000, 2048], [1024, 1256, 1999, 2000]],
        [[4096, 2560], [2560, 2048], [1024, 1256, 2000, 2560]],
    ]
    for x_shape, w_shape, group_list_d in iterations:
        x_data = np.random.normal(0, 0.01, x_shape).astype(np.float16)
        w_data = np.random.normal(0, 0.01, w_shape).astype(np.float16)
        group_list_data = np.array(group_list_d).astype(np.int64)
        expect = np_gmm_split_k(x_data, w_data, None, group_list_data)
        t.input(x, x_data)
        t.input(w, w_data)
        t.input(group_list, group_list_data)
        t.run()
        assert(t.check(out, expect + 1, 1e-3))
