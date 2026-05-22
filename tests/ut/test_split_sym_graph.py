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
import pytest
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_vec():
    t = Tester("split:priv1")
    in_shape, red_dims = [10, 4096], (0,)
    a = np.random.normal(0, 0.1, in_shape).astype(np.float16)
    x = t.load(a)
    x = t.add(x, 0.1)
    t.store_expect(x, a + 0.1)
    x = t.cast(x, "float32")
    x0 = t.sum(x, red_dims, True)
    x = t.cast(x0, "float16")
    out1 = t.mul(x, 1.5)
    b = np.random.normal(0, 0.1, in_shape).astype(np.float32)
    out2 = t.add(x0, t.load(b))
    res_x0 = np.sum((a + 0.1).astype(np.float32), red_dims, keepdims=True)
    t.store_expect(out2, res_x0 + b)
    t.store_expect(out1, res_x0.astype(np.float16) * 1.5)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_vec_mix():
    t = Tester("split:priv1")
    x_data = np.random.normal(0, 0.1, [1000, 3333]).astype(np.float16)
    y_data = np.random.normal(0, 0.1, [3333, 1024]).astype(np.float16)
    x = t.load(x_data)
    y = t.load(y_data)
    x = t.mul(x, 0.5)
    x = t.matmul(x, y, False, False)
    x = t.add(x, 0.1)
    res = np.matmul((x_data * 0.5).astype(np.float32), y_data.astype(np.float32)).astype(np.float16) + 0.1
    t.store_expect(x, res)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_reshape():
    t = Tester("split:priv1")
    a = np.random.normal(0.0, 1.0, [32, 128]).astype(np.float32)
    x = t.load(a)
    y = t.mul(x, 0.6)
    z = t.reshape(y, [2, 16, 128])
    r = t.add(z, 0.1)
    t.store_expect(r, (a * 0.6).reshape([2, 16, 128]) + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_matmul_reduce_shared_rhs():
    t = Tester("split:priv1")
    k, n, m = 1024, 1280, 640
    a_data = np.random.normal(0, 0.02, (k, n)).astype(np.float16)
    b_data = np.random.normal(0, 0.02, (k, m)).astype(np.float16)
    c_data = np.random.normal(0, 0.02, (k, n)).astype(np.float16)
    d_data = np.random.normal(0, 0.02, (k, m)).astype(np.float16)

    a = t.load(a_data)
    b = t.load(b_data)
    c = t.load(c_data)
    d = t.load(d_data)

    lhs = t.maximum(t.add(a, c), 0.0)
    rhs = t.mul(b, d)
    mm = t.cast(t.matmul(lhs, rhs, True, False), "float32")
    bias = t.cast(t.sum(t.cast(rhs, "float32"), (0,), True), "float16")
    out = t.mul(t.maximum(t.add(mm, t.cast(bias, "float32")), 0.0), 0.5)

    lhs_expect = np.maximum(a_data + c_data, 0.0)
    rhs_expect = b_data * d_data
    mm_expect = np.matmul(lhs_expect.T.astype(np.float32), rhs_expect.astype(np.float32))
    bias_expect = np.sum(rhs_expect.astype(np.float32), axis=0, keepdims=True)
    bias_expect = bias_expect.astype(np.float16).astype(np.float32)
    out_expect = np.maximum(mm_expect + bias_expect, 0.0) * 0.5

    t.store_expect(out, out_expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_stage_local_shared_load():
    t = Tester("split:priv1")
    x_data = np.random.normal(0, 0.02, (768, 5)).astype(np.float32)

    x = t.load(x_data)
    x2 = t.mul(x, x)
    x3 = t.mul(x2, x)
    x4 = t.mul(x3, 0.044715)
    x5 = t.add(x, x4)
    x6 = t.mul(x5, -1.5957691216057308)
    x7 = t.exp(x6)
    x8 = t.add(x7, 1.0)
    out = t.div(x, x8)

    out_expect = x_data / (np.exp((x_data + x_data * x_data * x_data * 0.044715) * -1.5957691216057308) + 1.0)
    t.store_expect(out, out_expect, 1e-5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_cross_stage_shared_load():
    t = Tester("split:priv1")
    rng = np.random.default_rng(0)

    x_data = rng.normal(0.0, 0.02, size=(4, 128, 768)).astype(np.float32)
    gamma_data = rng.normal(0.0, 0.02, size=(768,)).astype(np.float32)
    sub_data = rng.normal(0.0, 0.02, size=(4, 128, 768)).astype(np.float32)
    tangents_data = rng.normal(0.0, 0.02, size=(4, 128, 768)).astype(np.float32)
    getitem_176_data = rng.normal(0.0, 0.02, size=(4, 128, 768)).astype(np.float32)
    sqrt_23_data = rng.normal(1.0, 0.02, size=(4, 128, 1)).astype(np.float32)
    sqrt_23_data[0, 0, 0] = 0.0
    expand_48_data = rng.normal(0.0, 0.02, size=(4, 128, 768)).astype(np.float32)
    getitem_174_data = rng.normal(1.0, 0.02, size=(4, 128, 1)).astype(np.float32)
    full_default_12_data = np.array(0.0, dtype=np.float32)

    x = t.load(x_data)
    gamma = t.load(gamma_data)
    sub = t.load(sub_data)
    tangents = t.load(tangents_data)
    getitem_176 = t.load(getitem_176_data)
    sqrt_23 = t.load(sqrt_23_data)
    expand_48 = t.load(expand_48_data)
    getitem_174 = t.load(getitem_174_data)
    full_default_12 = t.load(full_default_12_data)

    neg = t.mul(x, -1.0)
    mul_67 = t.mul(gamma, sub)
    add_104 = t.add(tangents, getitem_176)
    mul_89 = t.mul(sqrt_23, 2.0)
    positive = t.greater(sqrt_23, 0.0)
    div_53 = t.div(expand_48, 768.0)
    div_46 = t.div(mul_67, getitem_174)
    div_50 = t.div(div_46, getitem_174)
    mul_86 = t.mul(neg, div_50)
    sum_4 = t.sum(mul_86, (2,), True)
    div_52 = t.div(sum_4, mul_89)
    where_12 = t.select(positive, div_52, full_default_12)
    mul_90 = t.mul(where_12, 0.002607561929595828)
    mul_91 = t.mul(mul_90, sub)
    add_105 = t.add(add_104, mul_91)
    out = t.add(add_105, div_53)

    neg_expect = -x_data
    mul_67_expect = gamma_data * sub_data
    add_104_expect = tangents_data + getitem_176_data
    mul_89_expect = sqrt_23_data * 2.0
    div_53_expect = expand_48_data / 768.0
    div_46_expect = mul_67_expect / getitem_174_data
    div_50_expect = div_46_expect / getitem_174_data
    mul_86_expect = neg_expect * div_50_expect
    sum_4_expect = np.sum(mul_86_expect, axis=2, keepdims=True)
    div_52_expect = np.divide(sum_4_expect, mul_89_expect, out=np.zeros_like(sum_4_expect), where=mul_89_expect != 0)
    where_12_expect = np.where(sqrt_23_data > 0.0, div_52_expect, full_default_12_data)
    mul_90_expect = where_12_expect * np.float32(0.002607561929595828)
    mul_91_expect = mul_90_expect * sub_data
    out_expect = add_104_expect + mul_91_expect + div_53_expect

    t.store_expect(out, out_expect, 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_dyn_shape():
    t = Tester("split:dyn,priv1")
    x0 = t.load([-1, -2], "float32")
    x1 = t.mul(x0, 0.6)
    x2 = t.load([-3, -1, -2], "float32")
    x3 = t.add(x1, x2)
    o1 = t.store(x3)
    x4 = t.load([-4, -1, -2], "float32")
    x5 = t.sub(x4, x1)
    o2 = t.store(x5)
    t.codegen()
    iterations = [[[256, 128], [4, 256, 128], [3, 256, 128]],
                  [[10, 32], [20, 10, 32], [40, 10, 32]]]
    for x0_shape, x2_shape, x4_shape in iterations:
        x0_data = np.random.normal(0, 0.1, x0_shape).astype(np.float32)
        x2_data = np.random.normal(0, 0.1, x2_shape).astype(np.float32)
        x4_data = np.random.normal(0, 0.1, x4_shape).astype(np.float32)
        t.input(x0, x0_data)
        t.input(x2, x2_data)
        t.input(x4, x4_data)
        t.run()
        assert(t.check(o1, x0_data * 0.6 + x2_data))
        assert(t.check(o2, x4_data - x0_data * 0.6))
