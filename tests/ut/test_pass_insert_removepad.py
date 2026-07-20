# Copyright 2024 Huawei Technologies Co., Ltd
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
@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_remove_pad_01(type):
    t = Tester()
    a = np.random.normal(0, 1, (32, 26, 26, 3, 1, 2)).astype(type)
    b = np.random.normal(0, 1, (32, 26, 26, 3, 1, 2)).astype(type)
    c = np.random.normal(0, 1, (32, 1, 1, 1, 50, 2)).astype(type)
    d = np.random.normal(0, 1, (32, 1, 1, 1, 50, 2)).astype(type)
    para766_Parameter_15926 = t.load(a)
    para767_Parameter_15928 = t.load(b)
    para768_Parameter_15930 = t.load(c)
    para769_Parameter_15932 = t.load(d)
    ret = np.maximum(np.minimum(a * 0.5 + b, c * 0.5 + d) - np.maximum(b - 0.5 * a, d - 0.5 * c), 0)
    y0 = t.mul(para766_Parameter_15926, 0.5)
    y1 = t.add(para767_Parameter_15928, y0)
    y2 = t.mul(para768_Parameter_15930, 0.5)
    y3 = t.add(para769_Parameter_15932, y2)
    y4 = t.minimum(y1, y3)
    y5 = t.sub(para767_Parameter_15928, y0)
    y6 = t.sub(para769_Parameter_15932, y2)
    y7 = t.maximum(y5, y6)
    y8 = t.sub(y4, y7)
    y9 = t.maximum(y8, 0)
    t.store_expect(y9, ret)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_02():
    t = Tester()
    a = np.random.normal(0, 1, (32, 1)).astype(np.float32)
    b = np.random.normal(0, 1, (32, 44)).astype(np.float32)
    x = t.load(a)
    y = t.load(b)
    y0 = t.mul(x, 0.5)
    y1 = t.add(x, y)
    t.store_expect(y0, 0.5 * a)
    t.store_expect(y1, a + b)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_03():
    t = Tester()
    tile_space = 40 + 39
    b = np.full([tile_space, 1, 1], 0.3, np.float32)
    y = t.load(b)
    z = t.broadcast(y, [tile_space, 1, 5])
    z = t.add(z, 0.1)
    z = t.broadcast(z, [tile_space, 32, 5])
    out = t.store_expect(z, 0.4)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_04():
    t = Tester()
    a = np.random.normal(0, 0.1, (20, 4096, 1)).astype(np.float32)
    b = np.random.normal(0, 0.1, (20, 4096, 77)).astype(np.float16)
    c = np.random.normal(0, 0.1, (20, 4096, 77)).astype(np.float16)
    expect = (c - a.astype(np.float16)) * b * 1.2
    a = t.load(a)
    b = t.load(b)
    c = t.load(c)
    y0 = t.cast(a, "float16")
    y1 = t.sub(c, y0)
    y2 = t.mul(b, y1)
    y3 = t.mul(1.2, y2)
    out = t.store_expect(y3, expect)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_05():
    t = Tester("vector:spec")
    shape = [80, 204, 204]
    red_dims = (2,)
    mask = (np.random.uniform(0, 1, shape) > 0.5).astype(np.bool_)
    scalar = np.array(-3.4028234663852886e38, dtype=np.float32)
    inp = np.random.normal(0.0, 1.0, shape).astype(np.float32)

    x_mask = t.load(mask)
    x_scalar = t.load(scalar)
    x_inp = t.load(inp)

    x_expand = t.broadcast(x_scalar, shape)
    x_div = t.div(x_inp, 8.0)
    x_where = t.select(x_mask, x_expand, x_div)
    x_amax = t.max(x_where, red_dims, True)

    t.spec_next()
 
    x_expand_amax = t.broadcast(x_amax, shape)
    x_sub = t.sub(x_where, x_expand_amax)
    x_exp = t.exp(x_sub)
    x_sum = t.sum(x_exp, red_dims, True)

    t.spec_next()

    x_expand_sum = t.broadcast(x_sum, shape)
    x_out = t.div(x_exp, x_expand_sum)


    np_where = np.where(mask, scalar, inp / 8.0)
    np_amax = np.max(np_where, axis=red_dims, keepdims=True)
    np_exp = np.exp(np_where - np_amax)
    np_sum = np.sum(np_exp, axis=red_dims, keepdims=True)
    np_out = np_exp / np_sum

    t.store_expect(x_out, np_out.astype(np.float32), 1e-4)
    t.set_passes("InsertRemovePad")
    
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_multi_user():
    t = Tester()
    a0 = np.random.normal(0, 1, [1, 4096, 4]).astype(np.float32)
    a1 = np.abs(np.random.normal(0, 10, [1, 4096, 1]).astype(np.float32)) + 1e-4
    e1 = a0 / a1
    e2 = e1 * 2.0
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.div(x0, x1)
    t.store_expect(y0, e1, 1e-4)
    y1 = t.mul(y0, 2.0)
    y2 = t.cast(y1, "bfloat16")
    t.store_expect(y2, e2, 1e-2)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_reduce_sum_after_cast():
    t = Tester()
    a = np.random.normal(0, 1, (32, 1, 4096, 1)).astype(np.float16)
    b = np.random.normal(0, 1, (32, 1, 1, 24)).astype(np.float16)
    expect = np.sum((a + b).astype(np.float32), axis=(0,), keepdims=False)
    x0 = t.load(a)
    x1 = t.load(b)
    y0 = t.add(x0, x1)
    y1 = t.cast(y0, "float32")
    y2 = t.sum(y1, (0,), False)
    t.store_expect(y2, expect, 1e-4)
    t.set_passes("InsertRemovePad")
    t.codegen()
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remove_pad_reduce_sum_after_cast_deterministic():
    t = Tester()
    a = np.random.normal(0, 1, (32, 1, 4096, 1)).astype(np.float16)
    b = np.random.normal(0, 1, (32, 1, 1, 24)).astype(np.float16)
    expect = np.sum((a + b).astype(np.float32), axis=(0,), keepdims=False)
    x0 = t.load(a)
    x1 = t.load(b)
    y0 = t.add(x0, x1)
    y1 = t.cast(y0, "float32")
    y2 = t.sum(y1, (0,), False)
    t.store_expect(y2, expect, 1e-4)
    t.set_deterministic(True)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())
    t.set_deterministic(False)
