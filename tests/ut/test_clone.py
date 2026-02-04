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
def test_clone_vector():
    t = Tester()
    a = np.random.normal(0, 1, [10, 512]).astype(np.float32)
    ax = t.load(a)
    x = t.add(ax, 0.01)
    x = t.sum(x, [0], True)
    out = t.store_expect(x, np.sum(a + 0.01, (0,), keepdims=True), 1e-4)
    assert (t.run_check())

    t2 = Tester()
    ax2, out2 = t2.clone(t, [ax, out])
    a2 = np.random.normal(0, 1, [10, 512]).astype(np.float32)
    t2.input(ax2, a2)
    t2.run()
    assert(t2.check(out2, np.sum(a2 + 0.01, (0,), keepdims=True), 1e-4))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_clone_vector_dyn():
    t = Tester("vector:dyn")
    ax = t.load([], "float32")
    x = t.add(ax, 0.01)
    dims = t.int_array()
    x = t.sum(x, dims, True)
    out = t.store(x)
    a = np.random.normal(0, 1, [10, 512]).astype(np.float32)
    t.input(ax, a)
    dims.update([0])
    t.run()
    assert(t.check(out, np.sum(a + 0.01, (0,), keepdims=True), 1e-4))

    t2 = Tester("vector:dyn")
    ax2, out2, dims2 = t2.clone(t, [ax, out, dims])
    a2 = np.random.normal(0, 1, [64, 800]).astype(np.float32)
    t2.input(ax2, a2)
    dims2.update([1])
    t2.run()
    assert(t2.check(out2, np.sum(a2 + 0.01, (1,), keepdims=True), 1e-4))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_clone_vector_spec():
    t = Tester("vector:spec")
    a = np.random.normal(0, 1, [10, 6000]).astype(np.float32)
    expect = np.sum(a, (0,), keepdims=True) + 0.01
    ax = t.load(a)
    x = t.sum(ax, [0], True)
    t.spec_next()
    x = t.add(x, 0.01)
    out = t.store_expect(x, expect)
    assert (t.run_check())

    t2 = Tester("vector:spec")
    ax2, out2 = t2.clone(t, [ax, out])
    t2.input(ax2, a)
    t2.run()
    assert(t2.check(out2, expect))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["mix", "mix:dyn"])
def test_clone_mix(mode):
    t = Tester(mode)
    g0 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    g1 = np.random.normal(0, 0.01, [1024, 1536]).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, False, False)
    d = t.add(c, b)
    expect = np.matmul(g0.astype(np.float32), g1.astype(np.float32)).astype(np.float16) + g1
    out = t.store_expect(d, expect)
    assert (t.run_check())

    t2 = Tester(mode)
    a2, b2, out2 = t2.clone(t, [a, b, out])
    t2.input(a2, g0)
    t2.input(b2, g1)
    t2.run()
    assert(t2.check(out2, expect))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_clone_split_dyn():
    t = Tester("split:dyn")
    a = t.load([], "float16")
    b = t.load([], "float16")
    x = t.mul(a, 0.8)
    x = t.matmul(x, b, False, False)
    x = t.cast(x, "float32")
    dims = t.int_array()
    x = t.sum(x, dims, True)
    x = t.mul(x, 0.5)
    out = t.store(x)
    ax = np.random.normal(0, 0.1, [512, 1024]).astype(np.float16)
    bx = np.random.normal(0, 0.1, [1024, 1024]).astype(np.float16)
    t.input(a, ax)
    t.input(b, bx)
    dims.update([0])
    t.run()
    expect_ = np.matmul((ax * 0.8).astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    expect = np.sum(expect_.astype(np.float32), (0,), keepdims=True) * 0.5
    assert(t.check(out, expect, 1e-3))

    t2 = Tester("split:dyn")
    a2, b2, out2, dims2 = t2.clone(t, [a, b, out, dims])
    ax = np.random.normal(0, 0.1, [1024, 1024]).astype(np.float16)
    bx = np.random.normal(0, 0.1, [1024, 768]).astype(np.float16)
    t2.input(a2, ax)
    t2.input(b2, bx)
    dims2.update([1])
    t2.run()
    expect_ = np.matmul((ax * 0.8).astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    expect = np.sum(expect_.astype(np.float32), (1,), keepdims=True) * 0.5
    assert(t2.check(out2, expect, 1e-3))
