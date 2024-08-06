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

def test_stage_vec_vec():
    t = Tester("stages")
    t.stage_switch("static");
    ax = np.full([1024, 128], 0.05, np.float32)
    a = t.load(ax)
    b = t.binary("Add", a, 0.2)
    c = t.binary("Mul", a, b)
    d = t.stage_store(c)
    t.stage_switch("static");
    e = t.stage_load(d)
    f = t.binary("Mul", e, 0.7)
    t.store_expect(f, (ax + 0.2)*ax*0.7)
    assert(t.run_check())

@pytest.mark.mix
def test_stage_vec_mix():
    t = Tester("stages")
    t.stage_switch("static");
    ax = np.random.normal(0, 1, [1024,512]).astype(np.float16)
    a = t.load(ax)
    b = t.binary("Add", a, 0.02)
    c = t.binary("Mul", a, b)
    d = t.stage_store(c)
    t.stage_switch("mix");
    e = t.stage_load(d)
    fx = np.random.normal(0, 1, [512,1024]).astype(np.float16)
    f = t.load(fx)
    g = t.matmul(e, f, False, False)
    h = t.cast(g, "float32")
    expect = np.matmul(((ax + 0.02) * ax).astype(np.float32), fx.astype(np.float32))
    t.store_expect(h, expect, 1e-3)
    assert(t.run_check())

@pytest.mark.mix
def test_stage_mix_vec():
    t = Tester("stages")
    t.stage_switch("mix");
    ax = np.random.normal(0, 1, [768,512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512,128*7]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    e = t.stage_store(d)
    t.stage_switch("static");
    f = t.stage_load(e)
    g = t.binary("Add", f, 0.02)
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32)) + 0.02
    t.store_expect(g, expect, 1e-3)
    assert(t.run_check())

def test_stage_inplace_reuse():
    ax = np.full([1024], 0.05, np.float32)
    t = Tester("stages")
    t.stage_switch("static")
    a0 = t.load(ax)
    a1 = t.binary("Add", a0, 0.1)
    a2 = t.stage_store(a1)
    t.stage_switch("static")
    b0 = t.stage_load(a2)
    b1 = t.binary("Add", b0, 0.1)
    b2 = t.stage_store(b1)
    t.stage_switch("static")
    c0 = t.stage_load(b2)
    c1 = t.binary("Add", c0, 0.1)
    c2 = t.store_expect(c1, 0.05 + 0.3)
    assert(t.run_check())

def test_stage_inplace_stage():
    ax = np.full([1024], 0.05, np.float32)
    t = Tester("stages")
    t.stage_switch("static")
    a0 = t.load(ax)
    a1 = t.cast(a0, "float16")
    a2 = t.stage_store(a1)
    t.stage_switch("static")
    b0 = t.stage_load(a2)
    b1 = t.cast(b0, "float32")
    b2 = t.stage_store(b1)
    t.stage_switch("static")
    c0 = t.stage_load(b2)
    c1 = t.cast(c0, "float16")
    c2 = t.stage_store(c1)
    t.stage_switch("static")
    d0 = t.stage_load(c2)
    d1 = t.binary("Add", d0, 0.1)
    d2 = t.stage_store(d1)
    t.stage_switch("static")
    e0 = t.stage_load(d2)
    e1 = t.binary("Add", e0, 0.1)
    e2 = t.stage_store(e1)
    t.stage_switch("static")
    f0 = t.stage_load(e2)
    f1 = t.cast(f0, "float32")
    f2 = t.store_expect(f1, 0.05 + 0.2, 1e-3)
    assert(t.run_check())

@pytest.mark.mix
def test_stage_workspace_reuse():
    t = Tester("stages")
    t.stage_switch("mix");
    ax = np.random.normal(0, 1, [512,512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512,512]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    e = t.stage_store(d)
    t.stage_switch("static")
    f0 = t.stage_load(e)
    f1 = t.cast(f0, "float16")
    f2 = t.stage_store(f1)
    t.stage_switch("static")
    g0 = t.stage_load(f2)
    g1 = t.cast(g0, "float32")
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    g2 = t.store_expect(g1, expect, 1e-2)
    assert(t.run_check())

@pytest.mark.mix
def test_stage_extern_code():
    t = Tester("stages")
    t.stage_switch("mix");
    ax = np.random.normal(0, 1, [512,512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512,512]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.cast(c, "float32")
    e = t.stage_store(d)
    t.stage_switch("static")
    f = t.stage_load(e)
    for i in range(200):
      f = t.binary("Add", f, 0.01)
      f = t.binary("Sub", f, 0.01)
    expect = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    g2 = t.store_expect(f, expect, 1e-2)
    assert(t.run_check())

def test_stage_vec_reduce():
    np.random.seed(1)
    a0 = np.random.normal(-1, 1, [1, 2048, 5120]).astype(np.float32)
    a1 = np.random.normal(-1, 1, [5120]).astype(np.float32)
    e1 = a0 * a0
    e2 = np.sum(e1, (2,), keepdims=True)
    e3 = e2 * 0.000195313
    e4 = e3 + 1e-6
    e5 = 1.0 / np.sqrt(e4)
    e6 = a0 * e5
    e7 = e6 * a1

    t = Tester("stages")
    t.stage_switch("static")
    x0 = t.load(a0, "bfloat16")
    y0 = t.cast(x0, "float32")
    o0 = t.store_expect(y0, a0, 1e-2)
    y1 = t.binary("Mul", y0, y0)
    y2 = t.reduce("sum", y1, (2,), True)
    s2 = t.stage_store(y2)

    t.stage_switch("static")
    l2 = t.stage_load(s2)
    y3 = t.binary("Mul", l2, 0.000195313)
    y4 = t.binary("Add", y3, 1e-6)
    y5 = t.unary("Reciprocal", t.unary("Sqrt", y4))
    o5 = t.store_expect(y5, e5, 1e-2)

    t.stage_switch("static")
    l0 = t.stage_load(o0)
    l5 = t.stage_load(o5)
    y6 = t.binary("Mul", l0, l5)
    x1 = t.load(a1)
    y7 = t.binary("Mul", y6, x1)
    y8 = t.cast(y7, "bfloat16")
    t.store_expect(y8, e7, 1e-2)
    t.set_passes("CompactPeakLiveness", "ReorderLoad", "ReorderStore", "InsertRemovePad")
    assert(t.run_check())
