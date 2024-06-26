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
import dvm
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
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
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
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
def test_stage_mix_vec():
    t = Tester("stages")
    t.stage_switch("mix");
    ax = np.random.normal(0, 1, [512,512]).astype(np.float16)
    bx = np.random.normal(0, 1, [512,512]).astype(np.float16)
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
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
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
    print(t.dump())
    assert(t.run_check())
