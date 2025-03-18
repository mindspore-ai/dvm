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

def test_eager_split_end():
    ''' reduce -> broad '''
    t = Tester("eager")
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([10, 1024], 0.1, np.float32)
    expect = np.sqrt(np.sum((g0 + g1) * 2.0, (0,), keepdims=True) - g2)
    for i in range(3):
        x = t.binary("Add", t.load(g0), t.load(g1))
        x = t.binary("Mul", x, 2.0)
        x = t.reduce("sum", x, [0], True)  # [1, 1024]
        x = t.binary("Sub", x, t.load(g2))
        x = t.unary("Sqrt", x)
        t.store_expect(x, expect)
        assert(t.run_check())
        t.reset_eager()

def test_eager_split_middle():
    ''' {broad -> reduce} broad '''
    t = Tester("eager")
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([32, 512], 0.1, np.float32)
    ex = g0 + 0.1
    ey = (ex + g1) * 0.1
    ez = np.sum(g2 + ex, (1,), keepdims=True)
    for i in range(3):
        x = t.load(g0)
        x = t.binary("Add", x, 0.1)
        y = t.binary("Add", x, t.load(g1))
        z = t.load(g2)
        z = t.binary("Add", z, x)
        z = t.reduce("sum", z, [1], True)
        t.store_expect(z, ez)
        y = t.binary("Mul", y, 0.1)
        t.store_expect(y, ey)
        assert(t.run_check())
        t.reset_eager()

def test_eager_split_join():
    ''' {Broadcast, Reduce} -> broadcast'''
    t = Tester("eager")
    g0 = np.full([32, 1], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    g2 = np.full([32, 512], 0.1, np.float32)
    ex = g0 + 0.1
    ey = ex + g1
    ez = np.sum(g2 + ex, (1,), keepdims=True)
    es = (ey + ez) * 0.1
    for i in range(3):
        x = t.load(g0)
        x = t.binary("Add", x, 0.1)
        y = t.load(g1)
        y = t.binary("Add", x, y)
        z = t.load(g2)
        z = t.binary("Add", z, x)
        z = t.reduce("sum", z, [1], True) # [31, 1]
        s = t.binary("Add", z, y)
        s = t.binary("Mul", s, 0.1)
        s = t.store_expect(s, es)
        assert(t.run_check())
        t.reset_eager()

@pytest.mark.parametrize('shape1, shape2', [
    {(1, 1280, 28, 36), (1, 1280, 1, 1)}, # broadcast
    {(1, 8, 28, 512), (1, 8, 1, 1)}, # broadcast store
    {(1, 1000, 28, 128), (1, 1000)}, # not affine
    ])
def test_eager_disconnect(shape1, shape2):
    x0 = np.random.normal(0, 1, shape1).astype(np.float16)
    x1 = np.random.normal(0, 1, shape1).astype(np.float16)
    x2 = np.random.normal(0, 1, shape2).astype(np.float16)
    x3 = np.random.normal(0, 1, shape2).astype(np.float16)
    t = Tester("eager")
    a = t.binary("Add", t.load(x0), t.load(x1))
    b = t.binary("Add", t.load(x2), t.load(x3))
    t.store_expect(a, x0 + x1)
    t.store_expect(b, x2 + x3)
    assert(t.run_check())
    t.reset_eager()

def test_eager_dead_node():
    t = Tester("eager")
    g0 = np.full([32, 512], 0.1, np.float32)
    g1 = np.full([32, 512], 0.2, np.float32)
    g2 = np.full([32, 512], 0.3, np.float32)
    x0 = t.load(g0)
    x1 = t.load(g1)
    x2 = t.binary('Sub', x0, x1) #dead
    x3 = t.load(g2)
    x4 = t.binary("Add", x3, 0.1)
    x5 = t.binary("Mul", x4, 2.0)
    t.store_expect(x5, (0.3 + 0.1) * 2.0)
    assert(t.run_check())
    assert(t.das().count("load.") == 1)
    t.reset_eager()

def test_eager_split_load():
    t = Tester("eager")
    g0 = np.random.normal(0, 1, (4, 32)).astype(np.float32)
    g1 = np.random.normal(0, 1, [10, 4, 32]).astype(np.float32)
    g2 = np.random.normal(0, 1, [12, 4, 32]).astype(np.float32)
    x0 = t.load(g0)
    x1 = t.load(g1)
    x2 = t.load(g2)
    x3 = t.binary("Add", x0, x1)
    x4 = t.binary("Add", x0, x2)
    t.store_expect(x3, g0 + g1)
    t.store_expect(x4, g0 + g2)
    assert(t.run_check())
    t.reset_eager()

def test_eager_split_load_reduce():
    t = Tester("eager")
    g0 = np.random.normal(0, 1, (1, 4, 1024)).astype(np.float32)
    g1 = np.random.normal(0, 1, [1, 1, 1024]).astype(np.float32)
    g2 = np.random.normal(0, 1, [12, 4, 1024]).astype(np.float32)
    x0 = t.binary("Add", t.load(g0), t.load(g1))
    x1 = t.reduce("sum", t.load(g2), (0,), True)
    t.store_expect(x0, g0 + g1)
    t.store_expect(x1, np.sum(g2, axis=(0,), keepdims=True))
    assert(t.run_check())
    t.reset_eager()

def test_eager_stop_fuse_reduce():
    t = Tester("eager")
    x0_a = np.random.normal(0, 0.1, [428, 16, 8]).astype(np.float32)
    x0 = t.load(x0_a)
    x1 = t.unary('Abs', x0)
    x2 = t.reduce('sum', x1, (2,), False)
    x3_a = np.random.normal(0, 0.1, (1,)).astype(np.float32)
    x3 = t.load(x3_a)
    x4 = t.binary('Mul', x2, x3)
    x5_a = np.random.normal(0, 0.1, [428, 16]).astype(np.float32)
    x5 = t.load(x5_a)
    x6 = t.binary('Add', x5, x4)
    y7_numpy = np.add(x5_a, np.multiply(np.sum(np.abs(x0_a), axis=(2,), keepdims=False), x3_a))
    y7 = t.store_expect(x6, y7_numpy)
    assert(t.run_check())
    t.reset_eager()

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[128, 1024], [1024, 256]],  # no pad
    [[128, 500], [500, 256]],    # pad a
    [[128, 1024], [1024, 500]],  # pad b
    [[4096, 34816], [34816, 2560]],# split k
    [[1024, 1, 256], [256, 512]],# batch fold
])
def test_eager_mm(shape_a, shape_b):
    t = Tester("eager")
    a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    x0 = t.load(a)
    x0 = t.binary("Add", x0, 0.01)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.binary("Add", x2, 1.0)
    expect = np.matmul((a + 0.01).astype(np.float32), b.astype(np.float32)) + 1.0
    t.store_expect(x3, expect, 2e-3)
    assert(t.run_check())

def test_eager_mm_bias():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    c = np.random.normal(0, 0.01, [256]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.load(c)
    x3 = t.matmul(x0, x1, False, False, x2)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32)) + c
    t.store_expect(x3, expect, 2e-3)
    assert(t.run_check())

def test_eager_mm_bias_fp16():
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float32)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float32)
    c = np.random.normal(0, 0.01, [256]).astype(np.float32)
    x0 = t.load(a, "bfloat16")
    x1 = t.load(b, "bfloat16")
    x2 = t.load(c, "bfloat16")
    x3 = t.matmul(x0, x1, False, False, x2)
    expect = np.matmul(a, b) + c
    t.store_expect(x3, expect, 5e-3)
    assert(t.run_check())

def test_eager_kernel_reuse():
    ''' 2 vec -> 1 cube -> 1 vec '''
    t = Tester("eager")
    # round 1
    a = np.random.normal(0, 0.1, [256, 2048]).astype(np.float32)
    x = t.reduce("sum", t.load(a), (0,), True)
    x = t.binary("Add", x, 0.2)
    t.store_expect(x, np.sum(a, (0,), keepdims=True) + 0.2)
    assert(t.run_check())
    t.reset_eager()
    # round 2
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x = t.matmul(t.load(a), t.load(b), False, False)
    t.store_expect(x, np.matmul(a.astype(np.float32), b.astype(np.float32)), 2e-3)
    assert(t.run_check())
    t.reset_eager()
    # round 3
    a = np.random.normal(0, 0.1, [256, 512]).astype(np.float32)
    x = t.binary("Add", t.load(a), 0.2)
    x = t.binary("Add", x, 0.2)
    t.store_expect(x, a + 0.4)
    assert(t.run_check())
    t.reset_eager()

def test_eager_extern_code():
    t = Tester("eager")
    ax = np.random.normal(0, 0.1, [512, 512]).astype(np.float32)
    bx = np.full([1, 512], 0.01, np.float32)
    a = t.load(ax)
    b = t.load(bx)
    for i in range(200):
        a = t.binary("Add", a, b)
    t.store_expect(a, ax + 0.01 * 200)
    assert(t.run_check())

def test_eager_workspace_inplace():
    t = Tester("eager")
    a = np.random.normal(0, 0.1, [128, 1024]).astype(np.float16)
    x0 = t.cast(t.load(a), "float32")
    x1 = t.reduce("sum", x0, (0,), False)
    x2 = t.cast(x1, "float16")
    t.store_expect(x2, np.sum(a.astype(np.float32), axis=(0,), keepdims=False).astype(np.float16), 2e-3)
    b = np.random.normal(0, 0.1, [1, 1024]).astype(np.float32)
    x3 = t.binary("Mul", t.load(b), x0)
    t.store_expect(x3, b * a.astype(np.float32))
    assert(t.run_check())

def test_eager_load_reuse():
    t = Tester("eager")
    ax = np.random.normal(0, 0.1, [288]).astype(np.float32)
    bx = np.random.normal(0, 0.1, [1]).astype(np.float32)
    x0 = t.load(ax)
    x1 = t.load(bx)
    x2 = t.binary("Mul", x0, x1)
    t.store_expect(x2, ax * bx)
    cx = np.random.normal(0, 0.1, [864, 1152]).astype(np.float32)
    x3 = t.load(cx)
    x4 = t.binary("Add", x3, x1)
    t.store_expect(x4, cx + bx)
    x5 = t.binary("Add", x2, 0.1)
    t.store_expect(x5, ax * bx + 0.1)
    assert(t.run_check())

def test_eager_lazy_tuner():
    Tester.set_online_tuning(True)
    t = Tester("eager")
    shape = [4096, 4096]
    a = np.random.normal(0, 0.01, shape).astype(np.float16)
    b = np.random.normal(0, 0.01, shape).astype(np.float16)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32))
    for i in range(3):
        x0 = t.load(a)
        x1 = t.load(b)
        x2 = t.matmul(x0, x1, False, False)
        t.store_expect(x2, expect, 1e-3)
        t.run_check()
        t.reset_eager()
    Tester.set_online_tuning(False)

def test_eager_pv_1():
    ''' parallel: {V, V} '''
    t = Tester("eager")
    g0 = np.full([32, 1024], 0.1, np.float32)
    g1 = np.full([32, 1024], 0.1, np.float32)
    x1 = t.binary("Add", t.load(g0), 1.0)
    x2 = t.binary("Add", t.load(g1), 2.0)
    t.store_expect(x1, 1.1)
    t.store_expect(x2, 2.1)
    assert(t.run_check())

def test_eager_pv_2():
    ''' parallel: V -> {V, V} '''
    t = Tester("eager")
    g0 = np.full([10, 1024], 0.1, np.float32)
    g1 = np.full([16, 1024], 0.2, np.float32)
    g2 = np.full([17, 1024], 0.3, np.float32)
    x0 = t.load(g0)
    x1 = t.reduce("sum", x0, (0,), True)
    x2 = t.binary("Add", t.load(g1), x1)
    x3 = t.binary("Add", t.load(g2), x1)
    t.store_expect(x2, 1.2)
    t.store_expect(x3, 1.3)
    assert(t.run_check())

def test_eager_pv_3():
    ''' parallel: V -> {V, V} -> V '''
    t = Tester("eager")
    g0 = np.full([4, 5, 3,1024], 0.1, np.float32)
    x0 = t.load(g0)
    x1 = t.reduce("sum", x0, (0,), True)
    x2 = t.reduce("sum", x1, (1,), True)
    x3 = t.reduce("sum", x1, (2,), True)
    x4 = t.binary("Add", x2, x3)
    t.store_expect(x4, 3.2)
    assert(t.run_check())

@pytest.mark.mix
def test_eager_cv_0():
    ''' matmul + area1 + area2 '''
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [256, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 256]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.binary("Add", x2, 0.1)
    x4 = t.binary("Add", x3, 0.1)
    expect_x2 = np.matmul(a, b)
    t.store_expect(x4, expect_x2 + 0.2)
    x5 = t.binary("Sub", x3, 0.05)
    t.store_expect(x5, expect_x2 + 0.05)
    assert(t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('broadcast_dim', [0, 1])
def test_eager_cv_broadcast(broadcast_dim):
    t = Tester("eager")
    a = np.random.normal(0, 0.01, [1024, 512]).astype(np.float16)
    b = np.random.normal(0, 0.01, [512, 1024]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    expect_x2 = np.matmul(a, b)
    c_shape = list(expect_x2.shape)
    c_shape[broadcast_dim] = 1
    c = np.random.normal(0, 0.01, c_shape).astype(np.float16)
    x3 = t.load(c)
    x4 = t.binary("Add", x2, x3)
    t.store_expect(x4, expect_x2 + c)
    assert(t.run_check())
