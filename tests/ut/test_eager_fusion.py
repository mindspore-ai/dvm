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
        t.run_check()
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
        t.run_check()
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
        t.run_check()
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
    t.run_check()
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
    t.run_check()
    assert(t.das().count("load.") == 1)
    t.reset_eager()
