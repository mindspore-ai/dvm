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

@pytest.mark.parametrize("shape1, shape2, shape3, reduce_dims", 
                         [[(10, 3, 1), (10, 3, 4), (1, 10, 12), (2,)],
                          [(10, 3, 1), (10, 3, 4), (1, 10, 12), (1,)]])
def test_reduce_forward(shape1, shape2, shape3, reduce_dims):
    t = Tester()
    b = np.full(shape1, 0.1, np.float32)
    y = t.load(b)
    z = t.broadcast(y, shape2)
    z = t.reshape(z, shape3)
    z = t.reduce("sum", z, reduce_dims, True)
    t.set_passes("EliminateReshape")
    c = np.copy(b)
    c = np.broadcast_to(c, shape2)
    print(c.shape)
    c = np.reshape(c, shape3)
    e = np.sum(c, reduce_dims, keepdims=True)
    t.store_expect(z, e)
    assert(t.run_check())

@pytest.mark.parametrize("shape1", [(16, 16)])
@pytest.mark.parametrize("shape2", [(32, 8)])
def test_unary(shape1, shape2):
    t = Tester()
    x = np.full(shape1, 81.0, np.float32)
    a = t.load(x)
    b = t.unary("Sqrt", a)
    c = t.reshape(b, shape2)
    d = t.unary("Sqrt", c)
    t.store_expect(d, 3.0)
    t.set_passes("EliminateReshape")
    assert (t.run_check())

@pytest.mark.parametrize("shape1", [(16, 16)])
@pytest.mark.parametrize("shape2", [(32, 8)])
def test_cast(shape1, shape2):
    t = Tester()
    x = np.full(shape1, 81.0, np.float32)
    a = t.load(x)
    b = t.unary("Sqrt", a)
    c = t.reshape(b, shape2)
    d = t.unary("Sqrt", c)
    d = t.cast(d, "float16")
    t.store_expect(d, 3.0)
    t.set_passes("EliminateReshape")
    assert (t.run_check())

@pytest.mark.parametrize("shape",[(3, 3)])
@pytest.mark.parametrize("shape2", [(9,)])
@pytest.mark.parametrize('type, eps', [(np.float32, 1e-5)])
def test_select_forward(shape, shape2, type, eps):
    t = Tester()
    a = np.full(shape2, 1.0).astype(type)
    b = np.full(shape, 2.0).astype(type)
    c = np.random.choice([True, False], shape).astype(bool)
    x = t.load(c)
    y = t.load(a)
    y = t.reshape(y, shape)
    z = t.load(b)
    z = t.select(x, y, z)
    t.store_expect_flat(z, np.select([c == True, c == False],[a.reshape(shape), b]), eps)
    t.set_passes("EliminateReshape")
    assert(t.run_check())

def test_broadcast_backward():
    t = Tester()
    b = np.full([1,8], 0.3, np.float32)
    y = t.load(b)
    z = t.broadcast(y, [10, 8])
    z = t.reshape(z, [10, 1, 8])
    z = t.broadcast(z, [10, 4, 8])
    t.set_passes("EliminateReshape")
    t.store_expect(z, 0.3000)
    assert(t.run_check())

def test_broadcast_forward():
    t = Tester()
    b = np.full([10, 3, 4], 0.3, np.float32)
    y = t.load(b)
    z = t.reshape(y, [1, 10, 12])
    z = t.broadcast(z, [8, 10 ,12])
    t.set_passes("EliminateReshape")
    t.store_expect(z, 0.3000)
    assert(t.run_check())

def test_broadcast_s():
    t = Tester()
    z = t.broadcast(0.3, [3, 10], "float32", True)
    z = t.reshape(z, [3, 1, 10])
    z = t.broadcast(z, [3, 10, 10])
    t.set_passes("EliminateReshape")
    t.store_expect(z, 0.3000)
    assert(t.run_check())

def test_multi_reshape_1():
    t = Tester()
    a = np.random.rand(54, 89).astype(np.float32)
    x = t.load(a)
    y = t.reshape(x, [89,54])
    y = t.unary("Sqrt", y)
    z = t.unary("Reciprocal", x)
    z = t.reshape(z, [89,54])
    z = t.binary("Add", y, z)
    expect = np.add(np.sqrt(a), np.reciprocal(a))
    t.store_expect_flat(z, expect)
    t.set_passes("EliminateReshape")
    assert(t.run_check())

def test_multi_reshape_2():
    t = Tester()
    a = np.random.rand(54, 89).astype(np.float32)
    x = t.load(a)
    y = t.reshape(x, [89,54])
    y = t.unary("Sqrt", y)
    z = t.reshape(y, [178, 27])
    t.store_expect_flat(z, np.sqrt(a))
    t.set_passes("EliminateReshape")
    assert(t.run_check())

def test_element_any_forward():
    t = Tester()
    a = np.full([10, 10], 0, np.float32)
    a[1, 1] = 1
    x = t.load(a)
    g = t.unary("Abs", x)
    g = t.reshape(g, [5, 10, 2])
    z = t.element_any(g)
    b = t.store(z)
    t.set_passes("EliminateReshape")
    t.run_check()
    assert b[0] == 1

def test_element_any_backward():
    t = Tester()
    a = np.full([10,8], 0, np.float32)
    a[1, 1] = 1
    y = t.load(a)
    z = t.element_any(y)
    z = t.broadcast(z, [10, 8])
    z = t.reshape(z, [10, 1, 8])
    z = t.broadcast(z, [10, 4, 8])
    t.set_passes("EliminateReshape")
    t.store_expect(z, 1.0)
    assert t.run_check()
