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

def test_multi_in_multi_out():
    np.random.seed(1)
    t = Tester()
    a0 = np.abs(np.random.normal(0, 1, [8, 32000]).astype(np.float32))
    a1 = np.abs(np.random.normal(0, 1, [8, 1]).astype(np.float32)) + 1e-6
    a2 = np.random.normal(0, 1, [8, 32000]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.binary("Div", x0, x1)
    expect0 = a0 / a1
    t.store_expect(y0, expect0)
    y1 = t.binary("Add", y0, 1e-24)
    y2 = t.unary("Log", y1)
    x2 = t.load(a2)
    y3 = t.binary("Mul", y2, x2)
    y4 = t.binary("Mul", y3, -1.0)
    expect1 = -(np.log(expect0 + 1e-24) * a2)
    t.store_expect(y4, expect1)
    assert(t.run_check())

def test_elemwise_reduce():
    t = Tester()
    a = np.full([32, 2048], 0.01, np.float32)
    x = t.load(a)
    x = t.binary("Add", x, x)
    y = t.reduce("sum", x, [1], True)
    t.store_expect(y, 0.02*2048)
    assert(t.run_check())

def test_backward_sync_overlap():
    np.random.seed(1)
    t = Tester()
    a0 = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    a1 = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.load(a1)
    b = t.binary("Sub", x0, x1)
    a = t.binary("Sub", x0, x1)
    f = t.binary("Sub", x1, x0)
    c = t.store(a)
    g = t.store(f)
    h = t.store(b)
    t.run_check()
    assert np.allclose(h, a0-a1, 1e-5, 1e-5)

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
    ret = np.maximum(np.minimum(a*0.5+b,c*0.5+d) - np.maximum(b -0.5*a, d-0.5*c),0)
    y0 = t.binary("Mul",para766_Parameter_15926, 0.5)
    y1 = t.binary("Add",para767_Parameter_15928, y0)
    y2 = t.binary("Mul",para768_Parameter_15930, 0.5)
    y3 = t.binary("Add",para769_Parameter_15932, y2)
    y4 = t.binary("Minimum",y1, y3)
    y5 = t.binary("Sub",para767_Parameter_15928, y0)
    y6 = t.binary("Sub",para769_Parameter_15932, y2)
    y7 = t.binary("Maximum",y5, y6)
    y8 = t.binary("Sub",y4, y7)
    y9 = t.binary("Maximum",y8, 0)
    t.store_expect(y9, ret)
    t.set_passes("InsertRemovePad")
    assert(t.run_check())

def test_remove_pad_02():
    t = Tester()
    a = np.random.normal(0, 1, (32, 1)).astype(np.float32)
    b = np.random.normal(0, 1, (32, 44)).astype(np.float32)
    x = t.load(a)
    y = t.load(b)
    y0 = t.binary("Mul",x, 0.5)
    y1 = t.binary("Add",x, y)
    t.store_expect(y0, 0.5*a)
    t.store_expect(y1, a+b)
    t.set_passes("InsertRemovePad")
    assert(t.run_check())

def test_remove_pad_03():
    t = Tester()
    tile_space = 40+39
    b = np.full([tile_space, 1, 1], 0.3, np.float32)
    y = t.load(b)
    z = t.broadcast(y, [tile_space, 1, 5])
    z = t.binary("Add", z, 0.1)
    z = t.broadcast(z, [tile_space, 32, 5])
    out = t.store_expect(z, 0.4)
    t.set_passes("InsertRemovePad")
    assert(t.run_check())
