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


def test_remove_pad_04():
    t = Tester()
    a = np.random.normal(0, 1, (20, 4096, 1)).astype(np.float32)
    b = np.random.normal(0, 1, (20, 4096, 77)).astype(np.float16)
    c = np.random.normal(0, 1, (20, 4096, 77)).astype(np.float16)
    expect = (c - a.astype(np.float16))*b*1.2
    a = t.load(a)
    b = t.load(b)
    c = t.load(c)
    y0 = t.cast(a, "float16")
    y1 = t.binary("Sub", c, y0)
    y2 = t.binary("Mul", b, y1)
    y3 = t.binary("Mul", 1.2, y2)
    out = t.store_expect(y3, expect)
    t.set_passes("InsertRemovePad")
    assert (t.run_check())
