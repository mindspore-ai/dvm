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

@pytest.mark.parametrize("mask",[1, 0])
@pytest.mark.parametrize("shape",[(1024, 32), (127), (100, 100) ,(23, 23)])
def test_element_any(mask, shape):
    t = Tester()
    a = np.full(shape, mask, np.float32)
    x = t.load(a)
    z = t.element_any(x)
    b = t.store(z)
    t.clear_store_memory(b)
    t.run()
    assert t.output(b)[0] == mask

def test_element_any_01():
    t = Tester()
    a = np.full([127, 127], 0, np.float32)
    a[11, 11] = 1
    x = t.load(a)
    g = t.unary("Abs", x)
    z = t.element_any(g)
    b = t.store(z)
    t.clear_store_memory(b)
    t.run()
    assert t.output(b)[0] == 1

@pytest.mark.parametrize("shape, tile",[((1001,), 32), ((3184,), 33), ((5231,), 33)])
def test_element_any_02(shape, tile):
    t = Tester()
    a = np.full(shape, 0.0, np.float32)
    x = t.load(a)
    z = t.element_any(x)
    b = t.store(z)
    t.tile(0, 0, tile)
    t.clear_store_memory(b)
    t.run()
    assert t.output(b)[0] == 0

def test_element_any_03():
    t = Tester()
    a = np.full([256, 32, 64], 0, np.bool_)
    a[1,1,1] = 1
    x = t.load(a)
    g = t.cast(x, "float32")
    z = t.element_any(g)
    b = t.store(z)
    t.clear_store_memory(b)
    t.run()
    assert t.output(b)[0] == 1
