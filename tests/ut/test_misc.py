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

@pytest.mark.parametrize('shape1, shape2', [([2, 3, 64], [2, 6, 32]),
  ([1000, 4], [100, 40]), # lead dim
])
def test_reshape(shape1, shape2):
    t = Tester()
    a = np.full(shape1, 0.3, np.float32)
    x = t.load(a)
    y = t.binary("Mul", x, 0.6)
    z = t.reshape(y, shape2)
    r = t.unary("Sqrt", z)
    t.store_expect(r, 0.42426)
    assert(t.run_check())

@pytest.mark.parametrize('shape1, shape2', [([2, 3, 64], [2, 6, 32]), ([128], [128, 1])])
def test_reshape_direct_store(shape1, shape2):
    t = Tester()
    a = np.full(shape1, 0.3, np.float32)
    x = t.load(a)
    y = t.binary("Mul", x, 0.6)
    z = t.reshape(y, shape2)
    t.store_expect(z, 0.18)
    assert(t.run_check())

def test_copy():
    t = Tester()
    a = np.full([4, 32], 0.5, np.float32)
    x = t.load(a)
    y = t.copy(x)
    out = t.store_expect(y, 0.5)
    assert(t.run_check())

def test_event_overflow():
    shape = [512]
    t = Tester()
    loads = []
    x = t.load(np.full(shape, 0.1, np.float32))
    for i in range(16):
        loads.append(t.load(np.full(shape, 0.2, np.float32)))
    result = 0.1
    for i in range(16):
        x = t.binary("Add", x, loads[i])
        result += 0.2
        t.store_expect(x, result)
    assert(t.run_check())
