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

@pytest.mark.parametrize('shape1, shape2,  shape3', [
    ([256], [256], [1024]),
    ([8192*2], [8192*4], [4096])
    ])
def test_basic(shape1, shape2, shape3):
    t = Tester("parallel")
    #kernel 0
    a0 = np.full(shape1, 0.1, np.float32)
    a = t.load(a0)
    b = t.binary("Mul", a, 0.3)
    c = t.store_expect(b, 0.1*0.3)
    #kernel 1
    t.p_next()
    b0 = np.full(shape2, 0.1, np.float32)
    a = t.load(b0)
    b = t.binary("Add", a, 0.3)
    c = t.store_expect(b, 0.1+0.3)
    #kernel 3
    t.p_next()
    c0 = np.full(shape3, 0.1, np.float32)
    a = t.load(c0)
    b = t.binary("Add", a, 0.3)
    c = t.store_expect(b, 0.1+0.3)
    assert(t.run_check())

@pytest.mark.parametrize('determ', [False, True])
def test_reduce(determ):
    t = Tester("parallel")
    t.set_determ(determ)
    # kernel 0
    a0 = np.full((8192,), 0.1, np.float32)
    a = t.reduce("sum", t.load(a0), [0], False)
    t.store_expect(a, 8192 * 0.1)
    # kernel 1
    t.p_next()
    b0 = np.full((4, 4096), 0.1, np.float32)
    b = t.reduce("sum", t.load(b0), [0], False)
    t.store_expect(b, 4 * 0.1)
    # kernel 2
    t.p_next()
    c0 = np.full((1, ), 0.1, np.float32)
    c = t.binary("Add", t.load(c0), 1.0)
    t.store_expect(c, 0.1 + 1.0)
    assert(t.run_check())
    t.set_determ(False)

def test_reduce_determ_all():
    t = Tester("parallel")
    t.set_determ(True)
    # kernel 0
    a0 = np.full((8192,), 0.1, np.float32)
    a = t.reduce("sum", t.load(a0), [0], False)
    t.store_expect(a, 8192 * 0.1)
    # kernel 1
    t.p_next()
    b0 = np.full((4, 4096), 0.1, np.float32)
    b = t.reduce("sum", t.load(b0), [0], False)
    t.store_expect(b, 4 * 0.1)
    assert(t.run_check())
    t.set_determ(False)
