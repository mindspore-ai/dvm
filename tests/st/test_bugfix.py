# Copyright 2025 Huawei Technologies Co., Ltd
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

def test_reduce_clear_pad_fault():
    ''' r2.7 '''
    t = Tester(use_pass_opt=True)
    a0 = np.random.randn(2, 576, 224, 16, 2).astype(np.float32)
    a1 = np.random.randn(2, 576, 224, 16, 1).astype(np.float32)
    a2 = np.random.randn(2, 576, 224, 16, 1).astype(np.float32)
    a3 = np.random.randn(2, 576, 224, 16, 2).astype(np.float32)
    a4 = np.random.randn(2, 576, 224, 16, 1).astype(np.float32)
    a5 = np.random.randn(2, 576, 224, 16, 2).astype(np.float32)
    e0 = a0 + a1
    e1 = e0 * a1
    e2 = a3 * e1
    e3 = a4 * a5
    e4 = e2 * e3
    e5 = np.sum(e4, axis=(0, 1, 2, 3, 4))
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.load(a2)
    x3 = t.load(a3)
    x4 = t.load(a4)
    x5 = t.load(a5)
    y0 = t.binary("Add", x0, x1)
    y1 = t.binary("Mul", y0, x1)
    y2 = t.binary("Mul", x3, y1)
    y3 = t.binary("Mul", x4, x5)
    y4 = t.binary("Mul", y2, y3)
    y5 = t.reduce("sum", y4, [0, 1, 2, 3, 4], False)
    t.store_expect(y0, e0, eps=1e-4)
    t.store_expect(y1, e1, eps=1e-4)
    t.store_expect(y2, e2, eps=1e-4)
    t.store_expect(y3, e3, eps=1e-4)
    t.store_expect(y4, e4, eps=1e-4)
    t.store_expect(y5, e5, eps=1e-4)
    assert(t.run_check())

def test_select_overread_of_ub():
    ''' r2.7 '''
    t = Tester(use_pass_opt=True)
    ax = np.random.normal(0, 0.1, [1,4,8192,96]).astype(np.float16)
    a = t.load(ax)
    bx = np.random.normal(0, 0.1, [1,4,8192,96]).astype(np.float16)
    b = t.load(bx)
    x0 = t.binary("Add", b, a)
    x0_expect = ax + bx
    t.store_expect(x0, x0_expect)
    cx = np.full([1], 0, np.bool_)
    dx = np.full([1,4,8192,96], 0.1, np.float16)
    c = t.load(cx)
    d = t.load(dx)
    x2 = t.select(c, d, x0)
    x3 = t.store_expect(x2, x0_expect)
    t.run_check()


