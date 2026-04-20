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
    y0 = t.add(x0, x1)
    y1 = t.mul(y0, x1)
    y2 = t.mul(x3, y1)
    y3 = t.mul(x4, x5)
    y4 = t.mul(y2, y3)
    y5 = t.sum(y4, [0, 1, 2, 3, 4], False)
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
    x0 = t.add(b, a)
    x0_expect = ax + bx
    t.store_expect(x0, x0_expect)
    cx = np.full([1], 0, np.bool_)
    dx = np.full([1,4,8192,96], 0.1, np.float16)
    c = t.load(cx)
    d = t.load(dx)
    x2 = t.select(c, d, x0)
    x3 = t.store_expect(x2, x0_expect)
    assert(t.run_check())

def test_eager_inverse_execute_sequence():
    t = Tester("eager")
    x0_a = np.random.normal(0, 0.1, [2, 4]).astype(np.float32)
    x0 = t.load(x0_a)
    x1_a = np.random.normal(0, 0.1, [2, 4]).astype(np.float32)
    x1 = t.load(x1_a)
    x2 = t.mul(x1, -1)
    x3 = t.exp(x2)
    x4 = t.add(x3, 1)
    x5 = t.div(x4, 1)
    x6 = t.div(x1, x4)
    x7 = t.add(x5, x6)
    x8 = t.mul(x5, x6)
    x9 = t.sub(x7, x8)
    x10 = t.mul(x9, x0)
    y11_numpy = np.multiply(np.subtract(np.add(np.divide(np.add(np.exp(np.multiply(x1_a, -1)), 1), 1), np.divide(x1_a, np.add(np.exp(np.multiply(x1_a, -1)), 1))), np.multiply(np.divide(np.add(np.exp(np.multiply(x1_a, -1)), 1), 1), np.divide(x1_a, np.add(np.exp(np.multiply(x1_a, -1)), 1)))), x0_a)
    y11 = t.store_expect(x10, y11_numpy)
    assert t.run_check()


def test_eager_pv_blockdim_overflow():
    t = Tester("eager")
    x0_a = np.random.normal(0, 0.1, [4096, 40000]).astype(np.float32)
    x0 = t.load(x0_a)
    x1 = t.sum(x0, (0,), True)
    t.store_expect(x1, np.sum(x0_a, (0,), keepdims=True), 1e-3)
    for i in range(8):
        x2_a = np.random.normal(0, 0.1, [8]).astype(np.float32)
        x2 = t.load(x2_a)
        x3 = t.add(x2, 0.1)
        t.store_expect(x3, x2_a + 0.1)
    assert t.run_check()


def test_eager_determ_reloc_fail():
    t = Tester("eager:unify_ws")
    t.set_deterministic(True)
    x0_a = np.random.normal(0, 0.1, [512, 3072]).astype(np.float32)
    x0 = t.load(x0_a)
    x1 = t.sum(x0, (0,), False)
    y2_numpy = np.sum(x0_a, axis=(0,), keepdims=False)
    y2 = t.store_expect(x1, y2_numpy, 1e-4)
    x3_a = np.random.normal(0, 0.1, [4, 128, 768]).astype(np.float32)
    x3 = t.load(x3_a )
    x4 = t.sum(x3, (0, 1), False)
    y5_numpy = np.sum(x3_a, axis=(0, 1), keepdims=False)
    y5 = t.store_expect(x4, y5_numpy, 1e-4)
    x6 = t.add(x3, -1)
    x7_a = np.random.normal(0, 0.1, [4, 128, 768]).astype(np.float32)
    x7 = t.load(x7_a)
    x8_a = np.random.normal(0, 0.1, [4, 128, 1]).astype(np.float32)
    x8 = t.load(x8_a)
    x9 = t.div(x7, x8)
    x10 = t.div(x9, x8)
    x11 = t.add(x6, x10)
    x12 = t.div(x3, x8)
    x13 = t.sum(x11, (2,), False)
    y14_numpy = np.sum(np.add(np.add(x3_a, -1), np.divide(np.divide(x7_a, x8_a), x8_a)), axis=(2,), keepdims=False)
    y14 = t.store_expect(x13, y14_numpy, 1e-4)
    x15_a = np.random.normal(0, 0.1, [768]).astype(np.float32)
    x15 = t.load(x15_a)
    x16 = t.add(x12, x15)
    y17_numpy = np.add(np.divide(x3_a, x8_a), x15_a)
    y17 = t.store_expect(x16, y17_numpy, 1e-4)
    x18_a = np.random.normal(0, 0.1, [4, 128, 768]).astype(np.float32)
    x18 = t.load(x18_a)
    x19 = t.add(x12, x18)
    x20 = t.sum(x19, (0, 1), False)
    y21_numpy = np.sum(np.add(np.divide(x3_a, x8_a), x18_a), axis=(0, 1), keepdims=False)
    y21 = t.store_expect(x20, y21_numpy, 1e-4)
    x22 = t.add(x16, -1)
    x23 = t.sum(x22, (2,), False)
    y24_numpy = np.sum(np.add(np.add(np.divide(x3_a, x8_a), x15_a), -1), axis=(2,), keepdims=False)
    y24 = t.store_expect(x23, y24_numpy, 1e-4)
    assert t.run_check()