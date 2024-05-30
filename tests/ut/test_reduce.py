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
import dvm
from dvm.tester import Tester

@pytest.mark.parametrize('dim', [64, 256, 251])
def test_reduce_x(dim):
    t = Tester()
    a = np.full([10, dim], 0.0, np.float32)
    val = 0.001
    for i in range(10):
        for j in range(dim):
            a[i, j] = val
        val += 0.001
    x = t.load(a)
    y = t.reduce("sum", x, [1], True)
    def _check(x):
        val = 0.001
        for i in range(10):
            if abs(x[i, 0] - val * dim) > 0.001:
                print(x[i, 0] , ", ", val * dim, ", ", i)
                return False
            val += 0.001
        return True
    t.store_expect(y, _check)
    assert(t.run_check())

@pytest.mark.parametrize('r_dim, s_dim', [(6, 64), (6, 60), (2, 383*16)])
def test_reduce_y(r_dim, s_dim):
    t = Tester()
    a = np.full([10, r_dim, s_dim], 0.01, np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, [1], True)
    expect = 0.01 * float(r_dim)
    t.store_expect(y, expect)
    assert(t.run_check())

@pytest.mark.parametrize('i, j', [(32, 65), (31, 89), (31, 90), (31, 93), (30, 82), (30, 83), (30, 85)])
def test_reduce_i_j(i, j) :
    t = Tester()
    a = np.random.normal(-0.5, 0.5, [3, 1280, i, j]).astype(np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, [0, 2, 3], False)
    expect = np.sum(a, axis=(0, 2, 3), keepdims=False)
    t.store_expect(y, expect)
    assert(t.run_check())

@pytest.mark.parametrize('dims', [[1,2,3], [1,3,5], [1, 2, 5, 6], [5,6], [0,1,2,3,4,5,6]])
def test_reduce_normalize(dims):
    t = Tester()
    shape = [6, 4, 2, 5, 1, 3, 8]
    a = np.full(shape, 0.01, np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, dims, True)
    total = 1
    for d in dims:
        total *= shape[d]
    expect = 0.01 * float(total)
    t.store_expect(y, expect)
    assert(t.run_check())

@pytest.mark.parametrize('dim', [1001, 1024])
def test_reduce_atomic(dim):
    t = Tester()
    a = np.full([64, dim], 0.01, np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, [0], True)
    t.store_expect(y, 0.01*64)
    assert(t.run_check())

@pytest.mark.parametrize('in_shape, dims', [[[511, 1024],(1,)], [[521, 1024],(1,)],
    [[35053], (0,)],
    [[120, 1], (0,)], # lead 1 not include
    [[1, 4, 120, 136], (0,2,3)], # reduce with 1
    [[120, 20, 1], (2,)], # reduce one range with 1
    [[1, 1, 1, 1, 5, 1, 1, 300, 1, 100], (0, 2, 4, 6, 8)], # opensora bugfix
    [[3, 4, 120, 136], (0, 2, 3)], # from sdxl:  two atomic dim range
    [[11, 6000], (0,)]]) # reducey red_size = 1
def test_reduce(in_shape, dims):
    t = Tester()
    a = np.random.normal(-0.5, 0.5, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    assert(t.run_check())

@pytest.mark.skipif(dvm.device.arch() != "AscendC100", reason = "only support 910 tiling")
def test_reduce_store_with_lead_dim_tiling():
    in_shape = [521, 1024]
    dims = (1,)
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    t.tile(1, 1, 20);
    assert(t.run_check())

def test_reduce_x_tail():
    in_shape = [1949]
    dims = (0,)
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res)
    t.tile(0, 0, 122)
    assert(t.run_check())

def test_reduce_y_tail():
    in_shape = [1949, 512]
    dims = (0,)
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.load(a)
    y = t.reduce("sum", x, dims, True)
    res = np.sum(a, dims, keepdims=True)
    t.store_expect(y, res, 1e-4)
    t.tile(1, 1, 122);
    assert(t.run_check())

def test_reduce_fake_atomic():
    t = Tester()
    a = np.full([1, 640, 64, 64], 1.0, np.float32)
    x1 = t.load(a)
    x2 = t.reduce("sum", x1, [2, 3], True)
    x3 = t.reduce("sum", x2, [0], True)
    t.store_expect(x3, 4096.0)
    assert(t.run_check())
