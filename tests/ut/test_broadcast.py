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


@pytest.mark.parametrize('type, eps', [(np.float16, 1e-03),
                                       (np.float32, 1e-05)])
def test_broadcast_x(type, eps):
    t = Tester()
    a = np.full([16, 1], 0.5, type)
    for i in range(16):
        a[i, 0] = i
    x = t.load(a)
    y = t.broadcast(x, [16, 2])

    def _check(x):
        for i in range(16):
            if not np.isclose(x[i, 0], i, eps, eps):
                return False
        return True

    t.store_expect(y, _check)
    assert (t.run_check())


@pytest.mark.parametrize('type, eps', [(np.float16, 1e-03),
                                       (np.float32, 1e-05)])
def test_broadcast_x_2(type, eps):
    t = Tester()
    a = np.full([16, 1], 0.5, type)
    for i in range(16):
        a[i, 0] = i
    b = np.full([16, 64], 0.2, type)
    x = t.load(a)
    y = t.load(b)
    z = t.broadcast(x, [16, 64])
    r = t.binary("Add", z, y)

    def _check(x):
        for i in range(16):
            for j in range(64):
                if not np.isclose(x[i, j], i + 0.2, eps, eps):
                    return False
        return True

    t.store_expect(r, _check)
    assert (t.run_check())


@pytest.mark.parametrize('type, eps', [(np.float16, 1e-03),
                                       (np.float32, 1e-05)])
def test_broadcast_y(type, eps):
    t = Tester()
    a = np.full([1, 32], 0.5, type)
    for i in range(32):
        a[0, i] = i
    x = t.load(a)
    y = t.broadcast(x, [8, 32])

    def _check(x):
        for i in range(8):
            for j in range(32):
                if not np.isclose(x[i, j], j, eps, eps):
                    return False
        return True
    t.store_expect(y, _check)
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_broadcast_multi(type):
    t = Tester()
    a = np.full([10, 33, 64], 0.2, np.float32)
    b = np.full([10, 1, 1], 0.3, np.float32)
    y = t.load(b)
    z = t.broadcast(y, [10, 33, 1])
    z = t.binary("Add", z, 0.1)
    z = t.broadcast(z, [10, 33, 64])
    x = t.load(a)
    s = t.binary("Add", x, z)
    out = t.store_expect(s, 0.6000)
    assert(t.run_check())

def test_broadcast_large_shape():
    src_shape  = [20, 30, 1, 50]
    dst_shape = [20, 30, 40, 50]
    t = Tester()
    a = np.random.random(src_shape).astype(np.float32)
    np_res = np.broadcast_to(a, dst_shape)
    x = t.load(a)
    y = t.broadcast(x, dst_shape)
    t.store_expect(y, np_res)
    assert(t.run_check())

def test_broadcastx_repeat_overflow():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [1, 1]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [184965, 1])
    expect = np.broadcast_to(a0, (184965, 1))
    t.store_expect(z0, expect)
    assert(t.run_check())

def test_broadcastx_unalign_load():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [3, 4, 1]).astype(np.float32)
    la0 = t.load(a0)
    z0 = t.broadcast(la0, [3, 4, 5])
    expect = np.broadcast_to(a0, (3, 4, 5))
    t.store_expect(z0, expect)
    assert(t.run_check())

def test_broadcastx_bf16():
    t = Tester()
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [30, 40, 1]).astype(np.float32)
    la0 = t.load(a0, "bfloat16")
    z0 = t.broadcast(la0, [30, 40, 5])
    expect = np.broadcast_to(a0, (30, 40, 5))
    t.store_expect(z0, expect, 5e-3)
    assert(t.run_check())

def test_broadcast_s():
    t = Tester()
    x = t.broadcast(0.2, [2, 64], "float32", True)
    t.store_expect(x, 0.2)
    assert(t.run_check())

@pytest.mark.parametrize('lead_dim', [1024, 511])
def test_broadcast_store_rank_1(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1,lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.binary("Add", la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 96, lead_dim])
    e2 = np.broadcast_to(e1, (2, 96, lead_dim))
    t.store_expect(z0, e2)
    t.tile(2, 2, 2)
    t.tile(1, 1, 48)
    assert(t.run_check())

@pytest.mark.parametrize('lead_dim', [1024, 511])
def test_broadcast_store_rank_2(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4*lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.binary("Add", la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4*lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4*lead_dim))
    t.store_expect(z0, e2)
    t.tile(2, 2, 2)
    t.tile(1, 1, 3)
    t.tile(0, 0, 4)
    assert(t.run_check())

@pytest.mark.parametrize('lead_dim', [1024, 511])
def test_broadcast_store_rank_3(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.binary("Add", la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4, 6, lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4, 6, lead_dim))
    t.store_expect(z0, e2)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 3)
    assert(t.run_check())

@pytest.mark.parametrize('lead_dim', [1024, 511])
def test_broadcast_store_rank_4(lead_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, [2, 1, 4, 1, 2*lead_dim]).astype(np.float32)
    la0 = t.load(a0)
    la0 = t.binary("Add", la0, 0.5)
    e1 = a0 + 0.5
    t.store_expect(la0, e1)
    z0 = t.broadcast(la0, [2, 3, 4, 5, 2*lead_dim])
    e2 = np.broadcast_to(e1, (2, 3, 4, 5, 2*lead_dim))
    t.store_expect(z0, e2)
    t.tile(4, 4, 2)
    t.tile(3, 3, 3)
    t.tile(2, 2, 4)
    t.tile(1, 1, 5)
    t.tile(0, 1, 2)

def test_broadcast_bool():
    t = Tester()
    a0 = np.array([[True]]).astype(bool)
    x0 = t.load(a0)
    x1 = t.broadcast(x0, [2,3])
    t.store_expect(x1, True)
    assert(t.run_check())
