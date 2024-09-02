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
    assert np.allclose(t.output(h), a0-a1, 1e-5, 1e-5)

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

def test_wrap_inplace():
    t = Tester()
    a = np.random.normal(0, 1, (1024, 1)).astype(np.float32)
    x = t.load(a)
    x = t.broadcast(x, [1024, 23]);
    y = t.binary("Add", x, 1.0)
    out = t.store_expect(y, np.broadcast_to(a, [1024, 23]) + 1.0)
    t.set_passes("InsertRemovePad")
    assert(t.run_check())
    for line in t.das().split("\n"):
        if "Adds" in line:
            words = line.split(" 0x")
            assert(words[1] == words[2][:len(words[1])])
            break
    else:
        assert(0)

def test_wrap_flex_op():
    t = Tester()
    a = np.random.normal(0, 1, (1024, 1)).astype(np.float16)
    x = t.load(a)
    x = t.broadcast(x, [1024, 23]);
    y = t.unary("IsFinite", x)
    out = t.store_expect(y, np.isfinite(np.broadcast_to(a, [1024, 23])))
    t.set_passes("InsertRemovePad")
    assert(t.run_check())
