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

@pytest.mark.parametrize("shape",[(32, 32), (1024, 32), (1024, 2000)])
@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16])
@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("Less", np.less), ("Greater", np.greater),
                                      ("GreaterEqual", np.greater_equal), ("LessEqual", np.less_equal), ("NotEqual", np.not_equal)])
def test_cmp(shape, type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=shape).astype(type)
    b = np.random.randint(1024, size=shape).astype(type)
    if type != np.int32:
        a[0] = np.nan
        b[0] = np.nan
    x = t.load(a)
    y = t.load(b)
    y = t.copy(y)
    x = t.copy(x)
    z = t.binary(op,x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b).astype(type))
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16])
@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("Less", np.less), ("Greater", np.greater),
                                      ("GreaterEqual", np.greater_equal), ("LessEqual", np.less_equal), ("NotEqual", np.not_equal)])
def test_cmp_s_r(type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=(1024, 32)).astype(type)
    if type != np.int32:
        a[0] = np.nan
    b = 30
    x = t.load(a)
    x = t.copy(x)
    z = t.binary(op, x, b)
    z = t.copy(z)
    t.store_expect(z, func(a, b).astype(type))
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.int32, np.float32, np.float16])
@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("Less", np.less), ("Greater", np.greater),
                                      ("GreaterEqual", np.greater_equal), ("LessEqual", np.less_equal), ("NotEqual", np.not_equal)])
def test_cmp_s_l(type, op, func):
    t = Tester()
    a = np.random.randint(1024, size=(1024, 32)).astype(type)
    b = 30
    x = t.load(a)
    x = t.copy(x)
    z = t.binary(op, b, x)
    z = t.copy(z)
    t.store_expect(z, func(b, a).astype(type))
    assert(t.run_check())

@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("NotEqual", np.not_equal)])
def test_cmp_over_repeat(op, func):
    t = Tester()
    a = np.random.randint(1024, size=(100000, 1)).astype(np.float16)
    b = np.random.randint(1024, size=(100000, 7)).astype(np.float16)
    x = t.load(a)
    y = t.load(b)
    z = t.binary(op,x, y)
    t.store_expect(z, func(a, b).astype(np.float16))
    assert(t.run_check())


@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("Less", np.less), ("Greater", np.greater),
                                      ("GreaterEqual", np.greater_equal), ("LessEqual", np.less_equal), ("NotEqual", np.not_equal)])
def test_cmp_int(op, func):
    t = Tester()
    a = np.random.randint(-2147483648, 2147483647, (1024, 1024)).astype(np.int32)
    b = np.random.randint(-2147483648, 2147483647, (1024, 1024)).astype(np.int32)
    x = t.load(a)
    y = t.load(b)
    z = t.binary(op,x, y)
    t.store_expect(z, func(a, b).astype(np.int32))
    assert(t.run_check())
