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

@pytest.mark.parametrize("shape",[(1024, 32), (13, 131), (16, 11) ,(3, 3)])
@pytest.mark.parametrize('type, eps', [(np.float32, 1e-3),(np.float16, 1e-3)])
def test_greater(shape, type, eps):
    t = Tester()
    a = np.random.rand(*shape).astype(type)
    b = np.random.rand(*shape).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.binary("Greater",x, y)
    t.store_expect(z, np.greater(a, b).astype(type), eps)
    assert(t.run_check())


@pytest.mark.parametrize('type, eps', [(np.float32, 1e-3), (np.float16, 1e-3)])
@pytest.mark.parametrize('op, func', [("Equal", np.equal), ("Less", np.less), ("Greater", np.greater),
                                      ("GreaterEqual", np.greater_equal), ("LessEqual", np.less_equal), ("NotEqual", np.not_equal)])
def test_equal(type, eps, op, func):
    t = Tester()
    a = np.random.rand(16, 10).astype(type)
    b = np.random.rand(16, 10).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.binary(op, x, y)
    t.store_expect(z, func(a, b).astype(type), eps)
    assert (t.run_check())
