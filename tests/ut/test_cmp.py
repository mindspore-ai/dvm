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
    x = t.load(a)
    y = t.load(b)
    y = t.copy(y)
    x = t.copy(x)
    z = t.binary(op,x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b).astype(type))
    assert(t.run_check())
