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
from dvm.tester import Tester, ShapeRef

def test_dyn_shape():
    t = Tester('dyn')
    a = np.full([10], 0.1, np.float32)
    ref = ShapeRef([2, 5])
    x = t.load(a)
    y = t.reshape(x, ref)
    z = t.binary("Add", y, 0.2)
    e1 = a + 0.2
    t.store_expect_flat(z, e1)
    t.run_check()
    # second time launch
    a2 = np.random.normal(0, 1, (20, 40)).astype(np.float32)
    t.reload(x, a2)
    ref.update([10, 80])
    e2 = a2 + 0.2
    t.store_expect_flat(z, e2)
    t.run_check()
