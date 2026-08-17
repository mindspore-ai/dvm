# Copyright 2026 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dce_load():
    t = Tester()
    a = np.random.normal(0.0, 1.0, [100]).astype(np.float32)
    b = np.random.normal(0.0, 1.0, [100]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b) # dead code
    x2 = t.add(x0, 0.1)
    t.store_expect(x2, a + 0.1)
    t.set_passes("DeadCodeEliminate")
    assert (t.run_check())
    graph = t.dump()
    assert(graph.count("Load") == 1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dce_simd():
    t = Tester()
    a = np.random.normal(0.0, 1.0, [100]).astype(np.float32)
    b = np.random.normal(0.0, 1.0, [100]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.sqrt(x0)
    x2 = t.mul(x0, 0.6)
    t.store_expect(x2, a * 0.6)
    t.set_passes("DeadCodeEliminate")
    assert (t.run_check())
    graph = t.dump()
    assert(graph.count("Sqrt") == 0)
