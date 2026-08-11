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
def test_double_buffer_0():
    t = Tester()
    a = np.full([32], 2, np.float16)
    b = np.full([32], 2, np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.add(x0, 0.1)
    x3 = t.add(x1, 0.1)
    x4 = t.add(x2, x3)
    x5 = t.mul(x4, x0)
    t.store_expect(x5, 8.4)
    t.set_passes("VectorDoubleBuffer")
    assert (t.run_check())
    graph = t.dump()
    assert(graph.count("Copy") == 1)
