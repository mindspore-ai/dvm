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

import numpy as np
from dvm.tester import Tester
from utils import Graph
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reorder_store():
    t = Tester()
    a = np.full([8], 2, np.float16)
    b = np.full([8], 9, np.float16)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.reciprocal(x1)
    x4 = t.sqrt(x2)
    x5 = t.add(x4, x1)
    t.set_passes("ReorderStore")
    t.store_expect(x5, 5.0)
    t.store_expect(x3, 0.5)
    assert (t.run_check())
    g = Graph(t)
    assert (g[3].name == "Store")
    assert (g[3].input(0) == g[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reorder_multi_store():
    t = Tester()
    a = np.full([8], 9, np.float16)
    b = np.full([8], 2, np.float16)
    c = np.full([8], 3, np.float16)
    x1 = t.load(a)
    x2 = t.sqrt(x1)
    x3 = t.load(b)
    x4 = t.add(x3, x2)
    x5 = t.load(c)
    x6 = t.add(x4, x5)
    t.set_passes("ReorderStore")
    t.store_expect(x6, 8.0)
    t.store_expect(x2, 3.0)
    t.store_expect(x4, 5.0)
    assert (t.run_check())
    g = Graph(t)
    assert (g[2].name == "Store")
    assert (g[2].input(0) == g[1])
    assert (g[5].name == "Store")
    assert (g[5].input(0) == g[4])
