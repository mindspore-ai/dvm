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


def test_reorder_load_1():
    """load should be arranged accroding to the order they are used"""
    t = Tester()
    a = np.full([8], 2, np.float16)
    b = np.full([8], 9, np.float16)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.unary("Sqrt", x2)
    x4 = t.binary("Add", x3, x1)
    t.set_passes("ReorderLoad")
    t.store_expect(x4, 5.0)
    assert (t.run_check())
    g = Graph(t)
    assert (g[2].input(0) == g[0])


def test_reorder_load_2():
    """load should be arranged before other operations"""
    t = Tester()
    a = np.full([8], 9, np.float16)
    b = np.full([8], 2, np.float16)
    c = np.full([8], 3, np.float16)
    x1 = t.load(a)
    x2 = t.unary("Sqrt", x1)
    x3 = t.load(b)
    x4 = t.binary("Add", x3, x2)
    x5 = t.load(c)
    x6 = t.binary("Add", x4, x5)
    t.set_passes("ReorderLoad")
    t.store_expect(x6, 8.0)
    assert (t.run_check())
    g = Graph(t)
    assert (g[0].name == "Load")
    assert (g[1].name == "Load")
    assert (g[2].name == "Load")
