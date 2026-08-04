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
from tests.mark_utils import arg_mark


def assert_liveness_compacked(capfd):
    captured, _ = capfd.readouterr()
    lines = captured.split('\n')
    peak_live_before = int(lines[0][11:])
    peak_live_after = int(lines[1][11:])
    assert (peak_live_after < peak_live_before)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('ktype', ["vector", "vector:dyn"])
def test_elemwise_1(capfd, ktype):
    t = Tester(ktype)
    a = np.full([128, 32], 2, np.float16)
    x0 = t.load(a)
    x1 = t.exp(x0)
    x2 = t.log(x1)
    x3 = t.add(x2, x0)
    x4 = t.add(x2, x1)
    x5 = t.sqrt(x4)
    x6 = t.exp(x4)
    x7 = t.abs(x4)
    x8 = t.add(x2, x0)
    x9 = t.add(x3, x5)
    x10 = t.sqrt(x3)
    t.store(x6)
    t.store(x7)
    t.store(x8)
    t.store(x9)
    t.store(x10)
    t.set_passes("PrintPeakLive", "CompactPeakLiveness", "PrintPeakLive")
    assert (t.run_check())
    assert_liveness_compacked(capfd)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_elemwise_2(capfd):
    t = Tester()
    a = np.full([128, 32], 23, np.float16)
    x0 = t.load(a)
    x1 = t.log(x0)
    x2 = t.log(x1)
    x3 = t.log(x2)
    x4 = t.add(x2, x3)
    x5 = t.log(x2)
    x6 = t.add(x2, x0)
    x7 = t.log(x1)
    x8 = t.add(x2, x5)
    x9 = t.add(x3, x4)
    x10 = t.log(x6)
    t.store(x7)
    t.store(x8)
    t.store(x9)
    t.store(x10)
    t.set_passes("PrintPeakLive", "CompactPeakLiveness", "PrintPeakLive")
    assert (t.run_check())
    assert_liveness_compacked(capfd)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_select(capfd):
    t = Tester()
    a = np.full([128, 32], 23, np.float16)
    x0 = t.load(a)
    x1 = t.log(x0)
    x2 = t.reciprocal(x0)
    x3 = t.add(x1, x2)
    x4 = t.greater(x1, x2)
    x6 = t.cast(x1, "float32")
    x7 = t.cast(x3, "float32")
    x8 = t.add(x6, x7)
    x5 = t.select(x4, x2, x3)
    t.store(x5)
    t.store(x8)
    t.set_passes("PrintPeakLive", "CompactPeakLiveness", "PrintPeakLive")
    assert (t.run_check())
    assert_liveness_compacked(capfd)
