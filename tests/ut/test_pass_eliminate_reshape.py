# Copyright 2025 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_elim_reshape():
    t = Tester()
    a = np.random.normal(0.0, 1.0, [50, 256]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.2)
    x3 = t.reshape(x2, [25, 256, 2])
    x4 = t.mul(x3, 0.3)
    t.store_expect(x4, (a + 0.2).reshape([25, 256, 2]) * 0.3)
    t.set_passes("EliminateReshape")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_elim_reshape_load_store():
    t = Tester()
    a = np.random.normal(0.0, 1.0, [50, 256]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.reshape(x1, [25, 256, 2])
    t.store_expect(x2, a.reshape([25, 256, 2]))
    t.set_passes("EliminateReshape")
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_continuous_reshape_with_view():
    t = Tester()
    np.random.seed(12)
    physical_shape = [8, 10, 204, 64]
    view_shape = [10, 204, 8, 64]
    view_stride = [13056, 64, 130560, 1]
    out_shape = [2040, 512]
    a = np.random.normal(0, 1, physical_shape).astype(np.float32)
    expect = np.transpose(a, (1, 2, 0, 3)).copy().reshape(out_shape)
    x = t.view_load(view_shape, view_stride, a)
    x = t.copy(x)
    x = t.copy(x)
    x = t.reshape(x, [10, 204, 512])
    x = t.reshape(x, out_shape)
    t.store_expect(x, expect)
    t.set_passes("EliminateReshape")
    assert (t.run_check())
