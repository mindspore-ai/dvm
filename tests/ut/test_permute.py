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

import copy
import pytest
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


def _full_stride(shape):
    dim = len(shape)
    result = [1] * dim
    stride = 1
    for i in range(dim -1, -1, -1):
        result[i] = stride
        stride *= shape[i]
    return result


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_permute_from_load():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(-0.5, 0.5, [10, 20, 32]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.permute(x0, [1, 0, 2])
    t.store_expect(x1, np.transpose(a, (1, 0, 2)))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('in_shape, b_shape, perm', [
    ([10, 1, 200], [10, 20, 200], [1, 0, 2]),  # single broadcast
    ([10, 1, 1, 200], [10, 20, 5, 200], [0, 2, 1, 3]),  # multi broadcast, in range
    ([10, 1, 1, 200], [10, 20, 5, 200], [2, 1, 0, 3]),  # multi broadcast, range ext
    ([1, 1, 10, 200], [10, 20, 10, 200], [3, 1, 2, 0]),  # multi broadcast, range split
])
def test_permute_broadcast(in_shape, b_shape, perm):
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0, 0.5, in_shape).astype(np.float32)
    x0 = t.view_load(in_shape, _full_stride(in_shape), a)
    x1 = t.broadcast(x0, b_shape)
    x2 = t.permute(x1, perm)
    t.store_expect(x2, np.transpose(a, perm))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('in_shape, dims, perm', [
    ([10, 20, 512], (1,), [1, 0, 2]),  # single reduce
    ([10, 2, 40, 6000], (0, 3), [1, 0, 2, 3]),  # multi range, trans with no reduce
    ([10, 2, 40, 6000], (0, 3), [2, 1, 0, 3]),  # multi range, trans reduce
    ([10, 20, 512], (1,2), [1, 0, 2]),  # range split 
])
def test_permute_reduce(in_shape, dims, perm):
    t = Tester("vector:spec,priv1")
    a = np.random.normal(-0.3, 0.5, in_shape).astype(np.float32)
    x0 = t.view_load(in_shape, _full_stride(in_shape), a)
    x1 = t.sum(x0, dims, True)
    x2 = t.add(x1, 0.1)
    x3 = t.permute(x2, perm)
    t.store_expect(x3, np.transpose(np.sum(a, dims, keepdims=True) + 0.1, perm))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_permute_backward():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(-0.3, 0.5, [20, 1, 200]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.1)
    x2 = t.permute(x1, [2, 1, 0])
    x3 = t.broadcast(x2, [200, 30, 20])
    x4 = t.mul(x3, 0.5)
    e = np.broadcast_to(np.transpose(a + 0.1, [2, 1, 0]), [200, 30, 20]) * 0.5
    t.view_store_expect(x4, _full_stride([200, 30, 20]), e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_permute_nest():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.5, [10, 20, 1]).astype(np.float16)
    x0 = t.load(a)
    x1 = t.permute(x0, [1, 0, 2])
    x2 = t.broadcast(x1, [20, 10, 100])
    x3 = t.permute(x2, [0, 2, 1])
    t.store_expect(x3, np.transpose(np.broadcast_to(np.transpose(a, (1, 0, 2)), [20, 10, 100]), (0, 2, 1)))
    assert (t.run_check())
