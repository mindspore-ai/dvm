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

import pytest
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_spec_base():
    t = Tester("vector:spec")
    a0 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a0)
    x0 = t.mul(x0, 0.1)
    x0 = t.add(x0, 0.2)
    t.store_expect(x0, a0 * 0.1 + 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_spec_fall_reshape():
    t = Tester("vector:spec")
    shape_a, shape_b, shape_c, red_dims = [10, 4, 64], [10, 4, 1], [10, 32, 8], (1,)
    a = np.random.normal(0.0, 0.3, shape_a).astype(np.float32)
    b = np.random.normal(0.0, 0.3, shape_b).astype(np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.add(x1, x2)
    x4 = t.reshape(x3, shape_c)
    t.spec_next()
    x5 = t.sum(x4, red_dims, True)
    t.store_expect(x5, np.sum((a + b).reshape(shape_c), axis=red_dims, keepdims=True), 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape_a, dims', [[[2, 30, 8000], (1,)], [[4, 128], (0,)], [[3, 40], (1,)], [[500], (0,)]])
def test_spec_fall_reduce(shape_a, dims):
    t = Tester("vector:spec")
    a = np.random.normal(0.0, 0.3, shape_a).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.02)
    x3 = t.sum(x2, dims, False)
    t.spec_next()
    x3_expect = np.sum(a + 0.02, axis=dims, keepdims=False)
    b = np.random.normal(0.0, 0.3, x3_expect.shape).astype(np.float32)
    x4 = t.mul(x3, t.load(b))
    t.store_expect(x4, x3_expect * b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_spec_migrate_load_store():
    t = Tester("vector:spec")
    a = np.random.normal(0.0, 0.3, [10, 6000]).astype(np.float32)
    b = np.random.normal(0.0, 0.3, [1, 6000]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.load(b)
    x3 = t.add(x1, 0.02)
    x4 = t.sum(x3, (0,), True)
    t.spec_next()
    x5 = t.mul(x4, x2)
    x3_e = a + 0.02
    t.store_expect(x3, x3_e)
    t.store_expect(x5, np.sum(x3_e, axis=(0,), keepdims=True) * b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_spec_load_reloc():
    t = Tester("vector:spec")
    a = np.random.normal(0.0, 0.3, [10, 6000]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.02)
    x3 = t.sum(x2, (0,), True)
    t.spec_next()
    x4 = t.mul(x3, x1)
    t.store_expect(x4, np.sum(a + 0.02, axis=(0,), keepdims=True) * a)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dyn_spec_fall_reduce():
    t = Tester("vector:spec,dyn")
    x1 = t.load([-1], "float32")
    x2 = t.add(x1, 0.02)
    dims = t.int_array()
    x3 = t.sum(x2, dims, True)
    t.spec_next()
    x4 = t.mul(x3, x2)
    x5 = t.store(x4)
    iterations = [[[3, 10, 8000], (1,)],
                  [[100, 500], (1,)],
                  [[8, 100], (0,)],
                  [[4, 512], (1,)]]
    for x_shape, dims_shape in iterations:
        d1 = np.random.normal(0.0, 1.0, x_shape).astype(np.float32)
        t.input(x1, d1)
        dims.update(dims_shape)
        t.run()
        assert(t.check(x5, np.sum(d1 + 0.02, axis=dims_shape, keepdims=True) * (d1 + 0.02), 1e-4))
