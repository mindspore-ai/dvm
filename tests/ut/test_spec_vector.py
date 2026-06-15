# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
import dvm
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
def test_spec_connected_split():
    t = Tester("vector:spec")
    a = np.random.normal(0.0, 0.3, [10, 6000]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.02)
    x3 = t.sum(x2, (0,), False)
    t.spec_next()
    x4 = t.mul(x3, 0.5)
    x5 = t.add(x2, 0.3)
    t.store_expect(x4, np.sum(a + 0.02, axis=(0,), keepdims=False) * 0.5)
    t.store_expect(x5, a + 0.02 + 0.3)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_broadcast():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.3, [6000]).astype(np.float32)
    b = np.random.normal(0.0, 0.3, [6000]).astype(np.float32)
    c = np.random.normal(0.0, 0.3, [10000, 6000]).astype(np.float32)
    x1 = t.add(t.load(a), t.load(b))
    x2 = t.mul(x1, t.load(c)) 
    x3 = t.cast(x2, "float16")
    t.store_expect(x1, a + b)
    t.store_expect(x3, ((a + b) * c).astype(np.float16))
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape_a, dims', [[[2, 30, 8000], (1,)], [[3, 2, 20, 8000], (0, 2)]])
def test_auto_spec_reduce(shape_a, dims):
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, shape_a).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.01)
    x3 = t.sum(x2, dims, True)
    x3_expect = np.sum(a + 0.01, axis=dims, keepdims=True)
    b = np.random.normal(0.0, 0.3, x3_expect.shape[:-1] + (1,)).astype(np.float32)
    x4 = t.mul(x3, t.load(b))
    t.store_expect(x4, x3_expect * b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_multi_reduce():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 20, 1]).astype(np.float32)
    b = np.random.normal(0.0, 0.03, [10, 20, 6000]).astype(np.float32)
    x1 = t.add(t.load(a), t.load(b))
    x2 = t.sum(x1, (0,), True)
    x3 = t.mul(x2, 0.5)
    t.store_expect(x3, np.sum(a + b, axis=(0,), keepdims=True) * 0.5)
    x4 = t.sub(x1, 0.002)
    x5 = t.sum(x4, (0,), True)
    x6 = t.add(x5, 0.2)
    t.store_expect(x6, np.sum(a + b - 0.002, axis=(0,), keepdims=True) + 0.2)
    x7 = t.mul(x4, 0.8)
    t.store_expect(x7, (a + b - 0.002) * 0.8)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_load_clone():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [4, 3, 20, 6000]).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.01)
    x3 = t.sum(x2, (0,), True)
    x4 = t.add(x3, x1)
    t.store_expect(x4, np.sum(a + 0.01, axis=(0,), keepdims=True) + a)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_nest_reduce():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [4, 3, 20, 1]).astype(np.float32)
    b = np.random.normal(0.0, 0.03, [4, 3, 20, 6000]).astype(np.float32)
    x1 = t.add(t.load(a), t.load(b))
    x2 = t.sum(x1, (2,), True)
    x3 = t.mul(x2, 0.5)
    x4 = t.sum(x3, (0,), True)
    x5 = t.add(x4, x3)
    e3 = np.sum(a + b, axis=(2,), keepdims=True) * 0.5
    t.store_expect(x5, np.sum(e3, axis=(0,), keepdims=True) + e3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape1, shape2, shape3, red_dims', [
    [[1, 2000], [20, 2000], [20 * 2000], (0,)], # forward prop
    [[10, 200], [10, 200], [5, 10, 40], (2,)], # backward prop
    [[10, 1], [1, 200], [5, 10, 40], (1,)],  # all split
])
def test_auto_spec_reshape_reduce(shape1, shape2, shape3, red_dims):
    """ (shape1 + shape2).reshape(shape3).reduce(red_dims) """
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, shape1).astype(np.float32)
    b = np.random.normal(0.0, 0.03, shape2).astype(np.float32)
    x1 = t.add(t.load(a), t.load(b))
    x2 = t.reshape(x1, shape3)
    x3 = t.sum(x2, red_dims, True)
    x4 = t.mul(x3, 0.8)
    x5 = t.add(x4, x3)
    e3 = np.sum((a + b).reshape(shape3), axis=red_dims, keepdims=True)
    t.store_expect(x4, e3 * 0.8)
    t.store_expect(x5, (e3 * 0.8) + e3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_reshape_seq():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 1000]).astype(np.float32)
    x1 = t.add(t.load(a), 0.01)
    x2 = t.reshape(x1, [10000])
    x3 = t.mul(x2, 0.7)
    x4 = t.reshape(x3, [20, 500])
    x5 = t.add(x4, 0.2)
    t.store_expect(x5, ((a + 0.01).reshape([10000]) * 0.7).reshape([20, 500]) + 0.2)
    assert (t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_nest_reshape():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 1000]).astype(np.float32)
    x1 = t.add(t.load(a), 0.01)
    x2 = t.reshape(x1, [10000])
    x3 = t.mul(x2, 0.7)
    x4 = t.reshape(x3, [50, 200])
    x5 = t.sub(x4, 0.01)
    t.store_expect(x5, ((a + 0.01).reshape([10000]) * 0.7).reshape([50, 200]) - 0.01)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_reshape_elim():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 1000]).astype(np.float32)
    x1 = t.add(t.load(a), 0.01)
    x2 = t.reshape(x1, [10000])
    x3 = t.reshape(x2, [20, 500])
    x4 = t.add(x3, 0.2)
    t.store_expect(x4, (a + 0.01).reshape([20, 500]) + 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_reshape_cross_dim():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 1000]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.reshape(x0, [1, 10, 1000])
    x2 = t.reshape(x0, [10, 1, 1000])
    x3 = t.add(x1, x2)
    t.store_expect(x3, a.reshape([1, 10, 1000]) + a.reshape([10, 1, 1000]))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_reshape_side_connect():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [10, 1, 1000]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.1)
    x2 = t.reshape(x1, [1, 10, 1000])
    x3 = t.mul(x2, x1)
    t.store_expect(x3, (a + 0.1).reshape([1, 10, 1000]) * (a + 0.1))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_dyn_shape():
    t = Tester("vector:spec,dyn,priv1")
    x1 = t.load([-1], "float32")
    x2 = t.load([-1], "float32")
    x3 = t.add(x1, x2)
    x4 = t.store(t.mul(x3, 1.2))
    dims = t.int_array()
    x5 = t.sum(x3, dims, True)
    x6 = t.mul(x5, 0.01)
    x7 = t.store(x6)
    iterations = [[[30, 8000], [30, 1], (0,)],
                  [[512, 1000], [512, 1000], (1,)],
                  [[4, 1000, 1], [4, 1000, 1000], (1,)],
                  [[10, 1, 20, 512], [10, 20, 1, 512], (0, 3)]]
    for x_shape, y_shape, dims_shape in iterations:
        d1 = np.random.normal(0.0, 1.0, x_shape).astype(np.float32)
        d2 = np.random.normal(0.0, 1.0, x_shape).astype(np.float32)
        t.input(x1, d1)
        t.input(x2, d2)
        dims.update(dims_shape)
        t.run()
        assert(t.check(x4, (d1 + d2) * 1.2))
        assert(t.check(x7, np.sum(d1 + d2, axis=dims_shape, keepdims=True) * 0.01, 1e-4))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ["vector:spec", "vector:spec,priv1"])
def test_spec_swap_with_store(mode):
    t = Tester(mode)
    shape_a, dims =  [2, 30, 8000], (1,)
    a = np.random.normal(0.0, 0.03, shape_a).astype(np.float32)
    x1 = t.load(a)
    x2 = t.add(x1, 0.01)
    x3 = t.sum(x2, dims, True)
    x3_expect = np.sum(a + 0.01, axis=dims, keepdims=True)
    t.store_expect(x3, x3_expect, 1e-4)
    t.spec_next()
    b = np.random.normal(0.0, 0.3, x3_expect.shape[:-1] + (1,)).astype(np.float32)
    x4 = t.mul(x3, t.load(b))
    t.store_expect(x4, x3_expect * b, 1e-4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_auto_spec_indirect_cut_depend():
    t = Tester("vector:spec,priv1")
    a = np.random.normal(0.0, 0.03, [4, 3, 20, 1]).astype(np.float32)
    b = np.random.normal(0.0, 0.03, [4, 3, 20, 6000]).astype(np.float32)
    x1 = t.add(t.load(a), t.load(b))
    x2 = t.sum(x1, (0,), True)
    x3 = t.mul(x2, 0.5)
    x4 = t.add(x3, x1)
    e1 = a + b
    e3 = np.sum(e1, axis=(0,), keepdims=True) * 0.5
    t.store_expect(x3, e3)
    t.store_expect(x4, e3 + e1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_spec_custom():
    t = Tester("vector:spec,priv1")
    a0 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a0)
    x0 = t.mul(x0, 0.1)
    x1 = t.store(x0)
    x2 = t.load([1024, 500], "float32")
    t.set_store_temp(x1)
    t.set_load_bind(x2, x1)
    x3 = t.add(x2, 0.2)
    t.store_expect(x3, a0 * 0.1 + 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('is_split', [True, False])
def test_spec_slice(is_split):
    t = Tester("vector:spec,priv1")
    a0 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a0)
    if is_split:
        x0 = t.mul(x0, 0.1)
        a0 = a0 * 0.1
    x3 = t.slice(x0, [100, 100], [600, 300])
    x4 = t.add(x3, 0.2)
    t.store_expect(x4, a0[100:700, 100:400] + 0.2)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('is_split, end', [[True, 700], [True, -2], [False, 100000]])
def test_spec_slice_dim(is_split, end):
    t = Tester("vector:spec,priv1")
    a0 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a0)
    if is_split:
        x0 = t.mul(x0, 0.1)
        a0 = a0 * 0.1
    b_ref = t.scalar(dvm.int64)
    b_ref.update(100)
    x2 = t.slice_dim(x0, 0, b_ref, end)
    x3 = t.add(x2, 0.2)
    t.store_expect(x3, a0[100:end, :] + 0.2)
    assert (t.run_check())
