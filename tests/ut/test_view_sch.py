
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
import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize("H, W", [
  (1024, 2048),  # body
  (38, 1024),  # body + h_tail
  (1024, 38),  # body + w_tail
  (100, 70),    # body + h_tail + w_tail
  (2000, 900),    # body + h_tail + w_tail + hw_tail
  (300, 10), # w_tail + hw_tail
  (7, 300), # h_tail + hw_tail
  (7, 12), # hw_tail
])
def test_trans_fractal_2d(type, H, W):
    t = Tester("vector:opt_fractal")
    a = np.random.normal(0, 1, (W, H)).astype(type)
    x0 = t.view_load([H, W], [1, H], a)
    t.view_store_expect(x0, [W, 1], a.T)
    assert (t.run_check())

def _continuous_stride(shape):
    stride = [1] * len(shape)
    for i in range(len(stride) - 1, 0, -1):
        stride[i - 1] = shape[i] * stride[i]
    return stride

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape1, swap1, shape2, swap2, view", [
    [[10, 64, 200], (1, 2), [10, 64, 200], (1, 2), True], # 3d. neighbor axis
    [[4, 70, 20, 200], (1, 3), [1, 70, 20, 200], (1, 3), False], # 4d. no neighbor axis
    [[10, 1, 200], (1, 2), [10, 128, 200], (1, 2), True],  # w broadcast
    [[10, 200, 1], (1, 2), [10, 200, 128], (1, 2), False],  # h broadcast
])
def test_trans_fractal_multi_input(shape1, swap1, shape2, swap2, view):
    t = Tester("vector:opt_fractal")
    a = np.random.normal(0, 1, shape1).astype(np.float32)
    b = np.random.normal(0, 1, shape2).astype(np.float32)
    swap_a = np.swapaxes(a, swap1[0], swap1[1])
    swap_b = np.swapaxes(b, swap2[0], swap2[1])
    stride1 = _continuous_stride(shape1)
    stride2 = _continuous_stride(shape2)
    stride1[swap1[0]], stride1[swap1[1]] = stride1[swap1[1]], stride1[swap1[0]]
    stride2[swap2[0]], stride2[swap2[1]] = stride2[swap2[1]], stride2[swap2[0]]
    x0 = t.view_load(swap_a.shape, stride1, a)
    x1 = t.view_load(swap_b.shape, stride2, b)
    x2 = t.add(x0, x1)
    expect = swap_a + swap_b
    if view:
        store_stride = _continuous_stride(expect.shape)
        t.view_store_expect(x2, store_stride, expect)
    else:
        t.store_expect(x2, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape1, shape2, shape3, shape4, axis, view", [
    [[3, 400], [3, 600], [3, 200], [3, 1200], 1, False], # lead
    [[3, 400, 32], [3, 600, 32], [3, 200, 32], [3, 1, 32], 1, True], # middle, cat broadcast
    [[10, 400], [20, 400], [30, 400], [60, 1], 0, False], # out, no-cat broadcast
])
def test_sch_concat(dtype, shape1, shape2, shape3, shape4, axis, view):
    t = Tester()
    a0 = np.random.normal(0, 1, shape1).astype(dtype)
    a1 = np.random.normal(0, 1, shape2).astype(dtype)
    a2 = np.random.normal(0, 1, shape3).astype(dtype)
    a3 = np.random.normal(0, 1, shape4).astype(dtype)
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.add(x0, 0.1)
    x3 = t.mul(x1, 0.6)
    x4 = t.load(a2)
    x5 = t.concat([x2, x3, x4], axis)
    expect = np.concatenate([a0 + 0.1, a1 * 0.6, a2], axis=axis) + a3
    if view:
        x6 = t.add(x5, t.view_load(shape4, _continuous_stride(shape4), a3))
        t.view_store_expect(x6, _continuous_stride(expect.shape), expect)
    else:
        x6 = t.add(x5, t.load(a3))
        t.store_expect(x6, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_exceed_core():
    # The first input accounts for 99.9% of total elements:
    # 137862*32 / (137862*32 + 69*2*32) = 4411584 / 4416000 = 0.999
    t = Tester()
    slice_num = 70
    cat_np, cat_ops = [], []
    first = np.random.normal(0, 1, [137862, 32]).astype(np.float32)
    x0 = t.load(first)
    x1 = t.add(x0, 0.1)
    cat_np.append(first + 0.1)
    cat_ops.append(x1)
    for _ in range(slice_num - 1):
        a = np.random.normal(0, 1, [2, 32]).astype(np.float32)
        x0 = t.load(a)
        x1 = t.add(x0, 0.1)
        cat_np.append(a + 0.1)
        cat_ops.append(x1)
    x2 = t.concat(cat_ops, 0)
    x3 = t.mul(x2, 0.5)
    t.store_expect(x3, np.concatenate(cat_np, axis=0) * 0.5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("view", [True, False])
def test_sch_concat_dyn(view):
    t = Tester("vector:dyn")
    x0 = t.load([-1], "float32")
    x1 = t.load([-1], "float32")
    x2 = t.add(x0, 0.1)
    x3 = t.mul(x1, 0.6)
    x4 = t.load([-1], "float32")
    x5 = t.concat([x2, x3, x4], 1)
    x6 = t.add(x5, 0.2)
    if view:
        strides = t.int_array()
        x7 = t.view_store(x6, strides)
    else:
        x7 = t.store(x6)
    iters = [[[3, 300], [3, 400],[3, 200]], [[3, 300, 16], [3, 400, 16],[3, 200, 16]], [[10, 1000], [10, 800],[10, 200]]]
    for shape1, shape2, shape3 in iters:
        a0 = np.random.normal(0, 1, shape1).astype(np.float32)
        a1 = np.random.normal(0, 1, shape2).astype(np.float32)
        a2 = np.random.normal(0, 1, shape3).astype(np.float32)
        t.input(x0, a0)
        t.input(x1, a1)
        t.input(x4, a2)
        expect = np.concatenate([a0 + 0.1, a1 * 0.6, a2], axis=1) + 0.2
        if view:
            strides.update(_continuous_stride(expect.shape))
            t.set_output(x7, np.ascontiguousarray(np.zeros_like(expect)))
        t.run()
        assert(t.check(x7, expect))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape, split_size, dim, view", [
    ([3, 900], 300, 1, False),    # lead, evenly divisible
    ([3, 500, 32], 200, 1, True),    # middle, body + tail
])
def test_sch_split(dtype, shape, split_size, dim, view):
    split_num = (shape[dim] + split_size - 1) // split_size
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(dtype)
    if view:
        x0 = t.view_load(shape, _continuous_stride(shape), a)
    else:
        x0 = t.load(a)
    x1 = t.mul(x0, 0.7)
    xout = t.split(x1, dim, split_size, split_num)
    expects = np.split(a * 0.7, [split_size * (i + 1) for i in range(split_num - 1)], dim)
    for i, s in enumerate(xout):
        val = 0.1 * (i + 1)
        y = t.add(s, val)
        e = expects[i] + val
        t.store_expect(y, e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape1, shape2", [
    ([3, 18911], [3, 1]),    # lead
    ([2, 18911, 3], [2, 1, 3]), # middle
    ([3, 15401, 3], [3, 1, 1]), # fold
])
@pytest.mark.parametrize("view", [True, False])
def test_sch_dup_tiling(shape1, shape2, view):
    t = Tester()
    a = np.random.normal(0, 1, shape1).astype(np.float32)
    b = np.random.normal(0, 1, shape2).astype(np.float32)
    if view:
        x0 = t.view_load(shape1, _continuous_stride(shape1), a)
        x1 = t.view_load(shape2, _continuous_stride(shape2), b)
    else:
        x0 = t.load(a)
        x1 = t.load(b)
    x2 = t.add(x0, x1)
    expect = a + b
    if view:
        t.view_store_expect(x2, _continuous_stride(expect.shape), expect)
    else:
        t.store_expect(x2, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape", [[3, 300], [2, 64, 1024], [4, 8192]])
def test_concat_int64_generalization(shape):
    # int64 (64-bit) concat generalization: actually supported in unconstrained cases (dims with size>1)
    t = Tester()
    a = np.random.randint(0, 1000, shape).astype(np.int64)
    b = np.random.randint(0, 1000, shape).astype(np.int64)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.concat([x0, x1], 1)
    t.store_expect(x2, np.concatenate([a, b], axis=1))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_concat_add_broadcast():
    # concat + binary(add) + broadcast fusion: add broadcast scalar to concat result
    t = Tester()
    a = np.random.normal(0, 1, [3, 300]).astype(np.float32)
    b = np.random.normal(0, 1, [3, 400]).astype(np.float32)
    s = np.random.normal(0, 1, [3, 1]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.concat([x0, x1], 1)
    x3 = t.add(x2, t.load(s))
    t.store_expect(x3, np.concatenate([a, b], axis=1) + s)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_mul_broadcast():
    # split + binary(mul) + broadcast fusion: add broadcast scalar to each split chunk then mul
    t = Tester()
    a = np.random.normal(0, 1, [3, 900]).astype(np.float32)
    x0 = t.load(a)
    xs = t.split(x0, 1, 300, 3)
    s = np.random.normal(0, 1, [3, 1]).astype(np.float32)
    for i, x in enumerate(xs):
        y = t.mul(t.add(x, t.load(s)), 0.5)
        t.store_expect(y, (np.split(a, [300, 600], 1)[i] + s) * 0.5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_tree_multi_branch():
    # three blocks each consumed by its own branch: add(a1,c), add(a2,d), add(a3,e)
    # each op consumes a single block -> should work
    t = Tester()
    a = np.random.normal(0, 1, [6, 300]).astype(np.float32)
    c = np.random.normal(0, 1, [2, 300]).astype(np.float32)
    d = np.random.normal(0, 1, [2, 300]).astype(np.float32)
    xa = t.load(a)
    xc = t.load(c)
    xd = t.load(d)
    xe = t.add(xa, 0.8)
    a1, a2, a3 = t.split(xa, 0, 2, 3)
    b1 = t.add(a1, xc)
    b2 = t.add(a2, xc)
    b3 = t.add(a3, xd)
    b4 = t.mul(b3, 0.6)
    split_e = np.split(a, 3, 0)
    t.store_expect(xe, a + 0.8)
    t.store_expect(b1, split_e[0] + c)
    t.store_expect(b2, split_e[1] + c)
    t.store_expect(b3, split_e[2] + d)
    t.store_expect(b4, (split_e[2] + d) * 0.6)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_concat_core_reserve():
    # concat quota imbalance: first segment (24576 rows) rounds up to all cores,
    # second segment (1 row) must be reserved at least 1 core, otherwise it gets
    # 0 cores and its output is all zeros (silent misalignment)
    t = Tester()
    lhs = np.random.normal(0, 1, [24576, 32]).astype(np.float32)
    rhs = np.random.normal(0, 1, [1, 32]).astype(np.float32)
    x0 = t.load(lhs)
    x1 = t.load(rhs)
    xout = t.concat([x0, x1], 0)
    t.store_expect(xout, np.concatenate([lhs, rhs], axis=0))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_shared_load():
    t = Tester()
    a0 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    a1 = np.random.normal(0, 1, [3, 600]).astype(np.float16)
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.concat([x0, x0, x1], 1)  # x0 shared by slices 0 and 1
    x3 = t.add(x2, 0.1)
    expect = np.concatenate([a0, a0, a1], axis=1) + 0.1
    t.store_expect(x3, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_slice_store():
    t = Tester()
    a0 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    a1 = np.random.normal(0, 1, [3, 600]).astype(np.float16)
    x0 = t.mul(t.load(a0), 0.5)
    x1 = t.load(a1)
    x2 = t.concat([x0, x1], 1)  # x0 shared by slices 0 and 1
    x3 = t.add(x2, 0.1)
    e0 = a0 * 0.5
    e3 = np.concatenate([e0, a1], axis=1) + 0.1
    t.store_expect(x0, e0)
    t.store_expect(x3, e3)
    assert (t.run_check())
