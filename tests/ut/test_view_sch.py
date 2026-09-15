
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
    [[5, 3, 32, 1], (2, 3), [5, 3, 32, 129], (2, 3), True],  # h broadcast + h tail
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
def test_dup_tiling_broadcast_zero_tail():
    t = Tester()
    lhs_src = np.random.normal(0, 1, [1, 1, 2, 1]).astype(np.float16)
    lhs = t.view_load([1, 2, 1, 1], [2, 1, 2, 1], lhs_src)
    rhs_src = np.random.normal(0, 1, [4, 1, 23, 257]).astype(np.float16)
    rhs = t.view_load([4, 1, 257, 23], [5911, 5911, 1, 257], rhs_src)
    result = t.copy(t.add(lhs, rhs))
    expected = np.swapaxes(lhs_src, 1, 2) + np.swapaxes(rhs_src, 2, 3)
    t.store_expect(result, expected)
    t.codegen()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape, swap", [
    ([4, 70, 20, 200], (1, 3)),   # first loop axis is 1
    ([2, 2048, 16, 128], (2, 3)), # middle loop axis is 1
    ([129, 128], (0, 1)) # w_tail is 1
])
def test_trans_fractal_inner_axis(shape, swap):
    t = Tester("vector:opt_fractal")
    a = np.random.normal(0, 1, shape).astype(np.float16)
    expect = np.swapaxes(a, swap[0], swap[1])
    stride = _continuous_stride(shape)
    stride[swap[0]], stride[swap[1]] = stride[swap[1]], stride[swap[0]]
    x = t.view_load(expect.shape, stride, a)
    t.view_store_expect(x, _continuous_stride(expect.shape), expect)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_split_shared_load_sideway_store():
    t = Tester()
    a0 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    a1 = np.random.normal(0, 1, [3, 1]).astype(np.float16)
    a5 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    ay = np.random.normal(0, 1, [3, 200]).astype(np.float16)
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.add(x0, x1)
    t.store_expect(x2, a0 + a1)
    x5 = t.load(a5)
    x6 = t.sub(x5, x1)
    t.store_expect(x6, a5 - a1)
    s0, s1 = t.split(x2, 1, 200, 2)
    y1 = t.load(ay)
    y2 = t.add(s0, y1)
    t.store_expect(y2, np.split(a0 + a1, [200], 1)[0] + ay)
    t.store_expect(s1, np.split(a0 + a1, [200], 1)[1])
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_shared_load_sideway_store():
    t = Tester()
    a0 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    a1 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    a5 = np.random.normal(0, 1, [3, 400]).astype(np.float16)
    ay = np.random.normal(0, 1, [3, 800]).astype(np.float16)
    x0 = t.load(a0)
    x1 = t.load(a1)
    x2 = t.add(x0, x1)
    t.store_expect(x2, a0 + a1)
    x5 = t.load(a5)
    x6 = t.sub(x5, x1)
    y0 = t.concat([x2, x6], 1)
    y1 = t.load(ay)
    y2 = t.add(y1, y0)
    t.store_expect(y2, np.concatenate([a0 + a1, a5 - a1], axis=1) + ay)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("in_shape, cat_dim, red_dim", [
    ([600, 128], 0, (1,)), # no atomic
    ([100, 4000], 1, (0,)), # red dim != cat dim
    ([32, 9000], 0, (0,)), # red dim == cat dim
])
def test_sch_concat_reduce(in_shape, cat_dim, red_dim):
    t = Tester()
    a0 = np.random.normal(0, 0.05, in_shape).astype(np.float32)
    a1 = np.random.normal(0, 0.04, in_shape).astype(np.float32)
    x0 = t.concat([t.load(a0), t.load(a1)], cat_dim)
    x1 = t.sum(x0, red_dim, True)
    t.store_expect(x1, np.sum(np.concatenate([a0, a1], axis=cat_dim), red_dim, keepdims=True))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape1, broad_shape1, shape2, cat_dim", [
    ([1, 512], [20, 512], [4, 5, 512], 1), # broadcast reshape, concat no reshape: prop ok
    ([1, 512], [20, 512], [20, 4, 128], 1), # broadcast no reshape, concat reshape: prop ok
    ([1, 512], [20, 512], [20 * 512], 0), # fallback
])
def test_sch_concat_reshape(shape1, broad_shape1, shape2, cat_dim):
    t = Tester()
    a0 = np.random.normal(0, 1, shape1).astype(np.float16)
    a1 = np.random.normal(0, 1, shape2).astype(np.float16)
    x0 = t.broadcast(t.load(a0), broad_shape1)
    x1 = t.reshape(x0, shape2)
    x2 = t.load(a1)
    x3 = t.concat([x1, x2], cat_dim)
    t.store_expect(x3, np.concatenate([np.broadcast_to(a0, broad_shape1).reshape(shape2), a1], axis=cat_dim))
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape1, shape2, reshape, split_dim", [
    ([1, 512], [20, 512], [4, 5, 512], 0), # backward prop ok
    ([1, 10, 64], [20, 10, 64], [200, 64], 1), # forward prop ok
    ([1, 512], [20, 512], [20 * 512], 0), # fallback
])
def test_sch_split_reshape(shape1, shape2, reshape, split_dim):
    t = Tester()
    a = np.random.normal(0, 1, shape1).astype(np.float32)
    b = np.random.normal(0, 1, shape2).astype(np.float32)
    x0 = t.add(t.load(a), t.load(b))
    x1 = t.reshape(x0, reshape)
    x2 = t.add(x1, 0.1)
    x3, x4 = t.split(x2, split_dim, reshape[split_dim] // 2, 2)
    expect = np.split((a + b).reshape(reshape) + 0.1, 2, split_dim)
    t.store_expect(x3, expect[0])
    t.store_expect(x4, expect[1])
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_update_slice():
    t = Tester(use_pass_opt=True)
    lhs = np.full([1, 8, 512], 0.5, dtype=np.float32)
    rhs = np.full([1, 8, 1, 32, 2], -0.25, dtype=np.float32)

    x0 = t.load(lhs, "bfloat16")
    x1 = t.load(rhs)
    x1 = t.reshape(x1, [1, 8, 1, 64])
    x1 = t.cast(x1, "bfloat16")
    x1 = t.reshape(x1, [1, 8, 64])

    out = t.concat([x0, x1], 2)

    expect = np.concatenate([lhs, rhs.reshape(1, 8, 64)], axis=2)
    t.store_expect(out, expect, eps=0)
    assert (t.run_check())


@pytest.mark.parametrize("shape1, shape2, start, slice1, slice2", [
    ([100, 2000], [1, 2000], [10, 30], [60, 1500], [60, 1500]),    # before slice broadcast
    ([100, 2000], [100, 2000], [10, 30], [60, 1500], [60, 1]),    # after slice broadcast
    ([100, 1000], [100, 1000], [0, 20], [100, 500], [100, 500]),    # start has 0
])
def test_sch_slice(shape1, shape2, start, slice1, slice2):
    t = Tester()
    a = np.random.normal(0, 1, shape1).astype(np.float32)
    b = np.random.normal(0, 1, shape2).astype(np.float32)
    c = np.random.normal(0, 1, slice2).astype(np.float32)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.mul(x0, x1)
    x3 = t.slice(x2, start, slice1)
    x4 = t.add(x3, t.load(c))
    expect = (a * b)[start[0] : start[0] + slice1[0], start[1] : start[1] + slice1[1]] + c
    t.store_expect(x4, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_store():
    t = Tester()
    a = np.random.normal(0, 1, [10, 1024]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.mul(x0, 0.6)
    x2 = t.slice(x1, [2, 0], [8, 512])
    x3 = t.add(x2, 0.1)
    x4 = t.mul(x3, 0.4)
    x1_e = a * 0.6
    x2_e = x1_e[2:10, :512]
    x3_e = x2_e + 0.1
    t.store_expect(x1, x1_e)
    t.store_expect(x3, x3_e)
    t.store_expect(x4, x3_e * 0.4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_broadcast():
    t = Tester()
    a0 = np.random.normal(0, 1, [10, 20]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.mul(x0, 0.1)
    s1 = t.slice(x1, [2, 5], [6, 1])
    a1 = np.random.normal(0, 1, [6, 20]).astype(np.float32)
    x2 = t.load(a1)
    add = t.add(s1, x2)
    z = t.exp(add)
    expect = np.exp(a0[2:8, 5:6] * 0.1 + a1)
    t.store_expect(z, expect)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_slice_chain():
    t = Tester()
    a = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    b = np.random.normal(0, 1, [600, 300]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.1)
    x2 = t.slice(x1, [100, 100], [600, 300])
    x3 = t.add(x2, t.load(b))
    x4 = t.slice(x3, [50, 0], [200, 160])
    x5 = t.exp(x4)
    e2 = (a + 0.1)[100:700, 100:400]
    t.store_expect(x2, e2)
    t.store_expect(x5, np.exp((e2 + b)[50:250, :160]))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_slice_v_branch():
    t = Tester()
    a0 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    a1 = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.mul(x0, 0.1)
    s0 = t.slice(x1, [100, 100], [200, 100])
    x2 = t.load(a1)
    x3 = t.mul(x2, 0.1)
    s1 = t.slice(x3, [200, 200], [200, 100])
    z = t.add(s0, s1)
    expect = a0[100:300, 100:200] * 0.1 + a1[200:400, 200:300] * 0.1
    t.store_expect(z, expect)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_reshape_slice_chain():
    t = Tester()
    a = np.random.normal(0, 1, [1024, 500]).astype(np.float32)
    x0 = t.load(a)
    x1 = t.add(x0, 0.1)
    x2 = t.slice(x1, [100, 100], [600, 300])
    x3 = t.mul(x2, 0.9)
    x3 = t.reshape(x3, [200, 3, 300])
    x4 = t.slice(x3, [0, 1, 0], [200, 2, 300])
    x5 = t.exp(x4)
    e2 = (a + 0.1)[100:700, 100:400]
    t.store_expect(x2, e2)
    t.store_expect(x5, np.exp((e2 * 0.9).reshape(200, 3, 300)[:, 1:, :]))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_concat():
    t = Tester()
    a0 = np.random.normal(0, 1, [600, 300]).astype(np.float32)
    a1 = np.random.normal(0, 1, [400, 200]).astype(np.float32)
    a2 = np.random.normal(0, 1, [200, 200]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.mul(x0, 0.1)
    s1 = t.slice(x1, [100, 100], [400, 200])
    x2 = t.load(a1)
    add = t.add(s1, x2)
    x3 = t.load(a2)
    c = t.concat([add, x3], 0)
    z = t.exp(c)
    expect = np.exp(np.concatenate([a0[100:500, 100:300] * 0.1 + a1, a2], axis=0))
    t.store_expect(z, expect)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_concat_slice():
    t = Tester()
    a0 = np.random.normal(0, 1, [600, 300]).astype(np.float32)
    a1 = np.random.normal(0, 1, [100, 300]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.mul(x0, 0.1)
    s1 = t.slice(x1, [100, 0], [300, 300])
    x2 = t.load(a1)
    c = t.concat([s1, x2], 0)
    a2 = np.random.normal(0, 1, [600, 300]).astype(np.float32)
    x3 = t.load(a2)
    x4 = t.mul(x3, 0.1)
    s2 = t.slice(x4, [100, 0], [400, 300])
    z = t.add(c, s2)
    expect = np.concatenate([a0[100:400] * 0.1, a1], axis=0) + a2[100:500] * 0.1
    t.store_expect(z, expect)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_concat_broadcast():
    t = Tester()
    a0 = np.random.normal(0, 1, [600, 200]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.mul(x0, 0.1)
    s1 = t.slice(x1, [100, 50], [400, 1])
    a1 = np.random.normal(0, 1, [400, 200]).astype(np.float32)
    x2 = t.load(a1)
    add = t.add(s1, x2)
    a2 = np.random.normal(0, 1, [200, 200]).astype(np.float32)
    x3 = t.load(a2)
    c = t.concat([add, x3], 0)
    z = t.exp(c)
    expect = np.exp(np.concatenate([a0[100:500, 50:51] * 0.1 + a1, a2], axis=0))
    t.store_expect(z, expect)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_split_concat():
    t = Tester()
    a0 = np.random.normal(0, 1, [500, 512]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.1)
    x2, x3, x4 = t.split(x1, 0, 200, 3)
    x5 = t.mul(x2, 0.8)
    x6 = t.concat([x5, x3], 1)
    e2, e3, e4 = np.split(a0 + 0.1, [200, 400])
    t.store_expect(x4, e4)
    t.store_expect(x6, np.concatenate([e2 * 0.8, e3], axis=1))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_split_slice():
    t = Tester()
    a0 = np.random.normal(0, 1, [500, 512]).astype(np.float32)
    x0 = t.load(a0)
    x1 = t.add(x0, 0.1)
    x2 = t.slice_dim(x1, 0, 50, 450)
    x3, x4 = t.split(x2, 0, 200, 2)
    x5 = t.exp(x3)
    x6 = t.slice_dim(x4, 1, 100, 300)
    e3, e4 = np.split((a0 + 0.1)[50:450, :], 2)
    t.store_expect(x5, np.exp(e3))
    t.store_expect(x6, e4[:, 100:300])
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_slice_same_src_same_op():
    t = Tester(use_pass_opt=True)
    a0 = np.random.normal(0, 1, [4, 600]).astype(np.float32)
    x0 = t.add(t.load(a0), 0.1)
    s0 = t.slice(x0, [0, 0], [4, 300])
    s1 = t.slice(x0, [0, 150], [4, 300])
    z = t.add(s0, s1)
    e = a0 + 0.1
    t.store_expect(z, e[:, 0:300] + e[:, 150:450])
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_slice_deep_diamond():
    t = Tester(use_pass_opt=True)
    a0 = np.random.normal(0, 1, [4, 600]).astype(np.float32)
    x0 = t.add(t.load(a0), 0.1)
    s0 = t.slice(x0, [0, 0], [4, 300])
    s1 = t.slice(x0, [0, 150], [4, 300])
    p0 = t.mul(s0, 0.5)
    p1 = t.add(s1, 0.2)
    z = t.add(p0, p1)
    e = a0 + 0.1
    t.store_expect(z, e[:, 0:300] * 0.5 + (e[:, 150:450] + 0.2))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_split_slice_same_src_concat():
    t = Tester(use_pass_opt=True)
    a0 = np.random.normal(0, 1, [4, 600]).astype(np.float32)
    x0 = t.add(t.load(a0), 0.1)
    s0a, s0b = t.split(x0, 1, 300, 2)
    sl1 = t.slice(x0, [0, 150], [4, 300])
    c = t.concat([s0a, sl1], 1)
    e = a0 + 0.1
    t.store_expect(t.exp(c), np.exp(np.concatenate([e[:, 0:300], e[:, 150:450]], axis=1)))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_diamond_nested_right():
    t = Tester(use_pass_opt=True)
    a0 = np.arange(240, dtype=np.float32).reshape(4, 60)
    x = t.add(t.load(a0), 0.0)
    s1 = t.slice(x, [0, 0], [4, 15])
    s2 = t.slice(x, [0, 30], [4, 30])
    s3 = t.slice(s2, [0, 0], [4, 15])
    s4 = t.slice(s2, [0, 15], [4, 15])
    d1 = t.add(s3, s4)
    d2 = t.add(s1, d1)
    e_d1 = a0[:, 30:45] + a0[:, 45:60]
    t.store_expect(d2, a0[:, 0:15] + e_d1)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_diamond_nested_both():
    t = Tester(use_pass_opt=True)
    a0 = np.arange(240, dtype=np.float32).reshape(4, 60)
    x = t.add(t.load(a0), 0.0)
    s1 = t.slice(x, [0, 0], [4, 30])
    s2 = t.slice(s1, [0, 0], [4, 15])
    s3 = t.slice(s1, [0, 15], [4, 15])
    d1 = t.add(s2, s3)
    s4 = t.slice(x, [0, 30], [4, 30])
    s5 = t.slice(s4, [0, 0], [4, 15])
    s6 = t.slice(s4, [0, 15], [4, 15])
    d2 = t.add(s5, s6)
    d3 = t.add(d1, d2)
    e_d1 = a0[:, 0:15] + a0[:, 15:30]
    e_d2 = a0[:, 30:45] + a0[:, 45:60]
    t.store_expect(d3, e_d1 + e_d2)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_slice_bcast_concat_direct_join():
    t = Tester(use_pass_opt=True)
    a0 = np.arange(240, dtype=np.float32).reshape(4, 60)
    a1 = np.arange(240, dtype=np.float32).reshape(4, 60) + 1000
    x = t.add(t.load(a0), 0.0)
    s1 = t.slice(x, [0, 30], [4, 1])
    c1 = t.concat([x, t.load(a1)], 1)
    b1 = t.broadcast(s1, [4, 120])
    z = t.add(b1, c1)
    e_c1 = np.concatenate([a0, a1], axis=1)
    e_s1 = np.broadcast_to(a0[0:4, 30:31], [4, 120])
    t.store_expect(z, e_s1 + e_c1)
    t.store_expect(c1, e_c1)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_diamond_view_load():
    t = Tester(use_pass_opt=True)
    a0 = np.arange(144, dtype=np.float32).reshape(4, 36)
    vl = t.view_load([4, 36], [36, 1], a0)
    x = t.add(vl, 0.0)
    s0a, s0b, s0c = t.split(x, 1, 12, 3)
    sl = t.slice(x, [0, 12], [4, 12])
    z = t.add(s0b, sl)
    t.store_expect(z, 2 * a0[:, 12:24])
    assert t.run_check(True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_diamond_join_view_load_non_contig():
    t = Tester(use_pass_opt=True)
    a0 = np.arange(64, dtype=np.float32).reshape(8, 8)
    vl = t.view_load([8, 4], [8, 2], a0)
    s0a, s0b = t.split(vl, 1, 2, 2)
    sl = t.slice(vl, [0, 2], [8, 2])
    z = t.add(s0b, sl)
    t.store_expect(z, 2 * a0[:, [4, 6]])
    assert t.run_check(True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sch_diamond_dyn():
    t = Tester("vector:dyn", use_pass_opt=True)
    x = t.load([-1, 120], "float32")
    y = t.add(x, 0.0)
    s0 = t.slice(y, [0, 0], [4, 60])
    s1 = t.slice(y, [0, 60], [4, 60])
    z = t.add(s0, s1)
    out = t.store(z)
    for nrows in [4, 8, 16]:
        a = np.arange(nrows * 120, dtype=np.float32).reshape(nrows, 120)
        t.input(x, a)
        expect = a[:4, 0:60] + a[:4, 60:120]
        t.set_output(out, np.zeros_like(expect))
        t.run()
        assert t.check(out, expect)
