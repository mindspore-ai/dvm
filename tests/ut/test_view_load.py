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
@pytest.mark.parametrize("in_shape, slice_shape, tile_depth, tile_tail", [
    ([30, 1000], [20, 500], 0, 0),
    ([30, 500], [20, 200], 1, 7),  # tile 1 with tail
    ([30, 4000], [20, 3000], 2, 3),  # tile 2
    ([43, 1000], [40, 500], 1, 40),  # tile 1 not tail
])
def test_slice_2d(in_shape, slice_shape, tile_depth, tile_tail):
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.view_load(slice_shape, [in_shape[1], 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[0:slice_shape[0], 0:slice_shape[1]] + 0.1)
    if tile_depth == 1:
        t.tile(1, 1, tile_tail)
    elif tile_depth == 2:
        t.tile(1, 1, slice_shape[0])
        t.tile(0, 0, tile_tail)
    else:
        pass
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("in_shape, slice_shape, tile_depth, tile_tail", [
    ([24, 20, 100], [20, 10, 60], 1, 10),  # tile 1
    ([24, 20, 1000], [20, 10, 600], 2, 5),  # tile 2
    ([24, 20, 1000], [20, 10, 600], 3, 3),  # tile 3
    ([24, 20, 1000], [20, 20, 600], 3, 3),  # tile continuous
    ([24, 20, 64], [20, 15, 64], 1, 20),  # loop continuous
    ([24, 15, 64], [20, 15, 64], 1, 20),  #
])
def test_slice_3d(in_shape, slice_shape, tile_depth, tile_tail):
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.view_load(slice_shape, [in_shape[1] * in_shape[2], in_shape[2], 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[0:slice_shape[0], 0:slice_shape[1], 0:slice_shape[2]] + 0.1)
    if tile_depth == 1:
        t.tile(2, 2, tile_tail)
    elif tile_depth == 2:
        t.tile(2, 2, slice_shape[0])
        t.tile(1, 1, tile_tail)
    elif tile_depth == 3:
        t.tile(2, 2, slice_shape[0])
        t.tile(1, 1, slice_shape[1])
        t.tile(0, 0, tile_tail)
    else:
        pass
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_slice_loop_2d():
    t = Tester()
    in_shape = [4, 4, 5, 3, 16]
    slice_shape = [3, 2, 3, 2, 12]
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.view_load(slice_shape, [16 * 3 * 5 * 4, 16 * 3 * 5, 16 * 3, 16, 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[0:slice_shape[0], 0:slice_shape[1], 0:slice_shape[2], 0:slice_shape[3], 0:slice_shape[4]] + 0.1)
    t.tile(4, 4, 3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_slice_loop_3d():
    t = Tester()
    in_shape = [4, 4, 4, 5, 6, 16]
    s_shape = [3, 2, 3, 4, 5, 12]
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.view_load(s_shape, [16 * 6 * 5 * 4 * 4, 16 * 6 * 5 * 4, 16 * 6 * 5, 16 * 6, 16, 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[0:s_shape[0], 0:s_shape[1], 0:s_shape[2], 0:s_shape[3], 0:s_shape[4], 0:s_shape[5]] + 0.1)
    t.tile(5, 5, 3)
    assert (t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_slice_dim_fold():
    t = Tester()
    in_shape = [4, 2, 64, 96]
    s_shape = [4, 2, 64, 64]
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x = t.view_load(s_shape, [12288, 6144, 96, 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[:, :, :, 0:64] + 0.1)
    t.tile(3, 3, 4)
    assert (t.run_check())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape, dim0, dim1", [
    ([8, 30, 512], 0, 1),
    ([8, 6, 20, 64], 1, 2),
    ([8, 6, 20, 64], 0, 2),
])
def test_transpose(shape, dim0, dim1):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(np.float32)
    stride = [1] * len(shape)
    for i in range(len(shape) - 1, 0, -1):
        stride[i - 1] = stride[i] * shape[i]
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    x = t.view_load(shape, stride, a)
    y = t.add(x, 0.1)
    t.store_expect(y, np.swapaxes(a, dim0, dim1) + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape, new_shape", [
    ([1, 10, 64], [40, 10, 64]),
    ([1, 1, 64], [40, 10, 64]),
    ([1, 3, 1, 64], [5, 3, 10, 64]),
])
def test_broadcast(shape, new_shape):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(np.float32)
    stride = [1] * len(shape)
    cur_stride = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = 0 if shape[i] != new_shape[i] else cur_stride
        cur_stride *= shape[i]
    x = t.view_load(new_shape, stride, a)
    y = t.add(x, 0.1)
    t.store_expect(y, np.broadcast_to(a, new_shape) + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_broadcast_tiling():
    t = Tester()
    shape, sshape, bshape = [1, 10, 1, 20, 8, 32], [1, 6, 1, 18, 7, 32], [4, 6, 8, 18, 7, 32]
    a = np.random.normal(0, 1, shape).astype(np.float32)
    b = np.random.normal(0, 1, bshape).astype(np.float32)
    stride = [1] * len(shape)
    for i in range(len(shape) - 1, 0, -1):
        stride[i - 1] = stride[i] * shape[i]
    x = t.view_load(sshape, stride, a)
    y = t.load(b)
    z = t.add(x, y)
    t.store_expect(z, a[:, :6, :, :18, :7, :] + b)
    t.tile(5, 5, 4)
    t.tile(4, 4, 6)
    t.tile(3, 3, 8)
    t.tile(2, 2, 9)
    assert (t.run_check())
