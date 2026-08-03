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
import dvm
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
    ([200, 20, 100], [200, 20, 30], 1, 11),  # tail fold
    ([200, 20, 60], [200, 10, 30], 0, 0),  # fold prop
    ([200, 20, 20], [200, 10, 20], 0, 0),  # align prop
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
def test_slice_dim_x():
    t = Tester()
    a = np.random.normal(0, 1, [100, 2048]).astype(np.float32)
    x = t.view_load([100, 200], [2048, 10], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[:, 0:2000:10] + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_slice_dim_x_fp16_ws_reserve():
    t = Tester()
    a = np.random.normal(0, 1, [4096, 20000]).astype(np.float16)
    x = t.view_load([4096, 10000], [20000, 2], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[:, 0:20000:2] + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_slice_dim_x_lead_1():
    t = Tester()
    a = np.random.normal(0, 1, [100, 2048, 1]).astype(np.float16)
    x = t.view_load([100, 199, 1], [2048, 10, 1], a)
    y = t.add(x, 0.1)
    t.store_expect(y, a[:, 0:1990:10, :] + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("stride_elem", [2, 3, 4, 5, 6, 7, 8])
def test_slice_dim_x_stride(stride_elem):
    t = Tester()
    shape = [8, 64]
    in_shape = [shape[0], shape[1] * stride_elem]
    row = ((np.arange(in_shape[0], dtype=np.float32) % 251) * 0.01).astype(np.float16)
    col = ((np.arange(in_shape[1], dtype=np.float32) % 257) * 0.001).astype(np.float16)
    a = (row[:, None] + col[None, :]).astype(np.float16, copy=False)
    x = t.view_load(shape, [in_shape[1], stride_elem], a)
    y = t.add(x, 0.5)
    t.store_expect(y, a[:, 0:shape[1] * stride_elem:stride_elem] + 0.5)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape, dim0, dim1", [
    ([8, 30, 512], 0, 1),
    ([8, 6, 20, 64], 1, 2),
    ([8, 6, 20, 64], 0, 2),
    ([1024, 512], 0, 1),
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
    ([8, 1, 1, 1], [8, 3, 10, 64]), # lead broadcast
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_lead_tiled():
    t = Tester()
    a = np.random.normal(0, 0.1, [32, 1, 1, 1]).astype(np.float32)
    b = np.random.normal(0, 0.1, [16, 10, 1, 4000]).astype(np.float32)
    x0 = t.view_load([16, 1, 1, 1], [2, 1, 1, 1], a)
    x1 = t.load(b)
    x2 = t.add(x0, x1)
    t.store_expect(x2, a[::2, :, :, :] + b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("shape, split_dim, split_num, split_idx", [
    ([40, 100, 64], 1, 5, 3),
    ([40, 1000], 1, 5, 0),
])
def test_split(shape, split_dim, split_num, split_idx):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(np.float32)
    new_shape = [1] * len(shape)
    split_size = int(shape[split_dim] // split_num)
    for i in range(len(shape) - 1, -1, -1):
        new_shape[i] = shape[i] if i != split_dim else split_size
    stride = [1] * len(shape)
    for i in range(len(shape) - 1, 0, -1):
        stride[i - 1] = shape[i] * stride[i]
    offset = split_idx * split_size * stride[split_dim]
    x = t.view_load(new_shape, stride, a, offset)
    y = t.add(x, 0.1)
    t.store_expect(y, np.split(a, split_num, split_dim)[split_idx] + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("slice_shape, in_shape, broadcast_shape", [
    ([3, 30, 100], [3, 60, 200], [4, 3, 30, 100]),  # ext broadcast
    ([4, 1, 512], [6, 2, 512], [4, 8, 512]),  # inner broadcat
])
def test_view_load_broadcast_3d(slice_shape, in_shape, broadcast_shape):
    t = Tester()
    a = np.random.normal(0, 1, in_shape).astype(np.float32)
    x0 = t.view_load(slice_shape, [in_shape[1] * in_shape[2], in_shape[2], 1], a)
    x1 = t.add(x0, 0.1)
    b = np.random.normal(0, 1, broadcast_shape).astype(np.float32)
    x2 = t.add(x1, t.load(b))
    t.store_expect(x2, a[:slice_shape[0], :slice_shape[1], :slice_shape[2]] + 0.1 + b)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_load_broadcast_data_cache():
    t = Tester()
    a = np.random.normal(0, 1, [4, 512]).astype(np.float32)
    x0 = t.view_load([4, 5, 500, 512], [512, 0, 0, 1], a)
    t.store_expect(x0, a[:, None, None, :])
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_load_x_broadcast_data_cache():
    t = Tester()
    a = np.random.normal(0, 1, [4, 512]).astype(np.float32)
    x0 = t.view_load([4, 5, 500, 128], [512, 0, 0, 4], a)
    t.store_expect(x0, a[:, None, None, :512:4])
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("out_shape, slice_shape, tile_depth, tile_tail", [
    ([30, 1000], [20, 500], 0, 0),
    ([30, 500], [20, 200], 1, 7),  # tile 1 with tail
    ([30, 4000], [20, 3000], 2, 3),  # tile 2
    ([43, 1000], [40, 500], 1, 40),  # tile 1 not tail
])
def test_view_store_2d(out_shape, slice_shape, tile_depth, tile_tail):
    t = Tester()
    a = np.random.normal(0, 1, slice_shape).astype(np.float32)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:slice_shape[0], :slice_shape[1]] = a + 0.1
    t.view_store_expect(y, [out_shape[1], 1], e)
    if tile_depth == 1:
        t.tile(1, 1, tile_tail)
    elif tile_depth == 2:
        t.tile(1, 1, slice_shape[0])
        t.tile(0, 0, tile_tail)
    else:
        pass
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("out_shape, slice_shape, tile_depth, tile_tail", [
    ([24, 20, 100], [20, 10, 60], 1, 10),  # tile 1
    ([24, 20, 1000], [20, 10, 600], 2, 5),  # tile 2
    ([24, 20, 1000], [20, 10, 600], 3, 3),  # tile 3
    ([24, 20, 1000], [20, 20, 600], 3, 3),  # tile continuous
    ([24, 20, 64], [20, 15, 64], 1, 20),  # loop continuous
    ([24, 15, 64], [20, 15, 64], 1, 20),  #
    ([200, 20, 100], [200, 20, 30], 1, 11),  # tail fold
    ([200, 20, 60], [200, 10, 30], 0, 0),  # fold prop
    ([200, 20, 20], [200, 10, 20], 0, 0),  # align prop
])
def test_view_store_3d(out_shape, slice_shape, tile_depth, tile_tail):
    t = Tester()
    a = np.random.normal(0, 1, slice_shape).astype(np.float32)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:slice_shape[0], :slice_shape[1], :slice_shape[2]] = a + 0.1
    t.view_store_expect(y, [out_shape[1] * out_shape[2], out_shape[2], 1], e)
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
def test_view_store_loop_2d():
    t = Tester()
    out_shape = [4, 4, 5, 3, 16]
    slice_shape = [3, 2, 3, 2, 12]
    a = np.random.normal(0, 1, slice_shape).astype(np.float32)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:slice_shape[0], :slice_shape[1], :slice_shape[2], :slice_shape[3], :slice_shape[4]] = a + 0.1
    t.view_store_expect(y, [16 * 3 * 5 * 4, 16 * 3 * 5, 16 * 3, 16, 1], e)
    t.tile(4, 4, 3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_store_loop_3d():
    t = Tester()
    out_shape = [4, 4, 4, 5, 6, 16]
    slice_shape = [3, 2, 3, 4, 5, 12]
    a = np.random.normal(0, 1, slice_shape).astype(np.float32)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:slice_shape[0], :slice_shape[1], :slice_shape[2], :slice_shape[3], :slice_shape[4], :slice_shape[5]] = a + 0.1
    t.view_store_expect(y, [16 * 6 * 5 * 4 * 4, 16 * 6 * 5 * 4, 16 * 6 * 5, 16 * 6, 16, 1], e)
    t.tile(5, 5, 3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_store_dim_fold():
    t = Tester()
    out_shape = [4, 2, 64, 96]
    s_shape = [4, 2, 64, 64]
    a = np.random.normal(0, 1, s_shape).astype(np.float32)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:, :, :, :64] = a + 0.1
    t.view_store_expect(y, [12288, 6144, 96, 1], e)
    t.tile(3, 3, 4)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("slice_shape, out_shape, broadcast_shape", [
    ([3, 30, 100], [3, 60, 200], [4, 3, 30, 100]),  # ext broadcast
    ([4, 1, 512], [6, 2, 512], [4, 8, 512]),  # inner broadcat
])
def test_view_store_broadcast_3d(slice_shape, out_shape, broadcast_shape):
    t = Tester()
    a = np.random.normal(0, 1, slice_shape).astype(np.float32)
    x0 = t.add(t.load(a), 0.1)
    e = np.full(out_shape, 0.0, np.float32)
    e[:slice_shape[0], :slice_shape[1], :slice_shape[2]] = a + 0.1
    t.view_store_expect(x0, [out_shape[1] * out_shape[2], out_shape[2], 1], e)
    b = np.random.normal(0, 1, broadcast_shape).astype(np.float32)
    x2 = t.add(x0, t.load(b))
    t.store_expect(x2, a + 0.1 + b)
    assert (t.run_check())


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_view_store_x(dtype):
    t = Tester()
    a = np.random.normal(0, 1, [400, 600]).astype(dtype)
    x = t.load(a)
    y = t.add(x, 0.1)
    e = np.full([400, 2000], 0.0, dtype)
    e[:, :1800:3] = a + 0.1
    t.view_store_expect(y, [2000, 3], e)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_store_x_transpose():
    t = Tester()
    a = np.random.normal(0, 1, [2000, 4000]).astype(np.float32)
    x = t.load(a)
    t.view_store_expect(x, [1, 2000], np.swapaxes(a, 0, 1))
    assert (t.run_check())


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
@pytest.mark.parametrize("shape1, shape2, shape3, shape4, axis, view", [
    [[3, 400], [3, 600], [3, 200], [3, 1200], 1, False], # lead
    [[3, 400, 32], [3, 600, 32], [3, 200, 32], [3, 1, 32], 1, True], # middle, cat broadcast
    [[10, 400], [20, 400], [30, 400], [60, 1], 0, False], # out, no-cat broadcast
])
def test_sch_concat(shape1, shape2, shape3, shape4, axis, view):
    t = Tester()
    a0 = np.random.normal(0, 1, shape1).astype(np.float32)
    a1 = np.random.normal(0, 1, shape2).astype(np.float32)
    a2 = np.random.normal(0, 1, shape3).astype(np.float32)
    a3 = np.random.normal(0, 1, shape4).astype(np.float32)
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
@pytest.mark.parametrize("shape, split_size, dim, view", [
    ([3, 900], 300, 1, False),    # lead, evenly divisible
    ([3, 500, 32], 200, 1, True),    # middle, body + tail
])
def test_sch_split(shape, split_size, dim, view):
    split_num = (shape[dim] + split_size - 1) // split_size
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(np.float32)
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
