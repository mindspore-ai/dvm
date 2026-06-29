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

import numpy as np
import pytest

import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad currently only supports C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("src_shape, index_shape, axis", [
    ((256, 512), (129,), 0),
    ((16, 128, 256), (9, 33), 1),
    ((8, 64, 64), (65,), -1),
    ((4, 16, 64, 32), (5, 7), 2),
    ((4, 8, 16, 128), (3, 4, 9), -2),
])
def test_gather_load_take_add(src_shape, index_shape, axis, dtype):
    t = Tester()
    if np.issubdtype(dtype, np.integer):
        x_np = np.random.randint(-7, 8, src_shape).astype(dtype)
        bias = 2
    else:
        x_np = np.random.normal(0, 1, src_shape).astype(dtype)
        bias = 0.25
    gather_axis = axis if axis >= 0 else len(src_shape) + axis
    index_np = np.random.randint(-src_shape[gather_axis], src_shape[gather_axis], index_shape).astype(np.int32)
    index_op = t.global_access(index_np)
    y = t.gather_load(x_np, index_op, axis=axis, gather_mode=1)
    z = t.add(y, bias)
    t.store_expect(z, np.take(x_np, index_np, axis=axis) + bias)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad currently only supports C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("src_shape, gather_size, axis", [
    ((256, 512), 129, 0),
    ((16, 128, 256), 33, 1),
    ((8, 64, 64), 65, -1),
    ((4, 16, 64, 32), 7, 2),
    ((4, 8, 16, 128), 9, -2),
])
def test_gather_load_add(src_shape, gather_size, axis, dtype):
    t = Tester()
    if np.issubdtype(dtype, np.integer):
        x_np = np.random.randint(-7, 8, src_shape).astype(dtype)
        bias = 2
    else:
        x_np = np.random.normal(0, 1, src_shape).astype(dtype)
        bias = 0.25
    gather_axis = axis if axis >= 0 else len(src_shape) + axis
    index_shape = list(src_shape)
    index_shape[gather_axis] = gather_size
    index_np = np.random.randint(-src_shape[gather_axis], src_shape[gather_axis], index_shape).astype(np.int32)
    index_op = t.global_access(index_np)
    y = t.gather_load(x_np, index_op, axis=axis)
    z = t.add(y, bias)
    t.store_expect(z, np.take_along_axis(x_np, index_np, axis=axis) + bias)
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad currently only supports C310")
@pytest.mark.parametrize("src_shape, index_np, axis, out_shape, dtype, tiles", [
    ((1, 7, 17), np.array([[1]], dtype=np.int32), 1, (7, 5, 3, 17), np.float16, [(1, 1, 3)]),
    ((1, 6, 64), np.array([0], dtype=np.int32), 1, (6, 5, 64), np.float32, [(1, 1, 3)]),
])
def test_gather_load_take_round_tile(src_shape, index_np, axis, out_shape, dtype, tiles):
    t = Tester()
    x_np = (np.arange(np.prod(src_shape)).reshape(src_shape) % 97).astype(dtype)
    x_np = (x_np / 10).astype(dtype)
    index_op = t.global_access(index_np)
    y = t.gather_load(x_np, index_op, axis=axis, gather_mode=1)
    y = t.broadcast(y, out_shape)
    z = t.add(y, 0.5)
    t.store_expect(z, np.broadcast_to(np.take(x_np, index_np, axis=axis), out_shape) + 0.5)
    for tile in tiles:
        t.tile(*tile)
    t.codegen()
    assert "gather_load" in t.das()
    assert "gather_mode(1)" in t.das()
    assert "rounds(" in t.das()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad currently only supports C310")
@pytest.mark.parametrize("src_shape, index_np, axis, out_shape, dtype, tiles", [
    ((1, 7, 17), np.full((1, 1, 17), 1, dtype=np.int32), 1, (7, 5, 17), np.float16, [(1, 1, 3)]),
    ((1, 6, 64), np.zeros((1, 1, 64), dtype=np.int32), 1, (6, 5, 64), np.float32, [(1, 1, 3)]),
])
def test_gather_load_round_tile(src_shape, index_np, axis, out_shape, dtype, tiles):
    t = Tester()
    x_np = (np.arange(np.prod(src_shape)).reshape(src_shape) % 97).astype(dtype)
    x_np = (x_np / 10).astype(dtype)
    index_op = t.global_access(index_np)
    y = t.gather_load(x_np, index_op, axis=axis)
    y = t.broadcast(y, out_shape)
    z = t.add(y, 0.5)
    t.store_expect(z, np.broadcast_to(np.take_along_axis(x_np, index_np, axis=axis), out_shape) + 0.5)
    for tile in tiles:
        t.tile(*tile)
    t.codegen()
    assert "gather_load" in t.das()
    assert "rounds(" in t.das()
    assert t.run_check()
