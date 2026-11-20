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

import re

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


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("tile_width", [128, 192, 512, 1536])
def test_gather_block_copy_reuses_scratch(dtype, tile_width):
    """Index batches, outer wrap, partial suffixes, OOB and two live gathers."""
    t = Tester()
    x = (np.arange(3 * 7 * 512).reshape(3, 7, 512) % 31).astype(dtype)
    index = np.resize(np.array([0, -1, 7, -8, 2, 2, -7], np.int32), (3, 43))
    valid = (index >= -7) & (index < 7)
    safe = np.where(valid, index, 0)
    expected = np.take(x, safe, axis=1) * valid[None, :, :, None]
    first = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    second = t.gather_load(x + 1, t.global_access(index), axis=1, gather_mode=1)
    bias = t.load(np.full(expected.shape, 2, dtype=dtype))
    result = t.add(t.add(first, second), bias)
    t.store_expect(result, expected * 2 + valid[None, :, :, None] + 2)
    if tile_width is not None:
        t.tile(0, 3, (expected.size + tile_width - 1) // tile_width, tile_width)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    scratch = re.findall(r"index_scratch\((\d+)\)", t.das())
    assert len(scratch) == 2
    # Both loads finish using their indices before the next load starts.
    assert scratch[0] == scratch[1]
    for _ in range(3):
        assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("inner", [1, 17, 63, 65, 127, 129])
def test_gather_block_copy_fallback(inner):
    t = Tester()
    x = np.arange(7 * inner, dtype=np.float32).reshape(7, inner)
    index = np.array([6, -1, 0, 2], dtype=np.int32)
    y = t.gather_load(x, t.global_access(index), axis=0, gather_mode=1)
    t.store_expect(t.add(y, 1), np.take(x, index, axis=0) + 1)
    t.codegen()
    assert "strategy(simt)" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gather_block_copy_broadcast(dtype):
    t = Tester()
    x = np.arange(7 * 512).reshape(1, 7, 512).astype(dtype)
    index = np.array([[-1]], dtype=np.int32)
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    y = t.broadcast(y, (5, 3, 4, 512))
    t.store_expect(t.add(y, 1), np.broadcast_to(np.take(x, index, axis=1), (5, 3, 4, 512)) + 1)
    t.tile(1, 1, 3)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    assert "rounds(" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gather_block_copy_index_batches(dtype):
    t = Tester()
    x = (np.arange(3 * 7 * 128).reshape(3, 7, 128) % 31).astype(dtype)
    index = (np.arange(129, dtype=np.int32) % 7) - 7
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    expected = np.take(x, index, axis=1) + 1
    t.store_expect(t.add(y, 1), expected)
    # More than 64 indices per tile, and tiles cross the index/outer boundary.
    width = 128 * 96
    t.tile(0, 2, (expected.size + width - 1) // width, width)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    for _ in range(3):
        assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_block_copy_rebind_bf16():
    t = Tester()
    x = (np.arange(7 * 512).reshape(7, 512) % 31).astype(np.float32)
    index = np.arange(5, dtype=np.int32)
    index_op = t.global_access(index)
    y = t.gather_load(x, index_op, axis=0, dtype=dvm.DataType.bfloat16, gather_mode=1)
    out = t.store(t.add(y, 1))
    t.tile(0, 1, 5, 512)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    t.run()
    assert t.check(out, np.take(x, index, axis=0) + 1)
    # Rebinding indices after codegen must not retain a stale UB/GM index cache.
    index = np.array([-1, 3, 3, -7, 2], dtype=np.int32)
    t.input(index_op, index)
    t.run()
    assert t.check(out, np.take(x, index, axis=0) + 1)


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("index_count", [3, 2305])
def test_gather_block_copy_scratch_capacity(dtype, index_count):
    """Short slices, >1024-index batches, outer wrap and reused vector storage."""
    t = Tester()
    inner = 16
    outer = 65 if index_count > 1024 else 3
    x = (np.arange(outer * 7 * inner).reshape(outer, 7, inner) % 31).astype(dtype)
    index = np.resize(np.array([0, -1, 7, -8, 2, -7], np.int32), index_count)
    valid = (index >= -7) & (index < 7)
    expected = np.take(x, np.where(valid, index, 0), axis=1) * valid[None, :, None]
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    # Multiple live intermediates exercise reuse of scratch by later SIMD ops.
    a = t.add(y, 1)
    b = t.mul(y, 2)
    t.store_expect(t.mul(a, b), (expected + 1) * (expected * 2))
    width = inner * (1100 if index_count > 1024 else 2)
    t.tile(0, 2, (expected.size + width - 1) // width, width)
    t.codegen()
    capacity = int(re.search(r"index_capacity\((\d+)\)", t.das()).group(1))
    assert capacity == (1024 if index_count > 1024 else 2)
    if dtype == np.float32 and index_count > 1024:
        scratch = int(re.search(r"index_scratch\((\d+)\)", t.das()).group(1))
        output = int(re.search(r"Adds\.fp32\.\d+ (0x[0-9a-f]+)", t.das()).group(1), 16)
        assert output == scratch
    for _ in range(3):
        assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("tile_width", [17, 24])
def test_gather_block_copy_partial_short_slice(tile_width):
    t = Tester()
    x = np.arange(3 * 7 * 16, dtype=np.float32).reshape(3, 7, 16)
    index = np.array([0, -1, 2, 3, 4], np.int32)
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    expected = np.take(x, index, axis=1) + 1
    t.store_expect(t.add(y, 1), expected)
    t.tile(0, 2, (expected.size + tile_width - 1) // tile_width, tile_width)
    t.codegen()
    strategy = "simt" if tile_width == 17 else "simd_block_copy"
    assert f"strategy({strategy})" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_block_copy_many_loads():
    """Account for scratch descriptors in code UB and retire many loads safely."""
    t = Tester()
    x = np.arange(7 * 16, dtype=np.float32).reshape(7, 16)
    index = np.array([0, -1, 2, 3, 1, 2, -7, 4], np.int32)
    result = None
    expected = np.zeros((8, 16), np.float32)
    for i in range(32):
        y = t.gather_load(x + i, t.global_access(index), axis=0, gather_mode=1)
        result = y if result is None else t.add(result, y)
        expected += np.take(x + i, index, axis=0)
    t.store_expect(result, expected)
    t.tile(0, 1, 1, 128)
    t.codegen()
    assert t.das().count("strategy(simd_block_copy)") == 32
    for _ in range(3):
        assert t.run_check()
