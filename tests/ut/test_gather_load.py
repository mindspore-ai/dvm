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
    t = Tester(use_pass_opt=True)
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
    t = Tester(use_pass_opt=True)
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
    t = Tester(use_pass_opt=True)
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
    t = Tester(use_pass_opt=True)
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
    t.set_passes("ReorderLoad")
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
def test_gather_short_unaligned_slices(inner):
    t = Tester()
    t.set_passes("ReorderLoad")
    x = np.arange(7 * inner, dtype=np.float32).reshape(7, inner)
    index = np.array([6, -1, 0, 2], dtype=np.int32)
    y = t.gather_load(x, t.global_access(index), axis=0, gather_mode=1)
    t.store_expect(t.add(y, 1), np.take(x, index, axis=0) + 1)
    t.codegen()
    assert "strategy(simd_gather)" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gather_block_copy_broadcast(dtype):
    t = Tester()
    t.set_passes("ReorderLoad")
    x = np.arange(7 * 512).reshape(1, 7, 512).astype(dtype)
    index = np.array([[-1]], dtype=np.int32)
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    y = t.broadcast(y, (5, 3, 4, 512))
    t.store_expect(t.add(y, 1), np.broadcast_to(np.take(x, index, axis=1), (5, 3, 4, 512)) + 1)
    # Split the four-wide broadcast dimension into two complete tiles.
    t.tile(1, 1, 2)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    assert "rounds(" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gather_block_copy_index_batches(dtype):
    t = Tester()
    t.set_passes("ReorderLoad")
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
    t.set_passes("ReorderLoad")
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
    """Short slices, >1024-index batches, outer wrap and workspace reuse."""
    t = Tester()
    t.set_passes("ReorderLoad")
    # Four ordinary buffers are live at the peak. Keep the manual tile inside UB.
    inner = 8 if index_count > 1024 and np.dtype(dtype).itemsize == 4 else 16
    outer = 65 if index_count > 1024 else 3
    x = (np.arange(outer * 7 * inner).reshape(outer, 7, inner) % 31).astype(dtype)
    index = np.resize(np.array([0, -1, 7, -8, 2, -7], np.int32), index_count)
    valid = (index >= -7) & (index < 7)
    expected = np.take(x, np.where(valid, index, 0), axis=1) * valid[None, :, None]
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    # Later SIMD operations may reuse ordinary UB borrowed by Gather's workspace.
    a = t.add(y, 1)
    b = t.mul(y, 2)
    t.store_expect(t.mul(a, b), (expected + 1) * (expected * 2))
    batch = 1760 if inner == 8 else 1100
    width = inner * (batch if index_count > 1024 else 2)
    t.tile(0, 2, (expected.size + width - 1) // width, width)
    t.codegen()
    capacity = int(re.search(r"index_capacity\((\d+)\)", t.das()).group(1))
    assert capacity == (batch if index_count > 1024 else 2)
    scratch = int(re.search(r"index_scratch\((\d+)\)", t.das()).group(1))
    assert scratch % 32 == 0
    for _ in range(3):
        assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("tile_width", [17, 24])
def test_gather_block_copy_partial_short_slice(tile_width):
    t = Tester()
    t.set_passes("ReorderLoad")
    x = np.arange(3 * 7 * 16, dtype=np.float32).reshape(3, 7, 16)
    index = np.array([0, -1, 2, 3, 4], np.int32)
    y = t.gather_load(x, t.global_access(index), axis=1, gather_mode=1)
    expected = np.take(x, index, axis=1) + 1
    t.store_expect(t.add(y, 1), expected)
    t.tile(0, 2, (expected.size + tile_width - 1) // tile_width, tile_width)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_block_copy_many_loads():
    """Account for scratch descriptors in code UB and retire many loads safely."""
    t = Tester()
    t.set_passes("ReorderLoad")
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


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
def test_gather_vector_shared_workspace_and_index(mode, dtype):
    t = Tester()
    t.set_passes("ReorderLoad")
    outer, gather, inner, count = 4, 19, 3, 1201
    src = (np.arange(outer * gather * inner) % 37).astype(dtype).reshape(outer, gather, inner)
    shape = (outer, count, inner) if mode == 0 else (count,)
    indices = np.resize(np.array([0, -1, 2, -19, -20, 19, 2147483647, -2147483648], np.int32), shape)
    shared = t.global_access(indices)
    first = t.gather_load(src, shared, axis=1, gather_mode=mode)
    second = t.gather_load(src + 5, shared, axis=1, gather_mode=mode)
    out = t.store(t.add(first, second))
    width = 1409  # Cross an outer boundary mid-tile and exceed a workspace index batch.
    t.tile(0, 2, (outer * count * inner + width - 1) // width, width)
    t.codegen()
    plans = re.findall(r"strategy\(simd_gather\), src_scratch\((\d+)\), index_scratch\((\d+)\)", t.das())
    assert len(plans) == 2 and plans[0] == plans[1]
    src_addr, index_addr = map(int, plans[0])
    assert index_addr - src_addr == (gather * inner * src.itemsize + 31) // 32 * 32
    for index in [indices, np.roll(indices, 1).copy()]:
        t.input(shared, index)
        valid = (index >= -gather) & (index < gather)
        safe = np.where(valid, index, 0)
        if mode == 0:
            expected = np.take_along_axis(src, safe, axis=1)
        else:
            expected = np.take(src, safe, axis=1)
            valid = valid[None, :, None]
        t.run()
        assert t.check(out, (expected * 2 + 5) * valid)


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_vector_b16_large_element_offset():
    t = Tester()
    t.set_passes("ReorderLoad")
    src = (np.arange(75000) % 97).astype(np.float16).reshape(1, 25000, 3)
    index = np.resize(np.array([22000, -1, 24999, 0, -25001], np.int32), (1, 2901, 3))
    y = t.gather_load(src, t.global_access(index), axis=1)
    valid = index >= -25000
    expected = np.take_along_axis(src, np.where(valid, index, 0), axis=1) * valid
    t.store_expect(t.add(y, 1), expected + 1)
    t.tile(0, 2, 2, 6144)
    t.codegen()
    assert "strategy(simd_gather)" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
@pytest.mark.parametrize("layout", ["rows", "flat"])
@pytest.mark.parametrize("inner", [25, 137])
def test_gather_element_workspace_index_padding_and_tail(dtype, layout, inner):
    t = Tester()
    t.set_passes("ReorderLoad")
    src = (np.arange(3 * 7 * inner) % 37).astype(dtype).reshape(3, 7, inner)
    index = np.resize(np.array([0, -1, 7, -8, -7, 2147483647, -2147483648], np.int32), (3, 13, inner))
    index_op = t.global_access(index)
    y = t.gather_load(src, index_op, axis=1)
    out = t.store(t.add(y, 1))
    if layout == "rows":
        # Five padded inner rows per tile; 137 elements exercise multiple vector groups per row.
        # The tail has four rows and tiles cross outer.
        t.tile(1, 2, 8, 5)
    else:
        # Tile boundaries split inner rows; the final tile is short.
        width = 2 * inner + 17
        t.tile(0, 2, (index.size + width - 1) // width, width)
    t.codegen()
    das = t.das()
    assert "strategy(simd_gather)" in das
    bits = src.itemsize * 8
    output = int(re.search(rf"gather_load\.b{bits}\.\d+x\d+ (0x[0-9a-f]+)", das).group(1), 16)
    src_addr = int(re.search(r"src_scratch\((\d+)\)", das).group(1))
    index_addr = int(re.search(r"index_scratch\((\d+)\)", das).group(1))
    assert index_addr == src_addr + (7 * inner * src.itemsize + 31) // 32 * 32
    assert output != index_addr
    if layout == "rows":
        assert f"gather_load.b{bits}.{inner}x5" in das and "pad_size(7)" in das and "tail_iter(4)" in das
    for data in [index, np.full_like(index, 7), np.roll(index, 1).copy()]:
        t.input(index_op, data)
        valid = (data >= -7) & (data < 7)
        expected = np.take_along_axis(src, np.where(valid, data, 0), axis=1) * valid + 1
        t.run()
        assert t.check(out, expected)


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("gather, strategy, capacity", [
    (2048, "simd_gather", 14336),
    (16376, "simd_gather", 8),
    (16384, "simt", 0),
    (32768, "simt", 0),
])
def test_gather_manual_tile_workspace_fallback(gather, strategy, capacity):
    t = Tester()
    t.set_passes("ReorderLoad")
    src = np.arange(gather, dtype=np.float32).reshape(1, gather)
    index = (np.arange(20352, dtype=np.int32) % gather).reshape(1, -1)
    y = t.gather_load(src, t.global_access(index), axis=1)
    t.store_expect(t.add(y, 1), np.take_along_axis(src, index, axis=1) + 1)
    t.tile(0, 1, 1, index.size)
    t.codegen()
    assert "gather_load.b32.20352x1" in t.das()
    assert f"strategy({strategy})" in t.das()
    if strategy == "simd_gather":
        # Two static outputs leave 64 KiB for source and index workspace.
        output = int(re.search(r"gather_load\.b32\.20352x1 (0x[0-9a-f]+)", t.das()).group(1), 16)
        src_addr = int(re.search(r"src_scratch\((\d+)\)", t.das()).group(1))
        index_addr = int(re.search(r"index_scratch\((\d+)\)", t.das()).group(1))
        assert index_addr == src_addr + src.nbytes and index_addr != output
        assert f"index_capacity({capacity})" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_block_copy_padded_row():
    t = Tester()
    t.set_passes("ReorderLoad")
    src = np.arange(175, dtype=np.float32).reshape(1, 7, 25)
    index = np.array([1, -1, 7, -8], dtype=np.int32)
    y = t.gather_load(src, t.global_access(index), axis=1, gather_mode=1)
    expected = np.take(src, [1, 6, 0, 0], axis=1)
    expected[:, 2:] = 0
    t.store_expect(t.add(y, 1), expected + 1)
    t.tile(0, 2, 4, 25)
    t.codegen()
    assert "strategy(simd_block_copy)" in t.das() and "pad_size(7)" in t.das()
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_block_copy_workspace_without_dynamic_slot():
    t = Tester()
    t.set_passes("ReorderLoad")
    src = np.arange(56, dtype=np.float32).reshape(1, 7, 8)
    index = np.resize(np.array([0, -1, 7, -8, 2], np.int32), 3560)
    valid = (index >= -7) & (index < 7)
    expected = np.take(src, np.where(valid, index, 0), axis=1) * valid[None, :, None]
    y = t.gather_load(src, t.global_access(index), axis=1, gather_mode=1)
    t.store_expect(t.add(y, 1), expected + 1)
    # Two 113920-byte static outputs leave 512 bytes; no full dynamic slot fits.
    t.tile(0, 2, 1, expected.size)
    t.codegen()
    das = t.das()
    assert "strategy(simd_block_copy)" in das
    assert "index_capacity(128)" in das
    output = int(re.search(r"Adds\.fp32\.28480 (0x[0-9a-f]+)", das).group(1), 16)
    scratch = int(re.search(r"index_scratch\((\d+)\)", das).group(1))
    assert scratch >= output + expected.nbytes
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("mode", [0, 1])
def test_gather_default_pipeline_reorders_load_before_compute(mode):
    t = Tester(use_pass_opt=True)
    values = (np.arange(1032, dtype=np.float32) % 17).reshape(1, 129, 8)
    prior = t.add(t.load(values), 2)
    src = np.arange(56, dtype=np.float32).reshape(1, 7, 8)
    index = np.resize(np.array([0, -1, 7, -8, 2], np.int32), 129)
    valid = (index >= -7) & (index < 7)
    indices = index if mode else np.broadcast_to(index[None, :, None], values.shape).copy()
    y = t.gather_load(src, t.global_access(indices), axis=1, gather_mode=mode)
    expected = values + 2 + np.take(src, np.where(valid, index, 0), axis=1) * valid[None, :, None]
    t.store_expect(t.add(prior, y), expected)
    t.tile(0, 2, 3, 512)
    t.codegen()
    das = t.das()
    # Construction order deliberately puts a computation before GatherLoad.
    assert das.index("gather_load") < das.index("Adds.fp32")
    assert f"strategy({'simd_block_copy' if mode else 'simd_gather'})" in das
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_workspace_shared_between_strategies():
    t = Tester()
    t.set_passes("ReorderLoad")
    src = np.arange(56, dtype=np.float32).reshape(1, 7, 8)
    index = np.resize(np.array([0, -1, 7, -8, 2], np.int32), 1025)
    valid = (index >= -7) & (index < 7)
    element_index = np.broadcast_to(index[None, :, None], (1, index.size, 8)).copy()
    first = t.gather_load(src, t.global_access(index), axis=1, gather_mode=1)
    second = t.gather_load(src + 5, t.global_access(element_index), axis=1)
    expected = (2 * np.take(src, np.where(valid, index, 0), axis=1) + 5) * valid[None, :, None]
    t.store_expect(t.add(first, second), expected)
    t.tile(0, 2, (expected.size + 1023) // 1024, 1024)
    t.codegen()
    das = t.das()
    block_index = re.search(r"strategy\(simd_block_copy\), index_scratch\((\d+)\)", das)
    vector_src = re.search(r"strategy\(simd_gather\), src_scratch\((\d+)\)", das)
    assert block_index and vector_src and block_index.group(1) == vector_src.group(1)
    for _ in range(3):
        assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_dynamic_source_and_index_shapes():
    t = Tester("vector:dyn", use_pass_opt=True)
    idx = t.global_access([-1], "int32")
    y = t.gather_load([-1], idx, dtype="float32", axis=-2)
    output = t.store(y)
    for outer, gather, count, inner in [(2, 19, 101, 3), (3, 11, 67, 5)]:
        src = np.arange(outer * gather * inner, dtype=np.float32).reshape(outer, gather, inner)
        index = (np.arange(outer * count * inner, dtype=np.int32) % gather).reshape(outer, count, inner)
        t.input(y, src)
        t.input(idx, index)
        t.run()
        assert t.check(output, np.take_along_axis(src, index, axis=1))


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
def test_gather_element_layout_validation():
    src = np.arange(3 * 7 * 5, dtype=np.float32).reshape(3, 7, 5)
    t = Tester()
    t.set_passes("ReorderLoad")
    index = (np.arange(2 * 9 * 5, dtype=np.int32) % 7).reshape(2, 9, 5)
    y = t.gather_load(src, t.global_access(index), axis=1)
    t.store_expect(y, np.take_along_axis(src[:2], index, axis=1))
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("width, gather, strategy", [
    (8192, 192, "simd_gather"),  # The old helper's conservative interval still fits.
    (16384, 504, "simt"),       # Overcounting places the old helper's start beyond UB.
    (16384, 512, "simt"),
])
def test_gather_workspace_static_owners(width, gather, strategy):
    t = Tester()
    t.set_passes("ReorderLoad")
    src = np.arange(gather, dtype=np.float32).reshape(1, gather)
    index = (np.arange(width, dtype=np.int32) % gather).reshape(1, -1)
    y = t.gather_load(src, t.global_access(index), axis=1)
    t.load(np.ones(index.shape, np.float32))  # A dead static Load follows Gather in the graph.
    result = t.add(y, 1)
    expected = np.take_along_axis(src, index, axis=1) + 1
    t.store_expect(result, expected)
    t.store_expect(result, expected)
    t.tile(0, 1, 1, index.size)
    t.codegen()
    das = t.das()
    assert f"strategy({strategy})" in das
    if strategy == "simd_gather":
        start = int(re.search(rf"Adds\.fp32\.{width} (0x[0-9a-f]+)", das).group(1), 16)
        src_scratch = int(re.search(r"src_scratch\((\d+)\)", das).group(1))
        index_scratch = int(re.search(r"index_scratch\((\d+)\)", das).group(1))
        assert src_scratch >= start + expected.nbytes
        output = int(re.search(rf"gather_load\.b32\.{width}x1 (0x[0-9a-f]+)", das).group(1), 16)
        assert src_scratch % 32 == 0 and index_scratch >= src_scratch + src.nbytes
        assert index_scratch != output
    assert t.run_check()


@pytest.mark.skipif(dvm.Device.arch() != "AscendC310", reason="GatherLoad requires C310")
@pytest.mark.parametrize("mode", [0, 1])
def test_gather_empty_axis(mode):
    t = Tester()
    t.set_passes("ReorderLoad")
    index = np.full((1, 5, 3) if mode == 0 else (5,), -1, np.int32)
    # A zero-length source has no GM binding to read.
    y = t.gather_load([1, 0, 3], t.global_access(index), axis=1, dtype="float32", gather_mode=mode)
    t.store_expect(y, np.zeros((1, 5, 3), np.float32))
    t.codegen()
    assert "strategy(simt)" in t.das()
    assert t.run_check()
