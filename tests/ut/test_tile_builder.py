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
from dvm.tester import TileBuilderTester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape, tile_space, tile_shape', [
    [[4096], [16], [256]],
    [[5000], ([20], 136), [256]],
    [[40, 1000], [20], [2, 1000]],
])
def test_tb_basic(shape, tile_space, tile_shape):
    t = TileBuilderTester()
    tile_list = tile_space if isinstance(tile_space, list) else tile_space[0]
    tile_space_size = 1
    for x in tile_list:
        tile_space_size *= x
    a = np.random.normal(1.0, 0.03, shape).astype(np.float32)
    b = np.random.normal(1.0, 0.03, shape).astype(np.float32)
    x0 = t.load(a, tile_shape, tile_space)
    x1 = t.load(b, tile_shape, tile_space)
    x2 = t.add(x0, x1)
    x3 = t.sqrt(x2)
    t.store_expect(x3, tile_space, expect=np.sqrt(a + b))
    max_tile_size = t.max_tile_size()
    assert max_tile_size > 0
    t.codegen(tile_space_size, 0, True)
    assert t.max_tile_size() == max_tile_size
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tb_fp32_stride_alignment():
    t = TileBuilderTester()
    a = np.random.normal(0.0, 0.03, [100]).astype(np.float32)
    b = np.random.normal(0.0, 0.03, [100]).astype(np.float32)
    tile_space = [1]
    tile_shape = [100]
    x0 = t.load(a, tile_shape, tile_space)
    x1 = t.load(b, tile_shape, tile_space)
    add = t.add(x0, x1)
    t.store_expect(add, tile_space, expect=a + b)

    t.codegen(tile_space_size=1, block_dim=1)
    das = t.das()
    # UpdateStride works in elements: one 32-byte block is 8 FP32 elements.
    assert "Add.fp32.104" in das
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tb_reorder_load_tied_first_use_by_last_use():
    """An input that dies earlier should be loaded first when first uses tie."""
    t = TileBuilderTester()
    x = np.full([256], 1.25, np.float32)
    y = np.full([256], 2.0, np.float32)
    tile_space = [1]
    tile_shape = [256]
    load_x = t.load(x, tile_shape, tile_space)
    load_y = t.load(y, tile_shape, tile_space)
    value = t.add(load_x, load_y)
    value = t.mul(value, load_y)
    value = t.sub(value, load_x)
    value = t.abs(value)
    value = t.sqrt(value)
    value = t.div(value, load_y)
    value = t.exp(value)
    value = t.mul(value, load_x)
    expected = np.exp(np.sqrt(np.abs((x + y) * y - x)) / y) * x
    t.store_expect(value, tile_space, expect=expected)

    t.codegen(tile_space_size=1, block_dim=1)
    dump = t.dump()
    # load_y becomes %0 and load_x becomes %1. Arithmetic dependencies must
    # remain unchanged because reordering moves TObject pointers, not operands.
    assert "Binary<68>(%1[256]<float32>, %0[256]<float32>)" in dump
    assert "Binary<82>(%" in dump
    das = t.das()
    div_begin = das.index("Div.fp32")
    exp_begin = das.index("Exp.fp32")
    final_mul_begin = das.rindex("Mul.fp32")
    store_begin = das.index("store.u8")

    def load_release_event(section):
        marker = "simd_load_sync(set, "
        begin = section.index(marker) + len(marker)
        end = section.index(")", begin)
        return int(section[begin:end])

    div_event = load_release_event(das[div_begin:exp_begin])
    final_mul_event = load_release_event(das[final_mul_begin:store_begin])
    # Event IDs are allocator details; only distinct, correctly placed release
    # events matter for overlapping the next tile's Loads with current SIMD.
    assert {div_event, final_mul_event} == {0, 1}
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tb_double_buffer():
    shape = [800000]
    tile_shape = [8000]
    tile_space = [shape[0] // tile_shape[0]]
    a = np.full(shape, 1.25, np.float32)
    b = np.full(shape, 1.25, np.float32)
    t = TileBuilderTester(flags=TileBuilderTester.F_DB)
    x0 = t.load(a, tile_shape, tile_space)
    x1 = t.load(b, tile_shape, tile_space)
    x2 = t.add(x0, x1)
    x3 = t.mul(x2, x0)
    x4 = t.sub(x3, x1)
    x5 = t.abs(x4)
    x6 = t.add(x5, x1)
    x7 = t.sqrt(x6)
    x8 = t.div(x7, x0)
    x9 = t.add(x8, x1)
    x10 = t.sub(x9, x0)
    x11 = t.mul(x10, x1)
    out = np.full(shape, 1.7677671, np.float32)
    t.store_expect(x11, tile_space, out)
    assert(tile_shape[0] * 4 < t.max_tile_size())
    t.codegen(tile_space_size=tile_space[0], block_dim=40)
    assert(t.run_check())
