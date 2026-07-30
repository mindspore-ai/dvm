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
def test_tb_basic():
    t = TileBuilderTester()
    a = np.random.normal(0.0, 0.03, [4096]).astype(np.float32)
    tile_space = [16]
    tile_shape = [256]
    tile_space_size = 16
    x0 = t.load(a, tile_shape, tile_space)
    x1 = t.add(x0, x0)
    x2 = t.sqrt(x1)
    t.store_expect(x2, tile_space, expect=np.sqrt(a + a))
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
