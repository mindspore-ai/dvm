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

def _gen_expect(indices, depth, axis, on_value, off_value):
    in_shape = [1, 1]
    axis = axis if axis >= 0 else len(in_shape) + axis + 1
    for i in range(len(indices.shape)):
        if i < axis:
            in_shape[0] = in_shape[0] * indices.shape[i]
        else:
            in_shape[1] = in_shape[1] * indices.shape[i]
    indices_ = np.reshape(indices, in_shape)
    out_shape = [in_shape[0], depth, in_shape[1]]
    out = np.full(out_shape, off_value, np.int32 if isinstance(on_value, int) else np.float32)
    for x in range(in_shape[0]):
        for y in range(in_shape[1]):
            out[x, indices_[x, y], y] = on_value
    out_shape = list(indices.shape)
    out_shape.insert(axis, depth)
    return np.reshape(out, out_shape)

@pytest.mark.parametrize('shape1, shape2, axis, depth, tile_depth, last_tile', [
  ([32], [32, 1], 1, 500, 1, 5),  # x
  ([8], [8, 1], 1, 8000, 2, 8),  # x_tile
  ([40, 16], [40,20,16], 1, 20, 1, 10),  # y
  ([40, 12], [40,20,12], 1, 20, 1, 10),  # y with pad
  ([4, 12], [4,200,1], 1, 200, 2, 10),  # y_tile
  ([4,30,12], [4,10,30,1], 1, 10, 3, 10),  # y_tile_2
])
def test_onehot_mode(shape1, shape2, axis, depth, tile_depth, last_tile):
    t = Tester()
    a = np.random.randint(0, depth, shape1, np.int32)
    b = np.full(shape2, 5.0, np.float32)
    x = t.one_hot(t.load(a), depth, axis, 2.0, 0.0, "float32")
    y = t.binary("Mul", x, t.load(b))
    expect = _gen_expect(a, depth, axis, 2.0, 0.0)
    t.store_expect(y, expect * b)
    if tile_depth > 0:
        out_shape = shape1
        out_shape.insert(axis if axis >= 0 else len(shape1), depth)
        for i in range(tile_depth):
            idx = len(out_shape) - i - 1
            tile_num = out_shape[i] if i + 1 < tile_depth else last_tile
            t.tile(idx, idx, tile_num)
    assert (t.run_check())

@pytest.mark.parametrize('shape, axis, depth, dtype, on_value, off_value', [
  ([32], -1, 1000, "float16", 3.0, 1.0),
  ([32], 0, 1000, "bfloat16", 3.0, 1.0),
  ([4,100], 1, 800, "float16", 3.0, 1.0),
  ([8,512], 1, 200, "int32", 2, 0),
])
def test_onehot_types(shape, axis, depth, dtype, on_value, off_value):
    t = Tester()
    a = np.random.randint(0, depth, shape, np.int32)
    x = t.one_hot(t.load(a), depth, axis, on_value, off_value, dtype)
    t.store_expect(x, _gen_expect(a, depth, axis, on_value, off_value))
    assert (t.run_check())

def test_onehot_reshape_elim():
    t = Tester(use_pass_opt=True)
    shape, depth, axis = [4*128], 512, 0
    a = np.random.randint(0, depth, shape, np.int32)
    x = t.one_hot(t.load(a), depth, axis, 0.05, 0.01, "float32")
    y = t.reshape(x, [512, 4, 128])
    z = t.reduce("sum", y, (2,), True)
    expect_x = _gen_expect(a, depth, axis, 0.05, 0.01)
    expect_z = np.sum(np.reshape(expect_x, [512, 4, 128]), (2,), keepdims=True)
    t.store_expect(z, expect_z)
    assert (t.run_check())

def test_update_domain():
    t = Tester()
    shape, depth, axis = [4096], 512, -1
    ax = np.random.normal(0.0, 1.0, [4096,1]).astype(np.float32)
    bx = np.random.normal(0.0, 1.0, [4096, depth]).astype(np.float32)
    cx = np.random.randint(0, depth, shape, np.int32)
    y0 = t.binary("Add", t.load(ax), 0.1)
    y1 = t.binary("Sub", t.load(bx), y0)
    expect_y1 = bx - (ax + 0.1)
    t.store_expect(y1, expect_y1)
    y2 = t.one_hot(t.load(cx), depth, axis, 1.0, 0.0, "float32")
    expect_y2 = _gen_expect(cx, depth, axis, 1.0, 0.0)
    t.store_expect(y2, expect_y2)
    y3 = t.binary("Mul", y1, y2)
    y4 = t.reduce("sum", y3, [1], False)
    expect_y4 = np.sum(expect_y1 * expect_y2, (1,), keepdims=False)
    t.store_expect(y4, expect_y4)
    assert (t.run_check())
