# Copyright 2024 Huawei Technologies Co., Ltd
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
from dvm.tester import Tester, ShapeRef

def test_reshape():
    t = Tester('dyn')
    x = t.load([-1], "float32")
    y = t.load([-1], "float32")
    ref = ShapeRef()
    a = t.reshape(x, ref)
    z = t.binary("Add", a, y)
    out = t.store(z)
    iterations = [[[10], [2, 5], [2, 5]],
        [[20, 40], [800, 1], [800, 16]],
        [[2, 40], [1, 80, 1], [10, 80, 32]]]
    for x_shape, ref_shape, y_shape in iterations:
        x_data = np.random.normal(0, 1, x_shape).astype(np.float32)
        y_data = np.random.normal(0, 1, y_shape).astype(np.float32)
        t.input(x, x_data)
        t.input(y, y_data)
        ref.update(ref_shape)
        t.run()
        t.check(out, x_data.reshape(ref_shape) + y_data)

def test_implicit_broadcast():
    t = Tester('dyn')
    x = t.load([-1], "float32")
    y = t.load([-1], "float32")
    a = t.binary("Add", x, 0.15)
    b = t.binary("Mul", a, y)
    out = t.store(b)
    iterations = [
        [[1000], [1000]],
        [[500, 1], [500, 1024]],
        [[500, 1], [1, 1024]],
        [[4, 1, 12, 1], [4, 32, 12, 64]]]
    for x_shape, y_shape in iterations:
        d1 = np.full(x_shape, 0.1, np.float32)
        d2 = np.full(y_shape, 0.4, np.float32)
        t.input(x, d1)
        t.input(y, d2)
        t.run()
        t.check(out, (d1 + 0.15) * d2)

def test_reduce():
    t = Tester('dyn')
    x = t.load([-1], "float32")
    dims = ShapeRef()
    a = t.reduce("sum", x, dims, True)
    out = t.store(a)
    iterations = [[[10, 64], [1]], [[12, 128], [0, 1]], [[3, 12, 20, 100], [1, 3]]]
    for x_shape, dims_shape in iterations:
        d1 = np.random.normal(0, 1, x_shape).astype(np.float32)
        t.input(x, d1)
        dims.update(dims_shape)
        t.run()
        t.check(out, np.sum(d1, tuple(dims_shape), keepdims=True), 1e-4)

def test_reduce_round_tile():
    t = Tester('dyn')
    x = t.load([-1,2,4,3000], "float32")
    a = t.reduce("sum", x, [0,2], True)
    out = t.store(a)
    din = np.random.normal(0, 1, [2,2,4,3000]).astype(np.float32)
    expect = np.sum(din, (0,2), keepdims=True)
    for i in range(3):
        t.input(x, din)
        t.run()
        t.check(out, expect, 1e-4)

def test_broadcast():
    t = Tester('dyn')
    x = t.load([-1], "float32")
    shape = ShapeRef()
    a = t.broadcast(x, shape)
    b = t.broadcast(0.2, shape)
    c = t.binary("Mul", a, b)
    out = t.store(c)
    iterations = [[[1], [10, 100]], [[1, 4, 1], [8, 4, 40]]]
    for x_shape, broad_shape in iterations:
        x_data = np.full(x_shape, 0.1, np.float32)
        t.input(x, x_data)
        shape.update(broad_shape)
        t.run()
        t.check(out, np.broadcast_to(x_data, broad_shape) * 0.2)

# llava dynamic shape inference
def test_atomic_reduce():
    t = Tester('dyn')
    para0 = t.load([-1], "float16")
    para1 = t.load([-1], "float16")
    y0 = t.binary("Add", para0, para1)
    out0 = t.store(y0)
    y1 = t.cast(y0, "float32")
    out1 = t.store(y1)
    y2 = t.binary("Mul", y1, y1)
    y3 = t.reduce("sum", y2,  [2], True)
    out2 = t.store(y3)
    iterations = [[[1, 2685, 4096], [1, 2685, 4096]],
                  [[1, 2686, 4096], [1, 2686, 4096]],
                  [[1, 2687, 4096], [1, 2687, 4096]]]
    for x_shape, y_shape in iterations:
        x_data = np.random.normal(0, 1, x_shape).astype(np.float16)
        y_data = np.random.normal(0, 1, y_shape).astype(np.float16)
        t.input(para0, x_data)
        t.input(para1, y_data)
        t.run()
        np_out0 = x_data + y_data
        np_out1 = np_out0.astype(np.float32)
        np_out2 = np.sum(np_out1 * np_out1, (2), keepdims=True)
        t.check(out0, np_out0)
        t.check(out1, np_out1)
        t.check(out2, np_out2, 1e-4)
