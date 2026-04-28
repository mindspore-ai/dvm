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
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_vec():
    t = Tester("split:priv1")
    in_shape, red_dims = [10, 4096], (0,)
    a = np.random.normal(0, 0.1, in_shape).astype(np.float16)
    x = t.load(a)
    x = t.add(x, 0.1)
    t.store_expect(x, a + 0.1)
    x = t.cast(x, "float32")
    x0 = t.sum(x, red_dims, True)
    x = t.cast(x0, "float16")
    out1 = t.mul(x, 1.5)
    b = np.random.normal(0, 0.1, in_shape).astype(np.float32)
    out2 = t.add(x0, t.load(b))
    res_x0 = np.sum((a + 0.1).astype(np.float32), red_dims, keepdims=True)
    t.store_expect(out2, res_x0 + b)
    t.store_expect(out1, res_x0.astype(np.float16) * 1.5)
    assert(t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_vec_mix():
    t = Tester("split:priv1")
    x_data = np.random.normal(0, 0.1, [1000, 3333]).astype(np.float16)
    y_data = np.random.normal(0, 0.1, [3333, 1024]).astype(np.float16)
    x = t.load(x_data)
    y = t.load(y_data)
    x = t.mul(x, 0.5)
    x = t.matmul(x, y, False, False)
    x = t.add(x, 0.1)
    res = np.matmul((x_data * 0.5).astype(np.float32), y_data.astype(np.float32)).astype(np.float16) + 0.1
    t.store_expect(x, res)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_static_reshape():
    t = Tester("split:priv1")
    a = np.random.normal(0.0, 1.0, [32, 128]).astype(np.float32)
    x = t.load(a)
    y = t.mul(x, 0.6)
    z = t.reshape(y, [2, 16, 128])
    r = t.add(z, 0.1)
    t.store_expect(r, (a * 0.6).reshape([2, 16, 128]) + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_dyn_shape():
    t = Tester("split:dyn,priv1")
    x0 = t.load([-1, -2], "float32")
    x1 = t.mul(x0, 0.6)
    x2 = t.load([-3, -1, -2], "float32")
    x3 = t.add(x1, x2)
    o1 = t.store(x3)
    x4 = t.load([-4, -1, -2], "float32")
    x5 = t.sub(x4, x1)
    o2 = t.store(x5)
    t.codegen()
    iterations = [[[256, 128], [4, 256, 128], [3, 256, 128]],
                  [[10, 32], [20, 10, 32], [40, 10, 32]]]
    for x0_shape, x2_shape, x4_shape in iterations:
        x0_data = np.random.normal(0, 0.1, x0_shape).astype(np.float32)
        x2_data = np.random.normal(0, 0.1, x2_shape).astype(np.float32)
        x4_data = np.random.normal(0, 0.1, x4_shape).astype(np.float32)
        t.input(x0, x0_data)
        t.input(x2, x2_data)
        t.input(x4, x4_data)
        t.run()
        assert(t.check(o1, x0_data * 0.6 + x2_data))
        assert(t.check(o2, x4_data - x0_data * 0.6))
