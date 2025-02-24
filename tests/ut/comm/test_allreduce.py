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

import numpy as np
from dvm.tester import Tester
import pytest
import functools


@pytest.mark.parametrize(
    "shape_size",
    [32, 9, 156, 129, 640, 6400, 64232, 3123, 343535, 222, 21313, 22, 23445, 129, 130],
)
def test_allreduce(comm, rank, size, shape_size):
    np.random.seed(1)

    t = Tester(comm=comm)
    inputs = []
    for i in range(size):
        inputs.append(np.random.normal(0.1, 1, [shape_size]).astype(np.float16))
    expect = inputs[0].copy()
    for i in range(1, size):
        expect += inputs[i]

    x1 = t.load(inputs[rank])
    x2 = t.allreduce(x1)
    x3 = t.unary("Abs", x2)

    t.store_expect(x3, np.abs(expect), 0.01)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("shape_size", [54857612])
def test_allreduce_big_shape(comm, rank, size, shape_size):
    np.random.seed(1)

    t = Tester(comm=comm)
    inputs = []
    for i in range(size):
        inputs.append(np.random.normal(0.1, 1, [shape_size]).astype(np.float16))
    expect = inputs[0].copy()
    for i in range(1, size):
        expect += inputs[i]
    x1 = t.load(inputs[rank])
    x2 = t.allreduce(x1)

    t.store_expect(x2, expect, 0.01)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("m", [16384, 32])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("n", [6400, 22])
def test_matmul_allreduce(comm, rank, size, m, k, n):
    np.random.seed(1)

    t = Tester("mix", comm=comm)
    shape_a = [m, k]
    shape_b = [k, n]

    inputs_a = []
    inputs_b = []
    for i in range(size):
        inputs_a.append(np.random.normal(0, 0.01, shape_a).astype(np.float16))
        inputs_b.append(np.random.normal(0, 0.01, shape_b).astype(np.float16))
    expect = np.matmul(
        inputs_a[0].astype(np.float32), inputs_b[0].astype(np.float32)
    ).astype(np.float16)
    for i in range(1, size):
        expect = expect + np.matmul(
            inputs_a[i].astype(np.float32), inputs_b[i].astype(np.float32)
        ).astype(np.float16)

    x1 = t.load(inputs_a[rank])
    x2 = t.load(inputs_b[rank])
    c = t.matmul(x1, x2, False, False)
    x3 = t.allreduce(c)
    t.store_expect(x3, expect, 0.01)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("m", [4096, 64])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [16384, 256])
def test_matmul_allreduce_indirect(comm, rank, size, m, k, n):
    np.random.seed(1)

    t = Tester("mix", use_pass_opt=True, comm=comm)
    shape_a = [m, k]
    shape_b = [k, n]

    inputs_a = []
    inputs_b = []
    for i in range(size):
        inputs_a.append(np.random.normal(0, 0.01, shape_a).astype(np.float16))
        inputs_b.append(np.random.normal(0, 0.01, shape_b).astype(np.float16))
    expect = np.matmul(
        inputs_a[0].astype(np.float32), inputs_b[0].astype(np.float32)
    ).astype(np.float16)
    expect = np.abs(expect)
    for i in range(1, size):
        expect = expect + np.abs(
            np.matmul(
                inputs_a[i].astype(np.float32), inputs_b[i].astype(np.float32)
            ).astype(np.float16)
        )
    expect = np.reciprocal(expect)

    x1 = t.load(inputs_a[rank])
    x2 = t.load(inputs_b[rank])
    c = t.matmul(x1, x2, False, False)
    c = t.unary("Abs", c)
    x2 = t.allreduce(c)
    x3 = t.unary("Reciprocal", x2)
    t.store_expect(x3, expect, 0.01)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("m", [4096])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [4096])
def test_matmul_allreduce_post(comm, rank, size, m, k, n):
    np.random.seed(1)

    t = Tester("mix", use_pass_opt=True, comm=comm)
    shape_a = [m, k]
    shape_b = [k, n]

    inputs_a = []
    inputs_b = []
    for i in range(size):
        inputs_a.append(np.random.normal(1, 2, shape_a).astype(np.float16))
        inputs_b.append(np.random.normal(1, 2, shape_b).astype(np.float16))
    expect = np.matmul(
        inputs_a[0].astype(np.float32), inputs_b[0].astype(np.float32)
    ).astype(np.float16)
    for i in range(1, size):
        expect = expect + np.matmul(
            inputs_a[i].astype(np.float32), inputs_b[i].astype(np.float32)
        ).astype(np.float16)
    expect = np.reciprocal(expect)

    x1 = t.load(inputs_a[rank])
    x2 = t.load(inputs_b[rank])
    c = t.matmul(x1, x2, False, False)
    x3 = t.allreduce(c)
    x4 = t.unary("Reciprocal", x3)
    t.store_expect(x4, expect, 1e-2)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("shape_size", [5120 * 5120])
def test_allreduce_post(comm, rank, size, shape_size):
    np.random.seed(1)

    t = Tester(use_pass_opt=True, comm=comm)
    inputs = []
    for i in range(size):
        inputs.append(np.random.normal(0.1, 1, [shape_size]).astype(np.float32))
    mul_rhs = np.random.normal(0, 1, [1]).astype(np.float16)
    expect = inputs[0].copy()
    for i in range(1, size):
        expect += inputs[i]
    expect = expect.astype(np.float16)
    expect = expect * mul_rhs

    x1 = t.load(inputs[rank])
    x2 = t.allreduce(x1)
    x3 = t.cast(x2, "float16")
    x4 = t.load(mul_rhs)
    x5 = t.binary("Mul", x3, x4)
    t.store_expect(x5, expect, 0.01)
    res = t.run_check()
    assert res


@pytest.mark.parametrize("shape_size", [5120])
def test_allreduce_multi_time(comm, rank, size, shape_size):
    """
    test if unique_id work
    """
    np.random.seed(1)

    t = Tester(comm=comm)
    inputs = []
    for i in range(4):
        inputs.append(np.full([shape_size], i).astype(np.float16))
    x1 = t.load(inputs[0])
    x2 = t.allreduce(x1)
    out = t.store(x2)
    res = t.run()
    for input in inputs:
        t.input(x1, input)
        t.bare_run()
        assert t.check(out, input * size, 1e-2)


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [256])
def test_matmul_allreduce_multi_time(comm, size, m, n, k):
    """
    test if unique_id work
    """
    np.random.seed(1)

    t = Tester("mix", comm=comm)
    shape_a = [m, n]
    shape_b = [n, k]
    inputs_a = []
    inputs_b = []
    for i in range(3):
        inputs_a.append(np.full(shape_a, i).astype(np.float16))
        inputs_b.append(np.full(shape_b, i).astype(np.float16))
    x1 = t.load(inputs_a[0])
    x2 = t.load(inputs_b[0])
    x3 = t.matmul(x1, x2, False, False)
    x4 = t.allreduce(x3)
    out = t.store(x4)
    t.run()
    for i in range(len(inputs_a)):
        t.input(x1, inputs_a[i])
        t.input(x2, inputs_b[i])
        t.bare_run()
        expect = np.matmul(
            inputs_a[i].astype(np.float32), inputs_b[i].astype(np.float32)
        ).astype(np.float16)
        assert t.check(out, expect * size, 1e-2)
