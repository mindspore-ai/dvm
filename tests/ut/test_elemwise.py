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
import dvm
from dvm.tester import Tester

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [("Sqrt", np.sqrt), ("Abs", np.abs), ("Log", np.log), ("Exp", np.exp), ("Reciprocal", np.reciprocal)])
def test_unary(type, op, func):
    t = Tester()
    a = np.full([32, 1024], 0.5, type)
    x = t.load(a)
    y = t.unary(op, x)
    t.store_expect(y, func(a))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [("Add", np.add), ("Sub", np.subtract),  ("Div", np.divide),
                                      ("Mul", np.multiply), ("Maximum", np.maximum), ("Minimum", np.minimum)])
def test_binary(type, op, func):
    t = Tester()
    a = np.full([32, 1024], 0.5, type)
    b = np.full([32, 1024], 0.4, type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.binary(op, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.bool_, np.float16, np.float32])
@pytest.mark.parametrize("shape",[(1024, 32), (1312, 131), (16, 11) ,(128, 7), (128, 777)])
@pytest.mark.parametrize('op, func', [("LogicalOr", np.logical_or), ("LogicalAnd", np.logical_and)])
def test_logical(type, shape, op, func):
    t = Tester()
    a = np.random.choice([True, False], shape).astype(type)
    b = np.random.choice([True, False], shape).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.binary(op, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b).astype(type))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.bool_, np.float16, np.float32, np.int32])
@pytest.mark.parametrize("shape",[(1024, 32), (1312, 131), (16, 11) ,(128, 7), (128, 777)])
@pytest.mark.parametrize('op, func', [("LogicalNot", np.logical_not)])
def test_logical_not(type, shape, op, func):
    t = Tester()
    a = np.random.choice([True, False], shape).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.unary(op, x)
    z = t.copy(z)
    t.store_expect(z, func(a).astype(type))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [("Add", np.add), ("Mul", np.multiply), ("Div", np.divide),
                                      ("Maximum", np.maximum), ("Minimum", np.minimum)])
def test_binary_s(type, op, func):
    t = Tester()
    a = np.random.normal(0, 1, [32, 1024]).astype(type)
    x = t.load(a)
    x = t.copy(x)
    y = t.binary(op, x, 0.1)
    y = t.copy(y)
    t.store_expect(y, func(a, 0.1))
    assert(t.run_check())


@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [("Div", np.divide), ("Sub", np.subtract)])
def test_binary_s_l(type, op, func):
    t = Tester()
    a = np.random.normal(0, 1, [32, 1024]).astype(type)
    x = t.load(a)
    x = t.copy(x)
    y = t.binary(op, 0.1, x)
    y = t.copy(y)
    t.store_expect(y, func(0.1, a))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16] if dvm.device.arch() != 'AscendC220' else [np.float16,np.float32])
@pytest.mark.parametrize("size",[1024, 24, 66666])
def test_isfinite(type, size):
    t = Tester()
    random_numbers = np.random.randn(size)
    special_values = np.array([np.inf, -np.inf, np.nan])
    a = np.concatenate((random_numbers, np.tile(special_values, size))).astype(type)
    np.random.shuffle(a)
    x = t.load(a)
    x = t.copy(x)
    y = t.unary("IsFinite", x)
    y = t.copy(y)
    t.store_expect(y, np.isfinite(a).astype(type), 0)
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_repeat_overflow(type):
    repeat = 257
    a = np.full([32*64*repeat], 0.5, type)
    t = Tester()
    x = t.load(a)
    y = t.binary("Add", x, 0.1)
    t.store_expect(y, 0.6)
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_scalar(type):
    t = Tester()
    a = np.full([1], 0.4, type)
    b = np.full([1], 0.2, type)
    x = t.load(a)
    y = t.load(b)
    z = t.binary("Sub", x, y)
    t.store_expect(z, 0.2)
    assert(t.run_check())

@pytest.mark.parametrize('type, eps, shape', [(np.float16, 1e-03, [2,3,4,5]), (np.float32, 1e-05, [1,1,4,5])])
def test_scalar_tensor_div(type, eps, shape):
    t = Tester()
    if (type == np.float16):
        x = t.broadcast(2, shape, "float16")
    else:
        x = t.broadcast(2, shape)
    a = np.random.random(shape).astype(type)
    y = t.load(a)
    z = t.binary("Div", x, y)
    np_res = 2 / a
    t.store_expect(z, np_res)
    assert(t.run_check())

@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_mul1(data_type):
    np.random.seed(1)
    t = Tester()
    a0 = np.random.normal(0, 1, [16000, 39, 1]).astype(data_type)
    a1 = np.random.normal(0, 1, [16000, 39, 1]).astype(data_type)
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.binary("Mul", x0, x1)
    expect = a0 * a1
    t.store_expect(y0, expect)
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_pow_1(type):
    t = Tester()
    a = np.array([-2, -3, -4, 0, -2, 9]).astype(type)
    b = np.array([1, 2, 3, 0, 0.5, 0.5]).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.binary("Pow", x, y)
    t.store_expect(z, np.array([-2, 9, -64, 1, np.nan, 3]))
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize("shape",[(1024, 32), (13, 131), (16, 11) ,(3, 3)])
def test_pow_2(type, shape):
    t = Tester()
    a = 10 * np.random.rand(*shape).astype(type) - 5
    b = np.random.randint(6, size = shape).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.binary("Pow", x, y)
    z = t.copy(z)
    t.store_expect(z, np.power(a, b))
    assert(t.run_check())

@pytest.mark.parametrize("shape",[(1024, 32), (13, 131)])
@pytest.mark.parametrize("exponent",[0.0, 1.2 ,5.0, -7.0, -3.0])
def test_pow_3(shape, exponent):
    t = Tester()
    a = np.abs(np.random.normal(0, 1, shape).astype(np.float32) - 2)
    x = t.load(a)
    y = t.binary("Pow", x, exponent)
    z = t.binary("Add", y, 2)
    z = t.copy(z)
    t.store_expect(z, np.power(a, exponent) + 2)
    assert(t.run_check())

@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)]) 
def test_rsqrt(type, eps):
    t = Tester()
    a = np.random.rand(32, 32).astype(type)
    x = t.load(a)
    z = t.unary("Sqrt", x)
    y = t.unary("Reciprocal", z)
    t.store_expect(y, np.reciprocal(np.sqrt(a)).astype(type), eps)
    assert(t.run_check())

@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.parametrize('op, func', [("Add", np.add), ("Sub", np.subtract), ("Mul", np.multiply), ("Maximum", np.maximum), ("Minimum", np.minimum)])
def test_binary_int(type, op, func):
    t = Tester()
    a = np.random.randint(low =-2**30, high = 2**30, size=(32, 1024)).astype(type)
    b = np.random.randint(low =-2**30, high = 2**30, size=(32, 1024)).astype(type)
    x = t.load(a)
    y = t.load(b)
    x = t.copy(x)
    y = t.copy(y)
    z = t.binary(op, x, y)
    z = t.copy(z)
    t.store_expect(z, func(a, b))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.parametrize('op, func', [("Add", np.add), ("Sub", np.subtract), ("Mul", np.multiply), ("Maximum", np.maximum), ("Minimum", np.minimum)])
def test_binarys_int(type, op, func):
    t = Tester()
    a = np.random.randint(low =-2**30, high = 2**30, size=(32, 1024)).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.binary(op, x, 11111)
    z = t.copy(z)
    t.store_expect(z, func(a, 11111))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.int32])
@pytest.mark.parametrize('op, func', [("Abs", np.abs)])
def test_unary_int(type, op, func):
    t = Tester()
    a = np.random.randint(low =-2**30, high = 2**30, size=(32, 1024)).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.unary(op, x)
    z = t.copy(z)
    t.store_expect(z, func(a))
    assert (t.run_check())

@pytest.mark.parametrize('num, expect', [(3, 2), (10, 4), (-8, 3), (11, 5), (15, 6), (20, 5)])
def test_power_s(num, expect):
    t = Tester()
    arg = np.random.random([1024,]).astype(np.float32)
    a = t.load(arg)
    b = t.binary("Pow", a, num)
    t.store_expect(b, np.power(arg, num))
    assert (t.run_check())
    assert (t.das().count("Mul.fp32") == expect)

@pytest.mark.parametrize('type', [np.float16, np.float32])
def test_pow_s_left(type):
    t = Tester()
    a = np.random.normal(0, 1, (1024, 32)).astype(type)
    x = t.load(a)
    y = t.binary("Pow", 0.91, x)
    t.store_expect(y, np.power(0.91, a))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('op, func', [("Round", np.round), ("Ceil", np.ceil), ("Floor", np.floor), ("Trunc", np.trunc)])
@pytest.mark.parametrize('shape', [[32, 32], [128*256*2*48]])
def test_trunc(type, op, func, shape):
    t = Tester()
    a = np.random.normal(-10, 10, shape).astype(type)
    x = t.load(a)
    x = t.copy(x)
    z = t.unary(op, x)
    z = t.copy(z)
    t.store_expect(z, func(a).astype(type))
    assert (t.run_check())
