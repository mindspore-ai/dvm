import numpy as np
import dvm
import pytest
from dvm.tester import Tester

@pytest.mark.mix
def test_gemm():
    m = np.random.randint(1, high = 1025)
    n = np.random.randint(1, high = 1025)
    k = np.random.randint(1, high = 1025)
    shape_a = [m, k]
    shape_b = [k, n]
    np_a = np.random.normal(0, 1, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 1, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    # compute pad size
    pad_size_a = [(128 - shape_a[1] % 128) % 128]
    pad_size_b = [(128 - shape_b[1] % 128) % 128]

    t = Tester("stages")
    t.stage_switch("static")
    a = t.load(np_a)
    a = t.copy(a)
    pad_a = t.stage_pad_store(a, pad_size_a)
    t.stage_switch("static")
    b = t.load(np_b)
    b = t.copy(b)
    pad_b = t.stage_pad_store(b, pad_size_b)
    t.stage_switch("mix")
    mat_a = t.stage_load(pad_a)
    mat_b = t.stage_load(pad_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, expect)
    assert(t.run_check())

@pytest.mark.mix
def test_gemm_post_fusion():
    m = np.random.randint(1, high = 1025)
    n = np.random.randint(2, high = 1025)  # 24/06/2024: n does not support 1
    k = np.random.randint(1, high = 1025)
    shape_a = [m, k]
    shape_b = [k, n]
    np_a = np.random.normal(0, 1, shape_a).astype(np.float32)
    np_b = np.random.normal(0, 1, shape_b).astype(np.float32)
    np_c = np.matmul(np_a.astype(np.float16).astype(np.float32), np_b.astype(np.float16).astype(np.float32))
    np_cc = np.random.normal(0, 1, np_c.shape).astype(np.float32)
    expect = np_c + np_cc

    # compute pad size
    pad_size_a = [(128 - shape_a[1] % 128) % 128]
    pad_size_b = [(128 - shape_b[1] % 128) % 128]

    t = Tester("stages")
    t.stage_switch("static")
    a = t.load(np_a)
    a = t.cast(a, "float16")
    pad_a = t.stage_pad_store(a, pad_size_a)
    t.stage_switch("static")
    b = t.load(np_b)
    b = t.cast(b, "float16")
    pad_b = t.stage_pad_store(b, pad_size_b)
    t.stage_switch("mix")
    mat_a = t.stage_load(pad_a)
    mat_b = t.stage_load(pad_b)
    res = t.matmul(mat_a, mat_b, False, False)
    c = t.load(np_cc)
    res = t.cast(res, "float32")
    res = t.binary("Add", res, c)
    t.store_expect(res, expect, 2e-3)
    assert(t.run_check())

@pytest.mark.mix
def test_gemm_row_nopad_transposeB():
    m = np.random.randint(1, high = 1025)
    n = np.random.randint(2, high = 1025)
    k = np.random.randint(1, high = 128) * 128
    shape_a = [m, k]
    shape_b = [n, k]
    np_a = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.astype(np.float32), np_b.transpose().astype(np.float32)) + 2.5

    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, True)
    res = t.binary("Add", res, 2.5)
    t.store_expect(res, np_c)
    assert(t.run_check())

@pytest.mark.mix
def test_gemm_row_nopad_transposeA():
    m = np.random.randint(1, high = 32) * 128
    n = np.random.randint(1, high = 32) * 128
    k = np.random.randint(1, high = 1025)
    shape_a = [k, m]
    shape_b = [k, n]
    np_a = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.transpose().astype(np.float32), np_b.astype(np.float32)) + 2.5

    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, True, False)
    res = t.binary("Add", res, 2.5)
    t.store_expect(res, np_c)
    assert(t.run_check())

@pytest.mark.mix
def test_gemm_row_nopad():
    m = np.random.randint(1, high = 1025)
    k = np.random.randint(1, high = 1025)
    n = np.random.randint(1, high = 128) * 128
    shape_a = [m, k]
    shape_b = [k, n]
    np_a = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    t = Tester("stages")
    t.stage_switch("static")
    a = t.load(np_a)
    a = t.copy(a)
    pad_a = t.stage_pad_store(a, [0, (128 - k % 128) % 128])
    t.stage_switch("mix")
    mat_a = t.stage_load(pad_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, np_c)
    assert(t.run_check())

@pytest.mark.mix
def test_batchgemm():
    m = np.random.randint(1, high = 1025)
    n = np.random.randint(1, high = 1025)
    k = np.random.randint(1, high = 1025)
    a = np.random.randint(1, high = 10)
    b = np.random.randint(1, high = 10)
    shape_a = [a, 1, m, k]
    shape_b = [1, b, k, n]
    np_a = np.random.normal(0, 1, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 1, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    # compute pad size
    pad_size_a = [(128 - shape_a[3] % 128) % 128]
    pad_size_b = [(128 - shape_b[3] % 128) % 128]

    t = Tester("stages")
    t.stage_switch("static")
    a = t.load(np_a)
    a = t.copy(a)
    pad_a = t.stage_pad_store(a, pad_size_a)
    t.stage_switch("static")
    b = t.load(np_b)
    b = t.copy(b)
    pad_b = t.stage_pad_store(b, pad_size_b)
    t.stage_switch("mix")
    mat_a = t.stage_load(pad_a)
    mat_b = t.stage_load(pad_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, expect)
    assert(t.run_check())