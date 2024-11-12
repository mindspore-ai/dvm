import numpy as np
import pytest
from dvm.tester import Tester


@pytest.mark.mix
@pytest.mark.parametrize('trans', [[False, False], [False, True], [True, False], [True, True]])
def test_matmul(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    g1 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    expect = np.matmul((g0 if not trans[0] else g0.transpose()).astype(np.float32),
                       (g1 if not trans[1] else g1.transpose()).astype(np.float32)).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect)
    assert (t.run_check())


@pytest.mark.mix
@pytest.mark.parametrize('m, n, k', [[222, 1111, 40111], [512, 256, 67584]])
@pytest.mark.parametrize('trans', [[False, True], [True, True], [False, False], [True, False]])
def test_matmul_split_k(m, n, k, trans):
    shape_a = [k, m] if trans[0] else [m, k]
    shape_b = [n, k] if trans[1] else [k, n]
    if shape_a[1] > 65535 or shape_b[1] > 65535:
        return
    g0 = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    g1 = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul((g0 if not trans[0] else g0.transpose()).astype(np.float32),
                       (g1 if not trans[1] else g1.transpose()).astype(np.float32)).astype(np.float16)
    t = Tester("mix")
    a1 = t.load(g0)
    b1 = t.load(g1)
    c1 = t.matmul(a1, b1, trans[0], trans[1])
    t.store_expect(c1, expect, 1e-3)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[2, 4, 256, 256], [2, 4, 256, 256]],  # no broadcast
    [[256, 256], [3, 1, 256, 256]],        # different dim, broadcast A
    [[1, 1, 256, 256], [3, 4, 256, 256]],  # same dim, broadcast A
    [[3, 4, 256, 256], [1, 4, 256, 256]],  # same dim, broadcast B
    [[3, 4, 256, 256], [256, 256]],        # differnet dim, broadcast B
    [[1, 4, 256, 256], [3, 1, 256, 256]]   # broadcast both A and B
])
def test_batchmatmul(shape_a, shape_b):
    t = Tester("mix")
    g0 = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    g1 = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul(g0.astype(np.float32), g1.astype(np.float32)).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, False, False)
    t.store_expect(c, expect)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[4, 1, 256, 256], [1, 8, 256, 256]],
    [[768, 1024], [1024, 10240]],
])
def test_matmul_post_fusion(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.01, np_c.shape).astype(np.float32)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    c = t.cast(c, "float32")
    d = t.binary("Add", c, z)
    e = t.unary("Abs", d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('trans', [[False, False]])
def test_matmul_bf16(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 0.01, [64, 64]).astype(np.float32)
    g1 = np.random.normal(0, 0.01, [64, 64]).astype(np.float32)
    expect = np.matmul(g0 if not trans[0] else g0.transpose(), g1 if not trans[1] else g1.transpose())
    a = t.load(g0, "bfloat16")
    b = t.load(g1, "bfloat16")
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect, 1e-1)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[211, 211], [211, 230]],      # gemm normal case
    [[193, 193], [193, 1930]],      # m0 == 1
    [[1, 1024], [1024, 320]],       # m == 1
    [[70000, 10], [10, 32]],
])
def test_unaligned_matmul(shape_a, shape_b):
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, expect)
    assert (t.run_check())
    assert (t.das().count("slice_store") > 0)

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [256, 256]],
    [[32, 256], [256, 128]],
    [[2048, 1024], [1024, 5120]],
])
def test_matmul_post_broadcast_fusion_0(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.01, [np_c.shape[1]]).astype(np.float32)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    cc = t.cast(c, "float32")
    d = t.binary("Add", cc, z)
    e = t.unary("Abs", d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("PingPongLoad") > 0)

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[512, 128], [128, 256]],
    [[768, 1024], [1024, 10240]],
])
def test_matmul_post_broadcast_fusion_1(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.01, [np_c.shape[0], 1]).astype(np.float32)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    cc = t.cast(c, "float32")
    d = t.binary("Add", cc, z)
    e = t.unary("Abs", d)
    expect = np.abs(np_c + zx)
    o = t.store_expect(e, expect, 2e-3)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[211, 211], [211, 230]],      # gemm normal case
    [[193, 100], [100, 257]],      # m0 == 1
    [[1, 127], [127, 127]],        # m == 1
    [[1, 1000], [1000, 4000]],     # m == 1
    [[123, 1], [1, 777]],        # k == 1
    [[123, 33333], [33333, 777]],
])
def test_unaligned_matmul_post_fusion(shape_a, shape_b):
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)) + 2.5
    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, False)
    res = t.cast(res, "float32")
    res = t.binary("Add", res, 2.5)
    t.store_expect(res, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("slice_store") > 0)

@pytest.mark.mix
def test_matmul_post_fusion_inplace():
    shape_a, shape_b = [1024, 512], [512, 1024]
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    zx = np.random.normal(0, 0.01, np_c.shape).astype(np.float16)
    z = t.load(zx)
    z = t.binary("Mul", z, 0.5)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.binary("Add", c, z)
    d = t.binary("Add", d, 0.1)
    d = t.binary("Sub", d, z)
    t.store_expect(d, np_c + 0.1)
    assert (t.run_check())
    assert (t.das().count("PingPongLoad") == 0)

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[1024, 512], [512, 1024]],
    [[123, 33333], [33333, 1111]],
])
def test_matmul_post_fusion_matmul_output(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    zx = np.random.normal(0, 0.01, np_c.shape).astype(np.float16)
    z = t.load(zx)
    z = t.binary("Mul", z, 0.5)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    t.store_expect(c, np_c, 2e-3)
    d = t.binary("Add", c, z)
    d = t.binary("Add", d, 0.1)
    d = t.binary("Sub", d, z)
    t.store_expect(d, np_c + 0.1)
    assert (t.run_check())

@pytest.mark.mix
def test_matmul_col_nopad():
    m = 800
    n = 700
    k = 1024
    shape_a = [m, k]
    shape_b = [n, k]
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.astype(np.float32), np_b.transpose().astype(np.float32)).astype(np.float16)

    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, True)
    t.store_expect(res, np_c)
    assert (t.run_check())
    assert (t.das().count("slice_store") == 0)

@pytest.mark.mix
def test_batchmatmul_col_nopad():
    m = 800
    n = 700
    k = 1024
    shape_a = [1, 2, m, k]
    shape_b = [3, 1, n, k]
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.astype(np.float32), np_b.transpose((0, 1, 3, 2)).astype(np.float32)).astype(np.float16)

    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, True)
    t.store_expect(res, np_c)
    assert (t.run_check())
    assert (t.das().count("slice_store") == 0)
    
@pytest.mark.mix
def test_sync_out_limit():
    a = np.random.normal(0, 0.01, (8192, 1152)).astype(np.float16)
    b = np.random.normal(0, 0.01, (1152, 4608)).astype(np.float16)
    c = np.random.normal(0, 0.01, (8192, 4608)).astype(np.float16)
    d = np.random.normal(0, 0.01, (4608,)).astype(np.float32)
    mm = np.matmul(a.astype(np.float32), b.astype(np.float32))
    t = Tester("mix")
    aa = t.load(a)
    bb = t.load(b)
    cc = t.load(c)
    dd = t.load(d)
    y0 = t.matmul(aa, bb, False, False)
    y1 = t.cast(y0, "float32")
    y2 = t.cast(cc, "float32")
    y3 = t.binary("Add", y2, dd)
    y4 = t.binary("Mul", y3, y3)
    y5 = t.binary("Mul", y3, y4)
    y6 = t.binary("Mul", y1, y5)
    for _ in range(100):
        y6 = t.binary("Add", y6, 0.1)
    y7 = t.cast(y6, "float16")
    t.store_expect(y3, c + d, 1e-3)
    t.store_expect(y7, (c + d) ** 3 * mm + 10, 1e-3)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[512, 40960], [40960, 256]],      # split k
    [[211, 211], [211, 230]],      # unalign
    [[2, 4, 256, 256], [2, 4, 256, 256]],       # batchmatmul
    [[256, 256], [256, 256]],    # same shape
])
def test_tuning_matmul(shape_a, shape_b):
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    t = Tester("mix")
    t.set_online_tuning(True)
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, expect)
    assert (t.run_check())
    t.set_online_tuning(False)

@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[2,16, 128, 128], [1,128, 256]],
    [[7680, 1024], [1024, 4096]],
    [[1024, 40960], [40960, 4096]],
])
def test_matmul_bias(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.01, [np_c.shape[len(np_c.shape)-1]]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    z = t.load(zx)
    c = t.matmul(a, b, False, False, z)
    expect = np_c + zx
    o = t.store_expect(c, expect, 2e-3)
    assert (t.run_check())
