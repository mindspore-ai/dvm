import numpy as np
import pytest
from dvm.tester import Tester
import dvm
from tests.mark_utils import arg_mark


# basic test for matmul functionality
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_matmul_basic():
    np.random.seed(1)
    t = Tester("mix")
    m = 128
    k = 64
    n = 32
    trans_a = False
    trans_b = False
    shape_a = (k, m) if trans_a else (m, k)
    shape_b = (n, k) if trans_b else (k, n)
    g0 = np.random.random(shape_a).astype(np.float16)
    # g0 = np.eye(*shape_a).astype(np.float16)
    # g0 = np.ones(shape_a).astype(np.float16)

    g1 = np.random.random(shape_b).astype(np.float16)
    # g1 = np.eye(*shape_b).astype(np.float16)
    # g1 = np.ones(shape_b).astype(np.float16)
    expect = np.matmul((g0 if not trans_a else g0.T).astype(np.float32),
                       (g1 if not trans_b else g1.T).astype(np.float32)).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, trans_a, trans_b)
    t.store_expect(c, expect)
    assert (t.run_check(True))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('trans', [[False, False], [False, True], [True, False], [True, True]])
def test_matmul(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    g1 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    expect = np.matmul((g0 if not trans[0] else g0.T).astype(np.float32),
                       (g1 if not trans[1] else g1.T).astype(np.float32)).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('m, n, k', [[300, 5000, 40000]])
@pytest.mark.parametrize('trans', [[False, True], [True, True], [False, False], [True, False]])
def test_matmul_split_k(m, n, k, trans):
    shape_a = [k, m] if trans[0] else [m, k]
    shape_b = [n, k] if trans[1] else [k, n]
    if shape_a[1] > 65535 or shape_b[1] > 65535:
        return
    g0 = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    g1 = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    expect = np.matmul((g0 if not trans[0] else g0.T).astype(np.float32),
                       (g1 if not trans[1] else g1.T).astype(np.float32)).astype(np.float16)
    t = Tester("mix")
    a1 = t.load(g0)
    b1 = t.load(g1)
    c1 = t.matmul(a1, b1, trans[0], trans[1])
    t.store_expect(c1, expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[2, 4, 256, 256], [2, 4, 256, 256]],  # no broadcast
    [[256, 444], [3, 1, 444, 256]],  # different dim, broadcast A
    [[1, 1, 256, 256], [3, 4, 256, 256]],  # same dim, broadcast A
    [[3, 4, 256, 256], [1, 4, 256, 256]],  # same dim, broadcast B
    [[3, 4, 256, 256], [256, 256]],  # differnet dim, broadcast B
    [[1, 4, 256, 256], [3, 1, 256, 256]],  # broadcast both A and B
    [[4, 1, 10, 256], [256, 256]],
    [[24, 1, 4, 4], [4, 4096]],
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[4, 1, 256, 256], [1, 8, 256, 256]],
    [[768, 1024], [1024, 10240]],
    [[4, 10, 1, 256], [256, 256]],
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
    d = t.add(c, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('trans', [[False, False]])
def test_matmul_bf16(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 0.01, [64, 64]).astype(np.float32)
    g1 = np.random.normal(0, 0.01, [64, 64]).astype(np.float32)
    expect = np.matmul(g0 if not trans[0] else g0.T, g1 if not trans[1] else g1.T)
    a = t.load(g0, "bfloat16")
    b = t.load(g1, "bfloat16")
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect, 5e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[211, 211], [211, 230]],  # gemm normal case
    [[193, 193], [193, 1930]],  # m0 == 1
    [[1, 1024], [1024, 320]],  # m == 1
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    d = t.add(cc, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("PingPongLoad") > 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    d = t.add(cc, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    o = t.store_expect(e, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[211, 211], [211, 230]],  # gemm normal case
    [[193, 100], [100, 257]],  # m0 == 1
    [[1, 127], [127, 127]],  # m == 1
    [[1, 1000], [1000, 4000]],  # m == 1
    [[123, 1], [1, 777]],  # k == 1
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
    res = t.add(res, 2.5)
    t.store_expect(res, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("slice_store") > 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_matmul_post_fusion_inplace():
    shape_a, shape_b = [1024, 512], [512, 1024]
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    zx = np.random.normal(0, 0.01, np_c.shape).astype(np.float16)
    z = t.load(zx)
    z = t.mul(z, 0.5)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    d = t.add(c, z)
    d = t.add(d, 0.1)
    d = t.sub(d, z)
    t.store_expect(d, np_c + 0.1)
    assert (t.run_check())
    assert (t.das().count("PingPongLoad") == 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[1024, 512], [512, 1024]],
    [[256, 40960], [40960, 5120]],
])
def test_matmul_post_fusion_matmul_output(shape_a, shape_b):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32)).astype(np.float16)
    zx = Tester.fast_random_normal(0, 0.01, np_c.shape).astype(np.float16)
    z = t.load(zx)
    z = t.mul(z, 0.5)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    t.store_expect(c, np_c, 2e-3)
    d = t.add(c, z)
    d = t.add(d, 0.1)
    d = t.sub(d, z)
    t.store_expect(d, np_c + 0.1)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_matmul_col_nopad():
    m = 800
    n = 700
    k = 1024
    shape_a = [m, k]
    shape_b = [n, k]
    np_a = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    np_b = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(np_a.astype(np.float32), np_b.T.astype(np.float32)).astype(np.float16)

    t = Tester("mix")
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, True)
    t.store_expect(res, np_c)
    assert (t.run_check())
    assert (t.das().count("slice_store") == 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_sync_out_limit():
    a = Tester.fast_random_normal(0, 0.01, (8192, 1152)).astype(np.float16)
    b = Tester.fast_random_normal(0, 0.01, (1152, 4608)).astype(np.float16)
    c = Tester.fast_random_normal(0, 0.01, (8192, 4608)).astype(np.float16)
    d = Tester.fast_random_normal(0, 0.01, (4608,)).astype(np.float32)
    mm = np.matmul(a.astype(np.float32), b.astype(np.float32))
    t = Tester("mix")
    aa = t.load(a)
    bb = t.load(b)
    cc = t.load(c)
    dd = t.load(d)
    y0 = t.matmul(aa, bb, False, False)
    y1 = t.cast(y0, "float32")
    y2 = t.cast(cc, "float32")
    y3 = t.add(y2, dd)
    y4 = t.mul(y3, y3)
    y5 = t.mul(y3, y4)
    y6 = t.mul(y1, y5)
    for _ in range(100):
        y6 = t.add(y6, 0.1)
    y7 = t.cast(y6, "float16")
    t.store_expect(y3, c + d, 1e-3)
    t.store_expect(y7, (c + d) ** 3 * mm + 10, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[5120, 40960], [40960, 256]],  # split k
    [[211, 211], [211, 230]],  # unalign
    [[2, 4, 256, 256], [2, 4, 256, 256]],  # batchmatmul
    [[256, 256], [256, 256]],  # same shape
    [[256, 1, 1000], [1000, 256]],  # batch fold
])
def test_tuning_matmul(shape_a, shape_b):
    np_a = Tester.fast_random_normal(0, 0.1, shape_a).astype(np.float16)
    np_b = Tester.fast_random_normal(0, 0.1, shape_b).astype(np.float16)
    expect = np.matmul(np_a.astype(np.float32), np_b.astype(np.float32)).astype(np.float16)
    t = Tester("mix")
    Tester.set_online_tuning(True)
    mat_a = t.load(np_a)
    mat_b = t.load(np_b)
    res = t.matmul(mat_a, mat_b, False, False)
    t.store_expect(res, expect)
    assert (t.run_check())
    Tester.set_online_tuning(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[2, 16, 128, 128], [1, 128, 256]],
    [[16, 16], [16, 4096]],
    [[7680, 1024], [1024, 4096]],
    [[1024, 40960], [40960, 4096]],
    [[1344, 1792], [1792, 64]],
])
def test_matmul_bias(shape_a, shape_b):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.1, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.01, [np_c.shape[len(np_c.shape) - 1]]).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    z = t.load(zx)
    c = t.matmul(a, b, False, False, z)
    expect = np_c + zx
    o = t.store_expect(c, expect, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [256, 256]],
    [[102, 40000], [40000, 5120]],
])
def test_matmul_bias_bf16(shape_a, shape_b):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.01, shape_a).astype(np.float32)
    bx = Tester.fast_random_normal(0, 0.01, shape_b).astype(np.float32)
    np_c = np.matmul(ax, bx)
    zx = np.random.normal(0, 0.01, [np_c.shape[len(np_c.shape) - 1]]).astype(np.float32)
    a = t.load(ax, "bfloat16")
    b = t.load(bx, "bfloat16")
    z = t.load(zx, "bfloat16")
    c = t.matmul(a, b, False, False, z)
    expect = np_c + zx
    o = t.store_expect(c, expect, 5e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[16, 16], [16, 4096]],
    [[4096, 16], [16, 16]],
    [[32, 4096, 16], [16, 16]],
    [[512, 64], [64, 8192]],
])
def test_matmul_skip_loadL1(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 0.01, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.01, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    o = t.store_expect(c, np_c, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [81960, 256]],
])
def test_matmul_n_big(shape_a, shape_b):
    t = Tester("mix")
    ax = Tester.fast_random_normal(0, 0.1, shape_a).astype(np.float16)
    bx = Tester.fast_random_normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32).T).astype(np.float16)
    zx = Tester.fast_random_normal(0, 0.1, np_c.shape).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, True)
    z = t.load(zx)
    d = t.add(c, z)
    o = t.store_expect(d, np_c + zx, 2e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
def test_dyn_matmul():
    t = Tester('mix:dyn')
    x = t.load([-1, -1], "float16")
    y = t.load([-1, -1], "float16")
    z = t.load([-1, -1], "float16")
    c = t.matmul(x, y, False, False)
    c = t.add(c, z)
    out = t.store(c)
    iterations = [[[256, 128], [128, 256], [256, 256]], [[256, 128], [128, 512], [256, 1]], [[444, 3333], [
        3333, 1111], [1, 1111]], [[1024, 1024], [1024, 2222], [1024, 2222]], [[2560, 128], [128, 512], [512]]]
    for x_shape, y_shape, z_shape in iterations:
        x_data = np.random.normal(0, 0.1, x_shape).astype(np.float16)
        y_data = np.random.normal(0, 0.1, y_shape).astype(np.float16)
        z_data = np.random.normal(0, 0.1, z_shape).astype(np.float16)
        np_c = np.matmul(x_data.astype(np.float32),
                         y_data.astype(np.float32)).astype(np.float16)
        t.input(x, x_data)
        t.input(y, y_data)
        t.input(z, z_data)
        t.run()
        assert (t.check(out, np_c + z_data, 2e-3))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, broadcast_shape', [
    [[2000, 512], [512, 3000], [1, 3000]],
    [[2000, 512], [512, 3000], [2000, 1]],
    [[2048, 512], [512, 4096], [2048, 1]],
    [[2048, 512], [512, 4096], [1, 1]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [1, 4, 256, 512]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [3, 1, 256, 512]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [256, 512]],
    [[3, 4, 256, 512], [512, 512], [1, 1, 256, 512]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [3, 1, 1, 512]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [3, 1, 256, 1]],
    [[3, 4, 256, 512], [3, 4, 512, 512], [1]],
])
def test_post_fusion_broadcast(shape_a, shape_b, broadcast_shape):
    t = Tester("mix")
    a = Tester.fast_random_normal(0, 0.1, shape_a).astype(np.float16)
    b = Tester.fast_random_normal(0, 0.1, shape_b).astype(np.float16)
    c = Tester.fast_random_normal(0, 0.1, broadcast_shape).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.load(c)
    x4 = t.add(x3, 0.2)
    x5 = t.mul(x4, x2)
    x4_expect = c + 0.2
    x5_expect = np.matmul(a.astype(np.float32), b.astype(np.float32)) * x4_expect
    t.store_expect(x5, x5_expect, 1e-3)
    t.store_expect(x4, x4_expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b, shape_bias', [
    [[256, 1, 1000], [1000, 256], None],
    [[256, 1, 1000], [1000, 256], [256, 1, 256]],
    [[256, 1, 1000], [1000, 256], [1, 256]],
    [[520, 1, 40960], [40960, 256], [520, 1, 256]],
    [[520, 1, 40960], [40960, 256], [1, 256]],
])
def test_batch_fold(shape_a, shape_b, shape_bias):
    t = Tester("mix")
    a = Tester.fast_random_normal(0, 0.1, shape_a).astype(np.float16)
    b = Tester.fast_random_normal(0, 0.1, shape_b).astype(np.float16)
    x0 = t.load(a)
    x1 = t.load(b)
    out = t.matmul(x0, x1, False, False)
    expect = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    if shape_bias:
        bias = Tester.fast_random_normal(0, 0.1, shape_bias).astype(np.float16)
        x2 = t.load(bias)
        out = t.add(out, x2)
        expect = expect + bias
    t.store_expect(out, expect, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_batch_fold_broadcast():
    t = Tester("mix")
    x0_a = Tester.fast_random_normal(0, 0.1, [632, 1, 3584]).astype(np.float16)
    x0 = t.load(x0_a)
    x1_a = Tester.fast_random_normal(0, 0.1, [4608, 3584]).astype(np.float16)
    x1 = t.load(x1_a)
    x2 = t.matmul(x0, x1, False, True)
    x3 = t.cast(x2, "float32")
    x4_a = np.random.normal(0, 0.1, [4608]).astype(np.float16)
    x4 = t.load(x4_a)
    x5 = t.cast(x4, "float32")
    x6 = t.add(x3, x5)
    x7 = t.cast(x6, "float16")
    x2_e = np.matmul(x0_a.astype(np.float32), x1_a.swapaxes(-1, -2).astype(np.float32)).astype(np.float16)
    y7_e = np.add(x2_e.astype(np.float32), x4_a.astype(np.float32)).astype(np.float16)
    t.store_expect(x7, y7_e, 1e-3)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support L0C->UB")
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [256, 256]],
    [[1024, 128], [128, 1024]],
    [[301, 256], [256, 401]],
])
def test_matmul_post_fusion_cc_ub_sync_0(shape_a, shape_b):
    t = Tester("mix")
    Tester.set_cube_store_type(1)
    ax = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.1, np_c.shape).astype(np.float16)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    d = t.add(c, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("cload") > 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support L0C->UB")
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [256, 256]],
    [[1024, 128], [128, 1024]],
    [[301, 256], [256, 401]],
])
def test_matmul_post_fusion_cc_ub_sync_1(shape_a, shape_b):
    t = Tester("mix")
    Tester.set_cube_store_type(1)
    ax = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.1, np_c.shape).astype(np.float32)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    c = t.cast(c, "float32")
    d = t.add(c, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("cload") > 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() != 'AscendC310', reason="c310 support L0C->UB")
@pytest.mark.mix
@pytest.mark.parametrize('shape_a, shape_b', [
    [[256, 256], [256, 256]],
    [[1024, 128], [128, 1024]],
    [[301, 256], [256, 401]],
])
def test_matmul_post_fusion_cc_ub_sync_2(shape_a, shape_b):
    t = Tester("mix")
    Tester.set_cube_store_type(2)
    ax = np.random.normal(0, 0.1, shape_a).astype(np.float16)
    bx = np.random.normal(0, 0.1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 0.1, np_c.shape).astype(np.float32)
    a = t.load(ax)
    b = t.load(bx)
    c = t.matmul(a, b, False, False)
    z = t.load(zx)
    c = t.cast(c, "float32")
    d = t.add(c, z)
    e = t.abs(d)
    expect = np.abs(np_c + zx)
    t.store_expect(e, expect, 2e-3)
    assert (t.run_check())
    assert (t.das().count("CopyCubeTile") > 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode, shape', [["mix", [1024, 2048]], ["mix:dyn", [1000, 2000]]])
def test_same_matmul_input(mode, shape):
    t = Tester(mode)
    g0 = np.random.normal(0, 0.01, shape).astype(np.float16)
    a = t.load(g0)
    c = t.matmul(a, a, False, True)
    expect = np.matmul(g0.astype(np.float32), g0.T.astype(np.float32)).astype(np.float16)
    t.store_expect(c, expect)
    assert (t.run_check())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.mix
@pytest.mark.parametrize('mode', ["mix", "mix:dyn"])
def test_same_matmul_vec_input(mode):
    t = Tester(mode)
    g0 = np.random.normal(0, 0.01, [1024, 1024]).astype(np.float16)
    g1 = np.random.normal(0, 0.01, [1024, 1536]).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, False, False)
    d = t.add(c, b)
    expect = np.matmul(g0.astype(np.float32), g1.astype(np.float32)).astype(np.float16) + g1
    t.store_expect(d, expect)
    assert (t.run_check())
