import numpy as np
import dvm
import pytest
from dvm.tester import Tester


@pytest.mark.mix
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
@pytest.mark.parametrize('trans', [[False, False], [False, True], [True, False], [True, True]])
def test_matmul(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 1, [1024, 1024]).astype(np.float16)
    g1 = np.random.normal(0, 1, [1024, 1024]).astype(np.float16)
    expect = np.matmul(g0 if not trans[0] else g0.transpose(), g1 if not trans[1] else g1.transpose())
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect)
    assert (t.run_check())


@pytest.mark.mix
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
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
    g0 = np.random.normal(0, 1, shape_a).astype(np.float16)
    g1 = np.random.normal(0, 1, shape_b).astype(np.float16)
    expect = np.matmul(g0, g1)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, False, False)
    t.store_expect(c, expect)
    assert (t.run_check())

@pytest.mark.mix
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
@pytest.mark.parametrize('shape_a, shape_b', [
    [[4, 1, 256, 256], [1, 8, 256, 256]],
    [[16, 2048, 1024], [16, 1024, 512]],
    [[2048, 1024], [1024, 5120]],
    [[768, 1024], [1024, 10240]],
])
def test_matmul_post_fusion(shape_a, shape_b):
    t = Tester("mix")
    ax = np.random.normal(0, 1, shape_a).astype(np.float16)
    bx = np.random.normal(0, 1, shape_b).astype(np.float16)
    np_c = np.matmul(ax.astype(np.float32), bx.astype(np.float32))
    zx = np.random.normal(0, 1, np_c.shape).astype(np.float32)
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
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
@pytest.mark.parametrize('trans', [[False, False]])
def test_matmul_bf16(trans):
    t = Tester("mix")
    g0 = np.random.normal(0, 1, [64, 64]).astype(np.float32)
    g1 = np.random.normal(0, 1, [64, 64]).astype(np.float32)
    expect = np.matmul(g0 if not trans[0] else g0.transpose(), g1 if not trans[1] else g1.transpose())
    a = t.load(g0, "bfloat16")
    b = t.load(g1, "bfloat16")
    c = t.matmul(a, b, trans[0], trans[1])
    t.store_expect(c, expect, 1e-1)
    assert (t.run_check())