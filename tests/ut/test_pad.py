import pytest
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
@pytest.mark.parametrize('shape, pad', [((200, 100), 128), ((220, 100), 202), ((2000, 1000), 1028), ((2000, 1), 1)])
def test_pad_2d(type, eps, shape, pad):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.load(a)
    z = t.abs(x)
    o = t.pad_store(z, pad)
    expect = np.abs(a)
    t.run_check()
    out = t.output(o)
    assert np.allclose(out[:shape[0], :shape[1]], expect,
                       rtol=eps, atol=eps, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
@pytest.mark.parametrize('shape, pad', [((1, 3, 200, 100), 128), ((1, 1, 200, 111), 139), ((4, 5, 111, 1), 1), ((33, 3000, 111, 1), 16)])
def test_pad_4d(type, eps, shape, pad):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.load(a)
    z = t.abs(x)
    o = t.pad_store(z, pad)
    expect = np.abs(a)
    t.run_check()
    out = t.output(o)
    assert np.allclose(out[:, :, :, :shape[3]],
                       expect, rtol=eps, atol=eps, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
def test_pad_broadcast(type, eps):
    t = Tester()
    a = np.random.normal(0, 1, (1024, 1)).astype(type)
    b = np.random.normal(0, 1, (1024, 30)).astype(type)
    x = t.load(a)
    y = t.load(b)
    z = t.add(x, y)
    o = t.pad_store(z, 15)
    expect = a + b
    t.run_check()
    out = t.output(o)
    assert np.allclose(out[:, :30], expect,
                       rtol=eps, atol=eps, equal_nan=True)
