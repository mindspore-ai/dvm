import pytest
import numpy as np
from dvm.tester import Tester


@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
@pytest.mark.parametrize('shape, pad', [((200, 100), (256, 128)), ((220, 100), (225, 202)), ((2000, 1000), (2560, 1028))])
def test_pad_2d(type, eps, shape, pad):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.load(a)
    z = t.unary("Abs", x)
    o = t.pad_store(z, pad)
    expect = np.abs(a)
    t.run_check()
    out = t.output(o)
    assert np.allclose(out[:shape[0], :shape[1]], expect,
                       rtol=eps, atol=eps, equal_nan=True)

@pytest.mark.parametrize('type, eps', [(np.float16, 1e-3), (np.float32, 1e-5)])
@pytest.mark.parametrize('shape, pad', [((1, 3, 200, 100), (0, 0, 256, 128)), ((1, 1, 200, 111), (0, 0, 356, 139))])
def test_pad_4d(type, eps, shape, pad):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.load(a)
    z = t.unary("Abs", x)
    o = t.pad_store(z, pad)
    expect = np.abs(a)
    t.run_check()
    out = t.output(o)
    assert np.allclose(out[:, :, :shape[2], :shape[3]],
                       expect, rtol=eps, atol=eps, equal_nan=True)

