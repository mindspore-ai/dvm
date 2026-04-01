import numpy as np
import pytest
from dvm.tester import Tester
import shutil
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["vector", "eager"])
def test_aiv(mode):
    path_name = "./profile_aiv"
    t = Tester(mode)
    a = np.random.normal(0, 0.1, [32, 256]).astype(np.float16)
    x = t.load(a)
    y = t.add(x, 1)
    t.store_expect(y, a + 1)
    assert (t.run_check())
    # cann is not stable in ci, need recheck
    try:
        info = t.run_msprof(path_name)
        shutil.rmtree(path_name)
        assert ("AI_VECTOR" in info)
        assert ("Dvm" in info)
    except:
        print("ms_prof failed, please recheck")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["mix", "eager"])
def test_aic(mode):
    path_name = "./profile_aic"
    t = Tester(mode)
    a = np.random.normal(0, 0.1, [256, 256]).astype(np.float16)
    b = np.random.normal(0, 0.1, [256, 256]).astype(np.float16)
    expect = np.matmul(a.astype(np.float32),
                       b.astype(np.float32))
    x = t.load(a)
    y = t.load(b)
    z = t.matmul(x, y, False, False)
    t.store_expect(z, expect, 1e-3)
    assert (t.run_check())
    info = t.run_msprof(path_name)
    shutil.rmtree(path_name)
    assert ("AI_CORE" in info)
    assert ("Dvm" in info)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["mix", "eager"])
def test_mix_aic(mode):
    path_name = "./profile_mix_aic"
    t = Tester(mode)
    a = np.random.normal(0, 0.1, [256, 256]).astype(np.float16)
    b = np.random.normal(0, 0.1, [256, 256]).astype(np.float16)
    c = np.random.normal(0, 0.1, [256, 256]).astype(np.float16)
    expect = np.matmul(a.astype(np.float32),
                       b.astype(np.float32))
    x = t.load(a)
    y = t.load(b)
    q = t.load(c)
    z = t.matmul(x, y, False, False)
    z = t.add(z, q)
    t.store_expect(z, expect + c, 1e-3)
    assert (t.run_check())
    info = t.run_msprof(path_name)
    shutil.rmtree(path_name)
    assert ("MIX_AIC" in info)
    assert ("Dvm" in info)
