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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msprof_static_refresh_io(tmp_path):
    path_name = str(tmp_path / "profile_static_refresh_io")
    t = Tester("vector")
    x_data = np.full((32, 256), 1.0, dtype=np.float32)
    y_data = np.full((32, 256), 2.0, dtype=np.float32)
    x = t.load(x_data)
    y = t.load(y_data)
    out = t.store(t.add(x, y))
    t.run()

    t.start_msprof(path_name)
    try:
        for value in (3.0, 4.0):
            x_data = np.full((32, 256), value, dtype=np.float32)
            y_data = np.full((32, 256), value + 1.0, dtype=np.float32)
            t.release_io()
            t.input(x, x_data)
            t.input(y, y_data)
            t.run()
    finally:
        info = t.stop_msprof()
    shutil.rmtree(path_name)
    assert t.check(out, x_data + y_data)
    assert "AI_VECTOR" in info
    assert "Dvm" in info


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msprof_dynamic_refresh_io(tmp_path):
    path_name = str(tmp_path / "profile_dynamic_refresh_io")
    t = Tester("vector:dyn")
    x = t.load([-1, -1], "float32")
    y = t.load([-1, -1], "float32")
    out = t.store(t.add(x, y))
    t.input(x, np.ones((16, 32), dtype=np.float32))
    t.input(y, np.ones((16, 32), dtype=np.float32))
    t.run()

    t.start_msprof(path_name)
    try:
        for shape in ((32, 64), (16, 128)):
            x_data = np.full(shape, 1.5, dtype=np.float32)
            y_data = np.full(shape, 2.5, dtype=np.float32)
            t.release_io()
            t.input(x, x_data)
            t.input(y, y_data)
            t.run()
    finally:
        info = t.stop_msprof()
    shutil.rmtree(path_name)
    assert t.check(out, x_data + y_data)
    assert "AI_VECTOR" in info
    assert "Dvm" in info


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msprof_split_kernel(tmp_path):
    path_name = str(tmp_path / "profile_split_kernel")
    t = Tester("split:priv1")
    x_data = np.ones((10, 4096), dtype=np.float16)
    x = t.load(x_data)
    value = t.add(x, 0.5)
    t.store(value)
    value = t.cast(value, "float32")
    t.store(t.sum(value, (0,), True))
    t.run()

    t.start_msprof(path_name)
    try:
        t.run()
        t.run()
    finally:
        info = t.stop_msprof()
    shutil.rmtree(path_name)
    assert "AI_VECTOR" in info
    assert "Dvm" in info
