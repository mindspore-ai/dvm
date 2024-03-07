import pytest
import numpy as np
from dvm.tester import Tester


@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('shape, start, size', [((32, 32), (3, 3), (16, 16)), 
                                                ((32, 32), (3, 3), (8, 13)), 
                                                ((4, 4097), (0, 0), (2, 4097)),  
                                                ((1111, 1111), (222, 222), (333, 333)),
                                                ((1621, 4096), (4, 222), (333, 333))])
def test_slice_load_2d(type, shape, start, size):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.slice_load(a, start, size)
    z = t.unary("Exp", x)
    t.store_expect(
        z, np.exp(a[start[0]:start[0]+size[0], start[1]:start[1]+size[1]]))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('shape, start, size', [((32,), (3,), (16,)), 
                                                ((32,), (3,), (8,)), 
                                                ((4,), (0,), (2,)),  
                                                ((1111,), (222,), (333,)),
                                                ((1621,), (4,), (333,))])
def test_slice_load_1d(type, shape, start, size):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.slice_load(a, start, size)
    z = t.unary("Exp", x)
    t.store_expect(
        z, np.exp(a[start[0]:start[0]+size[0]]))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float16, np.float32])
@pytest.mark.parametrize('shape, start, end', [((32,), (3,), (16,)), 
                                                ((32,), (3,), (8,)), 
                                                ((4,), (0,), (2,)),  
                                                ((1111,), (222,), (333,)),
                                                ((1621,), (4,), (333,))])
def test_stridedslice_load_1d(type, shape, start, end):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.stridedslice_load(a, start, end, (1,))
    z = t.unary("Exp", x)
    t.store_expect(z, np.exp(a[start[0]:end[0]]))
    assert (t.run_check())

@pytest.mark.parametrize('type', [np.float32, np.float16])
@pytest.mark.parametrize('shape, start, size', [((32, 32, 4), (3, 3, 1), (16, 16, 2)), 
                                                ((32, 32, 10), (3, 3, 1), (8, 13, 8)), 
                                                ((4, 4097, 20), (0, 0, 5), (2, 4097, 5)),  
                                                ((111, 111, 111), (22, 22, 22), (33, 33, 32)),
                                                ((16, 11,  4096), (4, 4, 222), (3, 3, 332)),
                                                ((32, 33, 34), (3, 5, 7), (27, 6, 5)),
])
def test_slice_load_3d(type, shape, start, size):
    t = Tester()
    a = np.random.normal(0, 1, shape).astype(type)
    x = t.slice_load(a, start, size)
    z = t.binary("Mul", x, 2)
    t.store_expect(
        z, 2*(a[start[0]:start[0]+size[0], start[1]:start[1]+size[1], start[2]:start[2]+size[2]]))
    assert (t.run_check())