import numpy as np
from dvm.tester import Tester

def test_matmul():
    t = Tester("mix")
    g0 = np.full([32, 128], 0.1, np.float32)
    g1 = np.full([128, 64], 0.1, np.float32)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, False, False)
    t.store(c) #TODO: store_check
    assert(t.run_check(True))

