import random
import numpy as np
from dvm.tester import Tester

# Generate tests automatically. TODO: support more operations
def fuzz(size, num_tests=1, seed=1, num_load=1):
    random.seed(seed)
    a0 = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    generators = [(Tester.sqrt, 1), (Tester.add, 2)]
    print("===== fuzz size: {}".format(size))
    for i in range(num_tests):
        print("--- test ", i, flush=True)
        t = Tester()
        unused = []
        pickable = []
        # load
        for j in range(num_load):
            load = t.load(a0)
            unused.append(load)
            pickable.append(load)
        for j in range(size):
            if len(pickable) == 1:
                gen = generators[0]
            else:
                gen = random.choice(generators)
            args = [t]
            for k in range(gen[1]):
                arg = random.choice(pickable)
                if arg in unused:
                    unused.remove(arg)
                args.append(arg)
            obj = gen[0](*args)
            unused.append(obj)
            pickable.append(obj)
        # store 
        for obj in unused:
            t.store(obj)
        t.set_passes("PrintPeakLive", "CompactPeakLiveness", "PrintPeakLive")
        t.run_check()

def test_size_10():
    fuzz(10, 10)