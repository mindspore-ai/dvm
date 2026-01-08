# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import inspect
import sys
import random
import numpy as np
from dvm.tester import Tester

def test_0(t):
    x0 = t.load(np.empty([128], np.float32))
    x1 = t.load(np.empty([128], np.float32))
    x2 = t.add(x0, x1)
    t.store(x2)

def test_1(t):
    x0 = t.load(np.empty([128], np.float16))
    x1 = t.load(np.empty([128], np.float16))
    x2 = t.add(x0, x1)
    t.store(x2)

def test_2(t):
    x0 = t.load(np.empty([1, 32, 1, 44, 80], np.float32))
    x1 = t.load(np.empty([1, 32, 100, 44, 80], np.float32))
    x2 = t.mul(x0, x1)
    x3 = t.load(np.empty([1, 1, 100, 44, 80], np.float32))
    x4 = t.mul(x3, x1)
    x5 = t.sum(x2, (1,), False)
    t.store(x5)
    x7 = t.sum(x4, (2,), False)
    t.store(x7)

def test_3(t):
    x0 = t.load(np.empty([96], np.float32))
    x1 = t.sum(x0, (0,), False)
    x2 = t.load(np.empty([96, 128], np.float32))
    x3 = t.sum(x2, (0,), False)
    x4 = t.load(np.empty([96, 128], np.float32))
    x5 = t.sum(x4, (0,), False)
    x6 = t.div(x3, x1)
    t.store(x6)
    x8 = t.div(x5, x1)
    x9 = t.mul(x6, x6)
    x10 = t.sub(x8, x9)
    x11 = t.add(x10, 1e-05)
    x12 = t.sqrt(x11)
    x13 = t.div(x12, 1)
    t.store(x13)
    x15 = t.load(np.empty([128], np.float32))
    x16 = t.mul(x15, 0.9)
    x17 = t.mul(x6, 0.1)
    x18 = t.add(x16, x17)
    t.store(x18)
    x20 = t.add(x1, -1)
    x21 = t.div(x1, x20)
    x22 = t.mul(x10, x21)
    x23 = t.load(np.empty([128], np.float32))
    x24 = t.mul(x23, 0.9)
    x25 = t.mul(x22, 0.1)
    x26 = t.add(x24, x25)
    t.store(x26)

def test_4(t):
    x0 = t.load(np.empty([24, 88, 160], np.float32))
    x1 = t.less(x0,  0)
    x2 = t.cast(x1, "float16")
    x3 = t.cast(x2, "bool")
    x4 = t.less(x0,  100)
    x5 = t.cast(x4, "float16")
    x6 = t.cast(x5, "bool")
    x7 = t.cast(x3, "float16")
    x8 = t.cast(x6, "float16")
    x9 = t.logical_or(x7, x8)
    x10 = t.less(x9,  0)
    x11 = t.cast(x10, "bool")
    x12 = t.isfinite(x0)
    x13 = t.less(x12,  0)
    x14 = t.cast(x13, "float16")
    x15 = t.cast(x14, "bool")
    x16 = t.cast(x15, "float16")
    x17 = t.mul(x16, -1)
    x18 = t.add(x17, 1)
    x19 = t.less(x18,  0)
    x20 = t.cast(x19, "bool")
    x21 = t.cast(x11, "float16")
    x22 = t.cast(x20, "float16")
    x23 = t.logical_or(x21, x22)
    x24 = t.less(x23,  0)
    x25 = t.cast(x24, "bool")
    x26 = t.cast(x25, "float16")
    x27 = t.mul(x26, -1)
    x28 = t.add(x27, 1)
    x29 = t.less(x28,  0)
    x30 = t.cast(x29, "bool")
    x31 = t.cast(x30, "float16")
    x32 = t.cast(x31, "float32")
    x33 = t.mul(x0, x32)
    x34 = t.cast(x25, "float16")
    x35 = t.cast(x34, "float32")
    x36 = t.mul(x35, 100)
    x37 = t.add(x33, x36)
    t.store(x37)

def test_5(t):
    x0 = t.load(np.empty([60, 288, 112], np.float32))
    x1 = t.less(x0,  1)
    x2 = t.cast(x1, "float16")
    x3 = t.cast(x2, "bool")
    x4 = t.cast(x3, "float16")
    x5 = t.cast(x4, "float32")
    t.store(x5)

def test_6(t):
    x0 = t.load(np.empty([2, 4, 512, 1024], np.float16))
    x1 = t.load(np.empty([1024, 512], np.float16))
    x2 = t.matmul(x0, x1, False, False)
    t.store(x2)

def test_7(t):
    x0 = t.load(np.empty([2, 4, 512, 1024], np.float16))
    x1 = t.load(np.empty([1024, 512], np.float16))
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 0.04)
    t.store(x3)

def test_8(t):
    x0 = t.load(np.empty([2, 4, 512, 1024], np.float16))
    x1 = t.load(np.empty([1024, 512], np.float16))
    x2 = t.matmul(x0, x1, False, False)
    x3 = t.add(x2, 0.04)
    t.store(x3)
    x4 = t.mul(x2, 0.04)
    t.store(x4)

def fuzz_consistent():
    all_tests = []
    mod = inspect.currentframe()
    for name, value in mod.f_globals.items():
        if name.startswith("test_") and inspect.isfunction(value):
            all_tests.append(value)
    iter_num = 5000000
    tester = Tester("eager", run_mode="das")
    test_num = len(all_tests)
    expects = [None] * test_num
    print("test_num={}, iter_num={}".format(test_num, iter_num), file=sys.stderr)
    for tidx in range(test_num):
        all_tests[tidx](tester)
        tester.codegen()
        expects[tidx] = tester.das()
        tester.reset()
    print("gen expect finish, start test", file=sys.stderr)
    for i in range(iter_num):
        if i % 1000 == 0:
            print(" (iter){}".format(i), file=sys.stderr)
        tidx = random.randint(0, test_num - 1)
        all_tests[tidx](tester)
        tester.codegen()
        das = tester.das()
        if expects[tidx] != das:
            print("[FAIL]", tidx, file=sys.stderr)
            print("******** [{}] **********".format(tidx))
            print("expect =>")
            print(expects[tidx])
            print("output =>")
            print(tester.dump())
            print(das)
        tester.reset()

if __name__ == "__main__":
    fuzz_consistent()
