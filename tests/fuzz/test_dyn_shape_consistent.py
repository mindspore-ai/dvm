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
import sys
import random
import numpy as np
from dvm.tester import Tester, ShapeRef

class Suite:
    def __init__(self, case_list):
        self.case_list = case_list
    def construct(self, t):
        pass
    def update(self, t, cs):
        pass

class BroadcastSuite(Suite):
    def __init__(self):
        self.input_x = None
        self.input_y = None
        case_list = [
            [[10, 4000], [10, 4000]],
            [[1, 5000], [10, 5000]],
            [[10, 6000], [1, 6000]],

            [[8, 10, 1024], [8, 10, 1024]],
            [[1, 10, 2000], [8, 10, 2000]],
            [[8, 10, 1024], [8, 1, 1024]],
            [[1, 10, 1], [8, 1, 1024]],
            [[8, 10, 1024], [1, 1024]],
            [[8, 10, 1024], [1]],

            [[3, 8, 4, 1024], [3, 8, 4, 1024]],
            [[1, 8, 4, 1024], [3, 8, 4, 1024]],
            [[1, 8, 4, 1], [3, 8, 4, 1024]],
            [[8, 4, 1024], [3, 8, 1, 1]],
            [[1, 8, 1, 1024], [3, 8, 4, 1024]],
            [[3, 8, 4, 1024], [1, 1, 4, 1]],
            [[1, 1024], [3, 8, 4, 1]],
        ]
        Suite.__init__(self, case_list)

    def construct(self, t):
        self.input_x = t.load([-1], "float32")
        self.input_y = t.load([-1], "float32")
        x1 = t.add(self.input_x, 0.12)
        x2 = t.mul(x1, self.input_y)
        t.store(x2)

    def update(self, t, cs):
        t.input(self.input_x, np.empty(cs[0], np.float32))
        t.input(self.input_y, np.empty(cs[1], np.float32))

class ReduceSuite(Suite):
    def __init__(self):
        self.input_x = None
        self.red_dims = ShapeRef()
        case_list = [
            [[10, 4000], [0]],
            [[10, 4000], [1]],
            [[10, 10000], [0]],
            [[10, 10000], [1]],

            [[10, 8, 2000], [0]],
            [[10, 8, 2000], [1]],
            [[10, 8, 2000], [2]],
            [[10, 8, 2000], [0, 1]],
            [[10, 8, 2000], [1, 2]],
            [[10, 8, 2000], [0, 2]],
            [[10, 8, 2000], [0, 1, 2]],
            [[10, 8, 6000], [0]],
            [[10, 8, 6000], [1]],
            [[10, 8, 6000], [2]],
            [[10, 8, 6000], [0, 1]],
            [[10, 8, 6000], [1, 2]],
            [[10, 8, 6000], [0, 2]],
            [[10, 8, 6000], [0, 1, 2]],

            [[2, 10, 8, 1024], [1]],
            [[2, 10, 8, 1024], [3]],
            [[2, 10, 8, 1024], [0, 2, 3]],

            [[3, 2, 6, 8, 512], [0, 2, 4]],
            [[3, 2, 6, 8, 512], [1, 3, 5]],
        ]
        Suite.__init__(self, case_list)

    def construct(self, t):
        self.input_x = t.load([-1], "float32")
        x1 = t.add(self.input_x, 0.12)
        x2 = t.sum(x1, self.red_dims, True)
        t.store(x2)

    def update(self, t, cs):
        self.red_dims.update(cs[1])
        t.input(self.input_x, np.empty(cs[0], np.float32))

def fuzz_consistent(suite_cls):
    suite = suite_cls()
    print("[suite] {}".format(suite.__class__.__name__))
    print("[suite] {}".format(suite.__class__.__name__), file=sys.stderr)
    test_num = len(suite.case_list)
    expects = [None] * test_num
    for tidx in range(test_num):
        t = Tester('vector:dyn', run_mode = 'das')
        suite.construct(t)
        suite.update(t, suite.case_list[tidx])
        t.run()
        expects[tidx] = t.das()
        t.reset()
    iter_num = test_num * 50
    tester = Tester('vector:dyn', run_mode = 'das')
    suite.construct(tester)
    for i in range(iter_num):
        if i % 1000 == 0:
            print(" (iter){}".format(i), file=sys.stderr)
        tidx = random.randint(0, test_num - 1)
        cs = suite.case_list[tidx]
        suite.update(tester, cs)
        tester.run()
        das = tester.das()
        print("case({}): {}".format(tidx, cs))
        if expects[tidx] != das:
            print("[FAIL]", cs, file=sys.stderr)
            print("expect =>")
            print(expects[tidx])
            print("output =>")
            print(das)
            print(tester.dump())
        tester.reset()

if __name__ == '__main__':
    fuzz_consistent(BroadcastSuite)
    fuzz_consistent(ReduceSuite)
