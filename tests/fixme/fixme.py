# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
from dvm.tester import Tester

def test_videochat_reducesum3():
    ''' bad tiling for inner divisible limit: [2048, 32001] -> [1,3]. B4: 361ms '''
    np.random.seed(1)
    a0 = np.random.normal(0, 1, [2048, 32001]).astype(np.float32)
    a1 = np.random.normal(0, 1, [2048, 1]).astype(np.float32)
    e1 = np.exp(a0 - a1)
    e2 = np.sum(e1, axis=(1,), keepdims=False)
    t = Tester()
    x0 = t.load(a0)
    x1 = t.load(a1)
    y0 = t.binary("Sub", x0, x1)
    y1 = t.unary("Exp", y0)
    t.store_expect(y1, e1)
    y2 = t.reduce("sum", y1, [1], False)
    t.store_expect(y2, e2)
    print(t.run_perf())

def test_slice_dit_00():
    '''
      fragmented load2 cause bad performace. for 910B1
      1. donot broadcast x of b tensor, time is 100us. tile_num=4200, simd_width=64. load(no padsize)=32x349
      2. do broadcast x of b tensor, time is 600us. tile_num=6300, simd_width=48. load2(padsize=16)=80x93
      suggestion: entire load to ub and split to diffrent area by vector?
    '''
    t = Tester("eager")
    a = np.random.normal(0, 1, (1, 14, 41850, 40)).astype(np.float16)
    #b = np.random.normal(0, 1, (1, 1, 41850, 40)).astype(np.float16)  # 100 us
    b = np.random.normal(0, 1, (1, 1, 41850, 1)).astype(np.float16)   # 600 us
    c = np.random.normal(0, 1, (1, 14, 41850, 40)).astype(np.float16)
    y = t.load(b)
    y = t.cast(y, "float32")
    x = t.load(a)
    x = t.cast(x, "float32")
    z = t.binary("Mul", x, y)
    zz = t.load(c)
    zz = t.cast(zz, "float32")
    z = t.binary("Add", zz, z)
    z = t.cast(z, "float16")
    t.store(z)
    print(t.run_perf())
