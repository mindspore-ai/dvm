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
