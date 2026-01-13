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
import sys
import numpy as np
import dvm
from dvm.tester import Tester

B1 = "Ascend910B1"
B4 = "Ascend910B4"
soc_name = dvm.Device.soc_name()

def run_perf(op_args):
    t = Tester("mix")
    shape_a, shape_b, trans_a, trans_b = op_args
    g0 = np.random.normal(0, 1, shape_a).astype(np.float16)
    g1 = np.random.normal(0, 1, shape_b).astype(np.float16)
    a = t.load(g0)
    b = t.load(g1)
    c = t.matmul(a, b, trans_a, trans_b)
    _ = t.store(c)
    t.codegen()
    min_time, max_time, avg_time = t.perf()
    return round(min_time, 2)

all_cases = [
([(256, 10240), (10240, 1280), False, False], {B1: 132.91, B4: 139.76}),
([(1024, 1280), (1024, 5120), True, False], {B1: 120.09, B4: 138.96}),
([(1024, 1280), (1280, 5120), False, False], {B1: 116.48, B4: 138.67}),
([(1024, 1280), (10240, 1280), False, True], {B1: 226.69, B4: 270.95}),
([(1024, 5120), (1280, 5120), False, True], {B1: 132.37, B4: 138.63}),
([(1024, 10240), (1024, 1280), True, False], {B1: 229.77, B4: 270.48}),
([(1024, 10240), (10240, 1280), False, False], {B1: 258.17, B4: 270.55}),
([(2048, 5120), (2048, 5120), True, False], {B1: 391.23, B4: 521.94}),
([(2048, 5120), (5120, 5120), False, True], {B1: 387.72, B4: 518.63}),
([(2048, 5120), (5120, 5120), False, False], {B1: 388.39, B4: 522.04}),
([(2048, 2048), (8192, 2048), False, True], {B1: 248.11, B4: 340.42}),
([(2048, 2560), (2560, 8192), False, False], {B1: 308.09, B4: 424.36}),
([(2048, 8192), (2560, 8192), False, True], {B1: 313.88, B4: 417.32}),
([(4096, 2560), (2560, 8192), False, False], {B1: 598.47, B4: 876.26}),
([(4096, 2560), (4096, 8192), True, False], {B1: 598.42, B4: 866.90}),
([(4096, 8192), (2560, 8192), False, True], {B1: 647.20, B4: 843.28}),
([(4096, 8192), (4096, 2048), True, False], {B1: 505.11, B4: 688.71}),
([(8192, 512), (512, 4096), False, False], {B1: 127.32, B4: 171.91}),
([(8192, 512), (4096, 512), False, True], {B1: 128.22, B4: 171.60}),
([(8192, 512), (8192, 4096), True, False], {B1: 136.74, B4: 210.61}),
([(8192, 4096), (8192, 512), True, False], {B1: 141.61, B4: 210.17}),
([(8192, 1280), (1280, 5120), False, False], {B1: 390.72, B4: 522.5}),
([(8192, 1280), (5120, 1280), False, True], {B1: 390.04, B4: 519.20}),
# this case perf in B4 is not stable:
([(8192, 1280), (8192, 5120), True, False], {B1: 399.32}),
([(8192, 5120), (1280, 5120), False, True], {B1: 402.33, B4: 521.17}),
([(16384, 4096), (4096, 4096), False, True], {B1: 1983.64, B4: 2644.11}),
([(16384, 4096), (4096, 4096), False, False], {B1: 1970.05, B4: 2651.09}),
([(16384, 4096), (4096, 11008), False, False], {B1: 5336.18, B4: 7749.19}),
([(16384, 4096), (11008, 4096), False, True], {B1: 5369.76, B4: 7590.22}),
([(16384, 4096), (16384, 4096), True, False], {B1: 2190.26, B4: 3388.11}),
([(16384, 4096), (16384, 11008), True, False], {B1: 6419.19, B4: 9311.79}),
([(16384, 11008), (4096, 11008), False, True], {B1: 5332.41, B4: 7676.42}),
([(16384, 11008), (11008, 4096), False, False], {B1: 5347.52, B4: 7645.59}),
([(16384, 11008), (16384, 4096), True, False], {B1: 6623.32, B4: 9861.94}),
([(8192, 13696), (13696, 5120), False, False], {}), #baichuan
([(8192, 13696), (8192, 5120), True, False], {}), #baichuan
([(8192, 32), (32, 1152), False, False], {}), # opensora
([(8192, 1152), (32, 1152), False, True], {}), # opensora
([(8192, 32), (8192, 1152), True, False], {}), # opensora
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if not sys.argv[1].isdigit():
            print("Usage: python3 {} [case_index]".format(sys.argv[0]))
        else:
            idx = int(sys.argv[1])
            cs = all_cases[idx]
            perf = run_perf(cs[0])
            print("{}: {} : {} -> {}".format(idx, cs[0], cs[1].get(soc_name, "None"), perf))
        exit(0)
    bads, goods, news = [], [], []
    for i, cs in enumerate(all_cases):
        perf = run_perf(cs[0])
        print("{}: {} : {}".format(i, cs[0], perf))
        if soc_name in cs[1]:
            expect = cs[1][soc_name]
            if perf > expect and (perf - expect) / expect > 0.02:
                bads.append((i, perf))
            elif perf < expect and (expect - perf) / expect > 0.02:
                goods.append((i, perf))
        else:
            news.append((i, perf))
    print("******** WORSE CASES({}) ********".format(len(bads)))
    for i, perf in bads:
        print("{}: {} : {} -> {}".format(i, all_cases[i][0], all_cases[i][1][soc_name], perf))
    print("******** BETTER CASES({}) ********".format(len(goods)))
    for i, perf in goods:
        print("{}: {} : {} -> {}".format(i, all_cases[i][0], all_cases[i][1][soc_name], perf))
    print("******** NEW CASES({}) ********".format(len(news)))
    for i, perf in news:
        print("{}: {} : {}".format(i, all_cases[i][0], perf))
