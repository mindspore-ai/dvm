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

from mpi4py import MPI
import numpy as np
import os
import sys
from dvm.tester import Tester


def run_perf(m, n, k):
    np.random.seed(1)

    comm = MPI.COMM_WORLD
    t = Tester("mix", comm=comm)
    shape_a = [m, k]
    shape_b = [k, n]
    a = np.random.normal(0, 1, shape_a).astype(np.float16)
    b = np.random.normal(0, 1, shape_b).astype(np.float16)

    x1 = t.load(a)
    x2 = t.load(b)
    c = t.matmul(x1, x2, False, False)
    x2 = t.allreduce("sum", c)
    t.store(x2)

    comm.Barrier()
    res = t.run_perf()
    comm.Barrier()
    return round(res.min, 2)


# m n k
all_cases = [
    (25600, 2046, 1280),
    (4096, 2560, 8192),
    (8192, 5120, 1280),
    (8192, 5120, 3456),
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        m = int(sys.argv[1])
        n = int(sys.argv[2])
        k = int(sys.argv[3])
        perf_res = run_perf(m, n, k)
        print(perf_res)
        exit(0)
    for i, cs in enumerate(all_cases):
        perf_res = run_perf(*cs)
        print("Case {}: {}".format(i, cs))
        print(perf_res)
    MPI.Finalize()
