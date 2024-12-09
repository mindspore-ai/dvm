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
import sys
from dvm.tester import Tester


all_cases = [[4,16384*40]]


def run_perf(shape):
    np.random.seed(1)

    comm = MPI.COMM_WORLD
    t = Tester(comm=comm)
    a = np.random.normal(0, 1, shape).astype(np.float16)

    x1 = t.load(a)
    x2 = t.reducescatter(x1)
    t.store(x2)

    comm.Barrier()
    res = t.run_perf()
    print(t.das())
    comm.Barrier()
    return round(res.min, 2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
        perf_res = run_perf(num)
        print(perf_res, " us")
        exit(0)
    for i, cs in enumerate(all_cases):
        perf_res = run_perf(cs)
        print("======= Case {}, shape: {}".format(i, cs))
        print(perf_res, "us")
    MPI.Finalize()
