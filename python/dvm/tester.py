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

import os
import sys
import inspect
import numpy as np
from .builder import DvmKernelBuilder as DvmKernelMod

class Tester(DvmKernelMod):
    __test__ = False
    def __init__(self, ker_type=""):
        dev_id = int(os.getenv("DEVICE_ID"))
        DvmKernelMod.__init__(self, dev_id, ker_type)
        self.results = []

    def store_expect(self, x, e, eps=None):
        out = DvmKernelMod.store(self, x)
        self.results.append([out, e, eps])
        return out

    def run_check(self, verbose=False):
        def _print_result_diff(out, expect, eps):
            error_cnt = 0
            max_error = 32
            out = out.flatten()
            if isinstance(expect, np.ndarray):
                expect = expect.flatten()
            error_ranges = [] # [(start, end),]
            print("******* first {} error data *******".format(max_error))
            print("idx: expect output")
            start, end = -1,-1
            for i in range(out.shape[0]):
                exp = expect[i] if isinstance(expect, np.ndarray) else expect
                if not np.isclose(out[i], exp, rtol=eps, atol=eps):
                    if error_cnt < max_error:
                        print("{}: {}  {}".format(i, exp, out[i]))
                    error_cnt += 1
                    if start == -1:
                        start = i
                    end = i
                elif start >= 0:
                    error_ranges.append([start, end])
                    start, end = -1, -1
            if start >= 0:
                error_ranges.append([start, out.shape[0]-1])
            print("********* error data ranges **********")
            for i in error_ranges:
                print("[{}, {}]: {}".format(i[0], i[1], i[1] - i[0] + 1))

        kernel = DvmKernelMod.get(self)
        if verbose:
            print("******* before tiling *******")
            print(kernel.dump())
            das = kernel.das()
            print("******* after tiling *******")
            print(kernel.dump())
            print("********* bytecode *********")
            print(das)
        kernel.run()
        for out, expect, eps in self.results:
            if inspect.isfunction(expect):
                if not expect(out):
                    print("********** OUTPUT **********")
                    print(out)
                    return False
            else:
                if eps == None:
                    if out.dtype == np.float32:
                        eps = 1e-5
                    elif out.dtype == np.float16:
                        eps = 1e-3
                    else:
                        eps = 0
                if not np.allclose(out, expect, rtol=eps, atol=eps, equal_nan=True):
                    if verbose:
                        _print_result_diff(out, expect, eps)
                    return False
        return True

    def run_perf(self):
        kernel = DvmKernelMod.get(self)
        perf = kernel.perf()
        print("kernel time(fun_min_max_avg, us): {}  {}  {}  {}".format(sys._getframe(1).f_code.co_name, perf[0], perf[1], perf[2]))
