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
from ._dvm_py import Kernel, ShapeRef

class Tester(Kernel):
    __test__ = False
    def __init__(self, ker_type=""):
        dev_id = int(os.getenv("DEVICE_ID"))
        Kernel.__init__(self, dev_id, ker_type)
        self.is_dyn = ker_type == "dyn"
        self.is_codegen = False
        self.expects = [] # [(op, expect, eps)]
        self.passes = None

    def load(self, shape_arr, dtype=None):
        if not isinstance(shape_arr, np.ndarray):
            # dynamic shape scenario
            return Kernel.load(self, shape_arr, dtype)
        if dtype == "bfloat16":
            shape_arr = Kernel.convert_to_bf16(self, shape_arr)
        elif dtype == None:
            dtype = str(shape_arr.dtype)
        shape = list(shape_arr.shape)
        op = Kernel.load(self, shape, dtype)
        self.input(op, shape_arr)
        return op

    def slice_load(self, shape_arr, start, size, dtype=None):
        if not isinstance(shape_arr, np.ndarray):
            # dynamic shape scenario
            return Kernel.slice_load(self, shape_arr, start, size, dtype)
        if dtype == "bfloat16":
            shape_arr = Kernel.convert_to_bf16(self, shape_arr)
        elif dtype == None:
            dtype = str(shape_arr.dtype)
        shape = list(shape_arr.shape)
        op = Kernel.slice_load(self, shape, start, size, dtype)
        self.input(op, shape_arr)
        return op

    def stridedslice_load(self, shape_arr, start, end, step, dtype=None):
        if dtype is not None:
            return Kernel.stridedslice_load(self, shape_arr, start, end, step, dtype)
        dtype = str(shape_arr.dtype)
        shape = list(shape_arr.shape)
        op = Kernel.stridedslice_load(self, shape, start, end, step, dtype)
        self.input(op, shape_arr)
        return op

    def store_expect(self, x, e, eps=None):
        op = Kernel.store(self, x)
        self.expects.append([op, e, eps])
        return op

    def codegen(self):
        if self.is_codegen:
            return
        Kernel.codegen(self, self.passes)
        if not self.is_dyn:
            self.is_codegen = True

    def run(self, verbose=False):
        if not self.is_dyn and verbose:
            print("******* before tiling *******")
            print(self.dump())
        self.codegen()
        Kernel.run(self)
        if verbose:
            print("******* after tiling *******")
            print(self.dump())
            print("********* bytecode *********")
            print(self.das())

    def check(self, store, expect, eps=None, verbose=False):
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
        out = self.output(store)
        if store.dtype() == "bfloat16":
            out = Kernel.convert_from_bf16(self, out)
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

    def run_check(self, verbose=False):
        self.run(verbose)
        for op, expect, eps in self.expects:
            if not self.check(op, expect, eps, verbose):
                return False
        return True

    def run_perf(self):
        self.codegen()
        perf = self.perf()
        print("kernel time(fun_min_max_avg, us): {}  {}  {}  {}".format(sys._getframe(1).f_code.co_name, perf[0], perf[1], perf[2]))

    def set_passes(self, *pass_names):
        self.passes = []
        for pass_name in pass_names:
            self.passes.append(pass_name)
