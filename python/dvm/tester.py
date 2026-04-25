# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
import subprocess
import csv
import inspect
import numpy as np
from . import DataType
from . import PyKernel as Kernel

_DTYPE_NAME_MAP = {
    "bool": DataType.bool,
    "float16": DataType.float16,
    "bfloat16": DataType.bfloat16,
    "float32": DataType.float32,
    "int32": DataType.int32,
    "int64": DataType.int64,
}


def _normalize_dtype(dtype):
    if isinstance(dtype, DataType):
        return dtype
    if isinstance(dtype, str):
        mapped = _DTYPE_NAME_MAP.get(dtype)
        if mapped is None:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return mapped
    return dtype


class PerformanceResult:
    def __init__(self, results):
        self.min = results[0]
        self.max = results[1]
        self.mean = results[2]

    def __repr__(self):
        return "fun_min_max_avg(us): {}  {}  {}  {}".format(
            sys._getframe(1).f_code.co_name,
            round(self.min, 2),
            round(self.max, 2),
            round(self.mean, 2),
        )

    def __str__(self):
        return (
            f"Kernel Time Summary: {sys._getframe(1).f_code.co_name}\n"
            f"{'min (us)':<15} {'max (us)':<15} {'mean (us)':<15}\n"
            f"{self.min:<15.4f} {self.max:<15.4f} {self.mean:<15.4f}\n"
        )


class Tester(Kernel):
    __test__ = False

    def __init__(self, ker_type="", use_pass_opt=False, run_mode="dev", comm=None):
        if comm:
            os.environ["DEVICE_ID"] = str(comm.Get_rank())
            os.environ["RANK_SIZE"] = str(comm.Get_size())
        self.comm = comm
        dev_conf = os.getenv("DEVICE_ID")
        dev_id = int(dev_conf) if dev_conf else 0
        Kernel.__init__(self, ker_type, run_mode, dev_id)
        self.is_dyn = "dyn" in ker_type
        self.is_codegen = False
        self.expects = []  # [(op, expect, eps)]
        self.passes = None if use_pass_opt else []
        if comm:
            self.init_comm(comm.Get_rank(), comm.Get_size(), "memory")

    @staticmethod
    def fast_random_normal(loc, scale, shape):
        row_random = np.random.normal(loc, scale, (shape[-1],)).astype(np.float32)
        return np.broadcast_to(row_random, shape).copy()

    @staticmethod
    def bf16_random_normal(loc, scale, shape):
        x = np.random.normal(loc, scale, shape).astype(np.float32)
        x_int = x.view(np.uint32)
        x_bf16_int = x_int & 0xFFFF0000
        x_bf32 = x_bf16_int.view(np.float32)
        return x_bf32

    def _prepare_array_input(self, array, dtype):
        if dtype is None:
            dtype_id = _normalize_dtype(str(array.dtype))
        else:
            dtype_id = _normalize_dtype(dtype)
        if dtype_id == DataType.bfloat16:
            array = Kernel.convert_to_bf16(self, array)
        shape = list(array.shape)
        return array, dtype_id, shape

    def load(self, shape_arr, dtype=None):
        if not isinstance(shape_arr, np.ndarray):
            # dynamic shape scenario
            return Kernel.load(self, shape_arr, _normalize_dtype(dtype))
        shape_arr, dtype_id, shape = self._prepare_array_input(shape_arr, dtype)
        op = Kernel.load(self, shape, dtype_id)
        self.input(op, shape_arr)
        return op

    def view_load(self, shape, stride, arr_dtype, offset = 0, real_dtype=None):
        if not isinstance(arr_dtype, np.ndarray):
            # dynamic shape scenario
            return Kernel.view_load(self, shape, stride, _normalize_dtype(arr_dtype))
        dtype_id = _normalize_dtype(str(arr_dtype.dtype))
        real_dtype_id = _normalize_dtype(real_dtype) if real_dtype is not None else None
        if real_dtype_id is not None:
            assert real_dtype_id == DataType.bfloat16
            arr_dtype = Kernel.convert_to_bf16(self, arr_dtype)
            dtype_id = real_dtype_id
        op = Kernel.view_load(self, shape, stride, dtype_id)
        self.input(op, arr_dtype, offset)
        return op

    def slice_load(self, shape_arr, start, size, dtype=None):
        if not isinstance(shape_arr, np.ndarray):
            # dynamic shape scenario
            return Kernel.slice_load(
                self, shape_arr, start, size, _normalize_dtype(dtype)
            )
        shape_arr, dtype_id, shape = self._prepare_array_input(shape_arr, dtype)
        op = Kernel.slice_load(self, shape, start, size, dtype_id)
        self.input(op, shape_arr)
        return op

    def stridedslice_load(self, shape_arr, start, end, step, dtype=None):
        if dtype is not None:
            dtype_id = _normalize_dtype(dtype)
            if isinstance(shape_arr, np.ndarray) and dtype_id == DataType.bfloat16:
                shape_arr = Kernel.convert_to_bf16(self, shape_arr)
            return Kernel.stridedslice_load(self, shape_arr, start, end, step, dtype_id)
        shape_arr, dtype_id, shape = self._prepare_array_input(shape_arr, None)
        op = Kernel.stridedslice_load(self, shape, start, end, step, dtype_id)
        self.input(op, shape_arr)
        return op

    def multi_load(self, shape_arr, dtype=None):
        if not isinstance(shape_arr, np.ndarray):
            # dynamic shape scenario
            return Kernel.multi_load(self, shape_arr, _normalize_dtype(dtype))
        shape_arr, dtype_id, shape = self._prepare_array_input(shape_arr, dtype)
        op = Kernel.multi_load(self, shape, dtype_id)
        self.input(op, shape_arr)
        return op

    def cast(self, x, dtype):
        return Kernel.cast(self, x, _normalize_dtype(dtype))

    def full(self, scalar, shape, dtype=None):
        dtype_id = _normalize_dtype(dtype)
        return Kernel.full(self, scalar, shape, dtype_id)

    def one_hot(self, indices, depth, axis, on_value, off_value, dtype):
        return Kernel.one_hot(
            self,
            indices,
            depth,
            axis,
            on_value,
            off_value,
            _normalize_dtype(dtype),
        )

    def store_expect(self, x, e, eps=None):
        op = Kernel.store(self, x)
        self.expects.append([op, e, eps])
        return op

    def codegen(self, verbose=False):
        if self.is_codegen:
            return
        if verbose:
            print("******* before tiling *******")
            print(self.dump())
        Kernel.codegen(self, self.passes)
        if verbose:
            print("******* after tiling *******")
            print(self.dump())
            print("********* bytecode *********")
            print(self.das())
        if not self.is_dyn:
            self.is_codegen = True

    def run(self, verbose=False):
        if not self.is_dyn and verbose:
            print("******* before tiling *******")
            print(self.dump())
        self.codegen()
        Kernel.run(self)
        self.barrier()
        if verbose:
            print("******* after tiling *******")
            print(self.dump())
            print("********* bytecode *********")
            print(self.das())

    def bare_run(self):
        Kernel.run(self)

    def check(self, store, expect, eps=None, verbose=False):
        def _print_result_diff(out, expect, eps):
            error_cnt = 0
            max_error = 32
            out = out.flatten()
            if isinstance(expect, np.ndarray):
                expect = expect.flatten()
            error_ranges = []  # [(start, end),]
            print("******* first {} error data *******".format(max_error))
            print("idx: expect output")
            start, end = -1, -1
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
                error_ranges.append([start, out.shape[0] - 1])
            print("********* error data ranges **********")
            for i in error_ranges:
                print("[{}, {}]: {}".format(i[0], i[1], i[1] - i[0] + 1))

        out = self.output(store)
        if store.dtype() == DataType.bfloat16:
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
        return PerformanceResult(perf)

    def dry_run(self, core_id=0, is_cube=False):
        self.codegen()
        Kernel.dry_run(self, core_id, is_cube)

    def run_msprof(self, path, test_num=10):
        self.codegen()
        self.msprof(path, test_num)
        if not os.path.isdir(path):
            print(f"Invalid directory: {path}")
            return
        subprocess.run(
            ["msprof", f"--export=on", f"--output={path}"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        output_lines = []
        for root, _, files in os.walk(path):
            for file_name in files:
                if file_name.startswith("op_statistic") and file_name.lower().endswith(
                    ".csv"
                ):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, "r", encoding="utf-8", newline="") as f:
                        reader = list(csv.reader(f))
                    col_widths = [
                        max(len(cell) for cell in col) for col in zip(*reader)
                    ]
                    table = "\n".join(
                        " | ".join(
                            cell.ljust(width) for cell, width in zip(row, col_widths)
                        )
                        for row in reader
                    )
                    output_lines.append(f"\n===== {file_path} =====\n{table}")
        return "\n\n".join(output_lines)

    def set_passes(self, *pass_names):
        self.passes = []
        for pass_name in pass_names:
            self.passes.append(pass_name)

    def reset(self):
        Kernel.reset(self)
        self.is_codegen = False
        self.expects = []

    def barrier(self):
        if self.comm:
            self.comm.Barrier()
        else:
            Kernel.barrier()


class CommScope:
    """
    Create an comm domain scope.

    Examples:
        >>> from dvm.tester import CommScope, Tester
        >>> with CommScope(0, 1, 2, 3):
        >>>     t = Tester()
        >>>     ...
    """

    def __init__(self, *ids):
        self.ids = ids
        self.comm_type = ""

    def __enter__(self):
        if self.ids:
            rank_size = len(self.ids)
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in self.ids]
            )
        else:
            ids_str = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
            rank_size = len(ids_str.split(","))
        Kernel.fork(rank_size, self.comm_type)
        os.environ["DEVICE_ID"] = str(Kernel.rank_id())
        return Kernel

    def __exit__(self, type, value, trace):
        Kernel.join()


class HcclScope(CommScope):
    """
    Create an hccl comm domain scope.

    Examples:
        >>> from dvm.tester import HcclScope, Tester
        >>> with HcclScope(0, 1, 2, 3):
        >>>     t = Tester()
        >>>     ...
    """

    def __init__(self, *ids):
        self.ids = ids
        self.comm_type = "hccl"
