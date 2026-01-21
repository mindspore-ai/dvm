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

""" dvm python """
from . import _dvm_py as _core
from ._dvm_py import DataType, Device, Kernel, KernelBase, NDObject, IntArrayRef, ScalarRef
from .jit import kernel

bool_ = DataType.bool
float16 = DataType.float16
bfloat16 = DataType.bfloat16
float32 = DataType.float32
int32 = DataType.int32
int64 = DataType.int64

__all__ = [
    "DataType",
    "Device",
    "Kernel",
    "KernelBase",
    "NDObject",
    "IntArrayRef",
    "ScalarRef",
    "bool_",
    "float16",
    "bfloat16",
    "float32",
    "int32",
    "int64",
]
