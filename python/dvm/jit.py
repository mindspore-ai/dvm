# Copyright 2026 Huawei Technologies Co., Ltd
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
from . import Kernel, NDObject, ShapeRef

class JitKernel(Kernel):
    def __init__(self, ker_type, dynamic):
        if dynamic:
            ker_type += ",dyn" if ":" in ker_type else ":dyn"
        dev_conf = os.getenv("DEVICE_ID")
        dev_id = int(dev_conf) if dev_conf else 0
        Kernel.__init__(self, ker_type, "dev", dev_id)
        self.inputs = None
        self.outputs = None
        self.dynamic = dynamic
    def build(self, func):
        arg_cnt = func.__code__.co_argcount - 1
        args = [i for i in range(arg_cnt)]
        self.inputs = [None] * arg_cnt
        self.outputs = func(self, *args)
        if not self.dynamic:
            self.codegen(None)
    def load(self, index, dtype, shape=[]):
        op = Kernel.load(self, shape, dtype)
        self.inputs[index] = op
        return op
    def scalar(self, index, dtype):
        if dtype == "float32":
            s = self.make_float()
        elif dtype == "int32":
            s = self.make_int()
        else:
            raise ValueError("invalid dtype")
        self.inputs[index] = s
        return s
    def dims(self, index):
        ref = ShapeRef()
        self.inputs[index] = ref
        return ref
    def __call__(self, *args):
        for inp, arg in zip(self.inputs, args):
            if isinstance(inp, NDObject):
                Kernel.input(self, inp, arg)
            else:
                inp.update(arg)
        if self.dynamic:
            self.codegen(None)
        self.run()
        if isinstance(self.outputs, (list, tuple)):
            return [self.output(x) for x in self.outputs]
        else:
            return self.output(self.outputs)

def kernel(ktype="split", dynamic=True):
    if callable(ktype):
        func = ktype
        kobj = JitKernel("split", True)
        kobj.build(func)
        return kobj
    def decorate(func):
        kobj = JitKernel(ktype, dynamic)
        kobj.build(func)
        return kobj
    return decorate
