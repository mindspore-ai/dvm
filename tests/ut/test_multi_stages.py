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

import pytest
import numpy as np
import dvm
from dvm.tester import Tester

def test_stage_vec_vec():
    t = Tester("stages")
    t.stage_switch("static");
    ax = np.full([1024, 128], 0.05, np.float32)
    a = t.load(ax)
    b = t.binary("Add", a, 0.2)
    c = t.binary("Mul", a, b)
    d = t.stage_store(c)
    t.stage_switch("static");
    e = t.stage_load(d)
    f = t.binary("Mul", e, 0.7)
    t.store_expect(f, (ax + 0.2)*ax*0.7)
    assert(t.run_check())

@pytest.mark.mix
@pytest.mark.skipif(dvm.device.arch() == "AscendC100", reason="matmul not support 910a")
def test_stage_vec_cube():
    t = Tester("stages")
    t.stage_switch("static");
    ax = np.random.normal(0, 1, [1024,512]).astype(np.float16)
    a = t.load(ax)
    b = t.binary("Add", a, 0.02)
    c = t.binary("Mul", a, b)
    d = t.stage_store(c)
    t.stage_switch("mix");
    e = t.stage_load(d)
    fx = np.random.normal(0, 1, [512,1024]).astype(np.float16)
    f = t.load(fx)
    g = t.matmul(e, f, False, False)
    expect = np.matmul(((ax + 0.02) * ax).astype(np.float32), fx.astype(np.float32))
    t.store_expect(g, expect.astype(np.float16))
    assert(t.run_check())
