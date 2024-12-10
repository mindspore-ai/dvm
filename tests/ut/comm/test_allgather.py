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
import pytest

@pytest.mark.parametrize(
    "shape", [[4, 256], [4, 16384], [4, 32, 256], [4, 1278487], [32, 256]]
)
def test_allgather(comm, shape):
    np.random.seed(1)
    rank = comm.Get_rank()
    size = comm.Get_size()
    t = Tester(comm=comm)
    inputs = []
    for i in range(size):
        inputs.append((np.random.random(shape)).astype(np.float16))
    expect = np.concatenate(inputs, axis=0)

    x1 = t.load(inputs[rank])
    x2 = t.allgather(x1)
    t.store_expect(x2, expect, 0.01)
    assert(t.run_check())
