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


def reducescatter(arrs, rank_id=None):
    rank_size = len(arrs)
    split_arrs = [np.array_split(arrs[i], rank_size) for i in range(rank_size)]
    if rank_id is not None:
        return np.sum([split_arrs[j][rank_id] for j in range(rank_size)], axis=0)
    return [
        np.sum([split_arrs[j][i] for j in range(rank_size)], axis=0)
        for i in range(rank_size)
    ]


@pytest.mark.parametrize(
    "shape", [[4, 256], [4, 16384], [4, 32, 256], [4, 1278487], [32, 256]]
)
def test_reducescatter(comm, rank, size, shape):
    np.random.seed(1)
    if shape[0] < size:
        shape[0] = size
    t = Tester(use_pass_opt=True, comm=comm)
    inputs = []
    for i in range(size):
        inputs.append(np.random.normal(0.1, 1, shape).astype(np.float32))
    expect = reducescatter(inputs, rank)

    x1 = t.load(inputs[rank])
    x2 = t.reducescatter(x1)
    x3 = t.unary("Abs", x2)
    t.store_expect(x3, np.abs(expect), 0.01)
    res = t.run_check()
    assert res
