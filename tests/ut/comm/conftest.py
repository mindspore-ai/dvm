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
import os
from dvm.tester import Tester


def is_in_mpi_environment():
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "PMI_SIZE",
        "PMI_RANK",
    ]
    for var in mpi_env_vars:
        if var in os.environ:
            return True
    return False


@pytest.fixture(scope="session", autouse=True)
def comm():
    in_mpi = is_in_mpi_environment()
    if in_mpi:
        from mpi4py import MPI

        yield MPI.COMM_WORLD
    else:
        from dvm.tester import CommScope

        with CommScope():
            yield None
    if in_mpi:
        MPI.Finalize()


@pytest.fixture(scope="session", autouse=True)
def rank(comm):
    in_mpi = is_in_mpi_environment()
    if in_mpi:
        return comm.Get_rank()
    else:
        return Tester.rank_id()


@pytest.fixture(scope="session", autouse=True)
def size(comm):
    in_mpi = is_in_mpi_environment()
    if in_mpi:
        return comm.Get_size()
    else:
        return Tester.rank_size()
