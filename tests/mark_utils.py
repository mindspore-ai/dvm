# Copyright 2025 Huawei Technologies Co., Ltd
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

""" define marks """
import pytest


def arg_mark(plat_marks, level_mark, card_mark, essential_mark):
    def decorator(func):
        for plat_mark in plat_marks:
            func = getattr(pytest.mark, plat_mark)(func)
        func = getattr(pytest.mark, level_mark)(func)
        func = getattr(pytest.mark, card_mark)(func)
        func = getattr(pytest.mark, essential_mark)(func)
        return func

    return decorator
