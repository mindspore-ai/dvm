# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""Basic FP32 Axpy functionality on A2."""
import numpy as np
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_axpy():
    t = Tester(use_pass_opt=True)
    a = np.arange(131, dtype=np.float32) / 16
    b = np.arange(131, dtype=np.float32)[::-1].copy() / 8
    alpha = -0.25
    x, y = t.load(a), t.load(b)
    t.store_expect(t.axpy(x, y, alpha), a + alpha * b)
    t.store_expect(x, a)
    t.store_expect(y, b)
    assert t.run_check()
