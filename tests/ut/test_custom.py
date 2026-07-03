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

import pytest
import numpy as np
import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() == 'AscendC310', reason="C310 temporarily does not support ViewStoreX")
def test_custom_op_basic():
    dev_code = r'''
extern "C" [aicore] void fuse_addmul(__gm__ uint64_t *__restrict__ pc, uint64_t head, uint64_t tile) {
  uint64_t data = pc[0];
  __ubuf__ float *xd = (__ubuf__ float *)(data >> 48);
  __ubuf__ float *xn = (__ubuf__ float *)((data >> 32) & 0xffffu);
  uint64_t count = data & 0xffffu;
  set_vector_mask(0, count);
  vadds(xd, xn, 0.1f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vmuls(xd, xd, 0.6f, 1, 1, 1, 8, 8);
}
'''
    host_code = r'''
#include "ops.h"
using namespace dvm;
class FusedAddMul: public CustomOp {
public:
  FusedAddMul(NDObject *input) : CustomOp(input, nullptr, input->type_id_) {
    func_id_ = GetFunction("foo/fuse_addmul");
  }
  void Normalize(std::vector<NDObject *> &) override {
    ndd_.dims = lhs_->nd_.dims();
    shape_ = *lhs_->shape_ref_;
  }
  uint64_t EmitEx(uint64_t *payload) override {
    *payload = xbuf_ << 48 | lhs_->xbuf_ << 32 | nd_.stride_back();
    return 1;
  }
  static NDObject *MakeOp(const std::vector<NDObject *> &inputs, const std::vector<ScalarRef> &attrs) {
    return new FusedAddMul(inputs[0]);
  }
};
CustomDef __ALL_OPS__[] = {{"FusedAddMul", &FusedAddMul::MakeOp}, {nullptr, nullptr}};
'''
    t = Tester()
    Tester.reg_custom("foo", host_code, dev_code)
    a = np.random.normal(-1, 1, [3, 1000]).astype(np.float32)
    x = t.load(a)
    y = t.custom("foo/FusedAddMul", [x])
    t.store_expect(y, (a + 0.1) * 0.6)
    assert (t.run_check())
