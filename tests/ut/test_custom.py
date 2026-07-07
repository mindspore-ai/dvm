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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(dvm.Device.arch() == 'AscendC310', reason="C310 temporarily does not support ViewStoreX")
def test_custom_op_multi_inout():
    dev_code = r'''
extern "C" [aicore] void fuse_min(__gm__ uint64_t *__restrict__ pc, uint64_t head, uint64_t tile) {
  uint64_t data = pc[0];
  __ubuf__ float *a = (__ubuf__ float *)(data >> 40);
  __ubuf__ float *b = (__ubuf__ float *)((data >> 20) & 0xfffffu);
  __ubuf__ float *c = (__ubuf__ float *)(data & 0xfffffu);
  uint64_t data1 = pc[1];
  __ubuf__ float *d = (__ubuf__ float *)(data1 >> 40);
  __ubuf__ float *out = (__ubuf__ float *)((data1 >> 20) & 0xfffffu);
  __ubuf__ float *ws = (__ubuf__ float *)(data1 & 0xfffffu);
  uint64_t count = pc[2];
  set_vector_mask(0, count);
  // (a + b) * (c + d)
  vadd(ws, a, b, 1, 1, 1, 1, 8, 8, 8);
  vadd(out, c, d, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmul(out, ws, out, 1, 1, 1, 1, 8, 8, 8);
}

extern "C" [aicore] void fuse_mout(__gm__ uint64_t *__restrict__ pc, uint64_t head, uint64_t tile) {
  uint64_t data = pc[0];
  __ubuf__ float *input = (__ubuf__ float *)(data >> 40);
  __ubuf__ float *out1 = (__ubuf__ float *)((data >> 20) & 0xfffffu);
  __ubuf__ float *out2 = (__ubuf__ float *)(data & 0xfffffu);
  uint64_t data1 = pc[1];
  __ubuf__ float *out3 = (__ubuf__ float *)(data1 >> 40);
  uint64_t count = data1 & 0xffffu;
  // out1 = input + 0.1
  // out2 = out1 + input
  // out3 = out2 * 0.7
  set_vector_mask(0, count);
  vadds(out1, input, 0.1f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vadd(out2, out1, input, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmuls(out3, out2, 0.7f, 1, 1, 1, 8, 8);
}
'''
    host_code = r'''
#include "ops.h"
using namespace dvm;
class FusedMin : public CustomOp {
public:
  FusedMin(NDObject *a, NDObject *b, NDObject *c, NDObject *d) : CustomOp(a, b, a->type_id_) {
    func_id_ = GetFunction("mio/fuse_min");
    xhs_data_.data[0] = c;
    xhs_data_.data[1] = d;
    SetXhs(&xhs_data_);
    ws_num_ = 1;
  }
  void Normalize(std::vector<NDObject *> &) override {
    ndd_.dims = lhs_->nd_.dims();
    shape_ = *lhs_->shape_ref_;
  }
  uint64_t EmitEx(uint64_t *payload) override {
    payload[0] = lhs_->xbuf_ << 40 |  rhs_->xbuf_ << 20 | xhs_->data[0]->xbuf_;
    payload[1] = xhs_->data[1]->xbuf_ << 40 | xbuf_ << 20 | wss_[0];
    payload[2] = nd_.stride_back();
    return 3;
  }
  static NDObject *MakeOp(const std::vector<NDObject *> &inputs, const std::vector<ScalarRef> &attrs) {
    return new FusedMin(inputs[0], inputs[1], inputs[2], inputs[3]);
  }
  XhsN<2> xhs_data_;
};

class FusedMout : public CustomOp {
public:
  FusedMout(NDObject *input) : CustomOp(input, nullptr, input->type_id_), xout_data_(this) {
    func_id_ = GetFunction("mio/fuse_mout");
    xout_data_.data[0]->shape_ref_ = shape_ref_;
    xout_data_.data[1]->shape_ref_ = shape_ref_;
    SetXOut(&xout_data_);
  }
  void Normalize(std::vector<NDObject *> &) override {
    ndd_.dims = lhs_->nd_.dims();
    shape_ = *lhs_->shape_ref_;
    xout_data_.data[0]->nd_ = nd_;
    xout_data_.data[1]->nd_ = nd_;
  }
  uint64_t EmitEx(uint64_t *payload) override {
    payload[0] = lhs_->xbuf_ << 40 |  xbuf_ << 20 | xout_data_.data[0]->xbuf_;
    payload[1] = xout_data_.data[1]->xbuf_ << 40 | nd_.stride_back();
    return 2;
  }
  static NDObject *MakeOp(const std::vector<NDObject *> &inputs, const std::vector<ScalarRef> &attrs) {
    return new FusedMout(inputs[0]);
  }
  XOutN<2> xout_data_;
};
CustomDef __ALL_OPS__[] = {{"FusedMin", &FusedMin::MakeOp}, {"FusedMout", &FusedMout::MakeOp}, {nullptr, nullptr}};
'''
    t = Tester()
    Tester.reg_custom("mio", host_code, dev_code)
    a = np.random.normal(0, 0.2, [2, 1000]).astype(np.float32)
    b = np.random.normal(0, 0.2, [2, 1000]).astype(np.float32)
    c = np.random.normal(0, 0.2, [2, 1000]).astype(np.float32)
    d = np.random.normal(0, 0.2, [2, 1000]).astype(np.float32)
    x0 = t.add(t.load(a), 0.1)
    x1 = t.load(b)
    x2 = t.mul(t.load(c), 0.6)
    x3 = t.load(d)
    x4 = t.custom("mio/FusedMin", [x0, x1, x2, x3])
    x5 = t.custom("mio/FusedMout", [x4])
    x6 = t.ext_out(x5, 1)
    e4 = (a + 0.1 + b) * (c * 0.6 + d)
    t.store_expect(x4, e4)
    t.store_expect(x5, e4 + 0.1)
    t.store_expect(x6, ((e4 + 0.1) + e4) * 0.7)
    assert (t.run_check())
