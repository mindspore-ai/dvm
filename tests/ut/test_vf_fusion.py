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
import subprocess
import sys

import numpy as np
import pytest

import dvm
from dvm.tester import Tester
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def vf_fusion_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("DVM_VF_JIT_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "vf_fusion"


def run_pytest_child(request, child_flag, env_defaults=None):
    if os.environ.get(child_flag) == "1":
        return False
    env = os.environ.copy()
    for name, value in (env_defaults or {}).items():
        env.setdefault(name, value)
    env[child_flag] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", request.node.nodeid],
        cwd=str(request.config.rootdir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def run_codegen(request):
    return run_pytest_child(
        request,
        "DVM_VF_CODEGEN_CHILD",
        {"DVM_SOC_NAME": "Ascend950PR_9599"},
    )


def make_f32_multi_output_graph(t, shape=(1024, 1024)):
    x_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    y_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    z_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    x = t.load(x_np)
    y = t.load(y_np)
    z = t.load(z_np)
    abs_x = t.abs(x)
    scaled = t.add(abs_x, 0.25)
    add = t.add(scaled, y)
    cond = t.greater(add, 0.0)
    selected = t.select(cond, add, z)
    out = t.cast(selected, "float16")

    add_expect = np.abs(x_np) + 0.25 + y_np
    out_expect = np.where(add_expect > 0.0, add_expect, z_np).astype(np.float16)
    # `add` is deliberately both an internal value and an externally stored
    # value, exercising Custom primary output plus ExtOut metadata.
    t.store_expect(add, add_expect)
    t.store_expect(out, out_expect)


def make_gelu_graph(t, shape=(1024, 1024)):
    x_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    x = t.load(x_np)
    value = t.mul(x, x)
    value = t.mul(value, x)
    value = t.mul(value, 0.044715)
    value = t.add(x, value)
    value = t.mul(value, -1.5957691)
    value = t.exp(value)
    value = t.add(value, 1.0)
    out = t.div(x, value)

    x_cubed = x_np * x_np * x_np
    denominator = np.exp(
        np.float32(-1.5957691) * (x_np + np.float32(0.044715) * x_cubed)
    ) + 1.0
    t.store_expect(out, x_np / denominator)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gelu_codegen(request, vf_fusion_dir):
    if run_codegen(request):
        return
    dvm.Kernel.set_vf_fusion(True)
    t = Tester()
    t.set_passes("VfFusion")
    make_gelu_graph(t)

    t.codegen()

    assert list(vf_fusion_dir.glob("vf_*.o"))
    cce_source = next(vf_fusion_dir.glob("vf_*.cce")).read_text()
    assert all(
        intrinsic in cce_source
        for intrinsic in ("vmul(", "vmuls(", "vadd(", "vexp(", "vadds(", "vdiv(")
    )


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_f16_tail_codegen(request, vf_fusion_dir):
    if run_codegen(request):
        return
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1025)
    input_np = np.random.normal(-1.0, 1.0, shape).astype(np.float16)
    y_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    z_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(input_np, "float16")
    y = t.load(y_np)
    z = t.load(z_np)
    x_f32 = t.cast(x, "float32")
    abs_x = t.abs(x_f32)
    add = t.add(abs_x, y)
    cond = t.greater(add, z)
    selected = t.select(cond, add, y)
    out = t.cast(selected, "float16")

    add_expect = np.abs(input_np.astype(np.float32)) + y_np
    out_expect = np.where(add_expect > z_np, add_expect, y_np)
    t.store_expect(out, out_expect.astype(np.float16))

    t.codegen()
    cce_source = next(vf_fusion_dir.glob("vf_*.cce")).read_text()
    assert "UNPK_B16" in cce_source and "PK_B32" in cce_source


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_long_chain_codegen(request, vf_fusion_dir):
    if run_codegen(request):
        return
    dvm.Kernel.set_vf_fusion(True)
    input_np = np.random.normal(-1.0, 1.0, (1024, 1024)).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    value = t.load(input_np)
    expect = input_np
    for _ in range(24):
        value = t.abs(value)
        expect = np.abs(expect)
    t.store_expect(value, expect)

    t.codegen()

    assert list(vf_fusion_dir.glob("vf_*.o"))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_multi_output():
    dvm.Kernel.set_vf_fusion(True)
    t = Tester()
    t.set_passes("VfFusion")
    make_f32_multi_output_graph(t)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_gelu():
    dvm.Kernel.set_vf_fusion(True)
    t = Tester()
    t.set_passes("VfFusion")
    make_gelu_graph(t)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_f16_tail():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1025)
    input_np = np.random.normal(-1.0, 1.0, shape).astype(np.float16)
    y_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)
    z_np = np.random.normal(-1.0, 1.0, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(input_np, "float16")
    y = t.load(y_np)
    z = t.load(z_np)
    x_f32 = t.cast(x, "float32")
    add = t.add(t.abs(x_f32), y)
    selected = t.select(t.greater(add, z), add, y)
    out = t.cast(selected, "float16")
    add_expect = np.abs(input_np.astype(np.float32)) + y_np
    out_expect = np.where(add_expect > z_np, add_expect, y_np).astype(np.float16)
    t.store_expect(out, out_expect)

    t.codegen()

    assert "Custom" in t.dump()
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_reduce_boundary():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)
    z_np = np.random.normal(0.0, 0.2, shape).astype(np.float16)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np, "float16")
    y = t.load(y_np, "float16")
    z = t.load(z_np, "float16")
    add = t.add(x, y)
    sub = t.sub(add, z)
    cast = t.cast(sub, "float32")
    out = t.sum(cast, (1,), True)
    expect = np.sum(
        (x_np + y_np - z_np).astype(np.float32), axis=1, keepdims=True
    )
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    assert "Custom" in dump
    assert "Reduce" in dump
    assert all(op not in dump for op in ("Add", "Sub", "Cast"))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_cycle_guard():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    first = t.add(t.abs(x), 0.25)
    # Contracting both VF regions would create Custom -> Reshape -> Custom.
    bridge = t.reshape(t.reshape(first, (512, 2048)), shape)
    out = t.abs(t.add(first, bridge))
    t.store_expect(out, np.abs(2 * (np.abs(x_np) + 0.25)))

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 2
    assert dump.count("Reshape") == 2
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_multi_fusion():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    z_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    z = t.load(z_np)
    left = t.add(t.abs(x), y)
    right = t.sub(t.abs(z), y)
    t.store_expect(left, np.abs(x_np) + y_np)
    t.store_expect(right, np.abs(z_np) - y_np)

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 2
    assert all(op not in dump for op in ("Abs", "Add", "Sub"))
    assert t.run_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_extra_io():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1024)
    inputs_np = [
        np.random.normal(0.0, 0.2, shape).astype(np.float32) for _ in range(4)
    ]

    t = Tester()
    t.set_passes("VfFusion")
    a, b, c, d = [t.load(value) for value in inputs_np]
    add = t.add(a, b)
    sub = t.sub(add, c)
    out = t.mul(sub, d)
    add_expect = inputs_np[0] + inputs_np[1]
    sub_expect = add_expect - inputs_np[2]
    t.store_expect(add, add_expect)
    t.store_expect(sub, sub_expect)
    t.store_expect(out, sub_expect * inputs_np[3])

    t.codegen()
    dump = t.dump()

    assert dump.count("Custom") == 1
    assert dump.count("ExtOut") == 2
    assert all(op not in dump for op in ("Add", "Sub", "Mul"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("shape", [(1024, 1024), (1024, 1025)])
def test_broadcast_chain(shape):
    dvm.Kernel.set_vf_fusion(True)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.2, shape).astype(np.float32)
    bias_np = np.random.normal(0.5, 0.1, (1, shape[1])).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    bias = t.broadcast(t.load(bias_np), shape)
    value = t.abs(x)
    value = t.add(value, 0.25)
    value = t.mul(value, bias)
    value = t.add(value, y)
    value = t.maximum(value, -0.5)
    value = t.mul(value, 0.5)
    value = t.sub(value, bias)
    value = t.abs(value)
    value = t.add(value, 0.125)
    value = t.minimum(value, 2.0)
    value = t.mul(value, value)
    out = t.add(value, 0.1)

    bias_expect = np.broadcast_to(bias_np, shape)
    expect = np.abs(x_np)
    expect = expect + np.float32(0.25)
    expect = expect * bias_expect
    expect = expect + y_np
    expect = np.maximum(expect, np.float32(-0.5))
    expect = expect * np.float32(0.5)
    expect = expect - bias_expect
    expect = np.abs(expect)
    expect = expect + np.float32(0.125)
    expect = np.minimum(expect, np.float32(2.0))
    expect = expect * expect
    expect = expect + np.float32(0.1)
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    # One Broadcast plus twelve elementwise nodes.
    assert dump.count("Broadcast") == 1
    assert dump.count("Custom") == 1
    assert all(op not in dump for op in ("Abs", "Add", "Sub", "Mul", "Maximum", "Minimum"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
@pytest.mark.parametrize("shape", [(1024, 1024), (1024, 2048)])
def test_broadcast_regions(shape):
    dvm.Kernel.set_vf_fusion(True)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    row_np = np.random.normal(0.0, 0.2, (1, shape[1])).astype(np.float32)
    column_np = np.random.normal(0.0, 0.2, (shape[0], 1)).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    row = t.mul(t.add(t.abs(t.load(row_np)), 0.1), 0.5)
    column = t.mul(t.add(t.abs(t.load(column_np)), 0.2), 0.25)
    row = t.broadcast(row, shape)
    column = t.broadcast(column, shape)
    value = t.add(x, row)
    value = t.sub(value, column)
    value = t.abs(value)
    value = t.mul(value, 0.75)
    value = t.add(value, row)
    value = t.maximum(value, column)
    value = t.minimum(value, 2.0)
    out = t.add(value, 0.05)

    row_expect = (np.abs(row_np) + np.float32(0.1)) * np.float32(0.5)
    column_expect = (np.abs(column_np) + np.float32(0.2)) * np.float32(0.25)
    row_expect = np.broadcast_to(row_expect, shape)
    column_expect = np.broadcast_to(column_expect, shape)
    expect = x_np + row_expect
    expect = expect - column_expect
    expect = np.abs(expect)
    expect = expect * np.float32(0.75)
    expect = expect + row_expect
    expect = np.maximum(expect, column_expect)
    expect = np.minimum(expect, np.float32(2.0))
    expect = expect + np.float32(0.05)
    t.store_expect(out, expect, 1e-4)

    t.codegen()
    dump = t.dump()

    # Fourteen elementwise nodes form three VF regions around two Broadcasts.
    assert dump.count("Custom") == 3
    assert all(op not in dump for op in ("Abs", "Add", "Sub", "Mul", "Maximum", "Minimum"))
    assert t.run_check()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skipif(
    dvm.Device.arch() != "AscendC310",
    reason="VF Fusion only supports C310",
)
def test_broadcast_select():
    dvm.Kernel.set_vf_fusion(True)
    shape = (1024, 1024)
    x_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    y_np = np.random.normal(0.0, 0.3, shape).astype(np.float32)
    bias_np = np.random.normal(0.0, 0.2, (1, shape[1])).astype(np.float32)

    t = Tester()
    t.set_passes("VfFusion")
    x = t.load(x_np)
    y = t.load(y_np)
    bias = t.add(t.abs(t.load(bias_np)), 0.125)
    bias = t.broadcast(bias, shape)
    left = t.add(x, bias)
    right = t.sub(y, bias)
    product = t.mul(left, right)
    magnitude = t.abs(product)
    condition = t.greater(x, 0.0)
    clipped = t.minimum(magnitude, 1.5)
    scaled = t.mul(magnitude, 0.5)
    selected = t.select(condition, clipped, scaled)
    out = t.cast(selected, "float16")

    bias_expect = np.broadcast_to(np.abs(bias_np) + np.float32(0.125), shape)
    magnitude_expect = np.abs((x_np + bias_expect) * (y_np - bias_expect))
    selected_expect = np.where(
        x_np > np.float32(0.0),
        np.minimum(magnitude_expect, np.float32(1.5)),
        magnitude_expect * np.float32(0.5),
    )
    t.store_expect(magnitude, magnitude_expect, 1e-4)
    t.store_expect(out, selected_expect.astype(np.float16))

    t.codegen()
    dump = t.dump()

    # Eleven elementwise nodes, one Broadcast, and an external fused output.
    assert dump.count("Broadcast") == 1
    assert dump.count("Custom") == 2
    assert dump.count("ExtOut") == 1
    assert all(op not in dump for op in ("Abs", "Add", "Sub", "Mul", "Compare", "Minimum", "Select", "Cast"))
    assert t.run_check()
