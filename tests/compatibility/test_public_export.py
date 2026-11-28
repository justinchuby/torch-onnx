"""Cross-release public export contracts without PyTorch's internal test harness."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F

import torch_onnx

pytestmark = pytest.mark.compatibility


class _Affine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor(
                [
                    [1.0, -0.5, 0.25],
                    [-0.25, 0.5, 1.0],
                    [0.125, -1.0, 0.75],
                    [0.5, 0.25, -0.25],
                ]
            ),
        )
        self.register_buffer("bias", torch.tensor([0.25, -0.125, 0.5]))

    def forward(self, x):
        return F.relu(x @ self.weight + self.bias), torch.stack((x, x + 0.125))


class _Dynamic(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2).reshape(x.shape[0], -1), x.sum(dim=1)


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = (torch.arange(28, dtype=torch.float32).reshape(7, 4) - 14) / 8
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=True)

    def forward(self, indices):
        values = self.embedding(indices)
        return values, values.mean(dim=1)


class _CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([0.5, 2.0, 1.0, 3.0]))

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=0.2)


def _outputs(value):
    return tuple(value) if isinstance(value, tuple | list) else (value,)


def _compare(expected, actual):
    expected = _outputs(expected)
    assert len(actual) == len(expected)
    for reference, output in zip(expected, actual, strict=True):
        expected_array = reference.detach().cpu().numpy()
        actual_array = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else output
        )
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=1e-5, atol=1e-6, equal_nan=True
        )


def _runtime_check(program, args, names, expected):
    """Check serialized ONNX and both direct ORT and public program execution."""
    proto = program.model_proto
    onnx.checker.check_model(proto, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in zip(names, args, strict=True)
    }
    feeds = {value.name: arrays[value.name] for value in session.get_inputs()}
    _compare(expected, session.run(None, feeds))
    _compare(expected, program(*args))


@pytest.mark.parametrize("optimize", [False, True])
def test_basic_export_executes_noncontiguous_inputs_and_saved_model(optimize, tmp_path):
    model = _Affine().eval()
    x = (torch.arange(24, dtype=torch.float32).reshape(4, 6) - 12).transpose(0, 1) / 5
    assert not x.is_contiguous()
    program = torch_onnx.export(
        model,
        (x,),
        input_names=["x"],
        opset_version=18,
        optimize=optimize,
        verbose=False,
    )
    try:
        _runtime_check(program, (x,), ["x"], model(x))
        path = tmp_path / "basic.onnx"
        program.save(path, external_data=False)
        onnx.checker.check_model(onnx.load(path), full_check=True)
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
def test_dynamic_export_executes_different_batch_and_sequence_lengths(optimize):
    model = _Dynamic().eval()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7
    program = torch_onnx.export(
        model,
        (x,),
        input_names=["x"],
        opset_version=18,
        optimize=optimize,
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch", min=1),
                1: torch.export.Dim("sequence", min=1),
            }
        },
        verbose=False,
    )
    try:
        for shape in ((2, 3, 4), (5, 7, 4)):
            alternative = (
                torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) / 11
            )
            _runtime_check(program, (alternative,), ["x"], model(alternative))
    finally:
        program.release()


def test_explicit_program_optimize_refreshes_a_warmed_runtime():
    model = _Affine().eval()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8
    program = torch_onnx.export(
        model, (x,), input_names=["x"], opset_version=18, optimize=False, verbose=False
    )
    try:
        _compare(model(x), program(x))
        # The public IR model is editable. Updating an initializer after warming
        # ORT makes a stale cached session observable, despite equivalent rewrites.
        weights = [
            value
            for value in program.model.graph.initializers.values()
            if value.shape == ir.Shape([4, 3])
        ]
        assert len(weights) == 1
        replacement = np.zeros((4, 3), dtype=np.float32)
        weights[0].const_value = ir.tensor(replacement, dtype=ir.DataType.FLOAT)
        assert program.optimize() is None
        expected = (
            F.relu(x @ torch.from_numpy(replacement) + model.bias),
            torch.stack((x, x + 0.125)),
        )
        _runtime_check(program, (x,), ["x"], expected)
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
def test_embedding_state_and_integer_indices_execute(optimize):
    model = _Embedding().eval()
    indices = torch.tensor([[0, 2, 6], [1, 1, 4]], dtype=torch.int64)
    program = torch_onnx.export(
        model,
        (indices,),
        input_names=["indices"],
        opset_version=18,
        optimize=optimize,
        verbose=False,
    )
    try:
        _runtime_check(program, (indices,), ["indices"], model(indices))
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("probability_target", [False, True])
def test_weighted_smoothed_cross_entropy_executes(optimize, probability_target):
    model = _CrossEntropy().eval()
    logits = torch.tensor(
        [[2.0, -1.0, 0.5, 0.25], [0.1, 0.2, 0.3, -0.5], [-0.1, -0.2, 1.5, 0.4]]
    )
    target = (
        torch.tensor(
            [[0.5, 0.2, 0.1, 0.2], [0.0, 0.1, 0.7, 0.2], [0.25, 0.25, 0.25, 0.25]]
        )
        if probability_target
        else torch.tensor([0, 2, -100], dtype=torch.int64)
    )
    args = (logits, target)
    program = torch_onnx.export(
        model,
        args,
        input_names=["logits", "target"],
        opset_version=18,
        optimize=optimize,
        verbose=False,
    )
    try:
        _runtime_check(program, args, ["logits", "target"], model(*args))
    finally:
        program.release()


def test_import_and_export_do_not_require_private_test_or_vision_packages():
    """A fresh interpreter prevents prior test collection from hiding imports."""
    script = r"""
import importlib.abc
import importlib.util
import sys

# logging_tensor is bundled runtime code used by PyTorch itself. Only exclude
# optional *test harness* dependencies, not the entire _internal namespace.
blocked = ("torch.testing._internal.common_utils", "torch.testing._internal.common_device_type", "torch.testing._internal.opinfo", "torchvision", "expecttest", "hypothesis")
def unavailable(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)

# Match genuine absence for capability probes, as well as for actual imports.
# Dynamo legitimately probes optional packages through find_spec.
original_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    return None if unavailable(name) else original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec

class NoOptionalTestPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if unavailable(fullname):
            raise ModuleNotFoundError("Public compatibility smoke does not install " + fullname, name=fullname)
        return None

sys.meta_path.insert(0, NoOptionalTestPackages())
import numpy as np
import onnx
import torch
import torch_onnx

class M(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + x

model = M().eval()
x = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7
program = torch_onnx.export(model, (x,), opset_version=18, optimize=False, verbose=False)
try:
    onnx.checker.check_model(program.model_proto, full_check=True)
    np.testing.assert_allclose(program(x)[0].numpy(), model(x).numpy(), rtol=1e-5, atol=1e-6)
finally:
    program.release()
"""
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("opset", ["omitted", None, 18, 23, 28])
def test_default_opset_28_and_explicit_overrides(patched, opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu() + 1

    model = Model().eval()
    x = torch.tensor([-2.0, 0.0, 3.0])
    original_export = torch.onnx.export
    if patched:
        torch_onnx.patch_torch(verbose=False)
    try:
        export = torch.onnx.export if patched else torch_onnx.export
        kwargs = {} if opset == "omitted" else {"opset_version": opset}
        program = export(model, (x,), verbose=False, **kwargs)
        try:
            expected_opset = 28 if opset in ("omitted", None) else opset
            assert program.model.opset_imports[""] == expected_opset
            onnx.checker.check_model(program.model_proto, full_check=True)
            torch.testing.assert_close(program(x)[0], model(x))
        finally:
            program.release()
    finally:
        if patched:
            torch_onnx.unpatch_torch()
    assert torch.onnx.export is original_export


def test_default_opset_selects_native_swish_translation():
    class Model(torch.nn.Module):
        def forward(self, x):
            return F.silu(x)

    program = torch_onnx.export(Model().eval(), (torch.randn(2, 3),), verbose=False)
    try:
        assert program.model.opset_imports[""] == 28
        assert [node.op_type for node in program.model.graph] == ["Swish"]
        onnx.checker.check_model(program.model_proto, full_check=True)
        # ORT 1.32's Swish function expansion fails above opset 24.
        torch_onnx.testing.assert_onnx_program(program, backend="reference")
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
def test_default_opset_native_function_uses_active_opset(optimize):
    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    def translate(op, x):
        formal = ir.Value(name="x", type=x.type, shape=x.shape)
        body = ir.Graph(
            [formal],
            [],
            nodes=[],
            name="negate",
            opset_imports={"": op.builder.opset},
        )
        builder = torch_onnx.GraphBuilder(body, opset=op.builder.opset)
        body.outputs.append(builder.call("Neg", formal)[0])
        function = ir.Function("test.default", "Negate", "", graph=body, attributes=[])
        return op.builder.call_function(function, x, version=1)[0]

    target = torch.ops.aten.neg.default
    torch_onnx.register_custom_translation(target, translate)
    try:
        from unittest import mock

        with mock.patch(
            "onnxscript.version_converter.convert_version",
            side_effect=AssertionError(
                "Native opset-28 export must not convert versions"
            ),
        ):
            x = torch.randn(2, 3)
            program = torch_onnx.export(
                Model().eval(), (x,), optimize=optimize, verbose=False
            )
        try:
            assert program.model.opset_imports[""] == 28
            assert all(
                fn.opset_imports[""] == 28 for fn in program.model.functions.values()
            )
            onnx.checker.check_model(program.model_proto, full_check=True)
            torch.testing.assert_close(program(x)[0], -x)
        finally:
            program.release()
    finally:
        torch_onnx.unregister_custom_translation(target)
