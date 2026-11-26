# Owner(s): ["module: onnx"]
"""Optimization must not erase observable IEEE behavior or tiny constants."""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx.reference import ReferenceEvaluator

import torch_onnx


class _Model(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


def _evaluate(function, x, optimize, *, opset=18, return_program=False):
    model = _Model(function).eval()
    expected = model(x).numpy()
    program = torch_onnx.export(
        model, (x,), optimize=optimize is True, opset_version=opset, verbose=False
    )
    if optimize == "explicit":
        assert program.optimize() is None
        assert program.optimize() is None
    proto = ir.to_proto(program.model)
    onnx.checker.check_model(proto, full_check=True)
    feeds = {program.model.graph.inputs[0].name: x.numpy()}
    reference = ReferenceEvaluator(proto).run(None, feeds)[0]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    result = (expected, reference, runtime)
    return (*result, program) if return_program else result


@pytest.mark.parametrize("optimize", [False, True, "explicit"])
def test_add_zero_before_reciprocal_keeps_signed_zero_semantics(optimize):
    x = torch.tensor([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")])
    expected, reference, runtime = _evaluate(
        lambda value: (value + 0.0).reciprocal(), x, optimize
    )
    for actual in (reference, runtime):
        np.testing.assert_array_equal(actual, expected)
        zero = expected == 0
        np.testing.assert_array_equal(
            np.signbit(actual[zero]), np.signbit(expected[zero])
        )
        infinite = np.isinf(expected)
        np.testing.assert_array_equal(
            np.signbit(actual[infinite]), np.signbit(expected[infinite])
        )


@pytest.mark.parametrize("optimize", [False, True, "explicit"])
@pytest.mark.parametrize("scale", [1.0000000001, 0.9999999999])
def test_float64_near_one_scale_is_not_an_identity(scale, optimize):
    # pytorch/pytorch#206455: a tolerant literal matcher used to drop this Mul.
    x = torch.tensor([1.0, -2.0, 1e100, 1e-100], dtype=torch.float64)
    expected, reference, runtime = _evaluate(lambda value: value * scale, x, optimize)
    assert not np.array_equal(expected, x.numpy())
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(runtime, expected)


@pytest.mark.parametrize("optimize", [False, True, "explicit"])
@pytest.mark.parametrize("opset", [18, 22, 23])
def test_rms_norm_nonzero_epsilon_is_not_rewritten_as_zero(optimize, opset):
    x = torch.tensor([[1e-4, -2e-4, 3e-4], [2e-5, 0.0, -1e-5]])

    def function(value):
        return torch.nn.functional.rms_norm(value, (3,), eps=1e-8)

    expected, reference, runtime = _evaluate(function, x, optimize, opset=opset)
    without_epsilon = torch.nn.functional.rms_norm(x, (3,), eps=0.0).numpy()
    assert not np.allclose(expected, without_epsilon, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(runtime, expected, rtol=1e-6, atol=1e-7)


def test_export_optimization_does_not_modify_process_global_rewrite_policy():
    from onnxscript import rewriter
    from onnxscript.rewriter.rules.common import _no_op

    original_defaults = rewriter._DEFAULT_REWRITE_RULES
    original_noops = tuple(_no_op.rules)
    original_conditions = tuple(rule._condition_function for rule in original_noops)
    _evaluate(
        lambda value: value * 1.0000000001,
        torch.tensor([1.0], dtype=torch.float64),
        True,
    )
    assert rewriter._DEFAULT_REWRITE_RULES is original_defaults
    assert tuple(_no_op.rules) == original_noops
    assert (
        tuple(rule._condition_function for rule in original_noops)
        == original_conditions
    )


@pytest.mark.parametrize("optimize", [False, True, "explicit"])
def test_subtract_negative_zero_before_reciprocal_is_not_removed(optimize):
    x = torch.tensor([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")])
    expected, reference, runtime = _evaluate(
        lambda value: (value - -0.0).reciprocal(), x, optimize
    )
    for actual in (reference, runtime):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_exact_integer_add_zero_still_optimizes():
    x = torch.tensor([-(2**63), -1, 0, 1, 2**63 - 1], dtype=torch.int64)
    expected, reference, runtime, program = _evaluate(
        lambda value: value + 0,
        x,
        True,
        return_program=True,
    )
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(runtime, expected)
    assert all(node.op_type != "Add" for node in program.model.graph)


def test_explicit_optimization_releases_cached_session_and_model_files(monkeypatch):
    from torch.onnx._internal.exporter import _onnx_program as upstream_program

    class WeightedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(2))

        def forward(self, x):
            return (x * self.weight) * 1.0

    # Exercise the real external-data cache with small weights.
    monkeypatch.setattr(upstream_program, "_LARGE_MODEL_THRESHOLD", 0)
    x = torch.tensor([1.0, -2.0])
    model = WeightedModel().eval()
    program = torch_onnx.export(model, (x,), optimize=False, verbose=False)
    exported_program = program.exported_program
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1

    def initializer(model):
        return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])

    try:
        program.initialize_inference_session(initializer)
        old_session = weakref.ref(program._inference_session)
        torch.testing.assert_close(program(x)[0], x)
        assert program._tempdir is not None
        model_files = Path(program._tempdir.name)
        original_muls = sum(node.op_type == "Mul" for node in program.model.graph)
        assert program.optimize() is None
        assert program._inference_session is None
        assert old_session() is None
        assert program._tempdir is None
        assert not model_files.exists()
        assert program.exported_program is exported_program
        assert (
            sum(node.op_type == "Mul" for node in program.model.graph) < original_muls
        )
        program.initialize_inference_session(initializer)
        assert program._inference_session is not None
        torch.testing.assert_close(program(x)[0], x)
    finally:
        program.release()
