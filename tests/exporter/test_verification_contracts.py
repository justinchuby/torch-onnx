"""Verification diagnostics cover scalar, empty, boolean and complex values."""

import math
from unittest import mock

import pytest
import torch

import torch_onnx
from torch_onnx import _verification


@pytest.mark.parametrize(
    "expected,actual,max_diff",
    [
        (2, 3, 1),
        (True, False, 1),
        (torch.empty(0), torch.empty(0), math.inf),
        (torch.tensor([1 + 2j]), torch.tensor([[1.0, 3.0]]), 1),
    ],
)
def test_verification_statistics_are_serializable(expected, actual, max_diff):
    info = _verification.VerificationInfo.from_tensors("result", expected, actual)
    assert info.max_abs_diff == max_diff
    result = info.asdict()
    assert result["name"] == "result"
    assert len(result["abs_diff_hist"][0]) == 9
    assert isinstance(result["actual_dtype"], str)


@pytest.mark.parametrize("intermediates", [False, True])
def test_verification_without_exported_program_is_actionable(intermediates):
    program = torch_onnx.export(
        torch.nn.ReLU().eval(), (torch.randn(2),), verbose=False
    )
    program.exported_program = None
    try:
        with pytest.raises(ValueError, match="does not contain an exported_program"):
            _verification.verify_onnx_program(
                program, compare_intermediates=intermediates
            )
    finally:
        program.release()


def test_verification_accepts_keyword_only_examples():
    class Model(torch.nn.Module):
        def forward(self, *, x):
            return x + 1

    x = torch.randn(2)
    program = torch_onnx.export(
        Model().eval(), (), kwargs={"x": x}, optimize=False, verbose=False
    )
    try:
        results = _verification.verify_onnx_program(program, kwargs={"x": x})
        assert len(results) == 1 and results[0].max_abs_diff == 0
    finally:
        program.release()


def test_intermediate_verification_records_failures_and_mismatches(caplog):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.relu(x))

    x = torch.tensor([-1.0, 2.0])
    program = torch_onnx.export(Model().eval(), (x,), optimize=False, verbose=False)
    try:
        with mock.patch.object(
            program,
            "compute_values",
            side_effect=RuntimeError("unavailable intermediate"),
        ):
            assert not _verification.verify_onnx_program(
                program, compare_intermediates=True
            )
        assert "Failed to compute value" in caplog.text
        original = program.compute_values

        def incorrect(names, args):
            return tuple(value + 2 for value in original(names, args))

        with mock.patch.object(program, "compute_values", side_effect=incorrect):
            results = _verification.verify_onnx_program(
                program, compare_intermediates=True
            )
        assert results and all(info.max_abs_diff > 1 for info in results)
        assert "max_abs_diff" in caplog.text
    finally:
        program.release()
