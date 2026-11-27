"""End-to-end reports, analysis and verification for successful and failing exports."""

import io
from unittest import mock

import pytest
import torch

import torch_onnx
from torch_onnx import _analysis, _registration


class WeightedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 3))
        self.register_buffer("counter", torch.tensor([1]))

    def forward(self, x, scale=2):
        return (x @ self.weight) * scale, self.counter, x.shape[0]


def test_analysis_reports_weights_buffers_and_symbolic_outputs():
    program = torch.export.export(
        WeightedModel().eval(),
        (torch.randn(2, 2), 2),
        dynamic_shapes=({0: torch.export.Dim("batch", min=2, max=8)}, None),
    )
    output = io.StringIO()
    torch_onnx.analyze(program, file=output)
    report = output.getvalue()
    assert "6 parameters and 1 buffers" in report
    assert "torch.float32" in report and "torch.int64" in report
    assert "SymInt" in report
    assert "scale" not in report.split("Inputs:")[1].split("Outputs:")[0]


def test_analysis_retains_unsupported_operator_and_dispatch_failure_details():
    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    empty = _registration.ONNXRegistry()
    missing = io.StringIO()
    torch_onnx.analyze(program, registry=empty, file=missing)
    assert "aten.relu.default" in missing.getvalue()
    assert "Errors grouped by operator" in missing.getvalue()
    broken = io.StringIO()
    with mock.patch.object(
        _analysis._dispatching, "dispatch", side_effect=RuntimeError("broken dispatch")
    ):
        torch_onnx.analyze(program, registry=empty, file=broken)
    assert "Critical Error in dispatcher" in broken.getvalue()
    assert "broken dispatch" in broken.getvalue()


def test_analysis_compares_operators_in_two_exported_programs():
    x = torch.randn(2)
    relu = torch.export.export(torch.nn.ReLU(), (x,))
    sigmoid = torch.export.export(torch.nn.Sigmoid(), (x,))
    left, right = _analysis.compare_ops(relu, sigmoid)
    assert left == {"aten.relu.default"}
    assert right == {"aten.sigmoid.default"}


def test_capture_failure_writes_a_report_with_the_original_error(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x):
            raise ValueError("deliberate capture failure")

    with pytest.raises(torch_onnx.errors.TorchExportError):
        torch_onnx.export(
            Model().eval(),
            (torch.ones(2),),
            report=True,
            artifacts_dir=tmp_path,
            verbose=False,
        )
    reports = list(tmp_path.glob("*.md"))
    assert len(reports) == 1
    assert "deliberate capture failure" in reports[0].read_text()
    assert "Conversion Error Report" in reports[0].read_text()


def test_native_lowering_failure_writes_a_conversion_report(tmp_path):
    def fail(op, self):
        raise ValueError("deliberate lowering failure")

    target = torch.ops.aten.relu.default
    torch_onnx.register_custom_translation(target, fail)
    try:
        with pytest.raises(torch_onnx.errors.ConversionError):
            torch_onnx.export(
                torch.nn.ReLU().eval(),
                (torch.ones(2),),
                report=True,
                artifacts_dir=tmp_path,
                verbose=False,
            )
        reports = list(tmp_path.glob("*_conversion.md"))
        assert len(reports) == 1
        text = reports[0].read_text()
        assert "deliberate lowering failure" in text
        assert "Exported program" in text and "Analysis" in text
    finally:
        torch_onnx.unregister_custom_translation(target)


def test_successful_profiled_export_reports_verification_and_analysis(tmp_path):
    program = torch_onnx.export(
        torch.nn.Linear(2, 3).eval(),
        (torch.randn(2, 2),),
        report=True,
        verify=True,
        profile=True,
        artifacts_dir=tmp_path,
        verbose=False,
    )
    try:
        reports = list(tmp_path.glob("*_success.md"))
        assert len(reports) == 1
        text = reports[0].read_text()
        assert "Verification results" in text
        assert "max_abs_diff=" in text
        assert "Profiling result" in text
        assert "ONNX model" in text
    finally:
        program.release()


def test_incorrect_custom_lowering_is_reported_as_an_accuracy_failure(tmp_path):
    def incorrect(op, self):
        return op.Abs(self)

    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    target = torch.ops.aten.neg.default
    torch_onnx.register_custom_translation(target, incorrect)
    program = None
    try:
        program = torch_onnx.export(
            Model().eval(),
            (torch.tensor([2.0, -1.0]),),
            report=True,
            verify=True,
            artifacts_dir=tmp_path,
            verbose=False,
        )
        reports = list(tmp_path.glob("*_accuracy.md"))
        assert len(reports) == 1
        assert "max_abs_diff=4.000000e+00" in reports[0].read_text()
    finally:
        if program is not None:
            program.release()
        torch_onnx.unregister_custom_translation(target)
