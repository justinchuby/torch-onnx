"""Observed executions must expose inaccurate or failing custom lowerings."""

import pytest
import torch

import torch_onnx


@pytest.mark.parametrize("failure", ["shape", "dtype", "values"])
def test_observer_marks_incorrect_custom_lowering_as_unsuccessful(failure):
    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    def incorrect(op, self):
        if failure == "shape":
            return op.Unsqueeze(self, [0])
        if failure == "dtype":
            return op.Cast(self, to=7)  # INT64.
        return op.Abs(self)

    model = Model().eval()
    observer = torch_onnx.InputObserver()
    with observer(model):
        model(torch.tensor([1.0, 2.0]))
    target = torch.ops.aten.neg.default
    torch_onnx.register_custom_translation(target, incorrect)
    program = None
    try:
        program = torch_onnx.export(model, observer.infer_arguments(), verbose=False)
        results = observer.check_discrepancies(program, progress_bar=True)
        assert len(results) == 1
        assert not results[0]["SUCCESS"]
        if failure == "values":
            assert results[0]["abs"] == 4
        assert results[0]["n_inputs"] == 1
    finally:
        if program is not None:
            program.release()
        torch_onnx.unregister_custom_translation(target)


def test_observer_keeps_inference_error_in_summary(monkeypatch):
    model = torch.nn.ReLU().eval()
    observer = torch_onnx.InputObserver()
    with observer(model):
        model(torch.randn(2))
    program = torch_onnx.export(model, observer.infer_arguments(), verbose=False)

    def fail(*args, **kwargs):
        raise RuntimeError("injected inference failure")

    try:
        monkeypatch.setattr(torch_onnx.ONNXProgram, "__call__", fail)
        results = observer.check_discrepancies(program)
        assert not results[0]["SUCCESS"]
        assert results[0]["error"] == "injected inference failure"
    finally:
        program.release()
