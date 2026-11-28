"""Exception identity and interoperability with the installed PyTorch exporter."""

import pickle
import subprocess
import sys

import pytest
import torch
from torch.onnx._internal.exporter import _errors as torch_errors

import torch_onnx
from torch_onnx import _errors, errors


@pytest.mark.parametrize(
    "name",
    ["TorchExportError", "ConversionError", "DispatchError", "GraphConstructionError"],
)
def test_modern_exception_identity_and_pickle(name):
    upstream = getattr(torch_errors, name)
    assert getattr(errors, name) is upstream
    assert getattr(_errors, name) is upstream
    error = upstream("export failed")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is upstream
    assert str(restored) == "export failed"


@pytest.mark.parametrize(
    "name",
    [
        "OnnxExporterError",
        "OnnxExporterWarning",
        "SymbolicValueError",
        "UnsupportedOperatorError",
    ],
)
def test_public_exception_identity(name):
    assert getattr(errors, name) is getattr(torch.onnx.errors, name)
    assert torch_onnx.OnnxExporterError is torch.onnx.OnnxExporterError


def test_import_does_not_mutate_upstream_exception_metadata():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch.onnx
from torch.onnx._internal.exporter import _errors
classes = [torch.onnx.OnnxExporterError, _errors.TorchExportError,
           _errors.ConversionError, _errors.DispatchError, _errors.GraphConstructionError]
before = [(cls.__module__, cls.__qualname__) for cls in classes]
import torch_onnx
assert before == [(cls.__module__, cls.__qualname__) for cls in classes]
assert torch_onnx.OnnxExporterError is torch.onnx.OnnxExporterError
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("exporter", [torch_onnx.export, torch.onnx.export])
def test_capture_errors_are_caught_by_the_same_handler(exporter):
    class Model(torch.nn.Module):
        def forward(self, x):
            raise ValueError("controlled capture failure")

    with pytest.raises(torch.onnx.OnnxExporterError) as raised:
        exporter(Model().eval(), (torch.ones(2),), verbose=False)
    assert type(raised.value) is torch_errors.TorchExportError
    assert raised.value.__cause__ is not None
    assert "controlled capture failure" in str(raised.value)


def test_native_lowering_error_preserves_upstream_hierarchy_and_cause():
    marker = ValueError("controlled native lowering failure")

    def broken(op, self):
        raise marker

    torch_onnx.register_custom_translation(torch.ops.aten.neg.default, broken)
    try:

        class Model(torch.nn.Module):
            def forward(self, x):
                return -x

        with pytest.raises(torch_errors.ConversionError) as raised:
            torch_onnx.export(Model().eval(), (torch.ones(2),), verbose=False)
        chain = []
        error = raised.value
        while error is not None:
            chain.append(error)
            error = error.__cause__
        assert any(
            isinstance(error, torch_errors.GraphConstructionError) for error in chain
        )
        assert chain[-1] is marker
    finally:
        torch_onnx.unregister_custom_translation(torch.ops.aten.neg.default)


def test_missing_lowering_preserves_the_upstream_dispatch_error():
    with torch.library._scoped_library("standalone_error_test", "FRAGMENT") as library:
        name = "standalone_error_test::missing"
        torch.library.define(name, "(Tensor x) -> Tensor", lib=library)
        torch.library.impl(name, "CompositeExplicitAutograd", lib=library)(torch.sin)
        torch.library.register_fake(name, lib=library)(torch.empty_like)

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.standalone_error_test.missing.default(x)

        with pytest.raises(torch_errors.ConversionError) as raised:
            torch_onnx.export(Model().eval(), (torch.ones(2),), verbose=False)
        error = raised.value
        while error is not None and not isinstance(error, torch_errors.DispatchError):
            error = error.__cause__
        assert isinstance(error, torch_errors.DispatchError)
        assert "standalone_error_test" in str(error)
