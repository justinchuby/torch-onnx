"""Focused contracts for the centralized PyTorch capability adapters."""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

from torch_onnx import _torch_compat


def test_ci_can_read_version_matrix_without_importing_torch_or_warning():
    path = Path(_torch_compat.__file__).resolve()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys,warnings; "
            "warnings.simplefilter('error'); "
            f"data=runpy.run_path({str(path)!r}); "
            "assert 'torch' not in sys.modules; "
            "assert len(data['SUPPORTED_TORCH_VERSIONS']) == 8",
        ],
        check=True,
    )


def test_dynamic_non_rnn_capture_does_not_require_new_rnn_helpers():
    with (
        mock.patch.object(_torch_compat, "_dynamic_rnn_helpers", return_value=None),
        mock.patch.object(_torch_compat, "warn_once") as warn,
        _torch_compat.dynamic_rnn_decompositions(torch.nn.Identity()),
    ):
        pass
    warn.assert_not_called()


def test_missing_dynamic_rnn_helpers_emit_compatibility_diagnostic():
    with (
        mock.patch.object(_torch_compat, "_dynamic_rnn_helpers", return_value=None),
        mock.patch.object(_torch_compat, "warn_once") as warn,
        _torch_compat.dynamic_rnn_decompositions(torch.nn.LSTM(3, 4)),
    ):
        pass
    warn.assert_called_once()
    assert "unavailable" in warn.call_args.args[0]


def test_available_dynamic_rnn_helpers_enter_and_restore_contexts():
    events = []

    @contextlib.contextmanager
    def helper(name):
        events.append(f"enter {name}")
        try:
            yield
        finally:
            events.append(f"exit {name}")

    with (
        mock.patch.object(
            _torch_compat,
            "_dynamic_rnn_helpers",
            return_value=(lambda: helper("lstm"), lambda: helper("gru")),
        ),
        pytest.raises(RuntimeError, match="capture failed"),
        _torch_compat.dynamic_rnn_decompositions(),
    ):
        events.append("capture")
        raise RuntimeError("capture failed")
    assert events == ["enter lstm", "enter gru", "capture", "exit gru", "exit lstm"]


def test_only_absent_helper_module_uses_legacy_capture_path():
    with mock.patch.object(
        _torch_compat.importlib,
        "import_module",
        side_effect=ModuleNotFoundError(name="torch.export._patches"),
    ):
        assert _torch_compat._dynamic_rnn_helpers() is None
    with (
        mock.patch.object(
            _torch_compat.importlib,
            "import_module",
            side_effect=ModuleNotFoundError(
                "broken helper dependency", name="unexpected_dependency"
            ),
        ),
        pytest.raises(ModuleNotFoundError, match="broken helper dependency"),
    ):
        _torch_compat._dynamic_rnn_helpers()


@pytest.mark.parametrize(
    "target", [torch.ops.aten.lstm.input, torch.ops.aten.gru.input]
)
def test_missing_helpers_diagnose_rnn_in_exported_fx_graph(target):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    result = graph.call_function(target, (x,))
    graph.output(result)
    module = torch.fx.GraphModule({}, graph)
    with (
        mock.patch.object(_torch_compat, "_dynamic_rnn_helpers", return_value=None),
        mock.patch.object(_torch_compat, "warn_once") as warn,
        _torch_compat.dynamic_rnn_decompositions(module),
    ):
        pass
    warn.assert_called_once()


@pytest.mark.parametrize("legacy", [False, True])
def test_fake_tensor_serialization_is_rejected_with_either_detection_api(legacy):
    from torch._subclasses import fake_tensor

    from torch_onnx import _core

    with fake_tensor.FakeTensorMode():
        value = torch.zeros(2)
    # Hide the capability only from our adapter: newer Torch's own is_fake
    # implementation internally calls is_fake_tensor and must remain intact.
    context = (
        mock.patch.object(
            _torch_compat,
            "getattr",
            side_effect=lambda module, name, default=None: None
            if name == "is_fake_tensor"
            else getattr(module, name, default),
            create=True,
        )
        if legacy
        else contextlib.nullcontext()
    )
    with context:
        assert not _torch_compat.is_fake_tensor(torch.zeros(2))
        assert _torch_compat.is_fake_tensor(value)
        with pytest.raises(
            TypeError, match="Cannot take content out from the FakeTensor"
        ):
            _core.TorchTensor(value).tobytes()


def test_exported_program_preparation_passes_graph_to_rnn_capability_adapter():
    from torch_onnx import _core

    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        torch.nn.ReLU(), (torch.randn(2, 3),), dynamic_shapes=({0: batch},)
    )
    with (
        mock.patch.object(
            _core._capture_strategies,
            "patch_dynamic_shape_rnn_decompositions",
            return_value=contextlib.nullcontext(),
        ) as adapter,
        mock.patch.object(
            _core._fx_passes, "decompose_with_registry", return_value=program
        ),
    ):
        _core._prepare_exported_program_for_export(
            program, registry=mock.sentinel.registry
        )
    adapter.assert_called_once_with(program.range_constraints, program.graph_module)
