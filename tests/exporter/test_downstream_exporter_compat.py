"""Downstream policies remain export-local and specialized flags are constants."""

from __future__ import annotations

import sys
import types
from unittest import mock

import onnx
import pytest
import torch

import torch_onnx
from torch_onnx import _exporter_compat


@pytest.fixture
def downstream(monkeypatch):
    module = types.ModuleType("transformers.exporters.exporter_onnx")
    table = {}
    monkeypatch.setattr(module, "_ONNX_TRANSLATION_TABLE", table, raising=False)
    callback = mock.Mock()
    monkeypatch.setattr(module, "apply_fx_node_fixes", callback, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return table, callback


def test_downstream_context_is_nested_and_restored_on_failure(downstream):
    table, callback = downstream
    resize = types.SimpleNamespace(op_type="Resize", domain="")
    add = types.SimpleNamespace(op_type="Add", domain="")
    marker = object()
    assert _exporter_compat.constant_fold_policy()(resize) is None
    with (
        pytest.raises(RuntimeError, match="controlled"),
        _exporter_compat.downstream_export(table),
    ):
        assert _exporter_compat.constant_fold_policy()(resize) is False
        assert _exporter_compat.constant_fold_policy()(add) is None
        _exporter_compat.postprocess_decomposed_graph(marker)
        with _exporter_compat.downstream_export({}):
            assert _exporter_compat.constant_fold_policy()(resize) is None
            _exporter_compat.postprocess_decomposed_graph(marker)
        assert _exporter_compat.constant_fold_policy()(resize) is False
        raise RuntimeError("controlled")
    assert _exporter_compat.constant_fold_policy()(resize) is None
    _exporter_compat.postprocess_decomposed_graph(marker)
    callback.assert_called_once_with("onnx", marker)


def test_real_export_runs_downstream_fx_pass_without_touching_torch_internals(
    downstream,
):
    from torch.onnx._internal.exporter import _core as torch_core

    table, callback = downstream
    upstream = torch_core._prepare_exported_program_for_export

    def rewrite(_backend, graph_module):
        for node in graph_module.graph.nodes:
            if node.target is torch.ops.aten.neg.default:
                node.target = torch.ops.aten.abs.default
        graph_module.recompile()

    callback.side_effect = rewrite

    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    x = torch.tensor([-2.0, 1.0])
    try:
        torch_onnx.patch_torch(verbose=False, optimize=False)
        program = torch.onnx.export(
            Model().eval(), (x,), custom_translation_table=table
        )
        try:
            assert [node.op_type for node in program.model.graph] == ["Abs"]
            torch.testing.assert_close(program(x)[0], x.abs())
        finally:
            program.release()
        plain = torch.onnx.export(Model().eval(), (x,))
        try:
            assert [node.op_type for node in plain.model.graph] == ["Neg"]
            torch.testing.assert_close(plain(x)[0], -x)
        finally:
            plain.release()
    finally:
        torch_onnx.unpatch_torch()
    assert torch_core._prepare_exported_program_for_export is upstream
    assert callback.call_count == 1


def test_downstream_resize_policy_reaches_constant_folder(downstream):
    table, _callback = downstream

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "position", torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
            )

        def forward(self, x):
            resized = torch.nn.functional.interpolate(
                self.position, size=(3, 3), mode="bicubic", align_corners=False
            )
            return x + resized

    model = Model().eval()
    x = torch.ones(2, 1, 3, 3)
    for custom_table, retain_resize in [(table, True), (None, False)]:
        program = torch_onnx.export(
            model, (x,), custom_translation_table=custom_table, verbose=False
        )
        try:
            assert (
                any(node.op_type == "Resize" for node in program.model.graph)
                == retain_resize
            )
            torch.testing.assert_close(program(x)[0], model(x))
        finally:
            program.release()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_specialized_bool_and_int_arguments_are_not_onnx_inputs(enabled, optimize):
    class Model(torch.nn.Module):
        def forward(self, enabled, x, scale, y):
            return (x + y) * scale if enabled else x - y

    model = Model().eval()
    x, y = torch.ones(2, 3), torch.full((2, 3), 0.5)
    args = (enabled, x, 3, y)
    program = torch_onnx.export(
        model, args, input_names=["x", "y"], optimize=optimize, verbose=False
    )
    try:
        assert [value.name for value in program.model.graph.inputs] == ["x", "y"]
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(*args)[0], model(*args))
        torch.testing.assert_close(program.call_reference(*args)[0], model(*args))
        with pytest.raises(ValueError, match="Specialized argument"):
            program(not enabled, x, 3, y)
    finally:
        program.release()


def test_specialized_keyword_arguments_preserve_captured_order():
    class Model(torch.nn.Module):
        def forward(self, *, x, enabled, y):
            return x - y if enabled else x + y

    model = Model().eval()
    x, y = torch.ones(2), torch.full((2,), 0.5)
    program = torch_onnx.export(
        model, (), kwargs={"x": x, "enabled": True, "y": y}, verbose=False
    )
    try:
        torch.testing.assert_close(program(y=y, enabled=True, x=x)[0], x - y)
    finally:
        program.release()
