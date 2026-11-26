"""Regression tests for the standalone package boundary."""

import onnx
import pytest
import torch

import torch_onnx


def test_patch_uses_standalone_translations_and_restores_torch(tmp_path):
    original_export = torch.onnx.export
    original_flag = torch.onnx.is_in_onnx_export

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    @torch_onnx.register_custom_translation(torch.ops.aten.add.Tensor)
    def subtract(op: torch_onnx.TranslationContext, self, other):
        return op.Sub(self, other)

    try:
        torch_onnx.patch_torch(optimize=False, verbose=False)
        program = torch.onnx.export(
            Model().eval(), (torch.ones(2), torch.zeros(2)), tmp_path / "model.onnx"
        )
        assert isinstance(program, torch_onnx.ONNXProgram)
        assert [node.op_type for node in program.model.graph] == ["Sub"]
        onnx.checker.check_model(onnx.load(tmp_path / "model.onnx"), full_check=True)
        torch.testing.assert_close(
            program(torch.ones(2), torch.zeros(2))[0], torch.ones(2)
        )
    finally:
        torch_onnx.unpatch_torch()
        torch_onnx.unregister_custom_translation(torch.ops.aten.add.Tensor)
    assert torch.onnx.export is original_export
    assert torch.onnx.is_in_onnx_export is original_flag


def test_direct_ir_conversion_and_optimization():
    model = torch.nn.ReLU().eval()
    args = (torch.randn(2, 3),)
    exported = torch.export.export(model, args)
    ir_model = torch_onnx.exported_program_to_ir(exported)
    program = torch_onnx.ONNXProgram(ir_model, exported)
    program.optimize()
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, strategy=None)


def test_legacy_export_is_explicitly_rejected():
    with pytest.raises(ValueError, match="only supports the modern exporter"):
        torch_onnx.export(torch.nn.ReLU(), (torch.ones(2),), dynamo=False)


def test_torch_export_state_is_visible_without_patching():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.sin() if torch.onnx.is_in_onnx_export() else x.cos()

    program = torch_onnx.export(
        Model().eval(), (torch.ones(2),), optimize=False, verbose=False
    )
    assert [node.op_type for node in program.model.graph] == ["Sin"]
    assert not torch_onnx.is_in_onnx_export()
    assert not torch.onnx.is_in_onnx_export()


def test_program_preserves_pytorch_type_compatibility():
    original_optimize = torch.onnx.ONNXProgram.optimize
    program = torch_onnx.export(torch.nn.Linear(3, 2).eval(), (torch.randn(4, 3),))
    assert isinstance(program, torch.onnx.ONNXProgram)
    assert type(program) is torch_onnx.ONNXProgram
    assert torch.onnx.ONNXProgram.optimize is original_optimize
    assert torch.onnx.ONNXProgram.__module__ == "torch.onnx"
    torch_onnx.testing.assert_onnx_program(program)
    program.release()
    torch_onnx.testing.assert_onnx_program(program)
