# Owner(s): ["module: onnx"]
"""Unit tests for the ONNXProgram class."""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification

import torch_onnx
from torch_onnx._lazy_import import onnx_ir as ir


class ONNXProgramRenameAxesTest(common_utils.TestCase):
    """Tests for ONNXProgram.rename_axes method."""

    hw_classification = HardwareClassification.GENERIC

    def _create_onnx_program_with_dynamic_shapes(
        self, input_shapes: list[list[str | int]]
    ) -> torch_onnx.ONNXProgram:
        """Helper to create an ONNXProgram with specified dynamic shapes."""
        from torch_onnx._onnx_program import ONNXProgram

        inputs = []
        for i, shape in enumerate(input_shapes):
            inputs.append(
                ir.Value(
                    name=f"input_{i}",
                    type=ir.DataType.FLOAT,
                    shape=ir.Shape(shape),
                )
            )

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[inputs[0]],
                nodes=[],
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )
        return ONNXProgram(model, exported_program=None)

    def test_rename_axes_with_string_keys(self):
        """Test renaming axes using string keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape,
            ir.Shape(["batch_size", "hidden_size"]),
        )
        # Outputs should also be renamed
        self.assertEqual(
            onnx_program.model.graph.outputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )

    def test_rename_axes_with_symbolic_dim_keys(self):
        """Test renaming axes using SymbolicDim objects as keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        # Get SymbolicDim objects from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]
        seq_dim = onnx_program.model.graph.inputs[0].shape[1]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                seq_dim: "sequence_length",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        # s0 should be renamed in both inputs
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape[0],
            ir.SymbolicDim("batch_size"),
        )

    def test_rename_axes_with_mixed_string_and_symbolic_dim_keys(self):
        """Test renaming axes using a mix of string and SymbolicDim keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1", "s2"]]
        )

        # Get SymbolicDim object from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length", "hidden_size"]),
        )

    def test_rename_axes_with_complex_shape_expressions(self):
        """Test renaming axes in complex shape expressions like 's1 + s2'."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1 + s2", "s1*2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "seq_len",
                "s2": "past_seq_len",
                "s1 + s2": "total_seq_len",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "total_seq_len", "seq_len*2"]),
        )

    def test_rename_axes_with_nonexistent_axis_is_ignored(self):
        """Test that renaming a non-existent axis is silently ignored."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        # Include a mapping for a non-existent axis
        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "nonexistent": "should_be_ignored",
            }
        )

        # Only s0 should be renamed
        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "s1"]),
        )

    def test_rename_axes_with_empty_mapping(self):
        """Test renaming with an empty mapping does nothing."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        original_shape = onnx_program.model.graph.inputs[0].shape

        onnx_program.rename_axes({})

        self.assertEqual(onnx_program.model.graph.inputs[0].shape, original_shape)

    def test_rename_axes_with_static_dimensions_unchanged(self):
        """Test that static (integer) dimensions are not affected by renaming."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", 10, "s1"]])

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", 10, "hidden_size"]),
        )


@pytest.mark.parametrize("backend", ["onnxruntime", "reference"])
@pytest.mark.parametrize(
    "arguments", ["missing", "extra", "bool_as_int", "int_as_float"]
)
def test_specialized_input_structure_and_types_are_validated(backend, arguments):
    class Model(torch.nn.Module):
        def forward(self, x, *, enabled=True, scale=2):
            return x * scale if enabled else x

    x = torch.ones(2)
    program = torch_onnx.export(
        Model().eval(), (x,), kwargs={"enabled": True, "scale": 2}, verbose=False
    )
    kwargs = {
        "missing": {"enabled": True},
        "extra": {"enabled": True, "scale": 2, "unexpected": 3},
        "bool_as_int": {"enabled": 1, "scale": 2},
        "int_as_float": {"enabled": True, "scale": 2.0},
    }[arguments]
    run = program if backend == "onnxruntime" else program.call_reference
    try:
        match = (
            "input structure"
            if arguments in ("missing", "extra")
            else "Specialized argument"
        )
        with pytest.raises(ValueError, match=match):
            run(x, **kwargs)
        torch.testing.assert_close(run(x, scale=2, enabled=True)[0], x * 2)
    finally:
        program.release()


def test_saved_program_runs_without_an_exported_program(tmp_path):
    x = torch.tensor([-2.0, 3.0])
    original = torch_onnx.export(torch.nn.ReLU().eval(), (x,), verbose=False)
    loaded = None
    try:
        path = tmp_path / "model.onnx"
        original.save(path)
        loaded = torch_onnx.ONNXProgram(ir.load(path), exported_program=None)
        torch.testing.assert_close(loaded(x)[0], x.relu())
        torch.testing.assert_close(loaded.call_reference(x)[0], x.relu())
    finally:
        original.release()
        if loaded is not None:
            loaded.release()


def test_optimize_invalidates_an_initialized_runtime_session():
    x = torch.tensor([-2.0, 3.0])
    program = torch_onnx.export(
        torch.nn.ReLU().eval(), (x,), optimize=False, verbose=False
    )
    try:
        torch.testing.assert_close(program(x)[0], x.relu())
        session = program._inference_session
        assert session is not None
        program.optimize()
        assert program._inference_session is None
        torch.testing.assert_close(program(-x)[0], (-x).relu())
        assert program._inference_session is not session
    finally:
        program.release()


if __name__ == "__main__":
    common_utils.run_tests()
