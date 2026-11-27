"""Renaming must preserve graph identity and reject ambiguous public names."""

import onnx_ir as ir
import pytest

from torch_onnx import _ir_passes


def _model():
    x = ir.Value(name="x", shape=ir.Shape(["s1", "s11", "2*s1 + s11", None, 3]))
    y = ir.Value(name="y")
    node = ir.Node("", "Identity", [x], outputs=[y])
    return ir.Model(ir.Graph([x], [y], nodes=[node]), ir_version=10)


@pytest.mark.parametrize(
    "rename", [_ir_passes.rename_inputs, _ir_passes.rename_outputs]
)
def test_duplicate_names_are_rejected_without_mutating_graph(rename):
    model = _model()
    with pytest.raises(ValueError, match=r"cannot be duplicated.*duplicate"):
        rename(model, ["duplicate", "duplicate"])
    assert model.graph.inputs[0].name == "x"
    assert model.graph.outputs[0].name == "y"


def test_axis_renaming_preserves_expression_boundaries_and_unknown_shapes():
    model = _model()
    _ir_passes.rename_axis(model, {ir.SymbolicDim("s1"): "batch", "s11": "length"})
    assert model.graph.inputs[0].shape == ir.Shape(
        ["batch", "length", "2*batch + length", None, 3]
    )
    assert model.graph.outputs[0].shape is None


@pytest.mark.parametrize("key", [3, ir.SymbolicDim(None)])
def test_invalid_axis_keys_fail_before_changing_shapes(key):
    model = _model()
    with pytest.raises(ValueError, match="Invalid"):
        _ir_passes.rename_axis(model, {key: "batch"})
    assert model.graph.inputs[0].shape[0].value == "s1"
