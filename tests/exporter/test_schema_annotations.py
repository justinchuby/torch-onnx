"""Python annotations must retain precise ONNX schema constraints."""

# Exercise both legacy typing forms and PEP 604 annotations.
# ruff: noqa: UP007, UP045

from collections.abc import Sequence
from typing import Optional, TypeVar, Union

import onnx_ir as ir
import pytest
from onnxscript.onnx_types import DOUBLE, FLOAT, INT64, TensorType

from torch_onnx import _schemas


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (
            FLOAT | DOUBLE,
            {ir.TensorType(ir.DataType.FLOAT), ir.TensorType(ir.DataType.DOUBLE)},
        ),
        (
            Union[FLOAT, INT64],
            {ir.TensorType(ir.DataType.FLOAT), ir.TensorType(ir.DataType.INT64)},
        ),
        (Optional[FLOAT], {ir.TensorType(ir.DataType.FLOAT)}),
        (FLOAT | None, {ir.TensorType(ir.DataType.FLOAT)}),
        (Sequence[FLOAT], {ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))}),
        (TensorType, {ir.TensorType(dtype) for dtype in ir.DataType}),
        (TypeVar("Bound", bound=FLOAT), {ir.TensorType(ir.DataType.FLOAT)}),
        (
            TypeVar("Choice", FLOAT, INT64),
            {ir.TensorType(ir.DataType.FLOAT), ir.TensorType(ir.DataType.INT64)},
        ),
    ],
)
def test_input_and_output_constraints_preserve_annotation(annotation, expected):
    def lowering(x):
        return x

    lowering.__annotations__ = {"x": annotation, "return": annotation}
    signature = _schemas.op_signature_from_function(lowering, "test")
    assert signature.params[0].type_constraint.allowed_types == expected
    assert signature.outputs[0].type_constraint.allowed_types == expected


def test_shared_type_variables_and_optional_sequence_constraints():
    Tensor = TypeVar("Tensor", FLOAT, INT64)

    def lowering(x, y=None):
        return x, y

    lowering.__annotations__ = {
        "x": Tensor,
        "y": Optional[Sequence[Tensor]],
        "return": tuple[Tensor, Sequence[Tensor]],
    }
    signature = _schemas.op_signature_from_function(
        lowering, "test", name="Shared", overload="typed", since_version=3
    )
    assert signature.name == "Shared" and signature.overload == "typed"
    assert signature.since_version == 3
    assert signature.outputs[0].type_constraint is signature.params[0].type_constraint
    assert signature.outputs[1].type_constraint is signature.params[1].type_constraint
    assert not signature.params[1].required
    assert signature.params[1].default is None


@pytest.mark.parametrize(
    "annotation,value,attribute_type",
    [
        (float, 1.5, ir.AttributeType.FLOAT),
        (int, 2, ir.AttributeType.INT),
        (bool, True, ir.AttributeType.INT),
        (str, "nearest", ir.AttributeType.STRING),
        (list[int], [1, 2], ir.AttributeType.INTS),
        (Sequence[float], [1.5], ir.AttributeType.FLOATS),
        (tuple[str], ("x",), ir.AttributeType.STRINGS),
    ],
)
def test_attribute_defaults_and_requiredness(annotation, value, attribute_type):
    def lowering(required, optional=value):
        pass

    lowering.__annotations__ = {"required": annotation, "optional": annotation}
    signature = _schemas.op_signature_from_function(lowering, "test")
    required, optional = signature.params
    assert required.type == optional.type == attribute_type
    assert required.required and required.default is None
    assert not optional.required
    assert optional.default.value == (
        tuple(value) if isinstance(value, list) else value
    )
    assert not signature.outputs


def test_unannotated_and_unbounded_inputs_remain_permissive():
    def lowering(x, y=None):
        return x

    lowering.__annotations__ = {"x": TypeVar("AnyTensor")}
    signature = _schemas.op_signature_from_function(lowering, "test")
    assert len(signature.params[0].type_constraint.allowed_types) > len(ir.DataType)
    assert len(signature.params[1].type_constraint.allowed_types) > len(ir.DataType)
    assert signature.params[0].required
    assert not signature.params[1].required
