"""Construct constants in their final dtype before any graph optimization."""

import math

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch

import torch_onnx
from torch_onnx._ir_builder import GraphBuilder, TranslationContext

pytestmark = pytest.mark.compatibility


@pytest.mark.parametrize(
    "dtype,value",
    [
        (ir.DataType.DOUBLE, 1.0000000001),
        (ir.DataType.DOUBLE, -0.0),
        (ir.DataType.INT64, 2**60 + 1),
        (ir.DataType.FLOAT16, 0.125),
        (ir.DataType.BFLOAT16, 0.5),
        (ir.DataType.BOOL, True),
        (ir.DataType.DOUBLE, [1.0000000001, -0.0]),
        (ir.DataType.INT64, []),
    ],
)
def test_cast_like_literals_have_no_cast_and_keep_precision(dtype, value):
    graph = ir.Graph([], [], nodes=[], name="typed")
    op = TranslationContext(GraphBuilder(graph))
    like = ir.Value(type=ir.TensorType(dtype), shape=ir.Shape([3, 4]))
    result = op.cast_like(value, like)
    assert result is op.cast_like(value, like)
    assert [node.op_type for node in graph] == ["Constant"]
    assert result.dtype == dtype
    actual = result.const_value.numpy()
    expected = ir.tensor(value, dtype=dtype).numpy()
    np.testing.assert_array_equal(actual, expected)
    if dtype == ir.DataType.DOUBLE:
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert actual.shape == np.asarray(value).shape


def test_cast_like_ir_values_and_unknown_dtype_retain_onnx_cast():
    graph = ir.Graph([], [], nodes=[], name="casts")
    op = TranslationContext(GraphBuilder(graph))
    like = ir.Value(type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape([2]))
    value = ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([7]))
    result = op.cast_like(value, like)
    assert result.producer().op_type == "CastLike"
    assert result.producer().inputs[0] is value
    assert result.dtype == ir.DataType.DOUBLE
    assert result.shape == value.shape
    unknown = ir.Value(name="unknown", shape=ir.Shape([]))
    result = op.cast_like(1.25, unknown)
    assert result.producer().op_type == "CastLike"
    assert result.producer().inputs[1] is unknown


@pytest.mark.parametrize(
    "fn",
    [
        torch.deg2rad,
        torch.rad2deg,
        lambda x: torch.xlogy(1.0000000001, x),
        lambda x: torch.xlogy(x, 1.0000000001),
    ],
)
def test_scalar_constants_export_without_castlike_or_precision_loss(fn):
    class Model(torch.nn.Module):
        def forward(self, x):
            return fn(x)

    x = torch.tensor([0.5, 1.25, math.pi], dtype=torch.float64)
    program = torch_onnx.export(Model().eval(), (x,), optimize=False, verbose=False)
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        assert all(node.op_type != "CastLike" for node in program.model.graph)
        # ORT may approximate Mul by 1.0000000001 as an identity. Verify
        # the exported graph's exact DOUBLE constants with ONNX reference.
        torch.testing.assert_close(
            program.call_reference(x)[0], fn(x), rtol=1e-14, atol=1e-15
        )
    finally:
        program.release()
