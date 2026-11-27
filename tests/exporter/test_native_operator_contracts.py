"""Check native translations directly when torch.export decomposes their overloads."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from torch_onnx import _ir_builder
from torch_onnx._torchlib.ops import core, factories, linalg, neural


def evaluate(translation, *tensors, **kwargs):
    graph = ir.Graph([], [], nodes=[], name="native_contract", opset_imports={"": 20})
    builder = _ir_builder.GraphBuilder(graph, opset=20)
    context = _ir_builder.TranslationContext(builder)
    values = [
        ir.Value(
            name=f"input_{i}",
            shape=ir.Shape(t.shape),
            type=ir.TensorType(ir.DataType.from_numpy(t.numpy().dtype)),
        )
        for i, t in enumerate(tensors)
    ]
    graph.inputs.extend(values)
    outputs = translation(context, *values, **kwargs)
    graph.outputs.extend([outputs] if isinstance(outputs, ir.Value) else outputs)
    model = ir.Model(graph, ir_version=10)
    model.functions.update(builder.functions)
    proto = onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    return [
        torch.from_numpy(x)
        for x in ReferenceEvaluator(proto).run(
            None, {v.name: t.numpy() for v, t in zip(values, tensors, strict=True)}
        )
    ]


@pytest.mark.parametrize(
    "shape,start,end",
    [
        ((), 0, -1),
        ((4,), 0, -1),
        ((2, 3, 4), 1, -1),
        ((2, 3, 4), 0, -2),
        ((2, 3, 4), -2, -1),
        ((2, 3, 4, 5), 1, 2),
        ((2, 3, 4), 0, -1),
    ],
)
def test_flatten_overload_preserves_scalar_and_partial_dimensions(shape, start, end):
    x = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    (actual,) = evaluate(core.aten_flatten, x, start_dim=start, end_dim=end)
    torch.testing.assert_close(actual, torch.flatten(x, start, end))


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_addr_ignores_nan_input_when_beta_is_zero(beta):
    x = torch.full((2, 3), float("nan")) if beta == 0 else torch.randn(2, 3)
    a, b = torch.randn(2), torch.randn(3)
    (actual,) = evaluate(linalg.aten_addr, x, a, b, alpha=2.0, beta=beta)
    torch.testing.assert_close(actual, torch.addr(x, a, b, alpha=2.0, beta=beta))


def test_complex_reciprocal_preserves_both_components():
    x = torch.tensor([1 + 2j, -3 + 0.5j, 2 - 4j])
    (actual,) = evaluate(core.aten_reciprocal_complex, torch.view_as_real(x))
    torch.testing.assert_close(actual, torch.view_as_real(torch.reciprocal(x)))


@pytest.mark.parametrize(
    "dtype", [None, int(ir.DataType.INT32), int(ir.DataType.FLOAT)]
)
def test_randint_like_preserves_shape_and_requested_dtype(dtype):
    x = torch.ones(2, 3, dtype=torch.int64)
    (actual,) = evaluate(factories.aten_randint_like, x, high=1, dtype=dtype)
    expected_dtype = {
        None: torch.int64,
        int(ir.DataType.INT32): torch.int32,
        int(ir.DataType.FLOAT): torch.float32,
    }[dtype]
    assert actual.dtype == expected_dtype
    torch.testing.assert_close(actual, torch.zeros_like(x, dtype=expected_dtype))


@pytest.mark.parametrize("dtype", [None, -1, int(ir.DataType.DOUBLE)])
def test_zero_variance_normal_is_exact_and_keeps_dtype(dtype):
    def translate(op, size):
        return factories.aten_normal_float_float(op, 2.5, 0.0, size, dtype=dtype)

    (actual,) = evaluate(translate, torch.tensor([2, 3], dtype=torch.int64))
    expected_dtype = (
        torch.float64 if dtype == int(ir.DataType.DOUBLE) else torch.float32
    )
    torch.testing.assert_close(actual, torch.full((2, 3), 2.5, dtype=expected_dtype))


@pytest.mark.parametrize("align_corners", [False, True])
def test_trilinear_explicit_output_size_matches_pytorch(align_corners):
    x = torch.randn(1, 2, 2, 3, 4)

    def translate(op, value, size):
        return neural.aten_upsample_trilinear3d(op, value, size, align_corners)

    (actual,) = evaluate(translate, x, torch.tensor([3, 4, 5]))
    expected = torch.nn.functional.interpolate(
        x, size=(3, 4, 5), mode="trilinear", align_corners=align_corners
    )
    torch.testing.assert_close(actual, expected)
