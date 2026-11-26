# Owner(s): ["module: onnx"]
"""Numerical regressions reported by cuiliaomei-beep in PyTorch ONNX issues."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx.reference import ReferenceEvaluator

import torch_onnx


class _Model(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


def _numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(ml_dtypes.bfloat16)
    return tensor.numpy()


def _export_and_compare(
    function, args, *, optimize, opset=18, exact=True, reference=False
):
    model = _Model(function).eval()
    expected_tensor = model(*args)
    program = torch_onnx.export(
        model, args, optimize=optimize, opset_version=opset, verbose=False
    )
    proto = ir.to_proto(program.model)
    onnx.checker.check_model(proto, full_check=True)
    feeds = {
        value.name: _numpy(arg)
        for value, arg in zip(program.model.graph.inputs, args, strict=True)
    }
    if reference or any(arg.dtype == torch.bfloat16 for arg in args):
        actual = ReferenceEvaluator(proto).run(None, feeds)[0]
        expected = _numpy(expected_tensor)
        if expected_tensor.dtype == torch.bfloat16:
            actual = actual.astype(np.float32)
            expected = expected.astype(np.float32)
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        actual = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)[0]
        expected = expected_tensor.numpy()
    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        double = expected.dtype == np.float64
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-14 if double else 2e-6,
            atol=1e-15 if double else 1e-7,
            equal_nan=True,
        )
    if expected.dtype.kind == "f":
        zero = expected == 0
        np.testing.assert_array_equal(
            np.signbit(actual[zero]), np.signbit(expected[zero])
        )
    return program


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("name", ["ceil", "floor", "round", "trunc"])
def test_integer_unary_preserves_all_int64_bits(name, optimize):
    # pytorch/pytorch#206815: a floating-point round trip cannot preserve these.
    x = torch.tensor([-(2**63), -(2**60 + 1), -1, 0, 1, 2**60 + 1, 2**63 - 1])
    program = _export_and_compare(getattr(torch, name), (x,), optimize=optimize)
    assert all(node.op_type != "Cast" for node in program.model.graph)


@pytest.mark.parametrize("optimize", [False, True])
def test_unsigned_negation_wraps_like_pytorch(optimize):
    _export_and_compare(
        torch.neg,
        (torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8),),
        optimize=optimize,
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("name", ["acos", "asin", "atan", "sin", "cos", "tan", "round"])
def test_bfloat16_unary_uses_legal_opset18_types(name, optimize):
    _export_and_compare(
        getattr(torch, name),
        (torch.tensor([-0.5, 0, 0.5], dtype=torch.bfloat16),),
        optimize=optimize,
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_division_does_not_round_operands_to_float(
    rounding_mode, dtype, optimize
):
    # pytorch/pytorch#206285: cover all quotient signs and values beyond float64.
    magnitude = 2**60 + 1 if dtype == torch.int64 else 2**30 + 1
    x = torch.tensor(
        [magnitude, -magnitude, magnitude, -magnitude, magnitude], dtype=dtype
    )
    y = torch.tensor([3, 3, -3, -3, 1], dtype=dtype)
    program = _export_and_compare(
        lambda a, b: torch.div(a, b, rounding_mode=rounding_mode),
        (x, y),
        optimize=optimize,
    )
    for node in program.model.graph:
        if node.op_type == "Cast":
            assert node.attributes["to"].as_int() not in (
                int(ir.DataType.FLOAT),
                int(ir.DataType.DOUBLE),
            )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "kind", ["scalar_add", "scalar_mul", "full", "full_like", "fill", "linspace"]
)
def test_float64_python_constants_are_materialized_without_narrowing(kind, optimize):
    # pytorch/pytorch#206455: only full/fill still needed a fix in this exporter.
    functions = {
        "scalar_add": lambda x: x + 0.001,
        "scalar_mul": lambda x: x * 1.0000000001,
        "full": lambda x: x + torch.full((3,), 0.001, dtype=torch.float64),
        "full_like": lambda x: torch.full_like(x, 0.001),
        "fill": lambda x: x.clone().fill_(0.001),
        "linspace": lambda x: x + torch.linspace(0.001, 0.003, 3, dtype=torch.float64),
    }
    _export_and_compare(
        functions[kind],
        (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),),
        optimize=optimize,
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_atan2_signed_zeros_infinities_and_nan(dtype, optimize):
    # pytorch/pytorch#206626: test every pairing, including both signed zeros.
    values = torch.tensor(
        [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")], dtype=dtype
    )
    y, x = torch.meshgrid(values, values, indexing="ij")
    _export_and_compare(
        torch.atan2,
        (y.contiguous(), x.contiguous()),
        optimize=optimize,
        exact=False,
        reference=dtype == torch.float64,  # ORT CPU has no Atan(double) kernel.
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -4, -1])
def test_squeeze_only_removes_requested_unit_dimension(dim, optimize):
    # pytorch/pytorch#206790.
    x = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3, 1)
    _export_and_compare(lambda value: value.squeeze(dim), (x,), optimize=optimize)


def test_squeeze_symbolic_dimension_preserves_nonunit_inputs():
    graph = ir.Graph([], [], nodes=[], name="dynamic_squeeze")
    op = torch_onnx.TranslationContext(torch_onnx.GraphBuilder(graph))
    value = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["batch", 3])
    )
    graph.inputs.append(value)
    from torch_onnx._torchlib.ops import core

    result = core.aten_squeeze_dim(op, value, 0)
    graph.outputs.append(result)
    evaluator = ReferenceEvaluator(ir.to_proto(ir.Model(graph, ir_version=10)))
    for batch in (1, 2):
        x = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
        (actual,) = evaluator.run(None, {"x": x})
        np.testing.assert_array_equal(actual, torch.from_numpy(x).squeeze(0).numpy())


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "left,right",
    [
        (torch.int8, torch.int64),
        (torch.float16, torch.float32),
        (torch.int32, torch.float32),
        (torch.bool, torch.bool),
    ],
)
def test_stack_mixed_dtypes_remains_supported(left, right, optimize):
    # pytorch/pytorch#206462 is already fixed by native sequence promotion.
    _export_and_compare(
        lambda a, b: torch.stack([a, b]),
        (torch.ones(2, 3, dtype=left), torch.ones(2, 3, dtype=right)),
        optimize=optimize,
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("...ij,...jk->...ik", ((2, 3, 4), (4, 5))),
        ("...ij,...jk->...ik", ((2, 2, 3, 4), (4, 5))),
        ("...ij,...jk->...ik", ((2, 3, 4), (2, 4, 5))),
        ("i...j,i...j->i...", ((2, 3, 4), (2, 4))),
    ],
)
def test_einsum_right_aligns_ellipsis_dimensions(equation, shapes, optimize):
    # pytorch/pytorch#206813: ONNX requires equal-width operand ellipses.
    generator = torch.Generator().manual_seed(5)
    args = tuple(torch.randn(shape, generator=generator) for shape in shapes)
    _export_and_compare(
        lambda a, b: torch.einsum(equation, a, b), args, optimize=optimize, exact=False
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_signbit_preserves_negative_zero_and_nan_at_opset26(dtype, optimize):
    # pytorch/pytorch#206457: comparisons cannot distinguish signed NaNs.
    x = torch.tensor(
        [0.0, -0.0, float("nan"), -float("nan"), float("inf"), -float("inf")],
        dtype=dtype,
    )
    _export_and_compare(torch.signbit, (x,), optimize=optimize, opset=26)


def test_signbit_refuses_lossy_floating_lowering_before_bitcast():
    with pytest.raises(torch_onnx.errors.ConversionError, match="Failed to convert"):
        torch_onnx.export(
            _Model(torch.signbit).eval(),
            (torch.tensor([-0.0]),),
            opset_version=18,
            verbose=False,
        )


@pytest.mark.parametrize("optimize", [False, True])
def test_trunc_preserves_negative_zero(optimize):
    x = torch.tensor(
        [-1.2, -0.2, -0.0, 0.0, 0.2, 1.2, -float("inf"), float("inf"), float("nan")]
    )
    _export_and_compare(torch.trunc, (x,), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
def test_full_without_explicit_dtype_follows_double_default(optimize):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _export_and_compare(
            lambda x: x + torch.full((3,), 0.001),
            (torch.zeros(3, dtype=torch.float64),),
            optimize=optimize,
        )
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signbit_after_trunc_observes_nan_and_zero_sign(dtype, optimize):
    x = torch.tensor([-float("nan"), float("nan"), -0.0, 0.0, -0.2, 0.2], dtype=dtype)
    _export_and_compare(
        lambda value: torch.signbit(torch.trunc(value)),
        (x,),
        optimize=optimize,
        opset=26,
    )


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "equation,shape",
    [
        ("ij->i", (2, 3)),
        ("ij->i", (2, 1)),
        ("ij->i", (2, 0)),
        ("ij->", (2, 3)),
        ("ij->ji", (2, 3)),
        ("ij", (2, 3)),
        ("ii->i", (3, 3)),
        ("ii->", (3, 3)),
        ("ii", (3, 3)),
        ("...i->i", (2, 3)),
        ("...i->...", (2, 3)),
        ("...i->i", (3,)),
        ("...i", (2, 3)),
        ("i...j->ij", (2, 3, 4)),
    ],
)
def test_single_operand_bool_einsum_reduction_dtype(equation, shape, optimize):
    x = torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 3 != 0
    _export_and_compare(lambda a: torch.einsum(equation, a), (x,), optimize=optimize)
