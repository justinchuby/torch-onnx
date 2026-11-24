"""Schema, dispatch and execution coverage for released and development opsets."""

from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch

import torch_onnx
from torch_onnx._ir_builder import validate_opset_version


@pytest.mark.parametrize("opset", range(24, 29))
def test_swish_export_and_execution(opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    program = torch_onnx.export(
        Model().eval(),
        (torch.randn(2, 3),),
        opset_version=opset,
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        verbose=False,
    )
    assert type(program) is torch.onnx.ONNXProgram
    assert program.model.opset_imports[""] == opset
    assert [node.op_type for node in program.model.graph] == ["Swish"]
    onnx.checker.check_model(program.model_proto, full_check=True)
    # ORT 1.32's function expansion fails for Swish above opset 24.
    backend = "onnxruntime" if opset == 24 else "reference"
    torch_onnx.testing.assert_onnx_program(program, backend=backend)
    torch_onnx.testing.assert_onnx_program(
        program, args=(torch.randn(5, 3),), kwargs={}, backend=backend
    )


@pytest.mark.parametrize("opset", [18, 23])
def test_swish_preserves_older_opset_lowering(opset):
    program = torch_onnx.export(
        torch.nn.SiLU().eval(), (torch.randn(3),), opset_version=opset, verbose=False
    )
    assert {node.op_type for node in program.model.graph} == {"Sigmoid", "Mul"}
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.parametrize("opset", range(24, 29))
@pytest.mark.parametrize("masked", [False, True])
def test_attention_schema_and_execution(opset, masked):
    class Model(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask if masked else None
            )

    q, k, v = (torch.randn(2, 3, 4, 8) for _ in range(3))
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:, :2] = True
    program = torch_onnx.export(
        Model().eval(), (q, k, v, mask), opset_version=opset, verbose=False
    )
    assert any(node.op_type == "Attention" for node in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.parametrize("opset", [26, 27, 28])
@pytest.mark.parametrize(
    ("x", "dim", "dtype"),
    [
        (torch.tensor([[2.0, 0.0, -3.0], [1.0, 2.0, 3.0]]), -1, None),
        (torch.tensor([2, 3, 4], dtype=torch.int32), 0, None),
        (torch.tensor([2, 3, 4], dtype=torch.int32), 0, torch.float64),
        (torch.tensor(3.0), -1, None),
        (torch.empty(2, 0), 1, None),
    ],
)
def test_cumprod_native_operator(opset, x, dim, dtype):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cumprod(x, dim, dtype=dtype)

    program = torch_onnx.export(
        Model().eval(), (x,), opset_version=opset, verbose=False
    )
    if x.ndim > 0:
        assert any(node.op_type == "CumProd" for node in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.parametrize("opset", [26, 27, 28])
def test_equal_width_bitcast(opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.view(torch.int32)

    x = torch.tensor([0.0, -0.0, float("inf"), -2.5])
    program = torch_onnx.export(
        Model().eval(), (x,), opset_version=opset, verbose=False
    )
    assert [node.op_type for node in program.model.graph] == ["BitCast"]
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


def test_range_27_new_half_schema():
    def build(opset):
        inputs = [
            ir.Value(
                name=name, type=ir.TensorType(ir.DataType.FLOAT16), shape=ir.Shape([])
            )
            for name in ("start", "limit", "delta")
        ]
        graph = ir.Graph(inputs, [], nodes=[], name="half_range")
        builder = torch_onnx.GraphBuilder(graph, opset=opset)
        (output,) = builder.call("Range", *inputs)
        output.shape = ir.Shape([None])
        graph.outputs.append(output)
        return ir.Model(graph, ir_version=10)

    with pytest.raises(TypeError, match="Insert an explicit Cast"):
        build(26)
    model = build(27)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    from onnx.reference import ReferenceEvaluator

    (result,) = ReferenceEvaluator(ir.to_proto(model)).run(
        None,
        {
            name: np.array(value, dtype=np.float16)
            for name, value in (("start", 0), ("limit", 3), ("delta", 1))
        },
    )
    np.testing.assert_array_equal(result, np.arange(3, dtype=np.float16))


def test_einsum_28_bfloat16_schema():
    def build(opset):
        inputs = [
            ir.Value(
                name=name,
                type=ir.TensorType(ir.DataType.BFLOAT16),
                shape=ir.Shape([2, 2]),
            )
            for name in ("x", "y")
        ]
        graph = ir.Graph(inputs, [], nodes=[], name="bfloat16_einsum")
        (output,) = torch_onnx.GraphBuilder(graph, opset=opset).call(
            "Einsum", *inputs, equation="ij,jk->ik"
        )
        output.shape = ir.Shape([2, 2])
        graph.outputs.append(output)
        return ir.Model(graph, ir_version=10)

    with pytest.raises(TypeError, match="Insert an explicit Cast"):
        build(27)
    onnx.checker.check_model(ir.to_proto(build(28)), full_check=True)


def test_unavailable_schema_target_is_actionable():
    with (
        mock.patch.object(onnx.defs, "onnx_opset_version", return_value=28),
        pytest.raises(ValueError, match="current ONNX main"),
    ):
        validate_opset_version(29)
    with pytest.raises(ValueError, match="supports through 29"):
        validate_opset_version(30)
    with (
        mock.patch.object(onnx.defs, "onnx_opset_version", return_value=29),
        mock.patch.object(
            onnx.defs, "get_schema", side_effect=onnx.defs.SchemaError("missing")
        ),
        pytest.raises(ValueError, match="older weekly builds have incomplete"),
    ):
        validate_opset_version(29)


@pytest.mark.opset29
@pytest.mark.parametrize("operator", ["expm1", "log1p", "logcumsumexp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimize", [False, True])
def test_29_precision_preserving_operators(operator, dtype, optimize):
    class Model(torch.nn.Module):
        def forward(self, x):
            if operator == "logcumsumexp":
                return torch.logcumsumexp(x, -1)
            return getattr(torch, operator)(x)

    x = torch.tensor([-1e-8, -0.0, 0.0, 1e-8, 0.25], dtype=dtype)
    if operator == "logcumsumexp":
        x = torch.tensor([-10000.0, 10000.0, -10000.0], dtype=dtype)
    program = torch_onnx.export(
        Model().eval(), (x,), opset_version=29, optimize=optimize, verbose=False
    )
    expected_op = {"expm1": "Expm1", "log1p": "Log1p", "logcumsumexp": "LogCumSumExp"}[
        operator
    ]
    assert any(node.op_type == expected_op for node in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(
        program,
        backend="reference",
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
        atol=0,
    )


@pytest.mark.opset29
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor(2.0),
        torch.empty(2, 0),
        torch.tensor([-float("inf"), -float("inf"), 0.0, float("inf")]),
        torch.tensor([0.0, float("nan"), 1.0]),
    ],
)
def test_29_logcumsumexp_edge_cases(x):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.logcumsumexp(x, -1)

    program = torch_onnx.export(
        Model().eval(), (x,), opset_version=29, optimize=False, verbose=False
    )
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, backend="reference")


@pytest.mark.opset29
def test_29_scatter_mean_uses_native_reduction():
    class Model(torch.nn.Module):
        def forward(self, x, index, src):
            return x.scatter_reduce(0, index, src, reduce="mean", include_self=True)

    program = torch_onnx.export(
        Model().eval(),
        (
            torch.tensor([2.0, 4.0, -3.0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([4.0, -1.0, 8.0]),
        ),
        opset_version=29,
        optimize=False,
        verbose=False,
    )
    scatter = [n for n in program.model.graph if n.op_type == "ScatterElements"]
    assert len(scatter) == 1
    assert scatter[0].attributes["reduction"].as_string() == "mean"
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, backend="reference")


@pytest.mark.opset29
def test_29_dynamic_conditional_roundtrip(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, pred):
            return torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

    program = torch_onnx.export(
        Model().eval(),
        (torch.randn(2, 3), torch.tensor(True)),
        dynamic_shapes=({0: torch.export.Dim("batch")}, None),
        opset_version=29,
        verbose=False,
    )
    assert any(node.op_type == "If" for node in program.model.graph)
    path = tmp_path / "conditional.onnx"
    program.save(path, external_data=False)
    reloaded = torch.onnx.ONNXProgram(ir.load(path), program.exported_program)
    onnx.checker.check_model(reloaded.model_proto, full_check=True)
    for batch, pred in ((2, False), (5, True)):
        torch_onnx.testing.assert_onnx_program(
            reloaded,
            args=(torch.randn(batch, 3), torch.tensor(pred)),
            kwargs={},
            strategy=None,
            backend="reference",
        )
