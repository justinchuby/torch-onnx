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
        output.dtype = ir.DataType.FLOAT16
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
        output.dtype = ir.DataType.BFLOAT16
        output.shape = ir.Shape([2, 2])
        graph.outputs.append(output)
        return ir.Model(graph, ir_version=10)

    with pytest.raises(TypeError, match="Insert an explicit Cast"):
        build(27)
    onnx.checker.check_model(ir.to_proto(build(28)), full_check=True)


def test_missing_standard_schema_warns_once_and_exports():
    for opset in (30, 50):
        validate_opset_version(opset)
    for invalid in (0, -1, True, 18.5, "29"):
        with pytest.raises(ValueError, match="positive integer"):
            validate_opset_version(invalid)  # type: ignore[arg-type]

    graph = ir.Graph([], [], nodes=[], name="missing_schema")
    builder = torch_onnx.GraphBuilder(graph, opset=30)
    op = torch_onnx.TranslationContext(builder)
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    with pytest.warns(
        UserWarning,
        match="Standard ONNX operator 'NotARealFutureOnnxOperator' at opset 30",
    ):
        default_out = op.NotARealFutureOnnxOperator(x)
    assert isinstance(default_out, ir.Value)
    assert default_out.type is None
    with pytest.raises(
        ValueError,
        match="Cannot determine output arity for standard ONNX operator",
    ):
        builder.call("AnotherUnregisteredFutureOp", x)

    real_get_schema = onnx.defs.get_schema
    scatter_18 = onnx.defs.get_schema("ScatterElements", 18, "")

    def fake_onnx_28_get_schema(name, max_inclusive_version, domain=""):
        if domain == "" and name in {"Expm1", "Log1p", "LogCumSumExp"}:
            raise onnx.defs.SchemaError(f"missing {name}")
        if domain == "" and name == "ScatterElements":
            return scatter_18
        return real_get_schema(name, min(max_inclusive_version, 28), domain)

    class MultiOpset29Model(torch.nn.Module):
        def forward(self, v, idx, src):
            y = torch.expm1(v) + torch.expm1(v * 2.0)
            y = torch.log1p(torch.abs(y))
            y = torch.logcumsumexp(y, 0)
            return y.scatter_reduce(0, idx, src, reduce="mean", include_self=True)

    with (
        mock.patch.object(onnx.defs, "onnx_opset_version", return_value=28),
        mock.patch.object(onnx.defs, "get_schema", side_effect=fake_onnx_28_get_schema),
        pytest.warns(UserWarning) as record,
    ):
        program = torch_onnx.export(
            MultiOpset29Model().eval(),
            (
                torch.tensor([0.2, 0.4], dtype=torch.float64),
                torch.tensor([0, 1]),
                torch.tensor([1.0, 3.0], dtype=torch.float64),
            ),
            opset_version=29,
            optimize=False,
            verbose=False,
        )

    user_warnings = [
        str(w.message) for w in record if issubclass(w.category, UserWarning)
    ]
    missing_warnings = [msg for msg in user_warnings if "Standard ONNX operator" in msg]
    assert len(missing_warnings) == 4
    for expected_op in ("'Expm1'", "'Log1p'", "'LogCumSumExp'", "'ScatterElements'"):
        assert sum(expected_op in msg for msg in missing_warnings) == 1
    assert program.model.opset_imports[""] == 29
    # ExportedProgram metadata populates returned ONNX values without ONNX schema inference.
    assert program.model.graph.outputs[0].dtype == ir.DataType.DOUBLE
    assert tuple(program.model.graph.outputs[0].shape) == (2,)
    op_types = [node.op_type for node in program.model.graph]
    assert op_types.count("Expm1") == 2
    assert op_types.count("Log1p") == 1
    assert op_types.count("LogCumSumExp") == 1
    scatter_nodes = [
        node for node in program.model.graph if node.op_type == "ScatterElements"
    ]
    assert len(scatter_nodes) == 1
    assert scatter_nodes[0].attributes["reduction"].as_string() == "mean"

    # When target opset <= installed ONNX schema ceiling, normal schema validation still raises.
    graph_28 = ir.Graph([], [], nodes=[], name="known_schema_validation")
    builder_28 = torch_onnx.GraphBuilder(graph_28, opset=28)
    with pytest.raises(ValueError, match="ScatterElements reduction='mean' requires"):
        builder_28.call(
            "ScatterElements",
            x,
            ir.Value(
                name="idx", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([2])
            ),
            x,
            axis=0,
            reduction="mean",
            _output_types=[x.type],
        )


def test_future_and_custom_opset_extensibility():
    @torch.library.custom_op("torch_onnx_test::affine_scale", mutates_args=())
    def affine_scale(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0

    @affine_scale.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def custom_affine_35(op, x):
        scaled = op.builder.node(
            "FutureScale",
            [x],
            domain="acme.ops",
            version=3,
            attributes={"factor": 2.0},
            output_types=[x.type],
        )[0]
        return op.Add(scaled, 1.0)

    def custom_expm1_35(op, x):
        return op.FutureStdExpm1(x, approx=0)

    torch_onnx.register_custom_translation(
        torch.ops.torch_onnx_test.affine_scale.default,
        custom_affine_35,
        opset_version=35,
        replace=True,
    )
    torch_onnx.register_custom_translation(
        torch.ops.aten.expm1.default,
        custom_expm1_35,
        opset_version=35,
        replace=True,
    )
    try:

        class CustomFutureModel(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.torch_onnx_test.affine_scale.default(x)
                z = torch.onnx.ops.symbolic(
                    "acme.symbolic::Bias",
                    (y,),
                    attrs={"bias": 0.5},
                    dtype=y.dtype,
                    shape=y.shape,
                    version=2,
                )
                return torch.expm1(z) + torch.expm1(y)

        with pytest.warns(
            UserWarning,
            match="Standard ONNX operator 'FutureStdExpm1' at opset 35",
        ) as record:
            program = torch_onnx.export(
                CustomFutureModel().eval(),
                (torch.randn(2, 3),),
                opset_version=35,
                optimize=False,
                verbose=False,
            )
        assert program.model.opset_imports[""] == 35
        assert program.model.opset_imports["acme.ops"] == 3
        assert program.model.opset_imports["acme.symbolic"] == 2
        assert sum("FutureStdExpm1" in str(w.message) for w in record) == 1
        assert [node.op_type for node in program.model.graph] == [
            "FutureScale",
            "Constant",
            "Add",
            "Bias",
            "FutureStdExpm1",
            "FutureStdExpm1",
            "Add",
        ]

        class StandardFutureOpsetModel(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x) + 2.0

        executable = torch_onnx.export(
            StandardFutureOpsetModel().eval(),
            (torch.tensor([0.0, 0.5, -0.25]),),
            opset_version=35,
            verbose=False,
        )
        assert executable.model.opset_imports[""] == 35
        onnx.checker.check_model(executable.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(executable, backend="reference")

        class PassthroughModel(torch.nn.Module):
            def forward(self, x):
                return x

        passthrough = torch_onnx.export(
            PassthroughModel().eval(),
            (torch.randn(2, 3),),
            opset_version=35,
            verbose=False,
        )
        assert passthrough.model.opset_imports[""] == 35
        onnx.checker.check_model(passthrough.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(passthrough, backend="reference")

        class LegacyTableModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.torch_onnx_test.affine_scale.default(x)

        legacy_table = torch_onnx.export(
            LegacyTableModel().eval(),
            (torch.tensor([1.0, -2.0]),),
            opset_version=35,
            custom_translation_table={
                torch.ops.torch_onnx_test.affine_scale.default: lambda x: x * 2.0 + 1.0
            },
            verbose=False,
        )
        assert legacy_table.model.opset_imports[""] == 35
        onnx.checker.check_model(legacy_table.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(legacy_table, backend="reference")
    finally:
        torch_onnx.unregister_custom_translation(
            torch.ops.aten.expm1.default, opset_version=35
        )
        torch_onnx.unregister_custom_translation(
            torch.ops.torch_onnx_test.affine_scale.default, opset_version=35
        )


@pytest.mark.parametrize("conversion_error", [False, True])
def test_future_local_function_opset_alignment(conversion_error):
    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    def translate(op: torch_onnx.TranslationContext, x: ir.Value) -> ir.Value:
        opset = 20 if conversion_error else op.builder.opset
        formal = ir.Value(name="x", type=x.type, shape=x.shape)
        body = ir.Graph(
            [formal], [], nodes=[], name="future_negate", opset_imports={"": opset}
        )
        builder = torch_onnx.GraphBuilder(body, opset=opset)
        (negated,) = builder.call("Neg", formal)
        negated.type = formal.type
        body.outputs.append(negated)
        function = ir.Function("acme.future", "Negate", "", graph=body, attributes=[])
        return op.builder.call_function(function, x, version=1)[0]

    def export():
        target = torch.ops.aten.neg.default
        torch_onnx.register_custom_translation(target, translate, opset_version=35)
        try:
            return torch_onnx.export(
                Model().eval(),
                (torch.randn(3),),
                opset_version=35,
                optimize=False,
                verbose=False,
            )
        finally:
            torch_onnx.unregister_custom_translation(target, opset_version=35)

    if conversion_error:
        with (
            mock.patch(
                "onnxscript.version_converter.convert_version",
                side_effect=ValueError("unsupported future conversion"),
            ),
            pytest.raises(
                torch_onnx.errors.ConversionError,
                match="Construct the function using the active translation builder",
            ),
        ):
            export()
    else:
        program = export()
        assert program.model.opset_imports[""] == 35
        assert next(iter(program.model.functions.values())).opset_imports[""] == 35
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(program, backend="reference")


@pytest.mark.opset29
@pytest.mark.parametrize(
    "operator", ["expm1", "special.expm1", "log1p", "special.log1p", "logcumsumexp"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimize", [False, True])
def test_29_precision_preserving_operators(operator, dtype, optimize):
    class Model(torch.nn.Module):
        def forward(self, x):
            if operator == "logcumsumexp":
                return torch.logcumsumexp(x, -1)
            if operator.startswith("special."):
                return getattr(torch.special, operator.removeprefix("special."))(x)
            return getattr(torch, operator)(x)

    x = torch.tensor([-1e-8, -0.0, 0.0, 1e-8, 0.25], dtype=dtype)
    if operator == "logcumsumexp":
        x = torch.tensor([-10000.0, 10000.0, -10000.0], dtype=dtype)
    program = torch_onnx.export(
        Model().eval(), (x,), opset_version=29, optimize=optimize, verbose=False
    )
    expected_op = {"expm1": "Expm1", "log1p": "Log1p", "logcumsumexp": "LogCumSumExp"}[
        operator.rsplit(".", 1)[-1]
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
def test_29_roi_pool_schema():
    import torchvision

    class Model(torch.nn.Module):
        def forward(self, x, rois):
            return torchvision.ops.roi_pool(x, rois, (3, 2), 1.0)

    program = torch_onnx.export(
        Model(),
        (torch.randn(2, 3, 6, 7), torch.tensor([[0.0, 0, 0, 6, 5]])),
        opset_version=29,
        verbose=False,
    )
    assert program.model.opset_imports[""] == 29
    assert any(node.op_type == "MaxRoiPool" for node in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)


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
