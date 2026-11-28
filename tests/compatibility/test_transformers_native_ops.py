"""Native coverage for fixes originally supplied by Transformers' exporter."""

import onnx
import onnx_ir as ir
import pytest
import torch

import torch_onnx

pytestmark = pytest.mark.compatibility


def check(
    model, args, *, opset=23, dynamic_shapes=None, extra=(), backend="onnxruntime"
):
    model = model.eval()
    program = torch_onnx.export(
        model, args, opset_version=opset, dynamic_shapes=dynamic_shapes, verbose=False
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        for data in (args, *extra):
            with torch.no_grad():
                expected = model(*data)
            actual = (
                program(*data)
                if backend == "onnxruntime"
                else program.call_reference(*data)
            )
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            expected = tuple(
                torch.view_as_real(x) if x.is_complex() else x for x in expected
            )
            torch.testing.assert_close(tuple(actual), expected, equal_nan=True)
        return program.model_proto
    finally:
        program.release()


@pytest.mark.parametrize("descending", [False, True])
def test_stable_sort_preserves_tie_indices(descending):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sort(x, stable=True, descending=descending)

    check(Model(), (torch.tensor([2.0, 1.0, 2.0, 1.0, 2.0]),))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_exp_and_unsqueeze(dtype):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.exp().unsqueeze(-2)

    x = torch.tensor([1 + 2j, -1 - 0.5j, 0j], dtype=dtype)
    check(Model(), (x,))


@pytest.mark.parametrize("value", [False, True])
def test_boolean_masked_fill(value):
    class Model(torch.nn.Module):
        def forward(self, x, mask):
            return x.masked_fill(mask, value)

    check(
        Model(),
        (torch.tensor([[True, False, True]]), torch.tensor([[True, True, False]])),
    )


def test_dynamic_boolean_masked_fill():
    class Model(torch.nn.Module):
        def forward(self, x, mask, value):
            return x.masked_fill(mask, value)

    check(
        Model(),
        (torch.tensor([True, False]), torch.tensor([True, True]), torch.tensor(False)),
        extra=[
            (
                torch.tensor([True, False]),
                torch.tensor([False, True]),
                torch.tensor(True),
            )
        ],
    )


@pytest.mark.parametrize("empty_source", [False, True])
def test_masked_scatter_broadcast_and_empty_source(empty_source):
    class Model(torch.nn.Module):
        def forward(self, x, mask, source):
            return x.masked_scatter(mask, source)

    source = torch.empty(0) if empty_source else torch.arange(20.0)
    mask = (
        torch.zeros(2, 3, 1, dtype=torch.bool)
        if empty_source
        else torch.tensor([[[True], [False], [True]], [[False], [True], [False]]])
    )
    check(Model(), (torch.ones(2, 3, 4), mask, source))


@pytest.mark.parametrize("operation", ["mean", "var"])
def test_masked_reductions_ignore_masked_nan(operation):
    class Model(torch.nn.Module):
        def forward(self, x, mask):
            return getattr(torch.masked, operation)(x, dim=-1, keepdim=True, mask=mask)

    check(
        Model(),
        (
            torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, 6.0]]),
            torch.tensor([[True, False, True], [True, True, False]]),
        ),
    )


@pytest.mark.parametrize("dim", [None, 0, -1])
def test_repeat_interleave_symbolic_repeats_and_empty_input(dim):
    class Model(torch.nn.Module):
        def forward(self, x, repeats):
            return x.repeat_interleave(repeats.shape[0], dim=dim)

    rows = torch.export.Dim("rows", min=0, max=8)
    repetitions = torch.export.Dim("repetitions", min=1, max=8)
    check(
        Model(),
        (torch.arange(6.0).reshape(2, 3), torch.ones(2)),
        dynamic_shapes=({0: rows}, {0: repetitions}),
        extra=[
            (torch.empty(0, 3), torch.ones(3)),
            (torch.arange(9.0).reshape(3, 3), torch.ones(1)),
        ],
    )


@pytest.mark.parametrize("operation", ["chunk", "split"])
def test_dynamic_partition_has_no_sequence_ops(operation):
    class Model(torch.nn.Module):
        def forward(self, x):
            if operation == "chunk":
                return x.chunk(3, dim=1)
            return x.split(x.shape[1] // 3, dim=1)

    columns = 3 * torch.export.Dim("pieces", min=1, max=8)
    proto = check(
        Model(),
        (torch.arange(18.0).reshape(2, 9),),
        dynamic_shapes=({1: columns},),
        extra=[(torch.arange(30.0).reshape(2, 15),)],
    )
    assert not any("Sequence" in node.op_type for node in proto.graph.node)


@pytest.mark.parametrize("n", [0, 1, 31])
@pytest.mark.parametrize("opset", [20, 23])
def test_randperm_is_a_permutation(n, opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.randperm(x.shape[0], dtype=torch.int64)

    program = torch_onnx.export(
        Model().eval(), (torch.ones(n),), opset_version=opset, verbose=False
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        (actual,) = (
            program(torch.ones(n))
            if opset == 20
            else program.call_reference(torch.ones(n))
        )
        torch.testing.assert_close(actual.sort().values, torch.arange(n))
    finally:
        program.release()


def test_bf16_conv_is_not_forced_to_float32():
    model = torch.nn.Conv1d(2, 3, 3).eval().to(torch.bfloat16)
    program = torch_onnx.export(
        model,
        (torch.ones(1, 2, 8, dtype=torch.bfloat16),),
        opset_version=28,
        verbose=False,
    )
    onnx.checker.check_model(program.model_proto, full_check=True)
    conv = next(node for node in program.model.graph if node.op_type == "Conv")
    assert conv.inputs[0].dtype == ir.DataType.BFLOAT16
    assert conv.inputs[1].dtype == ir.DataType.BFLOAT16
    assert not any(node.op_type == "Cast" for node in program.model.graph)


def test_mm_dtype_casts_before_accumulation():
    from torch_onnx._torchlib.ops.linalg import aten_mm_dtype

    x, y = (
        torch.randn(2, 4, dtype=torch.float16),
        torch.randn(4, 3, dtype=torch.float16),
    )
    values = [
        ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.FLOAT16),
            shape=ir.Shape(tensor.shape),
        )
        for name, tensor in zip(("x", "y"), (x, y), strict=True)
    ]
    graph = ir.Graph(values, [], nodes=[], name="mm_dtype", opset_imports={"": 23})
    builder = torch_onnx.GraphBuilder(graph, opset=23)
    graph.outputs.append(
        aten_mm_dtype(
            torch_onnx.TranslationContext(builder), *values, out_dtype=ir.DataType.FLOAT
        )
    )
    program = torch_onnx.ONNXProgram(ir.Model(graph, ir_version=10), None)
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(x, y)[0], x.float() @ y.float())
    finally:
        program.release()


@pytest.mark.parametrize("rows", [0, 5])
def test_transformers_grouped_mm_alias_bias_dtype_and_empty_group(rows):
    def implementation(a, b, offs, bias=None, out_dtype=None):
        a, b = a.to(out_dtype or a.dtype), b.to(out_dtype or b.dtype)
        start, outputs = 0, []
        for group, end in enumerate(offs.tolist()):
            output = a[start:end] @ b[group]
            if bias is not None:
                output = output + bias[group].to(output.dtype)
            outputs.append(output)
            start = end
        return torch.cat(outputs)

    args = (
        torch.randn(rows, 4, dtype=torch.float16),
        torch.randn(3, 4, 2, dtype=torch.float16),
        torch.tensor([0, 2, 5] if rows else [0, 0, 0]),
        torch.randn(3, 2, dtype=torch.float16),
    )
    values = [
        ir.Value(
            name=name,
            type=ir.TensorType(
                ir.DataType.INT64 if value.dtype == torch.int64 else ir.DataType.FLOAT16
            ),
            shape=ir.Shape(value.shape),
        )
        for name, value in zip(("a", "b", "offs", "bias"), args, strict=True)
    ]
    graph = ir.Graph(values, [], nodes=[], name="grouped_mm", opset_imports={"": 23})
    builder = torch_onnx.GraphBuilder(graph, opset=23)
    registry = torch_onnx.ONNXRegistry.from_torchlib(23)
    translation = registry.get_decomps("transformers::grouped_mm_fallback")[
        0
    ].onnx_function
    graph.outputs.append(
        translation(
            torch_onnx.TranslationContext(builder), *values, out_dtype=ir.DataType.FLOAT
        )
    )
    program = torch_onnx.ONNXProgram(ir.Model(graph, ir_version=10), None)
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(
            program(*args)[0], implementation(*args, out_dtype=torch.float32)
        )
    finally:
        program.release()


def test_symbolic_floor_division_crosses_zero():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + (x.shape[0] - 5) // 2

    rows = torch.export.Dim("rows", min=2, max=10)
    check(
        Model(),
        (torch.randn(2, 3),),
        dynamic_shapes=({0: rows},),
        extra=[(torch.randn(3, 3),), (torch.randn(7, 3),)],
    )


@pytest.mark.parametrize("shape", [(8,), (4, 8)])
@pytest.mark.parametrize("eps", [None, 1e-5])
def test_nonaffine_rms_norm_preserves_normalized_shape(shape, eps):
    model = torch.nn.RMSNorm(shape, elementwise_affine=False, eps=eps)
    proto = check(model, (torch.randn(2, 3, 4, 8),))
    rms = next(node for node in proto.graph.node if node.op_type == "RMSNormalization")
    assert next(attr.i for attr in rms.attribute if attr.name == "axis") == -len(shape)


@pytest.mark.parametrize("opset", [18, 23])
@pytest.mark.parametrize("kind", ["bool", "negative_infinity", "finite_minimum"])
def test_sdpa_fully_masked_rows_preserve_torch_semantics(opset, kind):
    class Model(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )

    torch.manual_seed(0)
    q, k, v = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4)
    mask = torch.tensor([[False] * 5, [True, False, True, False, True], [True] * 5])
    if kind != "bool":
        bound = (
            -float("inf")
            if kind == "negative_infinity"
            else torch.finfo(torch.float32).min
        )
        mask = torch.where(mask, 0.0, bound)
    # ORT's Attention-23 kernel zeros finite-minimum rows, unlike the ONNX
    # additive-mask formula and PyTorch. Verify that graph with ONNX reference;
    # do not absorb Transformers' patch that also zeros these valid finite rows.
    backend = "reference" if kind == "finite_minimum" and opset == 23 else "onnxruntime"
    check(Model(), (q, k, v, mask), opset=opset, backend=backend)


@pytest.mark.parametrize("kind", ["mixed_tensor_dtypes", "integer_tensor_float_scalar"])
def test_where_uses_torch_promotion_instead_of_branch_dtype(kind):
    class Model(torch.nn.Module):
        def forward(self, x, mask, y):
            if kind == "integer_tensor_float_scalar":
                return torch.where(mask, x, 0.5)
            return torch.where(mask, x, y)

    x = (
        torch.tensor([1, 2])
        if kind == "integer_tensor_float_scalar"
        else torch.tensor([1.0, 2.0])
    )
    check(
        Model(),
        (x, torch.tensor([True, False]), torch.tensor([3.0, 4.0], dtype=torch.float64)),
    )


@pytest.mark.parametrize("operation", ["mean", "var"])
def test_masked_reduction_empty_mask_retains_nan(operation):
    class Model(torch.nn.Module):
        def forward(self, x, mask):
            return getattr(torch.masked, operation)(x, dim=-1, mask=mask)

    check(Model(), (torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.bool)))


def test_masked_scatter_caps_unused_source_and_rejects_short_source():
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument

    class Model(torch.nn.Module):
        def forward(self, x, mask, source):
            return x.masked_scatter(mask, source)

    source_size = torch.export.Dim("source_size", min=0, max=2048)
    model = Model().eval()
    args = (torch.ones(4), torch.tensor([True, True, True, True]), torch.arange(8.0))
    program = torch_onnx.export(
        model,
        args,
        opset_version=23,
        dynamic_shapes=({}, {}, {0: source_size}),
        verbose=False,
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(
            program(args[0], args[1], torch.arange(1024.0))[0], torch.arange(4.0)
        )
        torch.testing.assert_close(
            program(args[0], torch.zeros(4, dtype=torch.bool), torch.empty(0))[0],
            args[0],
        )
        with pytest.raises((Fail, InvalidArgument), match="bounds"):
            program(args[0], args[1], torch.ones(3))
    finally:
        program.release()


def test_boolean_row_updates_broadcast_and_empty_rows_without_scatter():
    class Model(torch.nn.Module):
        def forward(self, x, mask, values):
            return torch.index_put(x, (mask,), values)

    updates = torch.export.Dim("updates", min=0, max=16)
    x = torch.arange(12.0).reshape(4, 3)
    initial = (
        x,
        torch.tensor([True, False, True, False]),
        torch.tensor([[1.0], [2.0]]),
    )
    proto = check(
        Model(),
        initial,
        dynamic_shapes=({}, {}, {0: updates}),
        extra=[
            (x, torch.ones(4, dtype=torch.bool), torch.tensor([[3.0]])),
            (x, torch.zeros(4, dtype=torch.bool), torch.empty(0, 1)),
        ],
    )
    assert not any(
        node.op_type in {"NonZero", "ScatterND"} for node in proto.graph.node
    )


@pytest.mark.parametrize("mask_rank", [1, 2])
def test_boolean_prefix_assignment_preserves_payload_broadcast(mask_rank):
    class Model(torch.nn.Module):
        def forward(self, x, mask, values):
            return torch.index_put(x, (mask,), values)

    x = torch.arange(24.0).reshape(2, 3, 4)
    mask = (
        torch.tensor(True),
        torch.tensor([True, False]),
        torch.tensor([[True, False, False], [False, True, False]]),
    )[mask_rank]
    check(Model(), (x, mask, torch.tensor([1.0, 2.0, 3.0, 4.0])))


def test_empty_boolean_prefix_does_not_allocate_a_dummy_payload():
    class Model(torch.nn.Module):
        def forward(self, x, mask, values):
            return torch.index_put(x, (mask,), values)

    rows = torch.export.Dim("rows", min=0, max=8)
    updates = torch.export.Dim("updates", min=0, max=8)
    check(
        Model(),
        (torch.ones(2, 4), torch.tensor([True, True]), torch.randn(2, 4)),
        dynamic_shapes=({0: rows}, {0: rows}, {0: updates}),
        extra=[
            (torch.empty(0, 4), torch.empty(0, dtype=torch.bool), torch.empty(0, 4))
        ],
    )


def test_boolean_prefix_with_empty_payload():
    class Model(torch.nn.Module):
        def forward(self, x, mask, values):
            return torch.index_put(x, (mask,), values)

    check(
        Model(),
        (torch.empty(3, 0), torch.tensor([True, False, True]), torch.empty(2, 0)),
    )


def test_irfft_retains_standard_inverse_onesided_dft():
    """ORT's kernel limitation must not change the standard ONNX lowering."""

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.fft.irfft(x, n=8)

    program = torch_onnx.export(
        Model(),
        (torch.randn(2, 5, dtype=torch.complex64),),
        opset_version=23,
        verbose=False,
    )
    try:
        proto = program.model_proto
        onnx.checker.check_model(proto, full_check=True)
        dfts = [node for node in proto.graph.node if node.op_type == "DFT"]
        assert len(dfts) == 1
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in dfts[0].attribute
        }
        assert attributes["inverse"] == 1
        assert attributes["onesided"] == 1
    finally:
        program.release()


def test_int16_right_shift_keeps_spec_supported_where_dtype():
    class Model(torch.nn.Module):
        def forward(self, x, shifts):
            return torch.bitwise_right_shift(x, shifts)

    proto = check(
        Model(),
        (
            torch.tensor([-32768, -1, 32767, 7], dtype=torch.int16),
            torch.tensor([0, 1, 8, 2], dtype=torch.int16),
        ),
        backend="reference",
    )
    assert any(node.op_type == "Where" for node in proto.graph.node)
    assert not any(
        node.op_type == "Cast"
        and any(
            attr.name == "to" and attr.i == onnx.TensorProto.INT32
            for attr in node.attribute
        )
        for node in proto.graph.node
    )
