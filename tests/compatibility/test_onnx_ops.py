"""Public torch.onnx.ops APIs survive native export and serialization."""

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

import torch_onnx


class CustomAffine(OpRun):
    op_domain = "test.torch_onnx"

    def _run(self, x, missing=None, bias=None, **attrs):
        assert missing is None
        return (x * attrs["scale"] + bias,)


class CustomPair(OpRun):
    op_domain = "test.torch_onnx"

    def _run(self, x):
        return x * 2, x.sum(axis=1).astype(np.int64)


@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_symbolic_custom_domain_roundtrip(tmp_path, patched, optimize):
    class Model(torch.nn.Module):
        def forward(self, x):
            if torch.onnx.is_in_onnx_export():
                value = torch.onnx.ops.symbolic(
                    "test.torch_onnx::CustomAffine",
                    (x + 1, None, x),
                    {
                        "scale": 2.0,
                        "label": "affine",
                        "enabled": True,
                        "axes": [0, 1],
                        "weights": [0.25, 0.75],
                        "labels": ["a", "b"],
                        "flags": [True, False],
                    },
                    dtype=x.dtype,
                    shape=x.shape,
                    version=3,
                    metadata_props={"source": "public-symbolic"},
                )
                first, second = torch.onnx.ops.symbolic_multi_out(
                    "test.torch_onnx::CustomPair",
                    (value,),
                    dtypes=(x.dtype, onnx.TensorProto.INT64),
                    shapes=(x.shape, (x.shape[0],)),
                    metadata_props={"source": "public-multi-out"},
                )
                return first.relu(), second
            value = (x + 1) * 2 + x
            return (value * 2).relu(), value.sum(dim=1).to(torch.int64)

    model = Model().eval()
    data = torch.arange(6.0).reshape(2, 3) - 3
    if patched:
        torch_onnx.patch_torch(verbose=False)
    try:
        export = torch.onnx.export if patched else torch_onnx.export
        program = export(
            model,
            (data,),
            opset_version=23,
            optimize=optimize,
            verbose=False,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},),
        )
    finally:
        if patched:
            torch_onnx.unpatch_torch()
    path = tmp_path / "custom.onnx"
    program.save(path)
    proto = onnx.load(path)
    onnx.checker.check_model(proto, full_check=True)
    first_type, second_type = (value.type.tensor_type for value in proto.graph.output)
    assert first_type.elem_type == onnx.TensorProto.FLOAT
    assert second_type.elem_type == onnx.TensorProto.INT64
    assert len(first_type.shape.dim) == 2
    assert len(second_type.shape.dim) == 1
    assert first_type.shape.dim[0].dim_param
    assert first_type.shape.dim[0].dim_param == second_type.shape.dim[0].dim_param
    assert {item.domain: item.version for item in proto.opset_import}[
        "test.torch_onnx"
    ] == 3
    custom = [node for node in proto.graph.node if node.domain == "test.torch_onnx"]
    assert [node.op_type for node in custom] == ["CustomAffine", "CustomPair"]
    assert list(custom[0].input)[1] == ""
    assert {item.key: item.value for item in custom[0].metadata_props}[
        "source"
    ] == "public-symbolic"
    assert {item.key: item.value for item in custom[1].metadata_props}[
        "source"
    ] == "public-multi-out"
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr) for attr in custom[0].attribute
    }
    assert attrs == {
        "scale": 2.0,
        "label": b"affine",
        "enabled": 1,
        "axes": [0, 1],
        "weights": [0.25, 0.75],
        "labels": [b"a", b"b"],
        "flags": [1, 0],
    }
    evaluator = ReferenceEvaluator(proto, new_ops=[CustomAffine, CustomPair])
    for batch in (1, 2, 5):
        x = torch.arange(batch * 3.0).reshape(batch, 3) - 3
        actual = evaluator.run(None, {proto.graph.input[0].name: x.numpy()})
        expected = model(x)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(torch.from_numpy(result), reference)
    program.release()


@pytest.mark.parametrize("operator", ["attention", "rotary_embedding"])
@pytest.mark.parametrize("optimize", [False, True])
def test_native_onnx_ops_preserved_and_execute(operator, optimize):
    class Model(torch.nn.Module):
        def forward(self, x, cos, sin):
            if operator == "attention":
                return torch.onnx.ops.attention(x, x, x, is_causal=True)[0]
            return torch.onnx.ops.rotary_embedding(x, cos, sin)

    model = Model().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 8)
    cos, sin = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
    program = torch_onnx.export(
        model, (x, cos, sin), opset_version=23, optimize=optimize, verbose=False
    )
    try:
        expected_op = "Attention" if operator == "attention" else "RotaryEmbedding"
        assert any(node.op_type == expected_op for node in program.model.graph)
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(x, cos, sin)[0], model(x, cos, sin))
    finally:
        program.release()
