"""Regression for pytorch/pytorch#206882: SDPA accepts non-4D inputs."""

import onnx
import onnx_ir as ir
import pytest
import torch
import torch.nn.functional as F

import torch_onnx

pytestmark = pytest.mark.compatibility


class Attention(torch.nn.Module):
    def __init__(self, *, causal=False, scale=None, gqa=False, dropout=0.0):
        super().__init__()
        self.causal, self.scale, self.gqa, self.dropout = causal, scale, gqa, dropout

    def forward(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            self.dropout,
            self.causal,
            scale=self.scale,
            enable_gqa=self.gqa,
        )


def _inputs(prefix=(2,), *, q_length=4, kv_length=6, dtype=torch.float32):
    generator = torch.Generator().manual_seed(206882)
    return tuple(
        torch.randn(*prefix, length, width, generator=generator, dtype=dtype)
        for length, width in ((q_length, 8), (kv_length, 8), (kv_length, 5))
    )


def _check(model, args, *, opset, optimize=True):
    model.eval()
    expected = model(*args)
    program = torch_onnx.export(
        model, args, opset_version=opset, optimize=optimize, verbose=False
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        (actual,) = program(*args)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-3 if expected.dtype == torch.float16 else 2e-4,
            atol=2e-3 if expected.dtype == torch.float16 else 2e-5,
        )
        assert not any("Sequence" in node.op_type for node in program.model.graph)
    finally:
        program.release()
    return program


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("self_attention", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_issue_206882_3d_attention(opset, self_attention, optimize):
    # Keep the issue's exact batch-one fixture as a regression.
    q, k, _ = _inputs((1,))
    v = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(206883))
    args = (q, q, q) if self_attention else (q, k, v)
    _check(Attention(), args, opset=opset, optimize=optimize)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize(
    "kind", ["bool_2d", "bool_3d", "float_2d", "float_3d", "causal", "scale", "dropout"]
)
def test_3d_attention_masks_and_options(opset, kind):
    args = _inputs()
    options = {
        "causal": kind == "causal",
        "scale": 0.3 if kind == "scale" else None,
        "dropout": 1.0 if kind == "dropout" else 0.0,
    }
    if "2d" in kind or "3d" in kind:
        mask = torch.ones((2, 4, 6) if "3d" in kind else (4, 6), dtype=torch.bool)
        mask[..., 0, :] = False  # Fully masked rows must be zero, not NaN/uniform.
        mask[..., -1] = False
        if "float" in kind:
            mask = torch.where(mask, 0.0, float("-inf"))
        args = (*args, mask)
    _check(Attention(**options), args, opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("prefix", [(), (2, 3, 2)])
def test_other_valid_ranks(opset, prefix):
    _check(Attention(), _inputs(prefix), opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize(
    "shape_case", ["singleton_query", "singleton_key", "mixed_ranks", "batch_prefix"]
)
def test_non_4d_broadcasting(opset, shape_case):
    if shape_case == "singleton_query":
        q, _, _ = _inputs((1,))
        _, k, v = _inputs((3,))
    elif shape_case == "singleton_key":
        q, _, _ = _inputs((3,))
        _, k, v = _inputs((1,))
    elif shape_case == "mixed_ranks":
        q, _, _ = _inputs(())
        _, k, v = _inputs((3,))
    else:
        q, _, _ = _inputs((2, 1, 3))
        _, k, v = _inputs((1, 4, 3))
    _check(Attention(), (q, k, v), opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("prefixes", [((4,), (2,)), ((2, 1, 4), (1, 3, 2))])
def test_gqa_uses_third_last_dimension_for_heads(opset, prefixes):
    q, _, _ = _inputs(prefixes[0])
    _, k, v = _inputs(prefixes[1])
    _check(Attention(gqa=True), (q, k, v), opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("empty", ["query", "key"])
def test_empty_sequences(opset, empty):
    _check(
        Attention(),
        _inputs(
            q_length=0 if empty == "query" else 4, kv_length=0 if empty == "key" else 6
        ),
        opset=opset,
    )


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("optimize", [False, True])
def test_dynamic_3d_mask_and_saved_model(tmp_path, opset, optimize):
    model = Attention().eval()
    args = (*_inputs(), torch.ones(2, 4, 6, dtype=torch.bool))
    batch, query, key = (
        torch.export.Dim(name, min=1, max=10)
        for name in ("batch", "query_length", "key_length")
    )
    dynamic_shapes = (
        {0: batch, 1: query},
        {0: batch, 1: key},
        {0: batch, 1: key},
        {0: batch, 1: query, 2: key},
    )
    program = torch_onnx.export(
        model,
        args,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        optimize=optimize,
        verbose=False,
    )
    path = tmp_path / "sdpa.onnx"
    try:
        program.save(path)
        onnx.checker.check_model(str(path), full_check=True)
        loaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        try:
            for b, query_length, key_length in ((3, 5, 7), (1, 2, 3)):
                changed = (
                    *_inputs((b,), q_length=query_length, kv_length=key_length),
                    torch.rand(b, query_length, key_length) > 0.3,
                )
                for candidate in (program, loaded):
                    torch.testing.assert_close(
                        candidate(*changed)[0], model(*changed), rtol=2e-4, atol=2e-5
                    )
        finally:
            loaded.release()
    finally:
        program.release()


@pytest.mark.parametrize("opset", [23, 28])
def test_4d_attention_keeps_fused_operator(opset):
    program = _check(Attention(), _inputs((2, 3)), opset=opset)
    assert [node.op_type for node in program.model.graph] == ["Attention"]


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_3d_dtype_preservation(opset, dtype):
    _check(Attention(), _inputs(dtype=dtype), opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
def test_gqa_empty_key_sequence(opset):
    q, _, _ = _inputs((4,))
    _, k, v = _inputs((2,), kv_length=0)
    _check(Attention(gqa=True), (q, k, v), opset=opset)


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("mask_kind", ["bool", "float"])
def test_3d_attention_matches_onnx_reference(opset, mask_kind):
    mask = torch.ones(2, 4, 6, dtype=torch.bool)
    mask[..., -1] = False
    if mask_kind == "float":
        mask = torch.where(mask, 0.0, float("-inf"))
    args = (*_inputs(), mask)
    model = Attention().eval()
    program = _check(model, args, opset=opset)
    torch.testing.assert_close(
        program.call_reference(*args)[0], model(*args), rtol=2e-4, atol=2e-5
    )
