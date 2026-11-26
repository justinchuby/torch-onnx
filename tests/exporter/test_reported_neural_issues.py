"""Numerical and schema regressions from PyTorch ONNX issue reports."""

from __future__ import annotations

import copy

import onnx
import onnx_ir as ir
import pytest
import torch
import torch.nn.functional as F
from onnx.reference import ReferenceEvaluator

import torch_onnx


class _Function(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


def _check(model, args, *, optimize, opset=18, rtol=1e-5, atol=1e-7):
    with torch.no_grad():
        expected = copy.deepcopy(model)(*args)
    program = torch_onnx.export(
        model, args, opset_version=opset, optimize=optimize, verbose=False
    )
    try:
        onnx.checker.check_model(ir.to_proto(program.model), full_check=True)
        actual = program(*args)[0]
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        return program.model
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("kind", ["module", "functional", "no_running_stats"])
def test_issue_206627_batch_norm_uses_batch_statistics(optimize, kind):
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 3, 4, 4, dtype=torch.float64, generator=generator)
    if kind == "functional":
        model = _Function(
            lambda t: F.batch_norm(
                t,
                torch.full((3,), 100.0, dtype=t.dtype),
                torch.full((3,), 4.0, dtype=t.dtype),
                training=True,
            )
        ).eval()
    else:
        model = torch.nn.BatchNorm2d(
            3, track_running_stats=kind != "no_running_stats"
        ).double()
        if kind == "module":
            model.running_mean.fill_(100.0)
            model.running_var.fill_(4.0)
            model.train()
        else:
            model.eval()
    _check(model, (x,), optimize=optimize, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("mask_kind", ["bool", "float"])
@pytest.mark.parametrize("mask_shape", [(1, 4), (4, 1), (2, 1, 4), (1, 1, 1, 4)])
def test_issue_206402_attention_broadcasts_sequence_mask(
    optimize, mask_kind, mask_shape
):
    generator = torch.Generator().manual_seed(11)
    q, k, v = (torch.randn(1, 2, 4, 8, generator=generator) for _ in range(3))
    mask = torch.ones(mask_shape, dtype=torch.bool)
    if mask_shape[-1] == 4:
        mask[..., -2:] = False
    if mask_kind == "float":
        mask = torch.where(mask, 0.0, float("-inf"))
    model = _Function(
        lambda a, b, c, m: F.scaled_dot_product_attention(a, b, c, attn_mask=m)
    ).eval()
    _check(model, (q, k, v, mask), opset=23, optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("probabilities", [False, True])
def test_issue_206286_cross_entropy_label_smoothing(
    optimize, weighted, reduction, probabilities
):
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(4, 3, generator=generator)
    target = (
        torch.randn(4, 3, generator=generator).softmax(1)
        if probabilities
        else torch.tensor([0, 2, -100, 1])
    )
    weight = torch.tensor([0.5, 2.0, 3.0]) if weighted else None
    model = _Function(
        lambda a, t: F.cross_entropy(
            a,
            t,
            weight=weight,
            label_smoothing=0.5,
            reduction=reduction,
        )
    ).eval()
    _check(model, (x, target), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("shape", [(3,), (2, 3, 2)])
@pytest.mark.parametrize("probabilities", [False, True])
def test_cross_entropy_smoothing_preserves_unbatched_and_spatial_shapes(
    optimize, shape, probabilities
):
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(shape, generator=generator)
    target = (
        torch.randn(shape, generator=generator).softmax(0 if len(shape) == 1 else 1)
        if probabilities
        else torch.tensor(1)
        if len(shape) == 1
        else torch.tensor([[0, 1], [2, -100]])
    )
    model = _Function(
        lambda a, t: F.cross_entropy(a, t, label_smoothing=0.2, reduction="none")
    ).eval()
    _check(model, (x, target), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0])
def test_cross_entropy_smoothing_endpoints(optimize, eps):
    x = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    model = _Function(lambda a, t: F.cross_entropy(a, t, label_smoothing=eps)).eval()
    _check(model, (x, torch.tensor([0, 2])), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_issue_206460_bce_logits_extremes_are_finite_and_keep_small_losses(
    optimize, reduction
):
    x = torch.tensor([-100.0, -20.0, 0.0, 20.0, 40.0, 80.0], dtype=torch.float64)
    target = torch.ones_like(x)
    model = _Function(
        lambda a, t: F.binary_cross_entropy_with_logits(a, t, reduction=reduction)
    ).eval()
    _check(model, (x, target), optimize=optimize, rtol=1e-12, atol=0)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "input_dtype,target_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
    ],
)
def test_bce_logits_weights_and_target_dtype(optimize, input_dtype, target_dtype):
    x = torch.tensor([[-100.0, -20.0, 0.0], [20.0, 40.0, 5.0]], dtype=input_dtype)
    target = torch.tensor([[1.0, 0.0, 0.3], [1.0, 0.0, 0.8]], dtype=target_dtype)
    model = _Function(
        lambda a, t: F.binary_cross_entropy_with_logits(
            a,
            t,
            weight=torch.tensor([0.5, 2.0, 3.0]),
            pos_weight=torch.tensor([1.0, 2.0, 0.5]),
            reduction="none",
        )
    ).eval()
    _check(model, (x, target), optimize=optimize, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("opset", [18, 19])
@pytest.mark.parametrize(
    "padding", [(1, 1, 1, 1), (-1, 2, 1, -1), (0, 0, 0, 0), (-5, 0, 0, 0)]
)
def test_issue_206814_circular_padding_uses_legal_target_ops(optimize, opset, padding):
    model = _Function(lambda t: F.pad(t, padding, mode="circular")).eval()
    graph = _check(
        model, (torch.arange(25.0).reshape(1, 1, 5, 5),), optimize=optimize, opset=opset
    )
    for node in graph.graph:
        if node.op_type == "Pad" and node.attributes.get("mode") is not None:
            assert node.attributes["mode"].as_string() != "wrap" or opset >= 19


@pytest.mark.parametrize("operation", ["conv", "matmul", "bf16_pool"])
def test_reported_unsupported_tensor_types_are_rejected_before_serialization(operation):
    if operation == "conv":
        model = _Function(lambda a, b: F.conv2d(a, b)).eval()
        args = (
            torch.ones(1, 1, 4, 4, dtype=torch.int8),
            torch.ones(1, 1, 2, 2, dtype=torch.int8),
        )
    elif operation == "matmul":
        model = _Function(torch.matmul).eval()
        args = (torch.ones(3, 4, dtype=torch.int8), torch.ones(4, 5, dtype=torch.int8))
    else:
        model = _Function(lambda t: F.max_pool2d(t, 2)).eval()
        args = (torch.ones(1, 1, 4, 4, dtype=torch.bfloat16),)
    with pytest.raises(torch_onnx.errors.ConversionError):
        torch_onnx.export(model, args, opset_version=18, verbose=False)


@pytest.mark.parametrize("optimize", [False, True])
def test_rms_norm_default_epsilon_uses_input_dtype(optimize):
    x = torch.tensor([1e-4, -1e-4] * 4, dtype=torch.float64).reshape(1, 1, 8)
    model = _Function(lambda t: F.rms_norm(t, (8,))).eval()
    _check(model, (x,), opset=23, optimize=optimize, rtol=1e-12, atol=0)


@pytest.mark.parametrize("optimize", [False, True])
def test_attention_singleton_mask_expands_to_dynamic_sequence_lengths(optimize):
    generator = torch.Generator().manual_seed(17)

    class Attention(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    model = Attention().eval()
    args = (
        torch.randn(1, 2, 4, 8, generator=generator),
        torch.randn(1, 2, 5, 8, generator=generator),
        torch.randn(1, 2, 5, 8, generator=generator),
        torch.tensor([[True, False, True, False, True]]),
    )
    query_length = torch.export.Dim("query_length", min=2)
    key_length = torch.export.Dim("key_length", min=2)
    program = torch_onnx.export(
        model,
        args,
        opset_version=23,
        optimize=optimize,
        verbose=False,
        dynamic_shapes=(
            {2: query_length},
            {2: key_length},
            {2: key_length},
            {1: key_length},
        ),
    )
    try:
        alternative = (
            torch.randn(1, 2, 3, 8, generator=generator),
            torch.randn(1, 2, 6, 8, generator=generator),
            torch.randn(1, 2, 6, 8, generator=generator),
            torch.tensor([[True, False, True, False, True, False]]),
        )
        torch.testing.assert_close(
            program(*alternative)[0], model(*alternative), rtol=1e-5, atol=1e-7
        )
    finally:
        program.release()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "input_dtype,target_dtype",
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
def test_cross_entropy_probability_targets_preserve_promoted_dtype(
    optimize, input_dtype, target_dtype
):
    x = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]], dtype=input_dtype)
    target = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]], dtype=target_dtype)
    model = _Function(lambda a, t: F.cross_entropy(a, t, label_smoothing=0.2)).eval()
    _check(model, (x, target), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_smoothed_cross_entropy_all_ignored(optimize, reduction):
    x = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    target = torch.tensor([-100, -100])
    model = _Function(
        lambda a, t: F.cross_entropy(a, t, label_smoothing=0.5, reduction=reduction)
    ).eval()
    _check(model, (x, target), optimize=optimize)


@pytest.mark.parametrize("optimize", [False, True])
def test_issue_206815_unsigned_relu_is_identity(optimize):
    x = torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8)
    graph = _check(_Function(F.relu).eval(), (x,), optimize=optimize, rtol=0, atol=0)
    assert not any(node.op_type == "Relu" for node in graph.graph)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "input_dtype,target_dtype",
    [(torch.float64, torch.float32), (torch.float32, torch.float64)],
)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_bce_logits_mixed_dtype_inplace_rounding_stages(
    optimize, input_dtype, target_dtype, weighted, reduction
):
    # ATen rounds the linear term to target dtype before cancellation, and
    # rounds a pos_weight-scaled log_sigmoid to input dtype before subtraction.
    x = torch.tensor([-1e6, -2e6, -314159.2], dtype=input_dtype)
    target = torch.tensor([1e-5, 1e-6, 0.0001], dtype=target_dtype)
    pos_weight = (
        torch.tensor([3.0, 0.5, 2.0], dtype=torch.float64) if weighted else None
    )
    weight = torch.tensor([0.5, 3.0, 1.5], dtype=torch.float64) if weighted else None
    model = _Function(
        lambda a, t: F.binary_cross_entropy_with_logits(
            a,
            t,
            pos_weight=pos_weight,
            weight=weight,
            reduction=reduction,
        )
    ).eval()
    _check(model, (x, target), optimize=optimize, rtol=1e-12, atol=0)


@pytest.mark.parametrize("optimize", [False, True])
def test_cross_entropy_large_class_count_does_not_round_through_logits_dtype(optimize):
    x = torch.zeros((1, 2049), dtype=torch.float16)
    target = torch.full((1, 2049), 1.0 / 2049, dtype=torch.float64)
    model = _Function(
        lambda a, t: F.cross_entropy(a, t, label_smoothing=1.0, reduction="none")
    ).eval()
    # ORT promotes the half LogSoftmax computation to float32 without retaining
    # its intermediate half rounding (also with ORT_DISABLE_ALL). Check the
    # graph's typed semantics exactly with ONNX reference inference as well.
    graph = _check(model, (x, target), optimize=optimize, rtol=2e-5, atol=0)
    feeds = {
        value.name: argument.numpy()
        for value, argument in zip(graph.graph.inputs, (x, target), strict=True)
    }
    (actual,) = ReferenceEvaluator(ir.to_proto(graph)).run(None, feeds)
    torch.testing.assert_close(
        torch.from_numpy(actual), model(x, target), rtol=1e-12, atol=0
    )
