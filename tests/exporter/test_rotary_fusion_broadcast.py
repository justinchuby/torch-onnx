"""RoPE fusion must preserve broadcasting, supported types and dynamic shapes."""

from __future__ import annotations

import onnx
import pytest
import torch

import torch_onnx


class _Rotary(torch.nn.Module):
    def forward(self, x, frequencies):
        repeated = torch.cat((frequencies, frequencies), -1)
        cos, sin = repeated.cos().unsqueeze(1), repeated.sin().unsqueeze(1)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
        return x * cos + rotated * sin


@pytest.mark.parametrize("broadcast_batch", [False, True])
@pytest.mark.parametrize("broadcast_sequence", [False, True])
def test_rotary_fusion_executes_broadcast_caches_at_multiple_shapes(
    broadcast_batch, broadcast_sequence
):
    model = _Rotary().eval()

    def inputs(batch, length):
        return torch.randn(batch, 3, length, 8), torch.randn(
            1 if broadcast_batch else batch, 1 if broadcast_sequence else length, 4
        )

    batch = torch.export.Dim("batch", min=1, max=8)
    sequence = torch.export.Dim("sequence", min=1, max=16)
    frequencies = {}
    if not broadcast_batch:
        frequencies[0] = batch
    if not broadcast_sequence:
        frequencies[1] = sequence
    args = inputs(2, 5)
    program = torch_onnx.export(
        model,
        args,
        opset_version=23,
        dynamic_shapes=({0: batch, 2: sequence}, frequencies),
        verbose=False,
    )
    try:
        assert any(node.op_type == "RotaryEmbedding" for node in program.model.graph)
        onnx.checker.check_model(program.model_proto, full_check=True)
        for shape in [(2, 5), (3, 7), (1, 1)]:
            args = inputs(*shape)
            torch.testing.assert_close(program(*args)[0], model(*args))
    finally:
        program.release()


@pytest.mark.parametrize(
    "dtype,expand_input", [(torch.float64, False), (torch.float32, True)]
)
def test_rotary_fusion_rejects_unsupported_types_and_input_expansion(
    dtype, expand_input
):
    model = _Rotary().eval()
    args = (
        torch.randn(1 if expand_input else 2, 3, 5, 8, dtype=dtype),
        torch.randn(2, 5, 4, dtype=dtype),
    )
    program = torch_onnx.export(model, args, opset_version=23, verbose=False)
    try:
        assert not any(
            node.op_type == "RotaryEmbedding" for node in program.model.graph
        )
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(*args)[0], model(*args))
    finally:
        program.release()
