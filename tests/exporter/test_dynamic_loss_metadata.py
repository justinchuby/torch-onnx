"""Independent reproduction of PyTorch's dynamic unreduced CE retracing bug."""

from __future__ import annotations

import pytest
import torch


class _UnreducedLoss(torch.nn.Module):
    def forward(self, logits, targets):
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        return loss, loss.flatten(), loss.sum()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="PyTorch run_decompositions corrupts dynamic CE-none metadata and lowers flatten to view([1])",
)
def test_torch_dynamic_unreduced_cross_entropy_retracing_preserves_metadata():
    model = _UnreducedLoss().eval()
    batch = torch.export.Dim("batch", min=1, max=16)
    captured = torch.export.export(
        model,
        (torch.randn(2, 5), torch.tensor([0, 4])),
        dynamic_shapes=({0: batch}, {0: batch}),
    )
    decomposed = captured.run_decompositions({})
    loss = next(
        node
        for node in decomposed.graph.nodes
        if node.target is torch.ops.aten.cross_entropy_loss.default
    )
    assert loss.meta["val"].ndim == 1, (
        "Unreduced loss metadata became scalar during retracing"
    )
    # If metadata is repaired upstream, also require correct consumer execution.
    logits, targets = torch.randn(4, 5), torch.tensor([0, 1, 2, 3])
    for expected, actual in zip(
        model(logits, targets), decomposed.module()(logits, targets), strict=True
    ):
        torch.testing.assert_close(actual, expected)
