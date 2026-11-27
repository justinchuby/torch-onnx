"""Model-local ONNX fusion policy, preserving broadcast semantics for RoPE."""

from __future__ import annotations

from typing import Any

import onnx
import onnx_ir as ir


def _same_dimension(left: Any, right: Any) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return (
        isinstance(left, ir.SymbolicDim)
        and isinstance(right, ir.SymbolicDim)
        and left.value is not None
        and left.value == right.value
    )


def _rotary_rule() -> Any:
    from onnxscript import rewriter
    from onnxscript.rewriter.rules.fusion import _rotary_embedding

    base = _rotary_embedding.RotaryEmbedding23Fusion()
    signature = ir.schemas.OpSignature.from_op_schema(
        onnx.defs.get_schema("RotaryEmbedding", 23)
    )
    tensor_parameter = signature.params[0]
    assert isinstance(tensor_parameter, ir.schemas.Parameter)
    allowed_types = tensor_parameter.type_constraint.allowed_types

    def check(op: Any, x: Any, freqs: Any, **kwargs: Any) -> Any:
        result = base.check(op, x, **kwargs)
        if not result:
            return result
        if x.type not in allowed_types or freqs.type != x.type:
            return result.fail(
                "RoPE fusion would violate its tensor type constraint", x
            )
        if freqs.shape is None or len(freqs.shape) != 3:
            return result.fail("RoPE frequencies must have known rank three", freqs)
        if freqs.shape[2] != x.shape[3] // 2:
            return result.fail(
                "RoPE frequency width does not match the head size", freqs
            )
        for source, target in zip(
            freqs.shape[:2], (x.shape[0], x.shape[2]), strict=True
        ):
            if source != 1 and not _same_dimension(source, target):
                return result.fail(
                    "RoPE cache broadcasting to the input is not proven", freqs
                )
        return result

    def rewrite(op: Any, x: Any, freqs: Any, **_: Any) -> Any:
        cos, sin = op.Cos(freqs), op.Sin(freqs)
        if not (
            _same_dimension(freqs.shape[0], x.shape[0])
            and _same_dimension(freqs.shape[1], x.shape[2])
        ):
            # RotaryEmbedding requires exact cache batch/sequence sizes, while
            # the original multiply/add pattern broadcasts singleton axes.
            shape = op.Concat(
                op.Shape(x, start=0, end=1),
                op.Shape(x, start=2, end=3),
                op.Constant(value_ints=[x.shape[3] // 2]),
                axis=0,
            )
            cos, sin = op.Expand(cos, shape), op.Expand(sin, shape)
        return op.RotaryEmbedding(x, cos, sin, interleaved=0, num_heads=x.shape[1])

    return rewriter.RewriteRule(
        base.pattern,
        rewrite,
        check,
        name="BroadcastAwareRotaryEmbedding23",
        remove_nodes=False,
    )


def fuse(model: ir.Model) -> None:
    from onnxscript.rewriter import onnx_fusions

    if model.graph.opset_imports.get("") != 23:
        onnx_fusions.fuse(model)
        return
    from onnxscript.rewriter import _fusion_utils
    from onnxscript.rewriter.rules.fusion import _gqa, _rms_normalization

    _rms_normalization.fuse_rms_normalization(model)
    _fusion_utils.apply_fusion_rules(_rotary_rule())(model)
    _gqa.fuse_gqa(model)
