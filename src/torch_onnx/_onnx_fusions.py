"""Public fusion and rewrite APIs with model-local RoPE broadcast repairs."""

from __future__ import annotations

from collections.abc import Iterator
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


def _graphs(model: ir.Model) -> Iterator[ir.Graph]:
    yield from model.graphs()
    for function in model.functions.values():
        yield function.graph
        yield from function.graph.subgraphs()


def _rotary_repairs(original: set[ir.Node]):
    from onnxscript import rewriter

    signature = ir.schemas.OpSignature.from_op_schema(
        onnx.defs.get_schema("RotaryEmbedding", 23)
    )
    parameter = signature.params[0]
    assert isinstance(parameter, ir.schemas.Parameter)
    allowed_types = parameter.type_constraint.allowed_types

    def pattern(op, x, freqs):
        return op.RotaryEmbedding(x, op.Cos(freqs), op.Sin(freqs))

    def recognized(context, x, freqs):
        node = context.root
        return (
            node not in original
            and node.attributes.get_int("interleaved", 0) == 0
            and node.attributes.get_int("rotary_embedding_dim", 0) == 0
            and x.shape is not None
            and len(x.shape) == 4
            and isinstance(x.shape[3], int)
        )

    def supported(x, freqs):
        depth = x.shape[3]
        return (
            depth > 0
            and depth % 2 == 0
            and x.type in allowed_types
            and freqs.type == x.type
            and freqs.shape is not None
            and len(freqs.shape) == 3
            and freqs.shape[2] == depth // 2
            and all(
                source == 1 or _same_dimension(source, target)
                for source, target in zip(
                    freqs.shape[:2], (x.shape[0], x.shape[2]), strict=True
                )
            )
        )

    def needs_broadcast(context, x, freqs):
        return (
            recognized(context, x, freqs)
            and supported(x, freqs)
            and not (
                _same_dimension(freqs.shape[0], x.shape[0])
                and _same_dimension(freqs.shape[1], x.shape[2])
            )
        )

    def broadcast(op, x, freqs):
        shape = op.Concat(
            op.Shape(x, start=0, end=1),
            op.Shape(x, start=2, end=3),
            op.Constant(value_ints=[x.shape[3] // 2]),
            axis=0,
        )
        return op.RotaryEmbedding(
            x,
            op.Expand(op.Cos(freqs), shape),
            op.Expand(op.Sin(freqs), shape),
            interleaved=0,
            num_heads=x.shape[1],
        )

    def needs_primitives(context, x, freqs):
        return recognized(context, x, freqs) and not supported(x, freqs)

    def primitives(op, x, freqs):
        half = x.shape[3] // 2
        axes = op.Constant(value_ints=[-1])
        first = op.Slice(
            x, op.Constant(value_ints=[0]), op.Constant(value_ints=[half]), axes
        )
        second = op.Slice(
            x, op.Constant(value_ints=[half]), op.Constant(value_ints=[2**63 - 1]), axes
        )
        rotated = op.Concat(op.Neg(second), first, axis=-1)
        cos = op.Cos(freqs)
        sin = op.Sin(freqs)
        cos = op.Unsqueeze(op.Concat(cos, cos, axis=-1), op.Constant(value_ints=[1]))
        sin = op.Unsqueeze(op.Concat(sin, sin, axis=-1), op.Constant(value_ints=[1]))
        return op.Add(op.Mul(x, cos), op.Mul(rotated, sin))

    return rewriter.RewriteRuleSet(
        [
            rewriter.RewriteRule(
                pattern,
                broadcast,
                needs_broadcast,
                name="BroadcastAwareRotaryEmbedding23",
                remove_nodes=False,
            ),
            rewriter.RewriteRule(
                pattern,
                primitives,
                needs_primitives,
                name="UnsupportedRotaryEmbedding23ToPrimitives",
                remove_nodes=False,
            ),
        ]
    )


def fuse(model: ir.Model) -> None:
    from onnxscript.rewriter import onnx_fusions

    if model.graph.opset_imports.get("") != 23:
        onnx_fusions.fuse(model)
        return
    original = {
        node
        for graph in _graphs(model)
        for node in graph
        if node.domain == "" and node.op_type == "RotaryEmbedding"
    }
    onnx_fusions.fuse(model)
    _rotary_repairs(original).apply_to_model(model)
