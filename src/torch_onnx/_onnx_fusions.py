"""Public ONNX fusion entry point with model-local RoPE broadcast repairs."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from typing import Any

import onnx
import onnx_ir as ir

from torch_onnx._ir_builder import GraphBuilder


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


def _can_use_rotary(node: ir.Node, allowed_types: AbstractSet[ir.TypeProtocol]) -> bool:
    x, cos, sin = node.inputs[:3]
    if x is None or cos is None or sin is None:
        return False
    if x.type not in allowed_types or cos.type != x.type or sin.type != x.type:
        return False
    if x.shape is None or len(x.shape) != 4:
        return False
    depth = x.shape[3]
    if not isinstance(depth, int) or depth <= 0 or depth % 2:
        return False
    for cache in (cos, sin):
        if cache.shape is None or len(cache.shape) != 3 or cache.shape[2] != depth // 2:
            return False
        for source, target in zip(
            cache.shape[:2], (x.shape[0], x.shape[2]), strict=True
        ):
            if source != 1 and not _same_dimension(source, target):
                return False
    return True


def _repair_rotary(node: ir.Node, allowed_types: AbstractSet[ir.TypeProtocol]) -> None:
    """Repair only newly fused non-interleaved 4D rotate-half patterns."""
    graph = node.graph
    assert graph is not None
    x, cos, sin = node.inputs[:3]
    assert x is not None and cos is not None and sin is not None
    # Match the public pass's rotate-half fusion, rather than modifying other
    # RotaryEmbedding implementations a future upstream pass may introduce.
    trig = (cos.producer(), sin.producer())
    if (
        node.attributes.get_int("interleaved", 0) != 0
        or node.attributes.get_int("rotary_embedding_dim", 0) != 0
        or x.shape is None
        or len(x.shape) != 4
        or not isinstance(x.shape[3], int)
        or any(
            producer is None
            or producer.domain != ""
            or producer.op_type != kind
            or len(producer.inputs) != 1
            or producer.inputs[0] is None
            for producer, kind in zip(trig, ("Cos", "Sin"), strict=True)
        )
    ):
        return
    assert trig[0] is not None and trig[1] is not None
    if trig[0].inputs[0] is not trig[1].inputs[0]:
        return
    # The public rewriter leaves these fresh elementwise values untyped.
    for cache, producer in zip((cos, sin), trig, strict=True):
        assert producer is not None
        source = producer.inputs[0]
        assert source is not None
        cache.type = source.type
        cache.shape = source.shape
    # Build detached nodes using the public builder/IR APIs. Captured inputs
    # remain the original graph values; no graph or global rule is monkeypatched.
    # ONNX subgraphs inherit their parent's opset. Rewriter-created subgraphs
    # can carry a placeholder local import, which must not select our schemas.
    staging = ir.Graph([], [], nodes=[], opset_imports={**graph.opset_imports, "": 23})
    builder = GraphBuilder(staging, opset=23)
    if _can_use_rotary(node, allowed_types):
        assert x.shape is not None
        if all(
            cache.shape is not None
            and _same_dimension(cache.shape[0], x.shape[0])
            and _same_dimension(cache.shape[1], x.shape[2])
            for cache in (cos, sin)
        ):
            return
        batch = builder.call("Shape", x, start=0, end=1)[0]
        sequence = builder.call("Shape", x, start=2, end=3)[0]
        width = builder.constant([x.shape[3] // 2], dtype=ir.DataType.INT64)
        shape = builder.call("Concat", batch, sequence, width, axis=0)[0]
        expanded = [builder.call("Expand", cache, shape)[0] for cache in (cos, sin)]
        added = list(staging)
        staging.remove(added)
        graph.insert_before(node, added)
        node.replace_input_with(1, expanded[0])
        node.replace_input_with(2, expanded[1])
        return

    # The public fusion does not guard dtype or all broadcast relationships.
    # Recover its original primitive formula when RotaryEmbedding cannot express
    # that formula. Repeating cos/sin equals cos/sin of the repeated frequencies.
    assert x.shape is not None and isinstance(x.shape[3], int)
    half = x.shape[3] // 2
    first = builder.call("Slice", x, [0], [half], [-1])[0]
    second = builder.call("Slice", x, [half], [2**63 - 1], [-1])[0]
    negated = builder.call("Neg", second)[0]
    rotated = builder.call("Concat", negated, first, axis=-1)[0]
    terms = []
    for tensor, cache in ((x, cos), (rotated, sin)):
        repeated = builder.call("Concat", cache, cache, axis=-1)[0]
        expanded = builder.call("Unsqueeze", repeated, [1])[0]
        terms.append(builder.call("Mul", tensor, expanded)[0])
    result = builder.call("Add", *terms)[0]
    result.name = node.outputs[0].name
    result.metadata_props.update(node.outputs[0].metadata_props)
    producer = result.producer()
    assert producer is not None
    producer.metadata_props.update(node.metadata_props)
    producer.meta.update(node.meta)
    added = list(staging)
    staging.remove(added)
    ir.convenience.replace_nodes_and_values(
        graph, node, [node], added, node.outputs, [result]
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
    signature = ir.schemas.OpSignature.from_op_schema(
        onnx.defs.get_schema("RotaryEmbedding", 23)
    )
    tensor_parameter = signature.params[0]
    assert isinstance(tensor_parameter, ir.schemas.Parameter)
    allowed_types = tensor_parameter.type_constraint.allowed_types
    for graph in _graphs(model):
        for node in tuple(graph):
            if (
                node not in original
                and node.domain == ""
                and node.op_type == "RotaryEmbedding"
            ):
                _repair_rotary(node, allowed_types)
