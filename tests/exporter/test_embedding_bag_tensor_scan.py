"""Tensor scan embedding bags, including dynamic and empty auxiliary outputs."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from torch_onnx._ir_builder import GraphBuilder, TranslationContext
from torch_onnx._torchlib.ops import embedding

pytestmark = pytest.mark.compatibility


def build_model(
    function,
    *,
    mode=0,
    padding_idx=None,
    include_last_offset=False,
    weighted=False,
    opset=18,
):
    graph = ir.Graph([], [], nodes=[], name="embedding_bag", opset_imports={"": opset})
    op = TranslationContext(GraphBuilder(graph, opset=opset))
    specs = [
        ("weight", ir.DataType.FLOAT, [None, None]),
        ("indices", ir.DataType.INT64, [None]),
        ("offsets", ir.DataType.INT64, [None]),
    ]
    if weighted:
        specs.append(("per_sample_weights", ir.DataType.FLOAT, [None]))
    inputs = [
        ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))
        for name, dtype, shape in specs
    ]
    graph.inputs.extend(inputs)
    outputs = function(
        op,
        *inputs[:3],
        mode=mode,
        padding_idx=padding_idx,
        include_last_offset=include_last_offset,
        per_sample_weights=inputs[3] if weighted else None,
    )
    for index, output in enumerate(outputs):
        output.shape = ir.Shape(
            [None, None] if index == 0 or (index == 3 and mode == 2) else [None]
        )
    graph.outputs.extend(outputs)
    model = ir.to_proto(ir.Model(graph, ir_version=10))
    onnx.checker.check_model(model)
    return model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def node_types(graph):
    result = [node.op_type for node in graph.node]
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                result.extend(node_types(attribute.g))
    return result


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("padding_idx", [None, 0, -1])
@pytest.mark.parametrize("include_last_offset", [False, True])
@pytest.mark.parametrize("opset", [18, 23])
def test_dynamic_embedding_bag_scan(mode, padding_idx, include_last_offset, opset):
    weighted = mode == 0
    model = build_model(
        embedding.aten_embedding_bag_padding_idx,
        mode=mode,
        padding_idx=padding_idx,
        include_last_offset=include_last_offset,
        weighted=weighted,
        opset=opset,
    )
    assert not any("Sequence" in kind for kind in node_types(model.graph))
    assert "GreaterOrEqual" not in node_types(model.graph)
    runtime = session(model)
    generator = torch.Generator().manual_seed(0)
    for raw_indices, starts, final_offset, width in [
        ([1, 0, 2, 4, 0], [0, 2, 2], 5, 3),
        ([0, 0], [0, 0, 2], 2, 1),
        ([], [0, 0], 0, 7),
        ([], [], 0, 3),
        ([1, 0, 2, 4, 0], [0, 2], 3, 5),
        ([1, 2], [0, 2], 2, 0),
    ]:
        weight = torch.randn(5, width, generator=generator)
        indices = torch.tensor(raw_indices, dtype=torch.int64)
        offsets = torch.tensor(
            starts + ([final_offset] if include_last_offset else []), dtype=torch.int64
        )
        weights = torch.randn(len(indices), generator=generator) if weighted else None
        expected = torch.ops.aten.embedding_bag.padding_idx(
            weight,
            indices,
            offsets,
            False,
            mode,
            False,
            weights,
            include_last_offset,
            padding_idx,
        )
        feed = {
            "weight": weight.numpy(),
            "indices": indices.numpy(),
            "offsets": offsets.numpy(),
        }
        if weighted:
            feed["per_sample_weights"] = weights.numpy()
        actual = runtime.run(None, feed)
        for output, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(output, reference.numpy(), rtol=2e-5, atol=2e-6)
