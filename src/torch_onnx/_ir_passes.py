from __future__ import annotations

from typing import Sequence

from onnxscript import ir


def rename_inputs(model: ir.Model, new_names: Sequence[str]):
    # TODO: Ensure the names do not have duplicates
    for input, new_name in zip(model.graph.inputs, new_names):
        input.metadata_props["pkg.torch.onnx.original_node_name"] = input.name
        input.name = new_name


def rename_outputs(model: ir.Model, new_names: Sequence[str]):
    for output, new_name in zip(model.graph.outputs, new_names):
        output.metadata_props["pkg.torch.onnx.original_node_name"] = output.name
        output.name = new_name
