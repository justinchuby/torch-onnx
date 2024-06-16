"""Compatibility analyzer for PyTorch models."""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import onnxscript
import torch
import torch.fx
from _typeshed import SupportsWrite

from torch_onnx import _dispatching, _registration


@dataclasses.dataclass
class ModelInfo:
    """Information about the model."""

    parameter_count: defaultdict[torch.dtype, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    buffer_count: defaultdict[torch.dtype, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    fx_node_count: int = 0
    fx_node_op_count: defaultdict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    dispatch_failures: list[tuple[torch.fx.Node, str]] = dataclasses.field(
        default_factory=list
    )


def _count_weights(
    exported_program: torch.export.ExportedProgram,
) -> tuple[defaultdict[torch.dtype, int], defaultdict[torch.dtype, int]]:
    """Count the size of the parameters in the exported program."""

    parameter_count = defaultdict(int)
    buffer_count = defaultdict(int)
    for parameter in exported_program.parameters():
        dtype = parameter.dtype
        parameter_count[dtype] += parameter.numel()

    for buffer in exported_program.buffers():
        dtype = buffer.dtype
        buffer_count[dtype] += buffer.numel()

    return parameter_count, buffer_count


def _format_model_info(model_info: ModelInfo) -> str:
    """Format the information about the model."""
    lines = [
        f"Number of parameters: {sum(model_info.parameter_count.values())}",
        f"Number of buffers: {sum(model_info.buffer_count.values())}",
        f"Number of FX nodes: {model_info.fx_node_count}",
        "Number of FX nodes per op:",
    ]
    for op, count in model_info.fx_node_op_count.items():
        lines.append(f"  {op}: {count}")

    if model_info.dispatch_failures:
        lines.append("Dispatch failures:")
        for node, message in model_info.dispatch_failures:
            lines.append(f"  {node}: {message}")

    return "\n".join(lines)


def analyze(
    exported_program: torch.export.ExportedProgram,
    registry: _registration.OnnxRegistry | None = None,
    file: SupportsWrite[str] | None = None,
) -> None:
    """Analyze the compatibility of the exported program."""
    if registry is None:
        # Trigger op registration
        from onnxscript.function_libs.torch_lib import ops

        del ops
        registry = _registration.OnnxRegistry.from_torchlib(
            onnxscript.function_libs.torch_lib.registration.default_registry
        )

    # Get basic information about the model
    model_info = ModelInfo()
    model_info.parameter_count, model_info.buffer_count = _count_weights(
        exported_program
    )
    model_info.fx_node_count = len(exported_program.graph.nodes)

    # Try to find ops for every node in the graph
    for node in exported_program.graph.nodes:
        model_info.fx_node_op_count[node.op] += 1
        if node.op == "call_function":
            onnx_function, message = _dispatching.dispatch(node, registry)
            if onnx_function is None:
                model_info.dispatch_failures.append((node, message))

    # Write the results to a file
    print(_format_model_info(model_info), file=file)
