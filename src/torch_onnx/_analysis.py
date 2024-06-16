"""Compatibility analyzer for PyTorch models."""

from __future__ import annotations

import dataclasses
import textwrap
import traceback
import typing
from collections import defaultdict

import onnxscript
import torch
import torch._export.serde.schema
import torch.fx
from torch.export import graph_signature

from torch_onnx import _dispatching, _registration

if typing.TYPE_CHECKING:
    from _typeshed import SupportsWrite


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
    fx_node_target_count: defaultdict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    dispatch_failures: list[tuple[torch.fx.Node, str]] = dataclasses.field(
        default_factory=list
    )
    inputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(
        default_factory=dict
    )
    outputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(
        default_factory=dict
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
        textwrap.dedent(
            f"""\
            PyTorch ONNX Conversion Analysis

            ## Model Information

            The model has {sum(model_info.parameter_count.values())} parameters and {sum(model_info.buffer_count.values())} buffers (non-trainable parameters).
            """
        ),
        "Inputs:",
        *[f"- `{name}`: `{meta}`" for name, meta in model_info.inputs.items()],
        "",
        "Outputs:",
        *[f"- `{name}`: `{meta}`" for name, meta in model_info.outputs.items()],
        "",
        f"The FX graph has {model_info.fx_node_count} nodes in total. Number of FX nodes per op:",
    ]
    for op, count in model_info.fx_node_op_count.items():
        lines.append(f"- `{op}`: {count}")
    lines.append("\n")
    lines.append("Of the call_function nodes, the counts of operators used are:\n")
    sorted_targets = sorted(
        model_info.fx_node_target_count.items(), key=lambda x: x[1], reverse=True
    )
    for target, count in sorted_targets:
        lines.append(f"- `{target}`: {count}")

    lines.append("")
    lines.append("## ONNX Conversion Information")
    lines.append("")

    if model_info.dispatch_failures:
        lines.append(
            "The model contains operators the dispatcher could not find ONNX equivalents for. "
            "This may be due to missing implementations or a bug in the dispatcher."
        )
        lines.append("")
        lines.append("Errors grouped by operator:\n")

        target_to_nodes = defaultdict(list)
        for node, _ in model_info.dispatch_failures:
            target_to_nodes[str(node.target)].append(node)

        target_to_messages = {}
        for node, message in model_info.dispatch_failures:
            if str(node.target) not in target_to_messages:
                target_to_messages[str(node.target)] = message

        for target, nodes in sorted(
            target_to_nodes.items(), key=lambda x: x[0], reverse=True
        ):
            lines.append(
                f"- `{target}`: {target_to_messages[target]}. Example node: `{nodes[0].format_node()}`. All nodes: {nodes}"
            )
    else:
        lines.append("All operators in the model have ONNX equivalents.")

    return "\n".join(lines)


def _get_io_specs(exported_program: torch.export.ExportedProgram) -> tuple[dict, dict]:
    """Get the input and output specs of the exported program."""

    nodes: dict[str, torch.fx.Node] = {
        node.name: node for node in exported_program.graph.nodes
    }
    user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == graph_signature.InputKind.USER_INPUT
    ]
    user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind == graph_signature.OutputKind.USER_OUTPUT
    ]
    inputs: dict[str, torch._export.serde.schema.TensorMeta] = {}
    outputs: dict[str, torch._export.serde.schema.TensorMeta] = {}
    for spec in user_inputs:
        name = spec.arg.name
        inputs[name] = nodes[name].meta["tensor_meta"]
    for spec in user_outputs:
        name = spec.arg.name
        outputs[name] = nodes[name].meta["tensor_meta"]
    return inputs, outputs


def _count_fx_targets(
    exported_program: torch.export.ExportedProgram,
) -> defaultdict[str, int]:
    """Count the number of targets for each node in the exported program."""
    fx_node_target_count = defaultdict(int)
    for node in exported_program.graph.nodes:
        if node.op == "call_function":
            fx_node_target_count[str(node.target)] += 1
    return fx_node_target_count


def analyze(
    exported_program: torch.export.ExportedProgram,
    registry: _registration.OnnxRegistry | None = None,
    file: SupportsWrite[str] | None = None,
) -> str:
    """Analyze the compatibility of the exported program."""
    # Get basic information about the model
    model_info = ModelInfo()
    model_info.parameter_count, model_info.buffer_count = _count_weights(
        exported_program
    )
    model_info.fx_node_count = len(exported_program.graph.nodes)
    model_info.fx_node_target_count = _count_fx_targets(exported_program)
    inputs, outputs = _get_io_specs(exported_program)
    model_info.inputs = inputs
    model_info.outputs = outputs

    if registry is None:
        # Trigger op registration
        from onnxscript.function_libs.torch_lib import ops

        del ops
        registry = _registration.OnnxRegistry.from_torchlib(
            onnxscript.function_libs.torch_lib.registration.default_registry
        )

    # Try to find ops for every node in the graph
    for node in exported_program.graph.nodes:
        model_info.fx_node_op_count[node.op] += 1
        if node.op == "call_function":
            try:
                onnx_function, message = _dispatching.dispatch(node, registry)
            except Exception:
                message = "Critical Error in dispatcher:\n"
                message += textwrap.indent(
                    f"```pytb\n{traceback.format_exc()}\n```", "    "
                )
                onnx_function = None
            if onnx_function is None:
                model_info.dispatch_failures.append((node, message))

    # Print the results
    report = _format_model_info(model_info)
    print(report, file=file)
    return report
