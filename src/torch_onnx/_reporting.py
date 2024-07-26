from __future__ import annotations

import dataclasses
import os
import re

import torch
from onnxscript import ir

from torch_onnx import _analysis, _registration


@dataclasses.dataclass
class ExportStatus:
    # Whether torch.export.export.export() succeeds
    torch_export: bool | None = None
    # Whether torch.export.export.export(..., strict=False) succeeds
    torch_export_non_strict: bool | None = None
    # Whether torch.jit.trace succeeds
    torch_jit: bool | None = None
    # Whether ONNX translation succeeds
    onnx_translation: bool | None = None
    # Whether ONNX model passes onnx.checker.check_model
    onnx_checker: bool | None = None
    # Whether ONNX model runs successfully with ONNX Runtime
    onnx_runtime: bool | None = None
    # Whether the output of the ONNX model is accurate
    output_accuracy: bool | None = None


def _status_emoji(status: bool | None) -> str:
    if status is None:
        return "⚪"
    return "✅" if status else "❌"


def _format_export_status(status: ExportStatus) -> str:
    return (
        f"```\n"
        f"{_status_emoji(status.torch_export)} Obtain model graph with `torch.export.export`\n"
        f"{_status_emoji(status.torch_export_non_strict)} Obtain model graph with `torch.export.export(..., strict=False)`\n"
        f"{_status_emoji(status.torch_jit)} Obtain model graph with `torch.jit.trace`\n"
        f"{_status_emoji(status.onnx_translation)} Translate the graph into ONNX\n"
        f"{_status_emoji(status.onnx_checker)} Run `onnx.checker` on the ONNX model\n"
        f"{_status_emoji(status.onnx_runtime)} Execute the model with ONNX Runtime\n"
        f"{_status_emoji(status.output_accuracy)} Validate model output accuracy\n"
        f"```\n\n"
    )


def _strip_color_from_string(text: str) -> str:
    # This regular expression matches ANSI escape codes
    # https://github.com/pytorch/pytorch/blob/9554a9af8788c57e1c5222c39076a5afcf0998ae/torch/_dynamo/utils.py#L2785-L2788
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def _format_exported_program(exported_program: torch.export.ExportedProgram) -> str:
    # Adapted from https://github.com/pytorch/pytorch/pull/128476
    # to remove colors
    # Even though we can call graph_module.print_readable directly, since the
    # colored option was added only recently, we can't guarantee that the
    # version of PyTorch used by the user has this option. Therefore, we
    # still call str(ExportedProgram)
    text = f"```python\n{_strip_color_from_string(str(exported_program))}\n```\n\n"
    return text


def format_decomp_comparison(
    pre_decomp_unique_ops: set[str],
    post_decomp_unique_ops: set[str],
) -> str:
    """Format the decomposition comparison result.

    Args:
        unique_ops_in_a: The unique ops in the first program.
        unique_ops_in_b: The unique ops in the second program.

    Returns:
        The formatted comparison result.
    """
    return (
        f"Ops exist only in the ExportedProgram before decomposition: `{sorted(pre_decomp_unique_ops)}`\n\n"
        f"Ops exist only in the ExportedProgram after decomposition: `{sorted(post_decomp_unique_ops)}`\n"
    )


def create_torch_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    *,
    export_status: ExportStatus,
    profile_result: str | None,
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        if profile_result is not None:
            f.write("## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    decomp_comparison: str | None = None,
    export_status: ExportStatus,
    profile_result: str | None,
    model: ir.Model | None = None,
    registry: _registration.ONNXRegistry | None = None,
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("## Error message\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("\n```\n\n")
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        if model is not None:
            f.write("ONNX model:\n\n")
            f.write("```python\n")
            f.write(str(model))
            f.write("\n```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f, registry=registry)
        if decomp_comparison is not None:
            f.write("\n## Decomposition comparison\n\n")
            f.write(decomp_comparison)
            f.write("\n")
        if profile_result is not None:
            f.write("\n## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")
