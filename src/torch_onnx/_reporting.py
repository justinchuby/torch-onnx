from __future__ import annotations

import os
import re
import torch

from torch_onnx import _analysis, _registration
from onnxscript import ir


def _format_export_status(step: int, error: bool):
    def status_emoji(current_step: int, error: bool) -> str:
        if current_step < step:
            return "✅"
        if current_step == step:
            return "❌" if error else "✅"
        return "⚪"

    return (
        f"```\n"
        f"{status_emoji(0, error)} Obtain model graph with `torch.export.export`\n"
        f"{status_emoji(1, error)} Translate the graph into ONNX\n"
        f"{status_emoji(2, error)} Run `onnx.checker` on the ONNX model\n"
        f"{status_emoji(3, error)} Execute the model with ONNX Runtime\n"
        f"{status_emoji(4, error)} Validate model output accuracy\n"
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


def create_torch_export_error_report(
    filename: str | os.PathLike, formatted_traceback: str, *, profile_result: str | None
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(0, True))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        if profile_result is not None:
            f.write("## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    step: int,
    profile_result: str | None,
    model: ir.Model | None = None,
    registry: _registration.OnnxRegistry | None = None,
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(step, True))
        f.write("## Error message\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        if model is not None:
            f.write("ONNX model:\n\n")
            f.write("```python\n")
            f.write(str(model))
            f.write("\n```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f, registry=registry)
        if profile_result is not None:
            f.write("\n## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def crete_onnx_export_profile_report(
    filename: str | os.PathLike,
    program: torch.export.ExportedProgram,
    profile_result: str,
    step: int,
):
    """Create a report for the ONNX export profiling result.

    Args:
        filename: The file to write the report to.
        program: The exported program.
        profile_result: The profiling result.
        step: The current step in the conversion process.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Report\n\n")
        f.write(_format_export_status(step, False))
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        f.write("## Profiling result\n\n")
        f.write("```\n")
        f.write(profile_result)
        f.write("```\n")
