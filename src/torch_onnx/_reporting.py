import torch

from torch_onnx import _analysis


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


def create_torch_export_error_report(filename: str, formatted_traceback: str, *, profile_result: str | None):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(0, True))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        f.write("## Profile result\n\n")
        if profile_result is not None:
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_error_report(
    filename: str,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    step: int,
    profile_result: str | None,
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(step, True))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        f.write("Exported program:\n\n")
        f.write("```python\n")
        f.write(str(program))
        f.write("```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f)
        if profile_result is not None:
            f.write("## Profile result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def crete_onnx_export_profile_report(filename: str, profile_result: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Report\n\n")
        f.write(_format_export_status(4, False))
        f.write("## Profile result\n\n")
        f.write("```\n")
        f.write(profile_result)
        f.write("```\n")