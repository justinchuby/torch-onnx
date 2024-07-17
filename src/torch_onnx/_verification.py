from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch.utils import _pytree as pytree

from torch_onnx import _onnx_program


@dataclasses.dataclass
class VerificationInfo:
    name: str
    absolute_difference: float
    relative_difference: float
    expected_dtype: torch.dtype | None = None
    actual_dtype: torch.dtype | None = None


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float]:
    absolute_difference = torch.abs(expected - actual).max().item()
    eps = 1e-7
    relative_difference = torch.abs(absolute_difference / (expected + eps)).max().item()
    return absolute_difference, relative_difference


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    exported_program = onnx_program.exported_program
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    results = []
    for torch_output, onnx_output, output_val in zip(
        torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
    ):
        name = output_val.name
        absolute_difference, relative_difference = _compare_tensors(
            torch_output, onnx_output
        )
        results.append(
            VerificationInfo(
                name=name,
                absolute_difference=absolute_difference,
                relative_difference=relative_difference,
                expected_dtype=torch_output.dtype,
                actual_dtype=onnx_output.dtype,
            )
        )
    return results