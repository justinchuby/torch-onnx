"""Test utilities for ONNX export."""

from __future__ import annotations

import torch
from torch_onnx import _onnx_program, _verification
from typing import Any


def assert_model_accuracy(
    program: _onnx_program.ONNXProgram,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Asserts that the ONNX model has the same output as the PyTorch model."""

    verification_info = _verification.verify_onnx_program(
        program, args=args, kwargs=kwargs
    )
    for info in verification_info:
        if atol is not None:
            assert (
                info.absolute_difference <= atol
            ), f"Absolute difference is greater than {atol}"
        if rtol is not None:
            assert (
                info.relative_difference <= rtol
            ), f"Relative difference is greater than {rtol}"
    return None
