"""Test utilities for ONNX export."""

from __future__ import annotations

__all__ = ["assert_onnx_program"]

from typing import Any

import torch
from torch.utils import _pytree

from torch_onnx import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The :class:`torch_onnx.ONNXProgram` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
    """
    exported_program = program.exported_program
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = program(*args, **kwargs)
    # TODO(justinchuby): Include output names in the error message
    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(torch_outputs),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )
