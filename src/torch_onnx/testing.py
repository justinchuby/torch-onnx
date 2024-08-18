"""Test utilities for ONNX export."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils import _pytree

from torch_onnx import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Asserts that the ONNX model produces the same output as the PyTorch ExportedProgram."""

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
        tuple(torch_outputs),
        tuple(onnx_outputs),
        atol=atol,
        rtol=rtol,
        equal_nan=True,
        check_device=False,
    )
