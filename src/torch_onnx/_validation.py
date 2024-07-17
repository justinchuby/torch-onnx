from __future__ import annotations

from typing import Any

import torch
from torch.utils import _pytree as pytree

from torch_onnx import _onnx_program


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
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        # TODO: Use a data structure to store comparison results
        torch.testing.assert_close(torch_output, onnx_output, rtol=1e-3, atol=1e-4)
