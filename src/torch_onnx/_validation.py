from __future__ import annotations

from torch_onnx import _onnx_program
import torch
from torch.utils import _pytree as pytree


def verify_onnx_program(onnx_program: _onnx_program.ONNXProgram):
    exported_program = onnx_program.exported_program
    args, kwargs = exported_program.example_inputs
    torch_module = exported_program.module()
    torch_outputs, _ = pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        # TODO: Use a data structure to store comparison results
        torch.testing.assert_close(torch_output, onnx_output, rtol=1e-3, atol=1e-4)
