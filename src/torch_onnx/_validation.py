from __future__ import annotations

from torch_onnx import _onnx_program
import torch
from torch.utils import _pytree as pytree


def verify_onnx_program(onnx_program: _onnx_program.ONNXProgram):
    exported_program = onnx_program.exported_program
    inputs = exported_program.example_inputs
    torch_outputs, _ = pytree.tree_flatten(exported_program.graph_module(*inputs))
    onnx_outputs = onnx_program(*inputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        # TODO: Use a data structure to store comparison results
        torch.testing.assert_close(torch_output, onnx_output)
