from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from onnxscript import ir
from torch import nn

import torch_onnx


class MNISTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.fc1 = nn.Linear(9216, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.conv1(tensor_x)
        tensor_x = F.sigmoid(tensor_x)
        tensor_x = self.conv2(tensor_x)
        tensor_x = F.sigmoid(tensor_x)
        tensor_x = F.max_pool2d(tensor_x, 2)
        tensor_x = torch.flatten(tensor_x, 1)
        tensor_x = self.fc1(tensor_x)
        tensor_x = F.sigmoid(tensor_x)
        tensor_x = self.fc2(tensor_x)
        output = F.log_softmax(tensor_x, dim=1)
        return output


class FakeTensorTest(unittest.TestCase):
    def test_saving_model_with_fake_tensor_does_not_segfault(self):
        with torch.onnx.enable_fake_mode():
            model = MNISTModel()
            fake_input = torch.rand((64, 1, 28, 28), dtype=torch.float32)

        onnx_program = torch_onnx.export_compat(model, (fake_input,))

        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(Exception):
            onnx_program.save(os.path.join(tmpdir, "fake_model.onnx"))

    def test_apply_weights(self):
        model = MNISTModel()
        state_dict = model.state_dict()
        with torch.onnx.enable_fake_mode():
            fake_model = MNISTModel()
            fake_input = torch.rand((64, 1, 28, 28), dtype=torch.float32)

        onnx_program = torch_onnx.export_compat(fake_model, (fake_input,))
        onnx_program.apply_weights(state_dict)
        inputs = (torch.rand((64, 1, 28, 28), dtype=torch.float32),)
        ep = torch.export.export(model, inputs)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            onnx_program.save(model_path)
            saved_model = ir.load(model_path)
            onnx_program_loaded = torch_onnx.ONNXProgram(saved_model, ep)
            torch_onnx.testing.assert_onnx_program(onnx_program_loaded)


if __name__ == "__main__":
    unittest.main()
