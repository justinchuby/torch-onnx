import unittest

import torch
import torch_onnx


class Model(torch.nn.Module):
    def forward(self, arg0_1, arg1_1):
        add = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
        return add


class JitModelTest(unittest.TestCase):
    def test_jit_model_can_be_exported(self):
        traced = torch.jit.trace(Model(), (torch.rand(1), torch.rand(1)))
        torch_onnx.export(traced, (torch.rand(1), torch.rand(1)))


if __name__ == "__main__":
    unittest.main()
