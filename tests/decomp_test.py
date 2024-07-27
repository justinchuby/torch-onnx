from __future__ import annotations

import unittest

import torch
import torch._decomp
import torch_onnx


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.adaptive_avg_pool2d(x, [3, 3])


class DecompTest(unittest.TestCase):
    def test_decomp(self):
        exported_program = torch.export.export(Model(), (torch.rand(1, 3, 16, 16),))
        print("Before decomposition:")
        print(exported_program)
        # NOTE: It is important to include the `default` in keys
        decomposed = exported_program.run_decompositions(
            {
                torch.ops.aten._adaptive_avg_pool2d.default: torch._decomp.decompositions.adaptive_avg_pool2d,
                torch.ops.aten.adaptive_avg_pool2d.default: torch._decomp.decompositions.adaptive_avg_pool2d,
            }
        )
        print("After decomposition:")
        print(decomposed)

    def test_export(self):
        model = Model()
        input = torch.rand(1, 3, 16, 16)
        onnx_program = torch_onnx.export(model, (input,))
        print(onnx_program.exported_program)


if __name__ == "__main__":
    unittest.main()
