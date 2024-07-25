from __future__ import annotations

import unittest
import torch
import torch._decomp


class DecompTest(unittest.TestCase):
    def test_decomp(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool2d(x, [3, 3])

        exported_program = torch.export.export(Model(), (torch.rand(1, 3, 16, 16),))
        print("Before decomposition:")
        print(exported_program)
        decomposed = exported_program.run_decompositions(
            {
                torch.ops.aten._adaptive_avg_pool2d: torch._decomp.decompositions.adaptive_avg_pool2d,
                torch.ops.aten.adaptive_avg_pool2d: torch._decomp.decompositions.adaptive_avg_pool2d,
            }
        )
        print("After decomposition:")
        print(decomposed)


if __name__ == "__main__":
    unittest.main()
