import unittest

import torch

import torch_onnx


class MultiheadAttentionTest(unittest.TestCase):
    def test_export_multihead_attention(self):
        class MHAWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mha1 = torch.nn.MultiheadAttention(512, 8, batch_first=False)

            def forward(self, x, mask):
                return self.mha1(x, x, x, attn_mask=mask, need_weights=False)[0]

        mha = MHAWrapper()
        x = torch.rand(size=(62, 1, 512))
        mask = torch.randint(0, 2, size=(62, 62), dtype=bool)

        program = torch_onnx.export_compat(
            mha,
            (x, mask),
            "mha.onnx",
            verbose=True,
            input_names=["x", "mask"],
            output_names=["out"],
            dynamic_axes={
                "x": {0: "num_elem"},
                "mask": {0: "num_elem", 1: "num_elem"},
            },
        )

        torch_onnx.testing.assert_onnx_program(program)


if __name__ == "__main__":
    unittest.main()
