import unittest

import torch
import torch_onnx
import torchvision


class ResnetTest(unittest.TestCase):
    def test_resnet(self):
        torch_onnx.patch_torch(
            report=True,
            profile=True,
            verify=True,
            dump_exported_program=True,
            artifacts_dir="resnet18",
        )

        resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        ).eval()
        sample_input = (torch.randn(4, 3, 224, 224),)
        onnx_program = torch.onnx.export(
            resnet18,
            sample_input,
            "resnet18.onnx",
            opset_version=18,
        )
        assert onnx_program is not None
        print(onnx_program)
        torch_onnx.testing.assert_onnx_program(onnx_program, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
