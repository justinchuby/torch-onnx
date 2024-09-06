import unittest

import torch
import torchvision

import torch_onnx


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
        onnx_program = torch_onnx.export(
            resnet18,
            sample_input,
        )
        assert onnx_program is not None
        onnx_program.save(
            "resnet18.onnx", include_initializers=True, keep_initializers_as_inputs=True
        )
        onnx_program.save(
            "resnet18.onnx",
            include_initializers=False,
            keep_initializers_as_inputs=True,
        )
        onnx_program.save(
            "resnet18.onnx",
            include_initializers=True,
            keep_initializers_as_inputs=False,
        )
        onnx_program.save(
            "resnet18.onnx",
            include_initializers=False,
            keep_initializers_as_inputs=False,
        )
        torch_onnx.testing.assert_onnx_program(onnx_program, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
