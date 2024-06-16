import torch
import torchvision
import torch_onnx


torch_onnx.patch_torch(error_report=True, profile=True)

resnet18 = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
sample_input = (torch.randn(4, 3, 224, 224),)
torch.onnx.export(
    resnet18,
    sample_input,
    "resnet18.onnx",
    opset_version=18,
)
