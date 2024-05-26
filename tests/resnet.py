import torch
import torchvision
import torch_onnx
from onnxscript import ir
import onnx


resnet18 = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
sample_input = (torch.randn(4, 3, 224, 224),)
exported = torch.export.export(resnet18, sample_input)
model = torch_onnx.exported_program_to_ir(exported)
pb = ir.to_proto(model)
onnx.save(pb, "resnet18.onnx")
