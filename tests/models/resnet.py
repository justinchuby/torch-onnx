import onnx
import torch
import torchvision
from onnxscript import ir

import torch_onnx

lower = "at_conversion"
resnet18 = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
sample_input = (torch.randn(4, 3, 224, 224),)
exported = torch.export.export(resnet18, sample_input)
print(exported)
model = torch_onnx.exported_program_to_ir(exported, lower=lower)
model.display(page=False)
proto = ir.to_proto(model)
onnx.save(proto, f"resnet18_lower_{lower}.onnx")
onnx.checker.check_model(proto, full_check=True)
