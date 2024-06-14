import torch
import torchvision
import torch_onnx
from onnxscript import ir
import onnxscript
import onnx

lower = "at_conversion"
resnet18 = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
).eval()
sample_input = (torch.randn(4, 3, 224, 224),)
exported = torch.export.export(resnet18, sample_input)
print(exported)
model = torch_onnx.exported_program_to_ir(exported, lower=lower)
model.display(page=False)
proto = ir.to_proto(model)
proto = onnxscript.optimizer.optimize(proto)
onnx.save(proto, f"resnet18_lower_{lower}.onnx")
onnx.checker.check_model(proto, full_check=True)

torch.onnx.export(resnet18, sample_input, "resnet18_torchscript.onnx")
proto2 = onnx.load("resnet18_torchscript.onnx")
proto2 = onnx.shape_inference.infer_shapes(proto2)
onnx.save(proto2, "resnet18_torchscript.onnx")
onnx.checker.check_model(proto2, full_check=True)
