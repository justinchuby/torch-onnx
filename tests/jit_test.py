import torch
import torch_onnx

class Model(torch.nn.Module):
    def forward(self, arg0_1, arg1_1):
        add = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
        return add


traced = torch.jit.trace(Model(), (torch.rand(1), torch.rand(1)))
print(torch_onnx.export(traced, (torch.rand(1), torch.rand(1))))