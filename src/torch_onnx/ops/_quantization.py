"""Experimental quantization ops."""

from typing import Optional
import torch


@torch.library.custom_op("onnx::DequantizeLinear.opset23", mutates_args=())
def DequantizeLinear_23(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: Optional[torch.Tensor] = None,
    axis: int = 1,
    block_size: int = 0,
) -> torch.Tensor:
    # TODO: Use axis and block_size
    if x_zero_point is None:
        x_zero_point = torch.tensor(0, dtype=x.dtype, device=x.device)
    return (x - x_zero_point) * x_scale


@DequantizeLinear_23.register_fake
def DequantizeLinear_23_meta(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: Optional[torch.Tensor] = None,
    axis: int = 1,
    block_size: int = 0,
):
    return torch.empty_like(x).to(x_scale.dtype)


sample_inputs = [
    (torch.tensor([1,2,3]), torch.tensor(3.14)),
]


for args in sample_inputs:
    torch.library.opcheck(
        DequantizeLinear_23,
        args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )



# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = DequantizeLinear_23(x, torch.tensor(0.1))
        return x

# create a model instance
model = M()
model.eval()

input_fp32 = torch.tensor([1,2,3])

# dynamo export
program = torch.onnx.export(
    model,
    (input_fp32,),
    dynamo=True,
    report=True
)

print(program)
