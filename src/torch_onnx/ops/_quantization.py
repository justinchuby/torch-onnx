"""Experimental quantization ops."""

from typing import Optional
import torch


@torch.library.custom_op("onnx::DequantizeLinear_23", mutates_args=())
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
