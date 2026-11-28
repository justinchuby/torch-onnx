"""Optional torchao translations; imported only when torchao is installed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir

from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import quantized_decomposed as qdq

if TYPE_CHECKING:
    from torch_onnx._ir_builder import TranslationContext


def _expand_parameter(op: TranslationContext, value, input, block_size):
    shape = input.shape
    if shape is None or not all(isinstance(d, int) and d > 0 for d in shape):
        raise NotImplementedError(
            "torchao block quantization requires positive static input dimensions. "
            "Export with static shapes or use quantized_decomposed per-channel operators."
        )
    if len(block_size) != len(shape):
        raise ValueError(
            f"torchao block_size {block_size} must match input rank {len(shape)}."
        )
    expanded_shape = []
    parameter_shape = []
    for dim, block in zip(shape, block_size, strict=True):
        if not isinstance(block, int) or block <= 0 or dim % block:
            raise ValueError(
                f"torchao block_size {block_size} must divide input shape {shape} exactly."
            )
        expanded_shape.extend((dim // block, block))
        parameter_shape.extend((dim // block, 1))
    return op.Reshape(
        op.Expand(op.Reshape(value, parameter_shape), expanded_shape), [-1]
    )


def _parameters(op, input, block_size, scale, zero_point, dtype):
    if dtype == ir.DataType.INT16 and op.builder.opset < 21:
        raise NotImplementedError(
            "torchao int16 Q/DQ requires ONNX opset 21 or newer. "
            "Pass opset_version=21 or use int8/uint8."
        )
    if scale.dtype != ir.DataType.FLOAT:
        raise NotImplementedError(
            "torchao affine export currently requires float32 scales. "
            "Convert scales to float32 before exporting."
        )
    if dtype not in (
        ir.DataType.INT8,
        ir.DataType.UINT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
    ):
        raise NotImplementedError(
            f"torchao affine dtype {dtype} is not supported. Use int8/uint8/int16/int32 "
            "with unpacked integer storage; packed and tensor-subclass layouts require "
            "a layout-specific decoder, not ONNX DequantizePacked's GGUF byte format."
        )
    if zero_point is not None and zero_point.dtype not in (
        ir.DataType.INT8,
        ir.DataType.UINT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    ):
        raise NotImplementedError(
            "torchao affine export requires integer zero_point storage. "
            "Use int32 or int64 zero points; floating-domain offsets need a different representation."
        )
    scale = _expand_parameter(op, scale, input, block_size)
    if zero_point is not None:
        zero_point = _expand_parameter(op, zero_point, input, block_size)
    bounds = qdq._FULL_QUANTIZATION_RANGE[ir.DataType(dtype)]
    return scale, zero_point, bounds


@onnx_impl("torchao::quantize_affine")
def quantize_affine(
    op: TranslationContext,
    input,
    block_size,
    scale,
    zero_point,
    output_dtype: int,
    quant_min=None,
    quant_max=None,
):
    if input.dtype != ir.DataType.FLOAT:
        raise NotImplementedError(
            "torchao affine quantization currently requires float32 input. "
            "Convert input to float32 before exporting."
        )
    scale, zero_point, bounds = _parameters(
        op, input, block_size, scale, zero_point, output_dtype
    )
    if zero_point is None:
        raise NotImplementedError(
            "torchao quantize_affine requires an integer zero_point tensor. "
            "Pass explicit zeros for symmetric quantization."
        )
    flattened = op.Reshape(input, [-1])
    flattened.shape = ir.Shape([None])
    if output_dtype != ir.DataType.INT32:
        flattened = op.Mul(flattened, op.Reciprocal(scale))
        scale = op.Expand(op.cast_like(1.0, scale), op.Shape(scale))
    result = qdq.quantized_decomposed_quantize_per_channel(
        op,
        flattened,
        scale,
        zero_point,
        0,
        bounds[0] if quant_min is None else quant_min,
        bounds[1] if quant_max is None else quant_max,
        output_dtype,
    )
    return op.Reshape(result, op.Shape(input))


@onnx_impl("torchao::dequantize_affine")
def dequantize_affine(
    op: TranslationContext,
    input,
    block_size,
    scale,
    zero_point,
    input_dtype: int,
    quant_min=None,
    quant_max=None,
    output_dtype: int = ir.DataType.FLOAT,
):
    if output_dtype != ir.DataType.FLOAT:
        raise NotImplementedError(
            "torchao affine dequantization currently requires float32 output. "
            "Request output_dtype=torch.float32 before exporting."
        )
    scale, zero_point, bounds = _parameters(
        op, input, block_size, scale, zero_point, input_dtype
    )
    result = qdq.quantized_decomposed_dequantize_per_channel(
        op,
        op.Reshape(input, [-1]),
        scale,
        zero_point,
        0,
        bounds[0] if quant_min is None else quant_min,
        bounds[1] if quant_max is None else quant_max,
        input_dtype,
        output_dtype,
    )
    return op.Reshape(result, op.Shape(input))


def _float8_scale(op, tensor, scale):
    if scale.dtype != ir.DataType.FLOAT:
        raise NotImplementedError(
            "torchao float8 export requires float32 scales. Convert scales to float32."
        )
    if tensor.shape is None or scale.shape is None:
        raise NotImplementedError(
            "torchao float8 export requires known tensor and scale shapes. Use static shapes."
        )
    shape = tuple(tensor.shape)
    scale_shape = tuple(scale.shape)
    if all(isinstance(d, int) for d in (*shape, *scale_shape)):
        if len(scale_shape) == len(shape) and all(
            s > 0 and d % s == 0 for d, s in zip(shape, scale_shape, strict=True)
        ):
            return _expand_parameter(
                op,
                scale,
                tensor,
                [d // s for d, s in zip(shape, scale_shape, strict=True)],
            )
        if not scale_shape or all(d == 1 for d in scale_shape):
            return op.Reshape(op.Expand(scale, op.Shape(tensor)), [-1])
    raise NotImplementedError(
        f"Unsupported torchao float8 scale shape {scale_shape} for tensor {shape}. "
        "Use scalar scales or same-rank block scales dividing each tensor dimension."
    )


def _check_float8_opset(op):
    if op.builder.opset < 19:
        raise NotImplementedError(
            "torchao float8 Q/DQ requires ONNX opset 19 or newer. "
            "Pass opset_version=19 or newer."
        )


@onnx_impl("torchao::quantize_affine_float8_non_decomposed")
def quantize_float8(
    op: TranslationContext,
    tensor,
    scale,
    float8_dtype: int = ir.DataType.FLOAT8E4M3FN,
):
    _check_float8_opset(op)
    if float8_dtype not in (ir.DataType.FLOAT8E4M3FN, ir.DataType.FLOAT8E5M2):
        raise NotImplementedError(
            f"torchao float8 dtype {float8_dtype} is not supported. "
            "Use torch.float8_e4m3fn or torch.float8_e5m2."
        )
    scale = _float8_scale(op, tensor, scale)
    result = op.QuantizeLinear(
        op.Reshape(op.Cast(tensor, to=ir.DataType.FLOAT), [-1]),
        scale,
        op.Cast(0, to=float8_dtype),
        axis=0,
    )
    return op.Reshape(result, op.Shape(tensor))


@onnx_impl("torchao::dequantize_affine_float8_non_decomposed")
def dequantize_float8(
    op: TranslationContext, tensor, scale, output_dtype: int = ir.DataType.FLOAT
):
    _check_float8_opset(op)
    scale = _float8_scale(op, tensor, scale)
    result = op.DequantizeLinear(
        op.Reshape(tensor, [-1]),
        scale,
        axis=0,
        output_types=[ir.TensorType(ir.DataType.FLOAT)],
    )
    return op.Cast(op.Reshape(result, op.Shape(tensor)), to=output_dtype)
