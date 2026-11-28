# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR factories translations."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import onnx_ir as ir
import torch

from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import _promotion
from torch_onnx._torchlib.ops import common as common_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_onnx._ir_builder import TranslationContext


aten = torch.ops.aten


def _int64_arange_non_integral_uses_float_length(
    op: TranslationContext,
) -> bool:
    """Whether torch computes the arange length in float when dtype is integral.

    Newer versions of torch return 4 elements for ``torch.arange(3.1, dtype=torch.int64)``,
    while older versions return 3. This is evaluated lazily to avoid running torch code
    at import time.
    """
    return _int64_arange_float_length_capability()


@functools.cache
def _int64_arange_float_length_capability() -> bool:
    return torch.arange(3.1, dtype=torch.int64, device="cpu").numel() == 4


def _range_supported(op: TranslationContext, dtype: int) -> bool:
    """Returns true if the dtype is supported by the ONNX Range op."""
    return dtype in {
        ir.DataType.DOUBLE,
        ir.DataType.FLOAT,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _is_integral_dtype(op: TranslationContext, dtype: int) -> bool:
    return dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _is_integral_scalar(op: TranslationContext, arg: ir.Value | int | float) -> bool:
    if isinstance(arg, int):
        return True
    if isinstance(arg, float):
        return False
    return arg.dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _arange_integral_dtype_with_non_integral_args(
    op: TranslationContext,
    start: ir.Value | int | float,
    end: ir.Value | int | float,
    step: ir.Value | int | float,
    dtype: int,
) -> ir.Value:
    """Implements torch.arange for integral dtypes when not all inputs are integral."""
    if dtype is None:
        dtype = -1
    start_float = op.Cast(start, to=ir.DataType.FLOAT)
    end_float = op.Cast(end, to=ir.DataType.FLOAT)
    step_float = op.Cast(step, to=ir.DataType.FLOAT)

    length = op.Cast(
        op.Ceil(
            op.Div(op.Sub(end_float, start_float), step_float),
        ),
        to=ir.DataType.INT64,
    )
    index = op.Range(
        op.constant(0, dtype=ir.DataType.INT64),
        length,
        op.constant(1, dtype=ir.DataType.INT64),
    )

    if _range_supported(op, dtype):
        index = op.Cast(index, to=dtype)
        start = op.Cast(start, to=dtype)
        step = op.Cast(step, to=dtype)
        return op.Add(start, op.Mul(step, index))

    start = op.Cast(start, to=ir.DataType.INT64)
    step = op.Cast(step, to=ir.DataType.INT64)
    result = op.Add(start, op.Mul(step, index))
    return op.Cast(result, to=dtype)


def _integral_to_be_adjusted(op: TranslationContext, dtype: int) -> bool:
    """Returns true if the dtype is special integral handled by torch."""
    return dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
    }


def _adjust_args_for_arange_int_dtype(
    op: TranslationContext,
    start: ir.Value | int | float,
    end: ir.Value | int | float,
    step: ir.Value | int | float,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    zero = op.Cast(0.0, to=ir.DataType.FLOAT)
    start_value: ir.Value = op.Cast(start, to=ir.DataType.FLOAT)
    end_value: ir.Value = op.Cast(end, to=ir.DataType.FLOAT)
    step_value: ir.Value = op.Cast(step, to=ir.DataType.FLOAT)

    start_value = op.Where(
        op.Less(start_value, zero),
        op.Ceil(start_value),
        start_value,
    )
    start_value = op.Where(
        op.Less(step_value, zero),
        op.Floor(start_value),
        start_value,
    )

    return (start_value, end_value, step_value)


@onnx_impl(aten.arange.default)
def aten_arange(
    op: TranslationContext,
    end: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE and not isinstance(end, ir.Value):
        end = op.constant(end, dtype=ir.DataType.DOUBLE)

    if dtype == -1:
        dtype = ir.DataType.INT64 if _is_integral_scalar(op, end) else ir.DataType.FLOAT
    if (
        _is_integral_dtype(op, dtype)
        and not _is_integral_scalar(op, end)
        and (
            dtype != ir.DataType.INT64
            or _int64_arange_non_integral_uses_float_length(
                op,
            )
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(op, 0, end, 1, dtype)
    elif _range_supported(op, dtype):
        end = op.Cast(end, to=dtype)
        zero = op.Cast(0, to=dtype)
        one = op.Cast(1, to=dtype)
        result = op.Range(zero, end, one)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=ir.DataType.FLOAT)
        zero = op.constant(0.0, dtype=ir.DataType.FLOAT)
        one = op.constant(1.0, dtype=ir.DataType.FLOAT)
        result = op.Cast(op.Range(zero, end, one), to=dtype)

    return result


@onnx_impl(aten.arange.start)
def aten_arange_start(
    op: TranslationContext,
    start: ir.Value | int | float,
    end: ir.Value | int | float,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else op.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else op.constant(end, dtype=ir.DataType.DOUBLE)
        )

    if dtype == -1:
        dtype = (
            ir.DataType.INT64
            if _is_integral_scalar(op, start) and _is_integral_scalar(op, end)
            else ir.DataType.FLOAT
        )
    if (
        _is_integral_dtype(op, dtype)
        and not (_is_integral_scalar(op, start) and _is_integral_scalar(op, end))
        and (
            dtype != ir.DataType.INT64
            or _int64_arange_non_integral_uses_float_length(
                op,
            )
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(op, start, end, 1, dtype)
    elif _range_supported(op, dtype):
        end = op.Cast(end, to=dtype)
        start = op.Cast(start, to=dtype)
        one = op.Cast(1, to=dtype)
        result = op.Range(start, end, one)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=ir.DataType.FLOAT)
        start = op.Cast(start, to=ir.DataType.FLOAT)
        one = op.constant(1.0, dtype=ir.DataType.FLOAT)
        result = op.Cast(op.Range(start, end, one), to=dtype)

    return result


@onnx_impl(aten.arange.start_step)
def aten_arange_start_step(
    op: TranslationContext,
    start: ir.Value,
    end: ir.Value,
    step: ir.Value | int | float = 1,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else op.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else op.constant(end, dtype=ir.DataType.DOUBLE)
        )
        step = (
            step
            if isinstance(step, ir.Value)
            else op.constant(step, dtype=ir.DataType.DOUBLE)
        )

    if dtype == -1:
        # Scalars remain Python values until an ONNX node consumes them;
        # dynamic inputs are already IR tensors here.
        # https://github.com/microsoft/onnxscript/issues/1914
        if isinstance(start, int | float):
            start_is_int = isinstance(start, int)
        else:
            start_is_int = start.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if isinstance(end, int | float):
            end_is_int = isinstance(end, int)
        else:
            end_is_int = end.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if isinstance(step, int | float):
            step_is_int = isinstance(step, int)
        else:
            step_is_int = step.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if start_is_int and end_is_int and step_is_int:
            start = op.Cast(start, to=ir.DataType.INT64)
            end = op.Cast(end, to=ir.DataType.INT64)
            step = op.Cast(step, to=ir.DataType.INT64)
            result = op.Range(start, end, step)
        else:
            # to float
            start = op.Cast(start, to=ir.DataType.FLOAT)
            end = op.Cast(end, to=ir.DataType.FLOAT)
            step = op.Cast(step, to=ir.DataType.FLOAT)
            result = op.Range(start, end, step)
    elif (
        _is_integral_dtype(op, dtype)
        and not (
            _is_integral_scalar(op, start)
            and _is_integral_scalar(op, end)
            and _is_integral_scalar(op, step)
        )
        and (
            dtype != ir.DataType.INT64
            or _int64_arange_non_integral_uses_float_length(
                op,
            )
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(
            op, start, end, step, dtype
        )
    elif dtype in (ir.DataType.INT64, ir.DataType.DOUBLE):
        end = op.Cast(end, to=dtype)
        start = op.Cast(start, to=dtype)
        step = op.Cast(step, to=dtype)
        result = op.Range(start, end, step)
    elif _integral_to_be_adjusted(op, dtype):
        # PyTorch arange op handles these integral types differently from INT64
        # when all arguments are integral.
        start, end, step = _adjust_args_for_arange_int_dtype(op, start, end, step)
        result = op.Cast(op.Range(start, end, step), to=dtype)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16,
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=ir.DataType.FLOAT)
        start = op.Cast(start, to=ir.DataType.FLOAT)
        step = op.Cast(step, to=ir.DataType.FLOAT)
        result = op.Cast(op.Range(start, end, step), to=dtype)

    return result


@onnx_impl(aten.linspace.default)
def aten_linspace(
    op: TranslationContext,
    start: ir.Value,
    end: ir.Value,
    steps: int | ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype == -1 or dtype is None:
        dtype = ir.DataType.FLOAT
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else op.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else op.constant(end, dtype=ir.DataType.DOUBLE)
        )

    if isinstance(steps, int) and steps == 0:
        return aten_full(
            op, op.constant([0], dtype=ir.DataType.INT64), 0.0, dtype=dtype
        )
    if isinstance(steps, int) and steps == 1:
        return aten_full(
            op, op.constant([steps], dtype=ir.DataType.INT64), start, dtype=dtype
        )

    # For integer output dtypes, cast start/end to the target dtype first
    # This matches PyTorch's behavior where fractional start/end values
    # are truncated before computing the linspace
    dtype = ir.DataType(dtype)
    if dtype.is_integer():
        # Use double precision for computation to match PyTorch's internal precision
        compute_dtype = ir.DataType.DOUBLE
        # Cast to integer dtype first (truncation), then to compute dtype
        start_int = op.Cast(start, to=dtype)  # Truncate to int32/int64
        end_int = op.Cast(end, to=dtype)
        start_f = op.Cast(start_int, to=compute_dtype)  # Then to double
        end_f = op.Cast(end_int, to=compute_dtype)
    else:
        compute_dtype = dtype
        start_f = op.Cast(start, to=compute_dtype)
        end_f = op.Cast(end, to=compute_dtype)

    rg = aten_arange_start(op, 0, steps, dtype=compute_dtype)
    steps_f = op.Cast(steps, to=compute_dtype)
    one = op.Constant(value=ir.tensor(1, dtype=compute_dtype))
    two = op.Constant(value=ir.tensor(2, dtype=compute_dtype))
    steps_minus_1 = op.Sub(steps_f, one)
    denominator = op.Max(steps_minus_1, one)
    step = op.Div(op.Sub(end_f, start_f), denominator)

    # Two-sided computation for numerical stability at endpoints
    # Use forward computation for first half, backward for second half
    lin_vals = op.Where(
        op.Less(rg, op.Div(steps_f, two)),
        op.Add(start_f, op.Mul(step, rg)),
        op.Sub(
            end_f,
            op.Mul(step, op.Sub(steps_minus_1, rg)),
        ),
    )

    return op.Cast(lin_vals, to=dtype)


@onnx_impl(aten.empty.memory_format)
def aten_empty(
    op: TranslationContext,
    size: Sequence[ir.Value],
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    target_size = common_ops.merge_dims(op, size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT

    # using Zeros to simulate empty()
    zero = op.Constant(value=ir.tensor(0, dtype=ir.DataType(dtype)))
    return op.Expand(zero, target_size)


@onnx_impl(aten.empty_like.default)
def aten_empty_like(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype == -1 or dtype is None:
        zero = op.CastLike(0, self)
    else:
        zero = op.Cast(0, to=dtype)

    shape = op.Shape(self)
    return op.Expand(zero, shape)


@onnx_impl(aten.empty_strided.default)
def aten_empty_strided(
    op: TranslationContext,
    size: Sequence[ir.Value],
    stride: Sequence[ir.Value],
    layout: str = "",
    dtype: int = ir.DataType.FLOAT,
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    # empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    # using Zeros to simulate empty()

    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    target_size = common_ops.merge_dims(op, size)
    zero = op.Constant(value=ir.tensor(0, dtype=ir.DataType(dtype)))
    return op.Expand(zero, target_size)


@onnx_impl((aten.fill.Tensor, aten.fill.Scalar))
def aten_fill(op: TranslationContext, self: ir.Value, value: ir.Value) -> ir.Value:
    """fill.Tensor(Tensor self, Tensor value) -> Tensor"""

    # Cast the value before Expand so it can be constant folded
    value = (
        op.constant(value, dtype=common_ops.require_dtype(op, self))
        if isinstance(value, bool | int | float)
        else op.CastLike(value, self)
    )
    shape = op.Shape(self)
    return op.Expand(value, shape)


@onnx_impl(aten.full.default)
def aten_full(
    op: TranslationContext,
    size: ir.Value,
    fill_value: ir.Value | bool | int | float | complex,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(op, size)

    if dtype != -1:
        fill_value = (
            op.constant(fill_value, dtype=ir.DataType(dtype))
            if isinstance(fill_value, bool | int | float)
            else op.Cast(fill_value, to=dtype)
        )
    elif isinstance(fill_value, bool | int | float):
        # In particular, a Python float follows torch's current default dtype.
        fill_value = _promotion.promote(op, fill_value)[0]

    size = op.Cast(size, to=ir.DataType.INT64)
    return op.Expand(fill_value, size)


@onnx_impl(aten.full_like.default)
def aten_full_like(
    op: TranslationContext,
    self: ir.Value,
    fill_value: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    target_dtype = (
        common_ops.require_dtype(op, self) if dtype == -1 else ir.DataType(dtype)
    )
    fill_value = (
        op.constant(fill_value, dtype=target_dtype)
        if isinstance(fill_value, bool | int | float)
        else op.Cast(fill_value, to=target_dtype)
    )

    self_shape = op.Shape(self)
    return op.Expand(fill_value, self_shape)


@onnx_impl(aten.new_empty.default)
def aten_new_empty(
    op: TranslationContext,
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(op, size)

    # using zero to simulate empty array
    result = op.ConstantOfShape(size)
    if dtype == -1:
        return op.CastLike(result, self)
    return op.Cast(result, to=dtype)


@onnx_impl(aten.new_empty_strided.default)
def aten_new_empty_strided(
    op: TranslationContext,
    self: ir.Value,
    size: Sequence[ir.Value],
    stride: Sequence[ir.Value],
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    target_size = common_ops.merge_dims(op, size)

    # using zero to simulate empty array
    zero = op.ConstantOfShape(target_size)
    if dtype == -1:
        return op.CastLike(zero, self)
    return op.Cast(zero, to=dtype)


@onnx_impl(aten.new_full.default)
def aten_new_full(
    op: TranslationContext,
    self: ir.Value,
    size: ir.Value,
    fill_value: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    # new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(op, size)
    if dtype == -1:
        fill_value = op.CastLike(fill_value, self)
    else:
        fill_value = op.Cast(fill_value, to=dtype)
    return op.Expand(fill_value, size)


@onnx_impl(aten.new_ones.default)
def aten_new_ones(
    op: TranslationContext,
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(op, size)

    one = op.constant(1.0, dtype=ir.DataType.FLOAT)
    result = op.Expand(one, size)
    if dtype == -1:
        return op.CastLike(result, self)
    return op.Cast(result, to=dtype)


@onnx_impl(aten.new_zeros.default)
def aten_new_zeros(
    op: TranslationContext,
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(op, size)

    result = op.ConstantOfShape(size)
    if dtype == -1:
        return op.CastLike(result, self)
    return op.Cast(result, to=dtype)


@onnx_impl(aten.ones.default)
def aten_ones(
    op: TranslationContext,
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
):
    """ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(op, size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT
    size = op.Cast(size, to=ir.DataType.INT64)
    one = op.constant(1.0, dtype=ir.DataType.FLOAT)
    one = op.Cast(one, to=dtype)
    return op.Expand(one, size)


@onnx_impl(aten.ones_like.default)
def aten_ones_like(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    Note: dtype is an onnx enum. Users should convert torch dtype to onnx dtype
    before calling this function.
    """
    if dtype is None:
        dtype = -1
    if dtype is None:
        dtype = -1

    one = op.CastLike(1, self) if dtype == -1 else op.Cast(1, to=dtype)
    shape = op.Shape(self)
    return op.Expand(one, shape)


@onnx_impl(aten.zeros.default)
def aten_zeros(
    op: TranslationContext,
    size: Sequence[ir.Value],
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = ir.DataType.FLOAT
    target_size = common_ops.merge_dims(op, size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT

    zero = op.Constant(value=ir.tensor(0, dtype=ir.DataType(dtype)))
    return op.Expand(zero, target_size)


@onnx_impl(aten.zeros_like.default)
def aten_zeros_like(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    # Each branch explicitly casts the fill value to the requested output type.
    if dtype is None:
        dtype = -1

    zero = op.CastLike(0, self) if dtype == -1 else op.Cast(0, to=dtype)

    shape = op.Shape(self)
    return op.Expand(zero, shape)


@onnx_impl(aten.tensor.bool)
def aten_tensor_bool(op: TranslationContext, self: bool, dtype: int) -> ir.Value:
    if dtype is None:
        dtype = ir.DataType.BOOL
    tensor = op.constant(self, dtype=ir.DataType.INT64)
    return op.Cast(tensor, to=dtype)


@onnx_impl(aten.tensor.float)
def aten_tensor_float(op: TranslationContext, self: float, dtype: int) -> ir.Value:
    if dtype is None:
        dtype = ir.DataType.FLOAT
    tensor = op.constant(self, dtype=ir.DataType.FLOAT)
    return op.Cast(tensor, to=dtype)


@onnx_impl(aten.tensor.int)
def aten_tensor_int(op: TranslationContext, self: int, dtype: int) -> ir.Value:
    if dtype is None:
        dtype = ir.DataType.INT64
    tensor = op.constant(self, dtype=ir.DataType.INT64)
    return op.Cast(tensor, to=dtype)


def _window_dtype(dtype: int | None) -> ir.DataType:
    if dtype is None or dtype == -1:
        dtype = _promotion._ONNX_DTYPES[torch.get_default_dtype()]
    dtype = ir.DataType(dtype)
    if dtype not in (
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
    ):
        raise TypeError("Window functions require a floating-point dtype")
    return dtype


@onnx_impl((aten.blackman_window.default, aten.blackman_window.periodic))
def aten_blackman_window(
    op: TranslationContext,
    window_length: int,
    periodic: bool = True,
    dtype: int | None = None,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    dtype = _window_dtype(dtype)
    if dtype != ir.DataType.BFLOAT16:
        return _window(op, "BlackmanWindow", window_length, dtype, periodic)
    phase = _window_phase(op, window_length, periodic, ir.DataType.FLOAT)
    result = op.Add(
        op.Sub(op.CastLike(0.42, phase), op.Mul(op.Cos(phase), 0.5)),
        op.Mul(op.Cos(op.Mul(phase, 2)), 0.08),
    )
    return _finish_window(op, result, window_length, dtype)


@onnx_impl(
    (
        aten.hamming_window.default,
        aten.hamming_window.periodic,
        aten.hamming_window.periodic_alpha,
        aten.hamming_window.periodic_alpha_beta,
    )
)
def aten_hamming_window(
    op: TranslationContext,
    window_length: int,
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
    dtype: int | None = None,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    dtype = _window_dtype(dtype)
    work_dtype = (
        ir.DataType.DOUBLE if dtype == ir.DataType.DOUBLE else ir.DataType.FLOAT
    )
    phase = _window_phase(op, window_length, periodic, work_dtype)
    result = op.Sub(op.constant(alpha, dtype=phase.dtype), op.Mul(op.Cos(phase), beta))
    return _finish_window(op, result, window_length, dtype)


@onnx_impl((aten.hann_window.default, aten.hann_window.periodic))
def aten_hann_window(
    op: TranslationContext,
    window_length: int,
    periodic: bool = True,
    dtype: int | None = None,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    dtype = _window_dtype(dtype)
    if dtype != ir.DataType.BFLOAT16:
        return _window(op, "HannWindow", window_length, dtype, periodic)
    return aten_hamming_window(op, window_length, periodic, 0.5, 0.5, dtype)


@onnx_impl((aten.bartlett_window.default, aten.bartlett_window.periodic))
def aten_bartlett_window(
    op: TranslationContext,
    window_length: int,
    periodic: bool = True,
    dtype: int | None = None,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    dtype = _window_dtype(dtype)
    work_dtype = (
        ir.DataType.DOUBLE if dtype == ir.DataType.DOUBLE else ir.DataType.FLOAT
    )
    fraction = _window_fraction(op, window_length, periodic, work_dtype)
    result = op.Sub(op.CastLike(1, fraction), op.Abs(op.Sub(fraction, 1)))
    return _finish_window(op, result, window_length, dtype)


@onnx_impl(
    (aten.kaiser_window.default, aten.kaiser_window.periodic, aten.kaiser_window.beta)
)
def aten_kaiser_window(
    op: TranslationContext,
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    dtype: int | None = None,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    from torch_onnx._torchlib.ops import _special_math

    dtype = _window_dtype(dtype)
    work_dtype = (
        ir.DataType.DOUBLE if dtype == ir.DataType.DOUBLE else ir.DataType.FLOAT
    )
    fraction = op.Sub(_window_fraction(op, window_length, periodic, work_dtype), 1)
    beta_value = op.constant(abs(beta), dtype=work_dtype)
    argument = op.Mul(
        op.Sqrt(
            op.Max(op.Sub(op.CastLike(1, fraction), op.Mul(fraction, fraction)), 0)
        ),
        beta_value,
    )
    result = op.Div(
        _special_math.bessel_i(op, argument), _special_math.bessel_i(op, beta_value)
    )
    return _finish_window(op, result, window_length, dtype)


def _window_fraction(op, length, periodic, dtype):
    if isinstance(length, int) and length < 0:
        raise ValueError("Window length must be nonnegative")
    positions = op.Cast(op.Range(0, length, 1), to=dtype)
    denominator = length if periodic else op.Sub(length, 1)
    denominator = op.Cast(op.Max(denominator, 1), to=dtype)
    return op.Div(op.Mul(positions, 2), denominator)


def _window_phase(op, length, periodic, dtype):
    return op.Mul(_window_fraction(op, length, periodic, dtype), math.pi)


def _finish_window(op, result, length, dtype):
    result = op.Where(op.Equal(length, 1), op.CastLike(1, result), result)
    return result if result.dtype == dtype else op.Cast(result, to=dtype)


def _window(
    op: TranslationContext,
    op_type: str,
    length: int | ir.Value,
    dtype: int,
    periodic: bool = True,
) -> ir.Value:
    if isinstance(length, int) and length <= 1:
        if length < 0:
            raise ValueError(f"{op_type} requires a nonnegative length, got {length}")
        return op.constant([1] * length, dtype=ir.DataType(dtype))
    safe_length = length if isinstance(length, int) else op.Max(length, 2)
    window = op.builder.call(
        op_type,
        safe_length,
        output_datatype=dtype,
        periodic=periodic,
        _output_types=[ir.TensorType(ir.DataType(dtype))],
    )[0]
    if isinstance(length, int):
        return window
    result = op.Slice(window, [0], op.Reshape(length, [1]))
    return _finish_window(op, result, length, dtype)


@onnx_impl(aten.bernoulli.default)
def aten_bernoulli(op: TranslationContext, self: ir.Value) -> ir.Value:
    """Proximal implementation of aten::bernoulli.default

    Note that due to the limitation of ONNX, we ignore the `generator` argument in
      aten::bernoulli.default(Tensor self, *, Generator? generator=None) -> Tensor
    """
    return op.Bernoulli(self)


@onnx_impl(aten.bernoulli.p)
def aten_bernoulli_p(op: TranslationContext, self: ir.Value, p: float) -> ir.Value:
    """Proximal implementation of aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None)

    Ignore `generator` due to the limit on ONNX expressiveness.
    """
    # NOTE: We will lose some precision when input is float64 but that's considered insignificant
    self_float = op.Cast(self, to=ir.DataType.FLOAT)
    rands = op.RandomUniformLike(
        self_float, high=1.0, low=0.0, output_types=[self_float.type]
    )
    sampled = op.Less(rands, p)
    return op.CastLike(sampled, self)


@onnx_impl(aten.normal_functional.default)
def aten_normal(
    op: TranslationContext,
    self: ir.Value,
    mean: float = 0.0,
    std: float = 1.0,
) -> ir.Value:
    """normal_functional(Tensor self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor"""

    result = op.RandomNormalLike(self, mean=mean, scale=std, output_types=[self.type])
    return result


@onnx_impl(aten.normal.float_float)
def aten_normal_float_float(
    op: TranslationContext,
    mean: float,
    std: float,
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(op, size)

    if dtype == -1:
        dtype = ir.DataType.FLOAT
    # Create a dummy tensor for RandomNormalLike to get the shape
    dummy_tensor = op.ConstantOfShape(size)
    result = op.RandomNormalLike(
        dummy_tensor, mean=mean, scale=std, output_types=[dummy_tensor.type]
    )
    return op.Cast(result, to=dtype)


@onnx_impl(aten.normal.float_Tensor)
def aten_normal_float_tensor(
    op: TranslationContext, mean: ir.Value, std: ir.Value
) -> ir.Value:
    """normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor"""

    mean_casted = op.CastLike(mean, std)
    sampled = op.RandomNormalLike(std, mean=0.0, scale=1.0, output_types=[std.type])
    # Transform the distribution to the mean and std
    return op.Add(op.Mul(std, sampled), mean_casted)


@onnx_impl(aten.normal.Tensor_float)
def aten_normal_tensor_float(
    op: TranslationContext, mean: ir.Value, std: ir.Value
) -> ir.Value:
    """normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor"""

    sampled = op.RandomNormalLike(mean, mean=0.0, scale=1.0, output_types=[mean.type])
    # Transform the distribution to the mean and std
    return op.Add(
        op.Mul(op.CastLike(std, sampled), sampled),
        mean,
    )


@onnx_impl(aten.normal.Tensor_Tensor)
def aten_normal_tensor_tensor(
    op: TranslationContext, mean: ir.Value, std: ir.Value
) -> ir.Value:
    """normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor"""

    std = _promotion.cast(op, std, _promotion.source_dtype(op, mean))
    sampled = op.RandomNormalLike(mean, mean=0.0, scale=1.0, output_types=[mean.type])
    # Transform the distribution to the mean and std
    return op.Add(op.Mul(std, sampled), mean)


@onnx_impl(aten.rand.default)
def aten_rand(
    op: TranslationContext,
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(op, size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT
    shaper = op.ConstantOfShape(size)
    return op.RandomUniformLike(
        shaper, dtype=dtype, output_types=[ir.TensorType(ir.DataType(dtype))]
    )


@onnx_impl(aten.rand_like.default)
def aten_rand_like(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype == -1:
        return op.RandomUniformLike(self, output_types=[self.type])
    return op.RandomUniformLike(
        self, dtype=dtype, output_types=[ir.TensorType(ir.DataType(dtype))]
    )


@onnx_impl(aten.randint.default)
def aten_randint(
    op: TranslationContext,
    high: ir.Value,
    size: ir.Value,
    dtype: int = ir.DataType.INT64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = ir.DataType.INT64
    size = common_ops.merge_dims(op, size)

    shaper = op.ConstantOfShape(size)
    rand = op.RandomUniformLike(shaper, output_types=[shaper.type])
    # Scale to [0, high] first
    rand_scaled = op.Mul(rand, op.CastLike(high, rand))
    # Round to ints
    rand_int = op.Floor(rand_scaled)
    return op.Cast(rand_int, to=dtype)


@onnx_impl(aten.randint.low)
def aten_randint_low(
    op: TranslationContext,
    low: ir.Value,
    high: ir.Value,
    size: ir.Value,
    dtype: int = ir.DataType.INT64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = ir.DataType.INT64
    size = common_ops.merge_dims(op, size)

    shaper = op.ConstantOfShape(size)
    rand = op.RandomUniformLike(shaper, output_types=[shaper.type])
    # Translate to [low, high] first
    high = op.Cast(high, to=ir.DataType.FLOAT)
    low = op.Cast(low, to=ir.DataType.FLOAT)
    rand_translated = op.Add(op.Mul(rand, op.Sub(high, low)), low)
    # Round to ints
    rand_int = op.Floor(rand_translated)
    return op.Cast(rand_int, to=dtype)


@onnx_impl(aten.randint_like.default)
def aten_randint_like(
    op: TranslationContext,
    self: ir.Value,
    high: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    self_float = op.Cast(self, to=ir.DataType.FLOAT)
    rand = op.RandomUniformLike(self_float, output_types=[self_float.type])
    # Scale to [0, high] first
    rand_scaled = op.Mul(rand, op.CastLike(high, rand))
    # Round to ints
    rand_int = op.Floor(rand_scaled)
    if dtype == -1:
        return op.CastLike(rand_int, self)
    return op.Cast(rand_int, to=dtype)


@onnx_impl(aten.randint_like.low_dtype)
def aten_randint_like_low_dtype(
    op: TranslationContext,
    self: ir.Value,
    low: ir.Value,
    high: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    This is the TorchLib overload for aten::randint_like.low_dtype when dtype is None.
    """
    if dtype is None:
        dtype = -1

    self_float = op.Cast(self, to=ir.DataType.FLOAT)
    rand = op.RandomUniformLike(self_float, output_types=[self_float.type])
    # Translate to [low, high] first
    high = op.Cast(high, to=ir.DataType.FLOAT)
    low = op.Cast(low, to=ir.DataType.FLOAT)
    rand_translated = op.Add(op.Mul(rand, op.Sub(high, low)), low)
    # Round to ints
    rand_int = op.Floor(rand_translated)
    if dtype == -1:
        return op.CastLike(rand_int, self)
    return op.Cast(rand_int, to=dtype)


@onnx_impl(aten.randn.default)
def aten_randn(
    op: TranslationContext,
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(op, size)

    shaper = op.ConstantOfShape(size)
    return op.RandomNormalLike(
        shaper, dtype=dtype, output_types=[ir.TensorType(ir.DataType(dtype))]
    )


@onnx_impl(aten.randn_like.default)
def aten_randn_like(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype == -1:
        return op.RandomNormalLike(self, output_types=[self.type])
    return op.RandomNormalLike(
        self, dtype=dtype, output_types=[ir.TensorType(ir.DataType(dtype))]
    )


@onnx_impl(aten.multinomial.default)
def aten_multinomial(
    op: TranslationContext,
    self: ir.Value,
    num_samples: int,
    replacement: bool = False,
) -> ir.Value:
    """multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor"""
    # ONNX Multinomial doesn't support 1D input
    if len(common_ops.require_shape(op, self)) == 1:
        unsqueezed_input = op.Unsqueeze(self, [0])
    else:
        unsqueezed_input = self
    # ONNX multinomial expects log probability
    log_input = op.Log(unsqueezed_input)
    result = op.Multinomial(
        log_input,
        dtype=ir.DataType.INT64,
        sample_size=num_samples,
        output_types=[ir.TensorType(ir.DataType.INT64)],
    )
    if len(common_ops.require_shape(op, self)) == 1:
        result = op.Squeeze(result, [0])
    return result


@onnx_impl((aten.randperm.default, aten.randperm.generator))
def aten_randperm(
    op: TranslationContext,
    n: ir.Value,
    generator=None,
    dtype: int = ir.DataType.INT64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """A permutation from sorted random keys, without promising Torch RNG identity."""
    if generator is not None:
        raise NotImplementedError(
            "Exporting randperm with an explicit Generator is unsupported."
        )
    n = op.Cast(n, to=ir.DataType.INT64)
    shaper = op.Expand(op.constant(0.0, dtype=ir.DataType.DOUBLE), op.Reshape(n, [1]))
    # Double precision keys make accidental ties much less likely than float32.
    keys = op.RandomUniformLike(
        shaper,
        dtype=ir.DataType.DOUBLE,
        output_types=[ir.TensorType(ir.DataType.DOUBLE)],
    )
    _, indices = op.TopK(
        keys, op.Reshape(n, [1]), axis=0, largest=0, sorted=1, outputs=2
    )
    return op.Cast(indices, to=ir.DataType.INT64 if dtype in (None, -1) else dtype)
