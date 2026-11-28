# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR special translations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import onnx_ir as ir
import torch

from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import _promotion, _special_math
from torch_onnx._torchlib.ops import common as common_ops

if TYPE_CHECKING:
    from torch_onnx._ir_builder import TranslationContext


aten = torch.ops.aten
_MATH_PI = math.pi


@onnx_impl((aten.erf.default, aten.special_erf.default), opset_introduced=18)
def aten_special_erf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """erf(Tensor self) -> Tensor"""
    (self,) = _promotion.promote(op, self, kind="INT_TO_FLOAT")
    return op.Erf(self)


@onnx_impl((aten.erfc.default, aten.special_erfc.default), opset_introduced=18)
def aten_special_erfc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """erfc(Tensor self) -> Tensor"""
    (self,) = _promotion.promote(op, self, kind="INT_TO_FLOAT")
    one = op.constant(1, dtype=common_ops.require_dtype(op, self))
    return op.Sub(one, op.Erf(self))


@onnx_impl(aten.special_erfcx.default, opset_introduced=18)
def aten_special_erfcx(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_erfcx(Tensor self) -> Tensor"""
    one = op.constant(1, dtype=common_ops.require_dtype(op, self))
    return op.Mul(op.Exp(op.Pow(self, 2)), op.Sub(one, op.Erf(self)))


@onnx_impl((aten.expm1.default, aten.special_expm1.default), opset_introduced=18)
def aten_special_expm1(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_expm1(Tensor self) -> Tensor"""
    (self,) = _promotion.promote(op, self, kind="INT_TO_FLOAT")
    one = op.constant(1, dtype=common_ops.require_dtype(op, self))
    return op.Sub(op.Exp(self), one)


@onnx_impl((aten.expm1.default, aten.special_expm1.default), opset_introduced=29)
def aten_special_expm1_opset29(op: TranslationContext, self: ir.Value) -> ir.Value:
    """Use the precision-preserving operator instead of subtracting from exp(x)."""
    (self,) = _promotion.promote(op, self, kind="INT_TO_FLOAT")
    return op.Expm1(self)


@onnx_impl(
    (aten.log_softmax.int, aten.special_log_softmax.default),
    opset_introduced=18,
)
def aten_special_log_softmax(
    op: TranslationContext, self: ir.Value, dim: int, dtype: int = -1
) -> ir.Value:
    """special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    self_is_scalar = len(common_ops.require_shape(op, self)) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, [0])
    result = op.LogSoftmax(self, axis=dim)
    if dtype != -1:
        result = op.Cast(result, to=dtype)
    if self_is_scalar:  # squeeze to scalar due to input is scalar
        result = op.Squeeze(result)
    return result


@onnx_impl((aten.special_sinc.default, aten.sinc.default), opset_introduced=18)
def aten_special_sinc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_sinc(Tensor self) -> Tensor"""
    (self,) = _promotion.promote(op, self, kind="INT_TO_FLOAT")
    # This computes the normalized sinc, where the input is multiplied by pi.
    # https://pytorch.org/docs/stable/special.html#torch.special.sinc
    pi_self = op.Mul(self, _MATH_PI)

    return op.Where(
        op.Equal(self, 0.0), op.CastLike(1, self), op.Div(op.Sin(pi_self), pi_self)
    )


@onnx_impl(
    (aten.xlogy.Tensor, aten.xlogy.Scalar_Self, aten.xlogy.Scalar_Other),
    opset_introduced=18,
)
def aten_special_xlogy(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """special_xlogy(Tensor self, Tensor other) -> Tensor"""
    self, other = _promotion.promote(op, self, other, kind="INT_TO_FLOAT")
    if not isinstance(self, ir.Value):
        self = op.CastLike(self, other)
    if not isinstance(other, ir.Value):
        other = op.CastLike(other, self)

    # https://pytorch.org/docs/stable/special.html#torch.special.xlogy
    # out := {
    #     NaN if other == NaN
    #     0 if self == 0
    #     self * log(other) otherwise
    # }

    nans = op.IsNaN(other)
    zeros = op.Equal(self, 0)
    xlogy = op.Mul(self, op.Log(other))
    xlogy_with_nans = op.Where(nans, other, xlogy)
    return op.Where(zeros, self, xlogy_with_nans)


# Shared numerical algorithms keep ATen calls from decomposing into unsupported
# prims; register the prims spellings too for already-decomposed programs.
prims = torch.ops.prims


def _unary_math(op, self, function, *args):
    (self,), dtype = _special_math.prepare(op, self)
    return _special_math.finish(op, function(op, self, *args), dtype)


@onnx_impl((aten.hypot.default, prims.hypot.default))
def aten_hypot(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    (self, other), dtype = _special_math.prepare(op, self, other, kind="DEFAULT")
    if not dtype.is_floating_point():
        raise TypeError("torch.hypot requires floating-point inputs")
    return _special_math.finish(op, _special_math.hypot(op, self, other), dtype)


@onnx_impl((aten.erfinv.default, aten.special_erfinv.default, prims.erf_inv.default))
def aten_erfinv(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.erfinv)


@onnx_impl((aten.i0.default, aten.special_i0.default, prims.bessel_i0.default))
def aten_i0(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.bessel_i)


@onnx_impl((aten.special_i1.default, prims.bessel_i1.default))
def aten_i1(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.bessel_i, 1)


@onnx_impl((aten.lgamma.default, aten.special_gammaln.default, prims.lgamma.default))
def aten_lgamma(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.lgamma)


@onnx_impl((aten.digamma.default, aten.special_digamma.default, prims.digamma.default))
def aten_digamma(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.digamma)


@onnx_impl(
    (aten.special_spherical_bessel_j0.default, prims.spherical_bessel_j0.default)
)
def aten_spherical_bessel_j0(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.spherical_j0)


@onnx_impl((aten.special_ndtri.default, prims.ndtri.default))
def aten_ndtri(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.ndtri)


@onnx_impl((aten.special_bessel_j0.default, prims.bessel_j0.default))
def aten_bessel_j0(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.bessel_j)


@onnx_impl((aten.special_bessel_j1.default, prims.bessel_j1.default))
def aten_bessel_j1(op: TranslationContext, self: ir.Value) -> ir.Value:
    return _unary_math(op, self, _special_math.bessel_j, 1)


@onnx_impl((aten.nextafter.default, prims.nextafter.default))
def aten_nextafter(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    (self, other), dtype = _special_math.prepare(op, self, other, kind="DEFAULT")
    if not dtype.is_floating_point():
        raise TypeError("torch.nextafter requires floating-point inputs")
    return _special_math.finish(
        op, _special_math.nextafter(op, self, other, dtype), dtype
    )


@onnx_impl((aten.special_zeta.default, prims.zeta.default))
def aten_zeta(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    (self, other), dtype = _special_math.prepare(op, self, other)
    # PyTorch's CPU Cephes implementation accumulates zeta in double. This also
    # keeps the Euler-Maclaurin terms representable when the result fits FP32
    # but the individual inverse powers do not.
    if self.dtype != ir.DataType.DOUBLE:
        self, other = (
            op.Cast(self, to=ir.DataType.DOUBLE),
            op.Cast(other, to=ir.DataType.DOUBLE),
        )
    return _special_math.finish(op, _special_math.zeta(op, self, other), dtype)


@onnx_impl((aten.igamma.default, aten.special_gammainc.default, prims.igamma.default))
def aten_igamma(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    (self, other), dtype = _special_math.prepare(op, self, other, kind="DEFAULT")
    if not dtype.is_floating_point():
        raise TypeError("torch.special.gammainc requires floating-point inputs")
    return _special_math.finish(op, _special_math.igamma(op, self, other), dtype)


@onnx_impl(
    (aten.igammac.default, aten.special_gammaincc.default, prims.igammac.default)
)
def aten_igammac(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    (self, other), dtype = _special_math.prepare(op, self, other, kind="DEFAULT")
    if not dtype.is_floating_point():
        raise TypeError("torch.special.gammaincc requires floating-point inputs")
    return _special_math.finish(
        op, _special_math.igamma(op, self, other, complement=True), dtype
    )
