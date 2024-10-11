from __future__ import annotations

import functools
import itertools
from typing import Any, List

import torch
import torchvision
from torch import testing as torch_testing
from torch.testing._internal import (
    common_device_type,
    common_dtype,
    common_methods_invocations,
)
from torch.testing._internal.opinfo import core as opinfo_core
import torch_onnx
import torch_onnx.ops


def _get_supported_dtypes(op: str, opset_version: int) -> common_dtype._dispatch_dtypes:
    raise NotImplementedError


class OnnxUnaryUfuncInfo(opinfo_core.UnaryUfuncInfo):
    def __init__(self, name: str, **kwargs) -> None:
        opinfo_name = f"ops.onnx.{name}"
        op = getattr(torch_onnx.ops, name)
        op_name, opset_version = name.rsplit("_", 1)
        dtypes = _get_supported_dtypes(op_name, int(opset_version))
        super().__init__(opinfo_name, op=op, dtypes=dtypes, **kwargs)


class OnnxBinaryUfuncInfo(opinfo_core.BinaryUfuncInfo):
    def __init__(self, name: str, **kwargs) -> None:
        opinfo_name = f"ops.onnx.{name}"
        op = getattr(torch_onnx.ops, name)
        op_name, opset_version = name.rsplit("_", 1)
        dtypes = _get_supported_dtypes(op_name, int(opset_version))
        super().__init__(opinfo_name, op=op, dtypes=dtypes, **kwargs)


class PlaceHolder(opinfo_core.OpInfo):
    pass

op_db = [
    OnnxUnaryUfuncInfo("Abs_13"),
    OnnxUnaryUfuncInfo("Acos_22"),
    OnnxUnaryUfuncInfo("Acosh_22"),
    OnnxUnaryUfuncInfo("Add_14"),
    PlaceHolder("AffineGrid_20"),
    OnnxUnaryUfuncInfo("And_7"),
    PlaceHolder("ArgMax_13"),
    PlaceHolder("ArgMin_13"),
    OnnxUnaryUfuncInfo("Asin_22"),
    OnnxUnaryUfuncInfo("Asinh_22"),
    OnnxUnaryUfuncInfo("Atan_22"),
    OnnxUnaryUfuncInfo("Atanh_22"),
    PlaceHolder("AveragePool_22"),
    PlaceHolder("BatchNormalization_15"),
    PlaceHolder("Bernoulli_22"),
    PlaceHolder("BitShift_11"),
    OnnxBinaryUfuncInfo("BitwiseAnd_18"),
    OnnxBinaryUfuncInfo("BitwiseNot_18"),
    OnnxBinaryUfuncInfo("BitwiseOr_18"),
    OnnxBinaryUfuncInfo("BitwiseXor_18"),
    PlaceHolder("BlackmanWindow_17"),
    PlaceHolder("Cast_21"),
    PlaceHolder("CastLike_21"),
    OnnxUnaryUfuncInfo("Ceil_13"),
    PlaceHolder("Celu_12"),
]
