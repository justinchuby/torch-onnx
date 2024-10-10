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
    def __init__(self, name: str) -> None:
        opinfo_name = f"ops.onnx.{name}"
        op = getattr(torch_onnx.ops, name)
        op_name, opset_version = name.rsplit("_", 1)
        dtypes = _get_supported_dtypes(op_name, int(opset_version))
        super().__init__(opinfo_name, op=op, dtypes=dtypes)


op_db = [
    OnnxUnaryUfuncInfo("Abs_13"),
    OnnxUnaryUfuncInfo("Acos_7"),
    OnnxUnaryUfuncInfo("Acosh_9"),
]
