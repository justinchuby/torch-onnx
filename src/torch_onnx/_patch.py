"""Patch torch.onnx.export to use the exported program"""

from __future__ import annotations

import io
import logging
from typing import Any, Mapping, Sequence, Union

import onnx
import torch
import torch.export
from onnxscript import ir

import torch_onnx
from torch_onnx import _ir_passes

logger = logging.Logger(__name__)


def torch_onnx_export_adaptor(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    f: Union[str, io.BytesIO],
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    **_,
) -> ir.Model:
    # Test: create an exported program first
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    # TODO: Support dynamic shapes
    try:
        program = torch.export.export(model, args, kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to export the model with torch.export. "
            "This is step 1/2 of exporting the model to ONNX. Please create an issue "
            "in the PyTorch GitHub repository against the *torch.export* component and "
            "attach the full error stack as well as reproduction scripts."
        ) from e

    try:
        ir_model = torch_onnx.exported_program_to_ir(program)

        if input_names:
            _ir_passes.rename_inputs(ir_model, input_names)
        if output_names:
            _ir_passes.rename_outputs(ir_model, output_names)

        proto = ir.serde.serialize_model(ir_model)
        if proto.ByteSize() >= 1 << 31:
            logger.warning(
                "The serialized ONNX model is larger than 2GB. "
                "Saving the weights in a separate file"
            )
            onnx.save_model(proto, f, save_as_external_data=True)
        else:
            onnx.save_model(proto, f)

    except Exception as e:
        raise RuntimeError(
            "Failed to convert the exported program to an ONNX model. "
            "This is step 2/2 of exporting the model to ONNX. Please create an issue "
            "in the PyTorch GitHub repository against the *onnx* component and "
            "attach the full error stack as well as reproduction scripts."
        ) from e

    return ir_model


def patch_torch():
    torch.onnx.export = torch_onnx_export_adaptor
