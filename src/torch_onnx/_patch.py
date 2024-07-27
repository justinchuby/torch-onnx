"""Patch torch.onnx.export to use the exported program"""

from __future__ import annotations

import inspect
import io
import logging
import os
import warnings
from typing import Any, Mapping, Sequence

import torch
import torch.export

from torch_onnx import _core, _onnx_program

logger = logging.getLogger(__name__)

_WRITE_REPORT = False
_PROFILE_EXECUTION = False
_DUMP_EXPORTED_PROGRAM = False
_VERIFY_ONNX_PROGRAM = False
_ARTIFACTS_DIR = "."


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _from_dynamic_axes_to_dynamic_shapes(
    model,
    dynamic_axes=None,
    input_names: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = {"x": {0: Dim("x_dim_0")}, "y": {1: Dim("y_dim_1")}}  # auto-generated dim names

    """
    # https://github.com/pytorch/pytorch/pull/128371
    # 1. The function does not need to provide dynamic_shapes to torch.export.export
    if dynamic_axes is None:
        return None

    if input_names is None:
        input_names = []

    sig = _signature(model)
    if len(input_names) > len(sig.parameters):
        raise ValueError(
            f"Number of input names ({len(input_names)}) should not be greater than the number of model inputs ({len(sig.parameters)})"
        )
    input_names_to_model_inputs = {}
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(input_names):
            input_names_to_model_inputs[input_names[idx]] = param_name
        else:
            input_names_to_model_inputs[param_name] = param_name

    # NOTE: torch.export.export does not support input names assignment,
    # so we need to map input names to model inputs to create dynamic_shapes
    # for the exported program
    dynamic_shapes_to_exported_program = {}
    for input_name, axes in dynamic_axes.items():
        # input_name can be either from inptu_names or from the model inputs
        if input_name not in input_names_to_model_inputs:
            raise ValueError(
                f"dynamix axis: {input_name} is not found in the input names: {input_names}"
            )
        model_input_name = input_names_to_model_inputs[input_name]
        if isinstance(axes, dict):
            dynamic_shapes_to_exported_program[model_input_name] = {
                k: torch.export.Dim(v) for k, v in axes.items()
            }
        elif isinstance(axes, list):
            dynamic_shapes_to_exported_program[model_input_name] = {
                k: torch.export.Dim(f"{model_input_name}_dim_{k}") for k in axes
            }
        else:
            raise TypeError(
                f"dynamic_axes value must be either a dict or a list, but got {type(axes)}"
            )
    # torch.export.export needs static dim to present in dynamic_shapes
    # for all input tensors, so we need to add them with None
    for input_name in sig.parameters:
        if input_name not in dynamic_shapes_to_exported_program:
            dynamic_shapes_to_exported_program[input_name] = None

    return dynamic_shapes_to_exported_program


def _get_torch_export_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    return args, kwargs


def _torch_onnx_export(
    model: torch.nn.Module | torch.export.ExportedProgram,
    args: tuple[Any, ...],
    f: str | io.BytesIO | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    external_data: bool = True,
    all_tensors_to_one_file: bool = True,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    **_,
) -> _onnx_program.ONNXProgram:
    # Set up the error reporting facilities
    report = _WRITE_REPORT or report
    verify = _VERIFY_ONNX_PROGRAM or verify
    profile = _PROFILE_EXECUTION or profile
    dump_exported_program = _DUMP_EXPORTED_PROGRAM or dump_exported_program
    artifacts_dir = _ARTIFACTS_DIR if artifacts_dir == "." else artifacts_dir

    if isinstance(model, torch.export.ExportedProgram):
        # We the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used
        dynamic_shapes = dynamic_shapes or {}
    else:
        args, kwargs = _get_torch_export_args(args, kwargs)
        if dynamic_shapes is None and dynamic_axes is not None:
            dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
                model, dynamic_axes, input_names
            )

    onnx_program = _core.export(
        model,
        args,
        kwargs,
        registry=None,
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        profile=profile,
        report=report,
        verify=verify,
        dump_exported_program=dump_exported_program,
        artifacts_dir=artifacts_dir,
    )

    if f is not None:
        # Always save the initializers as external data to reduce the size of the ONNX file
        onnx_program.save(
            f,
            include_initializers=export_params,
            external_data=external_data,
            all_tensors_to_one_file=all_tensors_to_one_file,
        )

    return onnx_program


def _torch_onnx_dynamo_export(
    model: torch.nn.Module | torch.export.ExportedProgram,
    /,
    *model_args,
    export_options: torch.onnx.ExportOptions | None = None,
    **model_kwargs,
) -> _onnx_program.ONNXProgram:
    if export_options and export_options.dynamic_shapes:
        warnings.warn("Dynamic shapes are not implemented yet.", stacklevel=1)

    return _core.export(
        model,
        model_args,
        kwargs=model_kwargs,
        report=_WRITE_REPORT,
        verify=_VERIFY_ONNX_PROGRAM,
        profile=_PROFILE_EXECUTION,
        dump_exported_program=_DUMP_EXPORTED_PROGRAM,
        artifacts_dir=_ARTIFACTS_DIR,
    )


_original_torch_onnx_export = torch.onnx.export
_original_torch_onnx_utils_export = torch.onnx.utils._export
_original_torch_onnx_dynamo_export = torch.onnx.dynamo_export


def patch_torch(
    *,
    report: bool = False,
    error_report: bool = False,  # deprecated
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    **_,
):
    if error_report:
        warnings.warn(
            "The 'error_report' argument is deprecated. Please use 'report' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        report = error_report
    global _WRITE_REPORT  # noqa: PLW0603
    _WRITE_REPORT = report
    global _PROFILE_EXECUTION  # noqa: PLW0603
    _PROFILE_EXECUTION = profile
    global _VERIFY_ONNX_PROGRAM  # noqa: PLW0603
    _VERIFY_ONNX_PROGRAM = verify
    global _DUMP_EXPORTED_PROGRAM  # noqa: PLW0603
    _DUMP_EXPORTED_PROGRAM = dump_exported_program
    global _ARTIFACTS_DIR  # noqa: PLW0603
    _ARTIFACTS_DIR = artifacts_dir
    torch.onnx.export = _torch_onnx_export
    torch.onnx.utils._export = _torch_onnx_export
    torch.onnx.dynamo_export = _torch_onnx_dynamo_export


def unpatch_torch():
    torch.onnx.export = _original_torch_onnx_export
    torch.onnx.utils._export = _original_torch_onnx_utils_export
    torch.onnx.dynamo_export = _original_torch_onnx_dynamo_export
