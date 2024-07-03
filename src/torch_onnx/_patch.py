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

from torch_onnx import _onnx_program, _core


logger = logging.getLogger(__name__)

WRITE_ERROR_REPORT = False
WRITE_PROFILE_REPORT = False
DUMP_EXPORTED_PROGRAM = False
ARTIFACTS_DIR = "."


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
    if dynamic_axes is None:
        return None

    input_names_set = set() if input_names is None else set(input_names)

    dynamic_shapes = {}
    for input_name, axes in dynamic_axes.items():
        if input_name in input_names_set:
            raise ValueError(
                "input names is not supported yet. Please use model forward signature."
            )
        if isinstance(axes, dict):
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(v) for k, v in axes.items()
            }
        elif isinstance(axes, list):
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(f"{input_name}_dim_{k}") for k in axes
            }
        else:
            raise TypeError(
                f"dynamic_axes value must be either a dict or a list, but got {type(axes)}"
            )
    # torch.export.export needs static dim to present in dynamic_shapes
    # for all input tensors, so we need to add them with None
    try:
        sig = _signature(model)
    except ValueError as e:
        warnings.warn(
            f"{e}, skipping auto filling None on static axes...", stacklevel=1
        )
        return dynamic_shapes
    for input_name in sig.parameters:
        if input_name not in dynamic_shapes:
            dynamic_shapes[input_name] = None
    return dynamic_shapes


def _get_torch_export_args(
    model: torch.nn.Module | torch.export.ExportedProgram,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None,
    input_names: Sequence[str] | None,
):
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]

    dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
        model, dynamic_axes, input_names
    )
    return args, kwargs, dynamic_shapes


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
    profile: bool = False,
    error_report: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    **_,
) -> _onnx_program.ONNXProgram:
    # Set up the error reporting facilities
    error_report = WRITE_ERROR_REPORT or error_report
    profile = WRITE_PROFILE_REPORT or profile
    dump_exported_program = DUMP_EXPORTED_PROGRAM or dump_exported_program
    artifacts_dir = ARTIFACTS_DIR if artifacts_dir == "." else artifacts_dir

    if not isinstance(model, torch.export.ExportedProgram):
        args, kwargs, dynamic_shapes = _get_torch_export_args(
            model, args, kwargs, dynamic_axes, input_names
        )
    else:
        args, kwargs, dynamic_shapes = _get_torch_export_args(
            model, args, kwargs, None, input_names
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
        error_report=error_report,
        dump_exported_program=dump_exported_program,
        artifacts_dir=artifacts_dir,
    )

    if f is not None:
        onnx_program.save(f, include_initializers=export_params)

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
    )


_original_torch_onnx_export = torch.onnx.export
_original_torch_onnx_utils_export = torch.onnx.utils._export
_original_torch_onnx_dynamo_export = torch.onnx.dynamo_export


def patch_torch(
    error_report: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    **_,
):
    global WRITE_ERROR_REPORT  # noqa: PLW0603
    WRITE_ERROR_REPORT = error_report
    global WRITE_PROFILE_REPORT  # noqa: PLW0603
    WRITE_PROFILE_REPORT = profile
    global DUMP_EXPORTED_PROGRAM  # noqa: PLW0603
    DUMP_EXPORTED_PROGRAM = dump_exported_program
    global ARTIFACTS_DIR  # noqa: PLW0603
    ARTIFACTS_DIR = artifacts_dir
    torch.onnx.export = _torch_onnx_export
    torch.onnx.utils._export = _torch_onnx_export
    torch.onnx.dynamo_export = _torch_onnx_dynamo_export


def unpatch_torch():
    torch.onnx.export = _original_torch_onnx_export
    torch.onnx.utils._export = _original_torch_onnx_utils_export
    torch.onnx.dynamo_export = _original_torch_onnx_dynamo_export
