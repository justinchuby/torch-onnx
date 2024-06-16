"""Patch torch.onnx.export to use the exported program"""

from __future__ import annotations

import datetime
import inspect
import io
import logging
import traceback
import warnings
from typing import Any, Mapping, Sequence

import onnx
import torch
import torch.export
from onnxscript import ir

import torch_onnx
from torch_onnx import _ir_passes, _reporting

_BLUE = "\033[96m"
_END = "\033[0m"

logger = logging.getLogger(__name__)

WRITE_ERROR_REPORT = False


class TorchExportError(RuntimeError):
    """Error during torch.export.export."""

    pass


class OnnxConversionError(RuntimeError):
    """Error during ONNX conversion."""

    pass


class OnnxCheckerError(RuntimeError):
    """Error during ONNX model checking."""

    pass


class OnnxRuntimeError(RuntimeError):
    """Error during ONNX Runtime execution."""

    pass


class OnnxValidationError(RuntimeError):
    """Output value mismatch."""

    pass


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
    model: torch.nn.Module,
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


def _torch_onnx_export_adaptor(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    f: str | io.BytesIO,
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
) -> tuple[ir.Model, torch.export.ExportedProgram]:
    args, kwargs, dynamic_shapes = _get_torch_export_args(
        model, args, kwargs, dynamic_axes, input_names
    )
    try:
        print("Obtain model graph with `torch.export.export`... ", end="")
        program = torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
        print("✅")
    except Exception as e:
        raise TorchExportError(
            "Failed to export the model with torch.export. "
            f"{_BLUE}This is step 1/2{_END} "
            "of exporting the model to ONNX. Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and "
            "attach the full error stack as well as reproduction scripts."
        ) from e

    try:
        print("Translate the graph into ONNX... ", end="")
        ir_model = torch_onnx.exported_program_to_ir(program)

        if input_names:
            _ir_passes.rename_inputs(ir_model, input_names)
        if output_names:
            _ir_passes.rename_outputs(ir_model, output_names)

        if not export_params:
            ir_model.graph.initializers.clear()

        proto = ir.serde.serialize_model(ir_model)
        if proto.ByteSize() >= 1 << 31:
            # TODO: Create an IR pass to handle external tensors conversion
            logger.warning(
                "The serialized ONNX model is larger than 2GB. "
                "Saving the weights in a separate file"
            )
            onnx.save_model(proto, f, save_as_external_data=True)
        else:
            onnx.save_model(proto, f)
        print("✅")

    except Exception as e:
        raise OnnxConversionError(
            "Failed to convert the exported program to an ONNX model. "
            f"{_BLUE}This is step 2/2{_END} "
            "of exporting the model to ONNX. Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. "
            "You can run `torch_onnx.analyze()` to produce an error report after obtaining "
            "an ExportedProgram with `torch.export.export()`."
        ) from e

    return ir_model, program


def _torch_onnx_export_adapter_with_error_report(
    *args,
    profile: bool = False,
    error_report: bool = False,
    **kwargs,
) -> ir.Model:
    error_report = WRITE_ERROR_REPORT or error_report
    if not error_report:
        ir_model, _ = _torch_onnx_export_adaptor(*args, **kwargs)
        return ir_model

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f = args[2]
    try:
        if profile:
            import pyinstrument

            profiler = pyinstrument.Profiler()
            profiler.start()
        ir_model, program = _torch_onnx_export_adaptor(*args, **kwargs, check=True)
    except TorchExportError:
        if profile:
            profiler.stop()
            profile_result = profiler.output_text(unicode=True)
        else:
            profile_result = None
        _reporting.create_torch_export_error_report(
            f"onnx_export_{timestamp}_pt_export.md",
            traceback.format_exc(),
            profile_result=profile_result,
        )
        raise
    except OnnxConversionError:
        if profile:
            profiler.stop()
            profile_result = profiler.output_text(unicode=True)
        else:
            profile_result = None

        # Run the analysis to get the error report
        model = args[0]
        arg_args = args[1]
        arg_kwargs = kwargs.get("kwargs", {})
        dynamic_axes = kwargs.get("dynamic_axes")
        arg_args, arg_kwargs, dynamic_shapes = _get_torch_export_args(
            model, arg_args, arg_kwargs, dynamic_axes, None
        )

        program = torch.export.export(
            model, arg_args, kwargs=arg_kwargs, dynamic_shapes=dynamic_shapes
        )
        _reporting.create_onnx_export_error_report(
            f"onnx_export_{timestamp}_conversion.md",
            traceback.format_exc(),
            program,
            step=1,
            profile_result=profile_result,
        )
        raise

    if profile:
        profiler.stop()
        profile_result = profiler.output_text(unicode=True)
        _reporting.crete_onnx_export_profile_report(
            f"onnx_export_{timestamp}_profile.md", profile_result
        )

    if not error_report:
        return ir_model

    try:
        print("Run `onnx.checker` on the ONNX model... ", end="")
        if not isinstance(f, io.BytesIO):
            onnx.checker.check_model(f, full_check=True)
        else:
            # Reset the file pointer to the beginning
            f.seek(0)
            proto = onnx.load_model(f)
            onnx.checker.check_model(proto, full_check=True)
        print("✅")
    except Exception as e:
        try:
            raise OnnxCheckerError(
                "Conversion successful but the ONNX model fails ONNX checker. "
                "Please create an issue "
                f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
                "attach the full error stack as well as reproduction scripts. "
            ) from e
        except OnnxCheckerError:
            _reporting.create_onnx_export_error_report(
                f"onnx_export_{timestamp}_checker.md",
                traceback.format_exc(),
                program,
                step=2,
                profile_result=profile_result,
            )

    # try:
    #     print("Execute the model with ONNX Runtime... ", end="")
    #     print("✅")
    # except Exception as e:
    #     raise OnnxConversionError(
    #         "Conversion successful but the ONNX model fails to execute with ONNX Runtime. "
    #         "Please create an issue "
    #         f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
    #         "attach the full error stack as well as reproduction scripts. "
    #     ) from e

    return ir_model


_original_torch_onnx_export = torch.onnx.export
_original_torch_onnx_utils_export = torch.onnx.utils._export


def patch_torch(error_report: bool = False, profile: bool = False):
    global WRITE_ERROR_REPORT
    if error_report:
        WRITE_ERROR_REPORT = True
    else:
        WRITE_ERROR_REPORT = False
    torch.onnx.export = _torch_onnx_export_adapter_with_error_report
    torch.onnx.utils._export = _torch_onnx_export_adapter_with_error_report


def unpatch_torch():
    torch.onnx.export = _original_torch_onnx_export
    torch.onnx.utils._export = _original_torch_onnx_utils_export
