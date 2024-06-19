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
from torch_onnx import _ir_passes, _reporting, _onnx_program

_BLUE = "\033[96m"
_END = "\033[0m"

logger = logging.getLogger(__name__)

WRITE_ERROR_REPORT = False
WRITE_PROFILE_REPORT = False


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


def _maybe_start_profiler(should_profile: bool) -> Any:
    if should_profile:
        import pyinstrument

        profiler = pyinstrument.Profiler(async_mode="disabled")
        profiler.start()
        return profiler
    return None


def _maybe_stop_profiler_and_get_result(profiler) -> str | None:
    if profiler is None:
        return None
    profiler.stop()
    return profiler.output_text(unicode=True)


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
    **_,
) -> _onnx_program.ONNXProgram:
    # Set up the error reporting facilities
    error_report = WRITE_ERROR_REPORT or error_report
    profile = WRITE_PROFILE_REPORT or profile
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    profiler = _maybe_start_profiler(profile)

    # Stage 1: Export the model with torch.export.export if the model is not already an ExportedProgram
    if not isinstance(model, torch.export.ExportedProgram):
        args, kwargs, dynamic_shapes = _get_torch_export_args(
            model, args, kwargs, dynamic_axes, input_names
        )
        try:
            print(
                "Obtain model graph with `torch.export.export`... ", end="", flush=True
            )
            program = torch.export.export(
                model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
            )
            print("✅")
        except Exception as e:
            profile_result = _maybe_stop_profiler_and_get_result(profiler)

            if error_report:
                error_report_path = f"onnx_export_{timestamp}_pt_export.md"
                _reporting.create_torch_export_error_report(
                    error_report_path,
                    traceback.format_exc(),
                    profile_result=profile_result,
                )
            else:
                error_report_path = None

            raise TorchExportError(
                "Failed to export the model with torch.export. "
                f"{_BLUE}This is step 1/2{_END} "
                "of exporting the model to ONNX. Please create an issue "
                f"in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and "
                "attach the full error stack as well as reproduction scripts."
                + f" Error report has been saved to '{error_report_path}'."
                if error_report
                else ""
            ) from e
    else:
        print("Obtain model graph with `torch.export.export`... ", end="", flush=True)
        program = model
        print("✅")

    # Stage 2: Convert the exported program to an ONNX model
    try:
        print("Translate the graph into ONNX... ", end="", flush=True)
        ir_model = torch_onnx.exported_program_to_ir(program)

        if input_names:
            _ir_passes.rename_inputs(ir_model, input_names)
        if output_names:
            _ir_passes.rename_outputs(ir_model, output_names)

        if not export_params:
            ir_model.graph.initializers.clear()

        onnx_program = _onnx_program.ONNXProgram(ir_model, program)
        if f is not None:
            onnx_program.save(f)
        print("✅")

    except Exception as e:
        profile_result = _maybe_stop_profiler_and_get_result(profiler)

        if error_report:
            error_report_path = f"onnx_export_{timestamp}_conversion.md"

            # Run the analysis to get the error report
            _reporting.create_onnx_export_error_report(
                error_report_path,
                traceback.format_exc(),
                program,
                step=1,
                profile_result=profile_result,
            )
        else:
            error_report_path = None

        raise OnnxConversionError(
            "Failed to convert the exported program to an ONNX model. "
            f"{_BLUE}This is step 2/2{_END} "
            "of exporting the model to ONNX. Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. "
            "You can run `torch_onnx.analyze()` to produce an error report after obtaining "
            "an ExportedProgram with `torch.export.export()`."
            + f" Error report has been saved to '{error_report_path}'."
            if error_report
            else ""
        ) from e

    profile_result = _maybe_stop_profiler_and_get_result(profiler)
    if not error_report:
        # Return if error report is not requested
        if profile:
            assert profile_result is not None
            _reporting.crete_onnx_export_profile_report(
                f"onnx_export_{timestamp}_profile.md",
                onnx_program.exported_program,
                profile_result,
                step=1,
            )
        return onnx_program

    # Stage 3: (When error report is requested) Check the ONNX model with ONNX checker
    try:
        print("Run `onnx.checker` on the ONNX model... ", end="", flush=True)
        if f is None:
            onnx.checker.check_model(onnx_program.model_proto, full_check=True)
        elif not isinstance(f, io.BytesIO):
            onnx.checker.check_model(f, full_check=True)
        else:
            # Reset the file pointer to the beginning
            f.seek(0)
            proto = onnx.load_model(f)
            onnx.checker.check_model(proto, full_check=True)
        print("✅")
    except Exception as e:
        if error_report:
            _reporting.create_onnx_export_error_report(
                f"onnx_export_{timestamp}_checker.md",
                traceback.format_exc(),
                onnx_program.exported_program,
                step=2,
                profile_result=profile_result,
                ir_model=onnx_program.model,
            )
        raise OnnxCheckerError(
            "Conversion successful but the ONNX model fails ONNX checker. "
            "Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. "
        ) from e

    # Stage 4: (When error report is requested) Execute the model with ONNX Runtime
    # try:
    #     print("Execute the model with ONNX Runtime... ", end="", flush=True)
    #     print("✅")
    # except Exception as e:
    #     raise OnnxConversionError(
    #         "Conversion successful but the ONNX model fails to execute with ONNX Runtime. "
    #         "Please create an issue "
    #         f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
    #         "attach the full error stack as well as reproduction scripts. "
    #     ) from e

    # Stage 5: (When error report is requested) Validate the output values
    # TODO

    if profile:
        assert profile_result is not None
        _reporting.crete_onnx_export_profile_report(
            f"onnx_export_{timestamp}_profile.md",
            onnx_program.exported_program,
            profile_result,
            step=4,
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
        raise NotImplementedError("Dynamic shapes are not implemented yet.")
    return _torch_onnx_export(
        model,
        model_args,
        kwargs=model_kwargs,
    )


_original_torch_onnx_export = torch.onnx.export
_original_torch_onnx_utils_export = torch.onnx.utils._export
_original_torch_onnx_dynamo_export = torch.onnx.dynamo_export


def patch_torch(error_report: bool = False, profile: bool = False):
    global WRITE_ERROR_REPORT  # noqa: PLW0603
    WRITE_ERROR_REPORT = error_report
    global WRITE_PROFILE_REPORT  # noqa: PLW0603
    WRITE_PROFILE_REPORT = profile
    torch.onnx.export = _torch_onnx_export
    torch.onnx.utils._export = _torch_onnx_export
    torch.onnx.dynamo_export = _torch_onnx_dynamo_export


def unpatch_torch():
    torch.onnx.export = _original_torch_onnx_export
    torch.onnx.utils._export = _original_torch_onnx_utils_export
    torch.onnx.dynamo_export = _original_torch_onnx_dynamo_export
