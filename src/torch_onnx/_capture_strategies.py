"""Strategies for capturing ExportedPrograms."""

from __future__ import annotations
from typing import Any, Callable

import torch
import abc
import dataclasses

from torch_onnx import _torchscript_converter


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None
    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


def _take_first_line(text: str) -> str:
    """Take the first line of a text."""
    lines = text.split("\n", maxsplit=1)
    first_line = lines[0]
    if len(lines) > 1:
        first_line += "[...]"
    return first_line


class CaptureStrategy(abc.ABC):
    """Strategy for capturing a module as ExportedProgram."""

    def __init__(self, verbose: bool = False):
        self._verbose_print = _verbose_printer(verbose)

    def __call__(
        self,
        model: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dynamic_shapes,
    ) -> Result:
        self.enter(model)
        try:
            exported_program = self.capture(model, args, kwargs, dynamic_shapes)
        except Exception as e:
            self.failure(model, e)
            return Result(exported_program=None, success=False, exception=e)
        self.success(model)
        return Result(exported_program)

    @abc.abstractmethod
    def capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        raise NotImplementedError

    def enter(self, model: torch.nn.Module) -> None:
        return

    def success(self, model: torch.nn.Module) -> None:
        return

    def failure(self, model: torch.nn.Module, e: Exception) -> None:
        return


@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    success: bool = True
    exception: Exception | None = None


class TorchExportStrategy(CaptureStrategy):
    def capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        return torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )

    def enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`..."
        )

    def success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`... ✅"
        )

    def failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`... ❌"
        )


class TorchExportNonStrictStrategy(CaptureStrategy):
    def capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        return torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False
        )

    def enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`..."
        )

    def success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ✅"
        )

    def failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ❌"
        )


class JitTraceConvertStrategy(CaptureStrategy):
    def capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        del dynamic_shapes  # Unused

        jit_model = torch.jit.trace(
            model, example_inputs=args, check_trace=False, strict=False
        )
        return _torchscript_converter.TS2EPConverter(jit_model, args, kwargs).convert()

    def enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script..."
        )

    def success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script... ✅"
        )

    def failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script... ❌"
        )
