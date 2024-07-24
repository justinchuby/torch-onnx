"""Strategies for capturing ExportedPrograms."""

from __future__ import annotations

import torch
import abc
import dataclasses

from torch_onnx import _torchscript_converter


class CaptureStrategy(abc.ABC):
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    @abc.abstractmethod
    def __call__(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        raise NotImplementedError

    def enter(self):
        return

    def success(self):
        return

    def failure(self, e: Exception):
        return


@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    success: bool = True
    exception: Exception | None = None


class TorchExportStrategy(CaptureStrategy):
    def __call__(self, model, args, kwargs, dynamic_shapes) -> Result:
        try:
            exported_program = torch.export.export(
                model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
            )
        except Exception as e:
            self.failure(e)
            return Result(exported_program=None, success=False, exception=e)

        self.success()
        return Result(exported_program)

    def success(self):
        if self._verbose:
            print("Exported program successfully captured.")


class TorchExportNonStrictStrategy(CaptureStrategy):
    def __call__(self, model, args, kwargs, dynamic_shapes) -> Result:
        try:
            exported_program = torch.export.export(
                model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False
            )
        except Exception as e:
            self.failure(e)
            return Result(exported_program=None, success=False, exception=e)

        self.success()
        return Result(exported_program)


class JitTraceConvertStrategy(CaptureStrategy):
    def __call__(self, model, args, kwargs, dynamic_shapes) -> Result:
        try:
            jit_model = torch.jit.trace(
                model, example_inputs=args, check_trace=False, strict=False
            )
            exported_program = _torchscript_converter.TS2EPConverter(
                jit_model, args, kwargs
            ).convert()
        except Exception as e:
            self.failure(e)
            return Result(exported_program=None, success=False, exception=e)

        self.success()
        return Result(exported_program)
