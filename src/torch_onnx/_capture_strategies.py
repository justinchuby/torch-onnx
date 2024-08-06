"""Strategies for capturing ExportedPrograms."""

from __future__ import annotations

import abc
import dataclasses
import datetime
import os
import pathlib
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.utils import _pytree

from torch_onnx import _torchscript_converter

if TYPE_CHECKING:
    import torch.fx


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


@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        return self.exported_program is not None


class CaptureStrategy(abc.ABC):
    """Strategy for capturing a module as ExportedProgram.

    To use a strategy, create an instance and call it with the model, args, kwargs, and dynamic_shapes.
    Example::

        strategy = TorchExportStrategy(verbose=True)
        result = strategy(model, args, kwargs, dynamic_shapes)
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        dump: bool = False,
        artifacts_dir: str | os.PathLike = ".",
        timestamp: str | None = None,
    ):
        """Initialize the strategy.

        Args:
            verbose: Whether to print verbose messages.
            dump: Whether to dump the intermediate artifacts to a file.
        """
        self._verbose_print = _verbose_printer(verbose)
        self._dump = dump
        self._artifacts_dir = pathlib.Path(artifacts_dir)
        self._timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )

    def __call__(
        self,
        model: torch.nn.Module | torch.jit.ScriptFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        dynamic_shapes,
    ) -> Result:
        self._enter(model)
        if kwargs is None:
            kwargs = {}
        try:
            exported_program = self._capture(model, args, kwargs, dynamic_shapes)
        except Exception as e:
            self._failure(model, e)
            return Result(
                exported_program=None,
                strategy=self.__class__.__name__,
                exception=e,
            )
        self._success(model)
        return Result(exported_program, strategy=self.__call__.__name__)

    @abc.abstractmethod
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        raise NotImplementedError

    def _enter(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _success(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _failure(
        self, model: torch.nn.Module | torch.jit.ScriptFunction, e: Exception
    ) -> None:
        return


class TorchExportStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        return torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export`... ❌"
        )


class TorchExportNonStrictStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        return torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=False
        )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ❌"
        )


class JitTraceConvertStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        del dynamic_shapes  # Unused

        flattened_args, spec = _pytree.tree_flatten((args, kwargs))
        flattened_args = tuple(flattened_args)

        class WrappedModel(torch.nn.Module):
            """Wrap the model so that it takes flattened arguments."""

            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, *_args):
                unflattened_args, unflattened_kwargs = _pytree.tree_unflatten(
                    _args, spec
                )
                return self.model(*unflattened_args, **unflattened_kwargs)

        jit_model = torch.jit.trace(
            WrappedModel(model),
            example_inputs=flattened_args,
            check_trace=False,
            strict=False,
        )
        if self._dump:
            program_path = self._artifacts_dir / f"onnx_export_{self._timestamp}.pt"
            try:
                torch.jit.save(jit_model, program_path)
            except Exception as e:
                self._verbose_print(
                    f"Failed to save Torch Script model due to an error: {e}"
                )
            else:
                self._verbose_print(
                    f"Torch Script model has been saved to '{program_path}'."
                )
        return _torchscript_converter.TS2EPConverter(
            jit_model, flattened_args
        ).convert()

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with Torch Script... ❌"
        )


class LegacyDynamoStrategy(CaptureStrategy):
    """Strategy implemented by the ONNX team using internal dynamo APIs and custom fx passes."""

    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        # Adapted from https://github.com/pytorch/pytorch/blob/ea42027e0ed7530386ae4222dc599fcaf84a8a05/torch/onnx/_internal/_exporter_legacy.py#L1491
        # NOTE: Import here to prevent circular dependency
        from torch.onnx._internal.fx import diagnostics, passes

        graph_module, _ = torch._dynamo.export(
            model,
            tracing_mode="symbolic",
            dynamic_shapes=dynamic_shapes,
        )(
            *args,
            **kwargs,
        )
        torch._dynamo.reset()

        diagnostic_context = diagnostics.DiagnosticContext(
            "torch.onnx.export",
            torch.__version__,
        )

        flattened_args, _ = _pytree.tree_flatten((args, kwargs))
        flattened_args = tuple(flattened_args)

        # ONNX does not support views and mutations.
        # Functionalize to get a semantically equivalent graph without mutations.
        graph_module = passes.Functionalize(
            diagnostic_context,
            graph_module,
            enable_dynamic_axes=bool(dynamic_shapes),
        ).run(*flattened_args)

        # Input mutations are detected and distilled after `Functionalize` pass.
        # Remove them since ONNX inference does not need them.
        graph_module = passes.RemoveInputMutation(diagnostic_context, graph_module).run(
            *flattened_args
        )

        # Use torch.export to recapture the GraphModule into an ExportedProgram.
        return torch.export.export(graph_module, flattened_args)

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with internal Dynamo apis..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with internal Dynamo apis... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with internal Dynamo apis... ❌"
        )


CAPTURE_STRATEGIES = (
    # TorchExportStrategy,
    # TorchExportNonStrictStrategy,
    JitTraceConvertStrategy,
    # LegacyDynamoStrategy,
)
