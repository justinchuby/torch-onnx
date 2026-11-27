"""Standalone native ONNX exporter for the latest stable PyTorch."""

from __future__ import annotations

# Bootstrap diagnostics before importing consumers of private PyTorch APIs.
# isort: off
from . import _torch_compat  # noqa: F401
# isort: on

from torch.onnx import ops

from . import _flags, errors
from . import _testing as testing
from . import _verification as verification
from ._analysis import analyze
from ._compat import export_compat
from ._compat import export_compat as export
from ._core import exported_program_to_ir
from ._custom_translation import (
    get_translation_builder,
    register_custom_translation,
    unregister_custom_translation,
)
from ._exportable_module import ExportableModule
from ._input_observer import InputObserver
from ._ir_builder import GraphBuilder, TranslationContext
from ._onnx_program import ONNXProgram
from ._patch import patch_torch, unpatch_torch
from ._registration import ONNXRegistry
from .errors import OnnxExporterError

__all__ = [
    "ExportableModule",
    "GraphBuilder",
    "InputObserver",
    "ONNXProgram",
    "ONNXRegistry",
    "OnnxExporterError",
    "TranslationContext",
    "analyze",
    "errors",
    "export",
    "export_compat",
    "exported_program_to_ir",
    "get_translation_builder",
    "is_in_onnx_export",
    "ops",
    "patch_torch",
    "register_custom_translation",
    "testing",
    "unpatch_torch",
    "unregister_custom_translation",
    "verification",
]

for _public_class in (
    ExportableModule,
    GraphBuilder,
    InputObserver,
    TranslationContext,
):
    _public_class.__module__ = __name__
del _public_class


def is_in_onnx_export() -> bool:
    """Return whether this standalone exporter is currently exporting a model."""
    return _flags._is_onnx_exporting
