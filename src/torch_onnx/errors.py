"""Aliases of PyTorch ONNX exceptions and warnings.

Both exporters share exception identities, so existing torch.onnx handlers also
catch failures raised by the standalone exporter.
"""

from torch.onnx._internal.exporter._errors import (
    ConversionError,
    DispatchError,
    GraphConstructionError,
    TorchExportError,
)
from torch.onnx.errors import (
    OnnxExporterError,
    OnnxExporterWarning,
    SymbolicValueError,
    UnsupportedOperatorError,
)

__all__ = [
    "ConversionError",
    "DispatchError",
    "GraphConstructionError",
    "OnnxExporterError",
    "OnnxExporterWarning",
    "SymbolicValueError",
    "TorchExportError",
    "UnsupportedOperatorError",
]
