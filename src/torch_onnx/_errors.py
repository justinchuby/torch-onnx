"""Internal aliases of the public standalone exporter exceptions."""

from torch_onnx.errors import (
    ConversionError,
    DispatchError,
    GraphConstructionError,
    TorchExportError,
)

__all__ = [
    "ConversionError",
    "DispatchError",
    "GraphConstructionError",
    "TorchExportError",
]
