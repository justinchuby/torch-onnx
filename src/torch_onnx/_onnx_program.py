"""Use PyTorch's ONNXProgram directly for downstream type compatibility."""

from torch.onnx import ONNXProgram
from torch.onnx._internal.exporter._onnx_program import _ort_session_initializer

__all__ = ["ONNXProgram", "_ort_session_initializer"]
