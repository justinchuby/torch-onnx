"""PyTorch-compatible ONNXProgram with the standalone optimization pipeline."""

import torch.onnx
from torch.onnx._internal.exporter._onnx_program import _ort_session_initializer

from torch_onnx import _optimization


class ONNXProgram(torch.onnx.ONNXProgram):
    """An ONNX program using the standalone exporter's optimizer."""

    def optimize(self) -> None:
        """Optimize the model in place and invalidate any cached runtime session."""
        if self._inference_session is not None or self._tempdir is not None:
            self.release()
        self.model = _optimization.optimize(self.model)


__all__ = ["ONNXProgram", "_ort_session_initializer"]
