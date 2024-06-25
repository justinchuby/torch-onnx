from __future__ import annotations

import io
import torch.export


def verify_model(
    exported_program: torch.export.ExportedProgram,
    onnx_model: str | io.BytesIO,
    args,
    kwargs,
):
    # TODO: Implement
    pass
