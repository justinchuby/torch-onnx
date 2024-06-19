import logging
import os
from typing import IO

import onnx
import torch
from onnxscript import ir

logger = logging.getLogger(__name__)


class ONNXProgram:
    """A substitute class for `torch.onnx.ONNXProgram`."""

    def __init__(self, model: ir.Model, exported_program: torch.export.ExportedProgram):
        self.model = model
        self.exported_program = exported_program

    @property
    def model_proto(self) -> onnx.ModelProto:
        """Compatibility property for `torch.onnx.ONNXProgram.model_proto`."""
        return ir.serde.serialize_model(self.model)

    def save(
        self,
        destination: str | os.PathLike | IO[bytes],
        *,
        include_initializers: bool = True,
        **_,
    ):
        if not include_initializers:
            raise NotImplementedError(
                "Developers: Please implement ir.Model copy() and remove initializers"
            )
        proto = ir.serde.serialize_model(self.model)
        if proto.ByteSize() >= 1 << 31:
            # TODO: Create an IR pass to handle external tensors conversion
            logger.warning(
                "The serialized ONNX model is larger than 2GB. "
                "Saving the weights in a separate file"
            )
            onnx.save_model(proto, destination, save_as_external_data=True)
        else:
            onnx.save_model(proto, destination)
