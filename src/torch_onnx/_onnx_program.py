import logging
import os
import pathlib
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
        external_data: bool | None = None,
        **_,
    ):
        if not include_initializers:
            self.model.graph.initializers.clear()
            logger.warning(
                "The initializers have been removed from the model. This is destructive. "
                "Developers: Please implement ir.Model copy() and remove initializers on the copied model."
            )
        proto = ir.serde.serialize_model(self.model)
        byte_size = proto.ByteSize()
        model_too_large = (byte_size) >= 1 << 31
        if external_data or model_too_large:
            # TODO: Create an IR pass to handle external tensors conversion
            if model_too_large:
                logger.warning(
                    "The serialized ONNX model is larger than 2GB (%s). "
                    "Saving the weights as external data in a separate file.",
                    byte_size,
                )
            if not isinstance(destination, (str, os.PathLike)):
                raise TypeError(
                    "Saving the weights as external data is only supported when destination is a file path"
                )
            destination_path = pathlib.Path(destination)
            data_path = destination_path.with_suffix(f"{destination_path.suffix}.data")
            onnx.save_model(
                proto,
                destination,
                save_as_external_data=True,
                location=os.fspath(data_path),
            )
        else:
            onnx.save_model(proto, destination)
