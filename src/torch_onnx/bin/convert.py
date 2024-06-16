#!/usr/bin/env python
"""Convert a pt2 PyTorch model to ONNX format."""

import argparse
import logging


def main(args):
    """Convert a pt2 PyTorch model to ONNX format."""

    # Delay import to improve startup time.
    import torch
    import torch_onnx
    from onnxscript import ir
    import onnx

    exported_program = torch.export.load(args.model_path)
    onnx_model = torch_onnx.exported_program_to_ir(exported_program)
    proto = ir.serde.serialize_model(onnx_model)
    proto_bytesize = proto.ByteSize()
    large_proto = proto_bytesize >= 1 << 31
    if large_proto or args.external_data:
        external_data_path = args.output_path + ".data"
        if large_proto:
            logging.warning(
                "The serialized ONNX model is larger than 2GB. "
                "Saving the weights in a separate file"
            )
        onnx.save_model(
            proto,
            args.output_path,
            save_as_external_data=True,
            location=external_data_path,
        )
    else:
        onnx.save_model(proto, args.output_path)

    # TODO: Display analysis info if the process fails.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pt2 PyTorch model to ONNX format."
    )
    parser.add_argument("model_path", type=str, help="Path to the pt2 PyTorch model.")
    parser.add_argument("--output_path", type=str, help="Path to save the ONNX model.")
    parser.add_argument(
        "--external_data", type=bool, help="Save the weights in a separate file."
    )
    args = parser.parse_args()
    main(args)
