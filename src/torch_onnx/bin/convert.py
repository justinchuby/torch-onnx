#!/usr/bin/env python  # noqa: EXE001
"""Convert a pt2 PyTorch model to ONNX format."""

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a pt2 PyTorch model to ONNX format."
    )
    parser.add_argument("model_path", type=str, help="Path to the pt2 PyTorch model.")
    parser.add_argument("output_path", type=str, help="Path to save the ONNX model.")
    parser.add_argument(
        "--external_data", type=bool, help="Save the weights in a separate file."
    )
    parser.add_argument(
        "--no_error_report",
        type=bool,
        help="Do not produce an error report if the conversion fails.",
    )
    args = parser.parse_args()
    return args


def main():
    """Convert a pt2 PyTorch model to ONNX format."""

    args = _parse_args()

    # Delay import to improve startup time.
    import torch
    from torch_onnx._core import export

    exported_program = torch.export.load(args.model_path)
    onnx_program = export(
        exported_program, (), {}, error_report=not args.no_error_report
    )
    del exported_program
    onnx_program.save(args.output_path)


if __name__ == "__main__":
    main()
