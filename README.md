# PyTorch ONNX Exporter
My experimental torch ONNX exporter

## Design

- Flat graph; Scope info as metadata, not functions
    - Because existing tools are not good at handling them
- Eager optimization where appropriate
    - Because exsiting tools are not good at optimizing
- Drop in replacement for torch.onnx.export
    - Minimum migration effort
- Use ExportedProgram
    - Rely on robustness of the torch.export implementation
    - This does not solve dynamo limitations, but it avoids introducing additional breakage by running fx passes
- Build graph eagerly, in place
    - Expose shape and dtype information to the op functions; build with IR

## Why is this doable?

- We need to verify coverage for torch.export on Huggingface Optimum https://github.com/huggingface/optimum/tree/main/optimum/exporters/onnx
- Path torch.onnx.export such that packages do not need to change a single line to use dynamo
- We have all operators implemented and portable
