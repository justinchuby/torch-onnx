# PyTorch to ONNX Exporter

Experimental torch ONNX exporter.

> [!WARNING]
> This is an experimental project and is not designed for production use.
> Use `torch.onnx.export` for these purposes.

## Installation

```bash
pip install torch-onnx
```

## Usage

```python
import torch
import torch_onnx
from onnxscript import ir
import onnx

# Get an exported program with torch.export
exported = torch.export.export(...)
model = torch_onnx.exported_program_to_ir(exported)
proto = ir.to_proto(model)
# This will give you an ATen dialect graph (un-lowered ONNX graph with ATen ops)
onnx.save(proto, "model.onnx")

# Or patch the torch.onnx export API
torch_onnx.patch_torch()
torch.onnx.export(...)
```

## Design

{ExportedProgram, jit} -> {ONNX IR} -> {torchlib} -> {ONNX}

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

- We need to verify torch.export coverage on Huggingface Optimum https://github.com/huggingface/optimum/tree/main/optimum/exporters/onnx; and they are not patching torch.onnx itself.
- Path torch.onnx.export such that packages do not need to change a single line to use dynamo
- We have all operators implemented and portable
