# PyTorch to ONNX Exporter

[![PyPI version](https://badge.fury.io/py/torch-onnx.svg)](https://badge.fury.io/py/torch-onnx)

Experimental torch ONNX exporter.

> [!WARNING]
> This is an experimental project and is not designed for production use.
> Use `torch.onnx.export` for these purposes.

## Installation

```bash
pip install --upgrade torch-onnx
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
onnx.save(proto, "model.onnx")

# Or patch the torch.onnx export API
# Set error_report=True to get a detailed error report if the export fails
torch_onnx.patch_torch(report=True, verify=True, profile=True)
torch.onnx.export(...)

# Use the analysis API to print an analysis report for unsupported ops
torch_onnx.analyze(exported)
```

## Design

{dynamo/jit} -> {ExportedProgram} -> {torchlib} -> {ONNX IR} -> {ONNX}

- Use ExportedProgram
  - Rely on robustness of the torch.export implementation
  - Reduce complexity in the exporter
  - This does not solve dynamo limitations, but it avoids introducing additional breakage by running fx passes
- Flat graph; Scope info as metadata, not functions
  - Because existing tools are not good at handling them
- Eager optimization where appropriate
  - Because exsiting tools are not good at optimizing
- Drop in replacement for torch.onnx.export
  - Minimum migration effort
- Build graph eagerly in the exporter
  - Give the exporter full control over the graph being built

## Why is this doable?

- We need to verify torch.export coverage on Huggingface Optimum https://github.com/huggingface/optimum/tree/main/optimum/exporters/onnx; and they are not patching torch.onnx itself.
- Patch torch.onnx.export such that packages do not need to change a single line to use dynamo
- We have all operators implemented and portable
