# PyTorch to ONNX Exporter

[![PyPI version](https://badge.fury.io/py/torch-onnx.svg)](https://badge.fury.io/py/torch-onnx)

Standalone native ONNX exporter, synchronized with the modern PyTorch exporter
and the native ONNX migration stack. **Version 0.2.0 targets PyTorch 2.15.x**
(latest stable: 2.15.0), Python 3.11+, not older releases or nightly/main.
This remains an experimental project.

## Installation

```bash
pip install --upgrade torch-onnx
```

ONNX Runtime is optional for execution and verification. ONNX Script is used for
optimization and legacy custom translation compatibility, not native built-ins.

## Usage

```python
import torch
import torch_onnx
import onnx_ir as ir

model = torch.nn.ReLU().eval()
inputs = (torch.randn(2, 3),)
onnx_program = torch_onnx.export(model, inputs)
onnx_program.optimize()
onnx_program.save("model.onnx")

# Convert an existing ExportedProgram directly to ONNX IR.
exported = torch.export.export(model, inputs)
ir_model = torch_onnx.exported_program_to_ir(exported)
proto = ir.to_proto(ir_model)

# Opt-in routing of the modern PyTorch API to this standalone exporter.
torch_onnx.patch_torch(report=True, verify=True)
onnx_program = torch.onnx.export(model, inputs)
onnx_program.save("model.onnx")
torch_onnx.unpatch_torch()

torch_onnx.analyze(exported)
```

The modern export options include dynamic shapes, export-local custom
translations, optimization, verification, profiling and reports. `ONNXProgram`,
`ExportableModule`, `InputObserver`, `testing` and `verification` are available.
`torch_onnx.ops` exposes PyTorch's symbolic operators without duplicate dispatcher
registration.

Exports return the exact `torch.onnx.ONNXProgram` class, also exposed as
`torch_onnx.ONNXProgram`, for downstream compatibility. See
[opset support and extensibility](docs/opsets.md) for native newer operators,
custom opset extensibility, and schema requirements, and the
[recovered regression inventory](docs/recovered-tests.md) for historical coverage.

Python scalar arguments remain in ONNX input order for compatibility with the
upstream program's callable interface. Their values and control flow are still
specialized by `torch.export`; retaining an input does not make it dynamic.

## Native translations

Native implementations receive an explicit `TranslationContext` and create
ONNX IR nodes directly. Operators own promotion; no FX FakeTensor graph rerun
is used. Constants/caches, subgraphs and captures belong to their graph builder.

```python
@torch_onnx.register_custom_translation(torch.ops.aten.add.Tensor)
def custom_add(op: torch_onnx.TranslationContext, self, other, alpha=1):
    if alpha != 1:
        other = op.Mul(other, alpha)
    return op.Add(self, other)

torch_onnx.unregister_custom_translation(torch.ops.aten.add.Tensor)
```

Custom translations must cast tensor inputs explicitly. Export-local
`custom_translation_table` entries override global registrations, which override
built-ins. Exports take locked snapshots of global registrations. Real and
complex variants are dispatched directly. `get_translation_builder` is only
for explicit raw-builder activation; native functions use their context.

## Breaking changes and provenance

0.2.0 removes pre-2.15 compatibility, the prototype TorchScript converter,
legacy `dynamo_export`/`ExportOptions`, and legacy fallback. `dynamo=False` is
unsupported; use unpatched PyTorch for its legacy exporter.

See [upstream synchronization and complete migration inventory](docs/upstream-sync.md)
for pinned source commits, included operator families and test coverage.
No package registry publication is part of this synchronization.
