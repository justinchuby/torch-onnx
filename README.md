# PyTorch to ONNX Exporter

[![PyPI version](https://badge.fury.io/py/torch-onnx.svg)](https://badge.fury.io/py/torch-onnx)

Standalone native ONNX exporter synchronized with the modern `torch.export`-based
PyTorch exporter. Development follows PyTorch 2.15 and requires Python 3.11+.
Compatibility CI tests the stable PyTorch releases published in the past year:
2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1 and 2.15.0.
Installation preserves your chosen PyTorch version. Other versions may work;
unvalidated or incompatible installations receive an actionable warning once.
This project is experimental.

Exporter releases use the current development PyTorch major/minor version and
increment their own patch number for each release; see [AGENTS.md](AGENTS.md).

[Release notes for 2.15.0](docs/releases/2.15.0.md) describe the version transition
and runtime boundaries.

## Installation

```bash
pip install --upgrade torch-onnx
```

Install `onnxruntime` as well to execute or numerically verify exported
`ONNXProgram` instances locally.

## Quick start

`torch_onnx.export` returns a `torch.onnx.ONNXProgram` (also exposed as
`torch_onnx.ONNXProgram`) targeting opset 28 with graph optimization enabled by default:

```python
import torch
import torch_onnx

model = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU()).eval()
x = torch.randn(4, 8, 32)

onnx_program = torch_onnx.export(model, (x,))
onnx_program.save("model.onnx")

# Execute with ONNX Runtime and compare against the captured PyTorch program
outputs = onnx_program(x)
torch_onnx.testing.assert_onnx_program(onnx_program)
onnx_program.release()
```

## Hugging Face Transformers

`torch_onnx.patch_torch()` supports Transformers’ `OnnxExporter`, including its
custom translations and export patches. See [integration and CI coverage](docs/transformers.md)
for forward and generation exports, dynamic shapes, and saving/reloading.
The [native implementation audit](docs/transformers-native.md) records operator
implementations and measured performance.

## Dynamic shapes

### Explicit constraints with `torch.export.Dim`

Use `torch.export.Dim` (or string axis names / `torch.export.Dim.DYNAMIC`) to
mark variable dimensions:

```python
import torch
import torch_onnx


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)


model = MLP().eval()
x = torch.randn(4, 8, 32)

batch = torch.export.Dim("batch", min=1, max=64)
seq_len = torch.export.Dim("seq_len", min=1)

onnx_program = torch_onnx.export(
    model,
    (x,),
    input_names=["x"],
    output_names=["y"],
    dynamic_shapes={"x": {0: batch, 1: seq_len}},
)
onnx_program.save("mlp_dynamic.onnx")

# String axis names are also accepted as shorthand:
# dynamic_shapes=({0: "batch", 1: "seq_len"},)
torch_onnx.testing.assert_onnx_program(onnx_program, args=(torch.randn(7, 13, 32),))
onnx_program.release()
```

### Automatic inference with `InputObserver`

`InputObserver` records sample forward calls and infers `dynamic_shapes` and
export arguments automatically:

```python
observer = torch_onnx.InputObserver()
with observer(model):
    model(torch.randn(2, 8, 32))
    model(torch.randn(5, 12, 32))

onnx_program = torch_onnx.export(
    model,
    observer.infer_arguments(),
    dynamic_shapes=observer.infer_dynamic_shapes(),
)
onnx_program.save("mlp_observed.onnx")
onnx_program.release()
```

## Common workflows

### Opset selection, verification, and diagnostics

```python
onnx_program = torch_onnx.export(
    model,
    (x,),
    opset_version=24,
    verify=True,  # numerically check outputs during export
    report=True,  # write an export report to artifacts_dir
)
ref_outputs = onnx_program.call_reference(x)  # evaluate with the ONNX reference runtime
onnx_program.release()
```

Pass `profile=True` (requires `pyinstrument`) to capture an HTML timing profile
alongside the export report.

### Fake tensors and deferred weights for large models

Export without materializing full weights in memory by disabling eager
optimization until real weights are applied:

```python
from torch._subclasses.fake_tensor import FakeTensorMode

state_dict = model.state_dict()
with FakeTensorMode():
    fake_model = MLP().eval()
    fake_x = torch.randn(4, 8, 32)
    onnx_program = torch_onnx.export(fake_model, (fake_x,), optimize=False)

onnx_program.apply_weights(state_dict)
onnx_program.optimize()
onnx_program.save("large_model.onnx", external_data=True)
onnx_program.release()
```

### `ExportedProgram` conversion and `torch.onnx` patching

```python
import onnx_ir as ir

# Convert an existing ExportedProgram directly to an onnx_ir.Model
exported = torch.export.export(model, (x,))
torch_onnx.analyze(exported)
ir_model = torch_onnx.exported_program_to_ir(exported)
proto = ir.to_proto(ir_model)

# Route torch.onnx.export calls to torch_onnx
torch_onnx.patch_torch()
onnx_program = torch.onnx.export(model, (x,))
onnx_program.save("patched.onnx")
onnx_program.release()
torch_onnx.unpatch_torch()
```

### Custom operator translations

Register native ONNX IR translations globally with `TranslationContext`. Native
translations own type promotion and must cast mismatched tensor inputs
explicitly:

```python
@torch_onnx.register_custom_translation(torch.ops.aten.add.Tensor)
def custom_add(op: torch_onnx.TranslationContext, self, other, alpha=1):
    if alpha != 1:
        other = op.Mul(other, alpha)
    return op.Add(self, other)


torch_onnx.unregister_custom_translation(torch.ops.aten.add.Tensor)
```

Use `op.constant(value, dtype=...)` to create explicitly typed constants. It
shares the builder's graph-local constant cache. Native nodes receive incremental
dtype and symbolic shape inference through `onnx-shape-inference`, including
shape-tensor propagation between operations. Missing rules or incomplete metadata
leave values partially inferred. Failed inference attempts roll back metadata and
propagated shape data before logging a debug diagnostic. Translations still own
tensor casts and type promotion. Construct nested graphs with
`op.builder.subgraph(graph)` so they share this export's symbol allocation and
captured shape data while retaining separate constant caches. When constructing
graphs incrementally, declare input shapes before inference or use
`builder.reserve_symbol_names(...)` to reserve symbols for graphs built later.

For per-export overrides, `custom_translation_table` accepts ONNX Script
functions and takes precedence over global native registrations, which override
built-in translations. PyTorch's `torch.onnx.ops.symbolic` and
`symbolic_multi_out` are supported through both export entry points;
`torch_onnx.ops` also re-exports these APIs. See the
[custom-domain example and CI coverage](docs/opsets.md#public-torchonnxops-custom-operators).

## Limitations and documentation

- **Modern exporter only**: `dynamo=False`, the legacy TorchScript exporter, and
  pre-0.2 `dynamo_export` / `ExportOptions` are unsupported. Use unpatched
  `torch.onnx.export(..., dynamo=False)` when legacy export is required.
- **Python scalar inputs**: Preserved in ONNX input order for `ONNXProgram`
  callable compatibility, though `torch.export` specializes scalar values and
  Python control flow during capture.
- [Opset support and custom-opset extensibility](docs/opsets.md) — built-in
  opsets 18–29, custom domains, and unbounded opset targets.
- [Recovered regression inventory](docs/recovered-tests.md) — migrated PyTorch
  ONNX regression suites.
- [Upstream synchronization and migration inventory](docs/upstream-sync.md) —
  pinned PyTorch commits, operator coverage, and architecture details.
