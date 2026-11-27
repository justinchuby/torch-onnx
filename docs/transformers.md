# Transformers exporter integration

Use the real [Transformers exporter](https://github.com/huggingface/transformers/blob/main/docs/source/en/exporters.md)
with the standalone ONNX backend by patching `torch.onnx.export` before export:

```python
import torch_onnx
from transformers.exporters import OnnxConfig, OnnxExporter

# model.eval() and sample_inputs are ordinary Transformers forward inputs.
torch_onnx.patch_torch(verbose=False)
try:
    exported = OnnxExporter().export(
        model.eval(), sample_inputs,
        config=OnnxConfig(dynamic=True, opset_version=23, optimize=True),
    )
finally:
    torch_onnx.unpatch_torch()

outputs = exported(**sample_inputs)
exported.save_pretrained("onnx_model")
```

`OnnxExporter.export_for_generation` also uses the patched backend for its
prefill and decode components. Reload either format with
`transformers.exporters.ExportedModel.from_pretrained`. The Transformers runner
retains its named inputs, model output types, and cache reconstruction.

## Downstream patches

Transformers keeps its reversible operator patches, custom ONNX translations,
FX fixes, and final IR fixes. The standalone exporter forwards the post-decomposition
FX callback and the policy preventing constant folding of `Resize` through
`src/torch_onnx/_exporter_compat.py`. Those hooks are scoped to an export identified
by Transformers' actual custom translation table, restore on failure, and do not
import Transformers or change PyTorch's private preparation function globally.

Specialized Python arguments, such as ViT's `interpolate_pos_encoding=True`,
are capture-time constants rather than ONNX graph inputs. `ONNXProgram` excludes
those constants when executing the original call signature and rejects changed
specialized arguments. Symbolic runtime inputs remain graph inputs.

The standalone opset-23 RoPE fusion broadcasts singleton cosine/sine cache axes
explicitly, because ONNX `RotaryEmbedding` requires exact cache batch and sequence
sizes. Unsupported dtypes and unproven broadcast relationships retain the original
tensor operations. Other fusion passes remain enabled. An isolated float32 CPU
benchmark with one thread, ten warmups, and 100 alternating measurements compared
the valid unfused graph with the fixed fusion:

| Input `[batch, heads, sequence, head_dim]` | Unfused | Fused | Speedup |
| --- | ---: | ---: | ---: |
| `[2, 2, 4, 8]` | 9.63 µs | 6.75 µs | 1.43× |
| `[2, 8, 128, 64]` | 102.92 µs | 54.38 µs | 1.89× |
| `[4, 16, 512, 64]` | 1.92 ms | 1.13 ms | 1.71× |

These are local kernel measurements, not whole-model latency guarantees.
Reproduce the comparison with `python benchmarks/rotary_embedding.py`; its JSON
output includes the environment, graph operations, and median runtimes.

## CI coverage

`.github/workflows/transformers.yaml` installs Transformers from both its rolling
`main` and the initially verified commit
`8e094f8970a60ca83052e2c199783b247b8fc13e` (5.25.0.dev0). It uses CPU PyTorch
2.15 and ONNX Runtime. No model checkpoint or tokenizer download is required.
Tests instantiate tiny BERT, Qwen3, Qwen3 MoE, Qwen3.5 hybrid linear/full attention,
and ViT models, retaining the real upstream exporter and its patches.

The suite checks static/unoptimized and dynamic/optimized exports, the default
configuration, optional CPU ORT fusions, prefill/decode with KV caches and both
single-token and multi-token decode, ONNX validity, every output tensor's type,
shape and values, and saved/reloaded execution. Dynamic forward cases execute a
second batch and sequence/image size. Float32 comparisons use `rtol=1e-4` and
`atol=1e-5`; this does not establish reduced-precision or GPU coverage.

The exporter defaults to opset 28. The default-configuration Qwen3 graph uses
ONNX reference execution because ORT 1.32 cannot expand Swish at opset 25–28.
The explicit-opset-23 Qwen3/MoE/hybrid and generation cases continue to exercise
the Transformers ORT runner and save/reload. Use `opset_version=23` for those
models when running with that ORT release.

For ViT interpolation, capture an image size different from the configured
training resolution. Capturing exactly the training resolution can specialize
Transformers' interpolation shortcut before ONNX translation; `Dim.AUTO` alone
cannot recover the discarded branch.

Run the integration suite explicitly in an environment with the checked-out
Transformers installed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/integration/test_transformers_exporter.py -m integration
```
