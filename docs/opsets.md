# ONNX opsets and custom-opset extensibility

Built-in translations cover ONNX opsets 18–29, and older targets still use the
existing version converter. The exporter does not enforce a maximum opset gate:
users can export custom operators, custom domains, and custom translations
targeting newer opset versions without breaking export. Schema-guided operator
calls still validate against the installed ONNX schema registry so missing
default-domain schemas fail with actionable guidance rather than silently
emitting invalid older-schema operators.

The development-schema authority is ONNX main
[`622adf94d4e720142950aa8178f7357d52e63db0`](https://github.com/onnx/onnx/commit/622adf94d4e720142950aa8178f7357d52e63db0).
Released ONNX 1.23.2 has schemas through 28. The available November 2 weekly
advertises 29 but lacks most of the new 29 operators; use the pinned source build
for the complete development-schema tests. ONNX IR 1.1.0 is required because the
newer schemas include datatype names absent from older IR releases.

| Target | Native graph generation and focused coverage |
| --- | --- |
| 24 | SiLU uses `Swish`; SDPA uses the target `Attention` schema, including boolean masks |
| 25 | Target-version schema resolution for `Attention`, casts, shapes and control-flow operators; Swish retains the 24 schema |
| 26 | `CumProd` with PyTorch dtype promotion, scalar/empty/negative-axis handling; equal-width `view(dtype)` uses `BitCast` |
| 27 | Updated `Range` schema accepts half inputs; earlier schemas reject them |
| 28 | Updated `Einsum` schema accepts bfloat16; earlier schemas reject it; dtype parsing supports the expanded Cast schema |
| 29 | Native precision-preserving `Expm1`, `Log1p`, stable `LogCumSumExp`, and floating `ScatterElements(reduction="mean")` when `include_self=True` |

Lower opsets retain their existing decompositions. Integer/exclude-self scatter
mean retains explicit sums/counts rather than assuming equivalent rounding or
contribution semantics. BitCast only supports equal element widths; resizing or
repacking a `view(dtype)` is not implemented. New stateful attention/convolution,
packed quantization, and grouped matmul schemas do not imply new PyTorch operator
translations. General recognition of a schema is not exhaustive PyTorch coverage.

## Validation and runtime boundaries

`tests/exporter/test_opsets.py` checks graph structure, target imports and full
ONNX schema validity for every target. Swish 24, Attention 24–28, CumProd/BitCast
26–28 execute against ONNX Runtime. Swish 25–28 uses ONNX reference evaluation:
ORT 1.32's function expansion reports a missing implementation for those targets.
The half Range case uses reference execution; bfloat16 Einsum has schema-only
coverage. Opset 29 uses ONNX main's reference evaluator, not an assertion that
released ORT implements development operators.

The default suite uses released dependencies. The separate `opset29` CI job
installs the exact ONNX main revision and runs the development tests:

```bash
pip install "onnx @ git+https://github.com/onnx/onnx.git@622adf94d4e720142950aa8178f7357d52e63db0"
pytest tests/exporter/test_opsets.py -m opset29
```

Default exports still use PyTorch's established default opset, not development 29.

## Semantic normalization operators

Batch normalization uses ONNX `BatchNormalization` in inference and training
translations. Training supplies the computed batch mean and biased variance to
the inference-mode operator, while separately preserving PyTorch's saved
statistics and unbiased running-variance update. This retains a fusible
normalization operator without requiring a training-capable runtime.
