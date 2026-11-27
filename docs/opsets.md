# ONNX opsets and custom-opset extensibility

Built-in translations cover ONNX opsets 18–29, and older targets still use the
existing version converter. The exporter does not enforce a maximum opset gate:
users can export custom operators, custom domains, and custom translations
targeting newer opset versions without breaking export.

When a standard ONNX operator or newer operator schema is missing from a
slightly older installed `onnx` package, `GraphBuilder` and `TranslationContext`
emit one `UserWarning` per missing standard operator per graph construction and
build the node directly rather than failing solely on installed `onnx` package
age:

- `GraphBuilder` does not infer output types from ONNX schemas. During
  `torch_onnx.export`, final node output shapes and dtypes are populated from
  the captured PyTorch `ExportedProgram` (`node.meta["val"]`) via
  `_set_shape_types` / `_set_shape_type` (matching upstream PyTorch's
  `torch/onnx/_internal/exporter/_core.py` and `_building.py`), while
  `Cast`, `BitCast`, `Constant`, and `ConstantOfShape` record their explicit
  attribute-defined output dtypes directly.
- Named calls on `TranslationContext` (for example `op.Expm1(x)`) default to one
  untyped output when no schema is present unless `outputs=` or `output_types=`
  is supplied explicitly for multi-output or sequence/subgraph cases. Note that
  a misspelled default-domain operator name on `TranslationContext` therefore
  emits the missing-schema `UserWarning` and constructs an unvalidated
  single-output node rather than raising at call time.
- Raw `GraphBuilder.call` stays strict when no schema is present and requires
  `_num_outputs=` or `_output_types=` so callers do not guess arity implicitly.
- Installed ONNX schemas within the available opset range are used only to
  validate input counts, typed tensor inputs, attributes, and Python constant
  conversions.

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

The default suite uses released dependencies. Note that while graph export and
serialization succeed on older `onnx` packages via the missing-schema fallback,
`onnx.checker.check_model`, `onnx.reference.ReferenceEvaluator`, and ONNX
Runtime still require an `onnx` / runtime build that recognizes the emitted
operators and opset version. The separate `opset29` CI job installs the exact
ONNX main revision and runs full checker + reference-evaluator tests:

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

## Public `torch.onnx.ops` custom operators

[`torch.onnx.ops.symbolic` and `symbolic_multi_out`](https://docs.pytorch.org/docs/stable/onnx_ops.html)
are supported directly by `torch_onnx.export` and by patched `torch.onnx.export`.
No custom translation registration is necessary:

```python
class Model(torch.nn.Module):
    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                "my.domain::CustomRelu", (x,),
                dtype=x.dtype, shape=x.shape, version=1,
                metadata_props={"implementation": "my-runtime-plugin"},
            )
        return x.relu()

program = torch_onnx.export(Model().eval(), (torch.randn(2, 3),))
program.save("custom.onnx")
```

These APIs preserve custom domain/opset imports, scalar/list attributes,
optional input holes, node metadata, and declared output shapes and dtypes.
`symbolic_multi_out` supports differently typed outputs, and declared shapes can
contain symbolic input dimensions. The symbolic operations do not implement eager
computation; use `torch.onnx.is_in_onnx_export()` to select the export branch.
Executing a custom-domain node requires its implementation in the chosen runtime.

The native `torch.onnx.ops.attention` and `rotary_embedding` APIs also export as
ONNX `Attention` and `RotaryEmbedding` at opset 23. Compatibility tests check these
against ONNX Runtime. Custom-domain tests use registered ONNX reference kernels
to check optimized/unoptimized graphs after save/reload at multiple batch sizes,
through both public export entry points. These tests live in
`tests/compatibility/test_onnx_ops.py`, so the annual PyTorch compatibility CI
matrix runs them on every supported stable release.
