# Native implementations of Transformers export fixes

The source audited here is Transformers
[`8e094f8970a60ca83052e2c199783b247b8fc13e`](https://github.com/huggingface/transformers/blob/8e094f8970a60ca83052e2c199783b247b8fc13e/src/transformers/exporters/exporter_onnx.py).
Useful operator implementations and custom translations are integrated into our
torchlib. Five integration cases disable Transformers' ONNX operator patches,
custom translations and final IR fixes, then check dynamic execution and
save/reload through its real exporter. Transformers retains its frontend
preparation and pre/post-decomposition FX graph fixes through the existing
scoped adapter. No pure FX graph repair is copied into this exporter.

No unconditional bf16 Conv cast is added. A regression checks that a native
opset-28 Conv graph keeps bf16 input and weight without Cast nodes.

## Operator implementation audit

| Transformers patch/lowering | Native behavior |
| --- | --- |
| `where` dtype/scalar patch | Retain Torch dtype promotion. Casting the second branch to the first branch's dtype or using `full_like` for a floating scalar can change results. Mixed tensor dtypes and integer-tensor/float-scalar cases are tested. |
| non-affine RMSNorm | Keep native RMSNormalization with Torch's epsilon and all normalized dimensions. Its implicit weight now covers only those dimensions, instead of materializing the entire input shape. |
| symbolic split/chunk | Retain existing fixed-arity tensor lowering, including dynamic partitions. Tests require no Sequence operators. Do not adopt the assumption that chunk always returns the requested count or divides evenly. |
| randperm | Add a native random-key sort, including empty results. Float64 keys reduce accidental ties. Explicit Generators are rejected; the Torch RNG sequence is not reproduced. |
| complex unsqueeze | Retain the existing packed-complex implementation. |
| complex exp | Add Euler-form native lowering. |
| fill_diagonal_ | Lower its functionalized aten/prims as_strided_scatter through shared strided indices and ScatterElements. Dense transposes, static base offsets, empty updates and dynamic sizes/strides are supported; gapped storage is explicitly unsupported. No FX patch is applied. |
| irfft | Retain the standard ONNX inverse/onesided DFT lowering. The ORT workaround stays in Transformers; do not rewrite a spec-valid graph for this runtime limitation. |
| integer `full` default dtype | Retain the existing dtype-aware factory implementation, including bool and the current default floating dtype. |
| masked mean/var | Retain Torch decomposition and native reductions/promotion. Masked NaNs and all-false masks are tested; do not adopt multiplication by zero or clamping that changes NaN/empty-mask semantics. |
| masked_scatter | Use CumSum, bounded source prefix, one-based Gather and Where. An empty source is valid with an all-false mask; a short selected source still errors. Temporary storage does not grow with unused source elements. |
| roll | Retain existing Slice/Concat lowering and safe modulo divisor for dynamic empty axes. |
| index_put | Scalar boolean-mask replacement uses Where/boolean primitives. Prefix masks gather whole rows and broadcast trailing payloads, avoiding per-element NonZero coordinates. Retain the general full-rank, multiple-index and accumulation paths. |
| bincount | Retain scatter-add, which handles empty inputs and avoids dense OneHot memory. Weighted bincount remains explicitly unsupported, rather than silently ignoring weights. |
| grouped_mm | Add the Transformers namespace alias and offset-group bias/output-dtype support to the existing bounded per-group MatMul implementation. Cast operands before accumulation when requested. Empty groups are tested. |
| mm.dtype | Add Cast-before-MatMul lowering to retain the requested accumulation dtype. |
| repeat_interleave | Retain existing flatten/Tile/Reshape logic, including symbolic repeats, explicit dimensions and empty tensors. |
| boolean masked_fill | Use boolean primitives, including tensor-valued dynamic fill values. |
| Python floor division | Reuse the signed floor-divide implementation, rather than truncating negative symbolic shape arithmetic. |
| per-channel dequantize | Retain the existing DequantizeLinear implementation with input-compatible zero-points, floating scales, and requested output conversion. |

## Attention and runtime boundaries

Export targets ONNX specification semantics. Runtime-only workarounds are not
part of torchlib: inverse/onesided DFT, int16 Where and broadcast Attention masks
retain their standard forms. As an explicit exception, function-local opset
imports are also mirrored into the model for ORT's nested-function inliner,
retaining each function's declarations. Control-flow values use normal IR naming
rather than ORT-specific names.
Circular padding uses native Pad wrap from opset 19; older schemas use tensor
operations because wrap is absent from those ONNX schemas. Negative-pad graphs
are schema-checked separately from runtimes that reject them.

Retain the native Attention path and the lower-opset math implementation. Bool
and negative-infinity fully masked rows are tested against Torch on ORT. A row
whose floating mask is entirely `finfo.min` has finite equal scores and returns
mean(value) in Torch. Transformers' patch changes that row to zero, so it is not
adopted. ORT 1.32's Attention-23 kernel also returns zero for that case; the ONNX
graph is verified against Torch with ONNX reference execution. This distinguishes
an exporter formula from that runtime limitation.

ORT 1.32 lacks RandomUniformLike-22; randperm executes on ORT at opset 20 and
uses reference execution for its opset-23 graph. Existing [opset boundaries](opsets.md)
continue to apply. Tests do not silently retry a failed backend or skip missing
operator support.

## Local performance measurements

`python benchmarks/transformers_ops.py` compares the old strategies and native
implementations with identical inputs, one CPU thread, ten warmups and 200
alternating measurements. Both results must match eager execution before timing.
One local macOS ARM64 / Torch 2.15 / ORT 1.32 run measured:

| Case | Input | Speedup |
| --- | --- | ---: |
| Non-affine RMSNorm | `[8, 128, 32]`, normalized width 32 | 1.99x |
| Scalar bool-mask index_put | `[64, 512]`, about half selected | 31.10x |
| Prefix row-mask index_put | `[64, 512]`, half the rows selected | 10.08x |
| masked_scatter | `[64, 512]`, about half selected | 1.31x |

These are CPU measurements of the listed cases, not claims about every shape or
GPU provider. RoPE repairs now use public `rewriter.RewriteRule` and
`RewriteRuleSet` APIs after the public fusion pass, preserving explicit user
RotaryEmbedding nodes and the existing broadcast/dtype checks.
