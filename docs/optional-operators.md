# Optional torchao and torchvision operators

Neither package is imported by `import torch_onnx` or required as a dependency.
Registry construction at export time checks for installed packages on every call,
including packages made available after the initial import. Installed-package
import failures raise an actionable error with the original exception as its cause;
only absent packages are ignored.

The integration suite uses actual torchao 0.18.0 and torchvision 0.29.2 with
PyTorch 2.15. Export returns the upstream `torch.onnx.ONNXProgram` unchanged.

## Torchao matrix

| Operator / frontend | Representation | Requirements and limits |
| --- | --- | --- |
| `quantize_affine`, `dequantize_affine`, int8/uint8 | Native `QuantizeLinear` / `DequantizeLinear` | Opset 18+, float32 input/scales/output; integer zero points; static positive shapes |
| Same, int16 | Native Q/DQ | Opset 21+; same shape and precision requirements; ONNX reference runtime coverage |
| Same, int32 | Shared rounded/clamped integer quantization, native DQ | Opset 18+; ONNX `QuantizeLinear` has no int32 output; ORT coverage at opset 18, reference at 21/29 |
| `quantize_affine_float8_non_decomposed`, `dequantize_affine_float8_non_decomposed` | Native Q/DQ | Opset 19+; E4M3FN/E5M2, float32 scales; scalar or same-rank block scales; static shapes; reference runtime coverage at opsets 19, 21, 29 |
| `quantize_(..., Int8WeightOnlyConfig())` on Linear | Torchao's captured scaled MatMul decomposition | Actual tensor-subclass frontend and runtime covered; not a claim that this frontend captures Q/DQ operators |

Affine block sizes must divide every input dimension. Parameters are reshaped and
expanded to per-element scales for a flattened native per-axis Q/DQ, then restored
to the original shape. This handles per-tensor, per-channel, groupwise, and
multi-axis blocks without a second quantization formula, but can increase scale
storage and runtime memory versus a backend-specific packed kernel.

Float16/bfloat16 affine arithmetic is deliberately rejected rather than changing
its intermediate rounding. Uint16 is not accepted by torchao 0.18's affine
frontend. Sub-byte packed weights, float-domain zero points, MX/NVFP4, sparse
CUDA kernels, and arbitrary quantized tensor subclasses/configurations are not
covered by this integration. Qparam selection can still depend on frontend
decomposition support. This is not support for every `torchao.quantize_` mode.

ONNX Runtime does not provide every integer kernel produced by optimizer
expansion (for example narrowed int16 clipping through `Where`). Float8 and
int16/int32 paths at newer opsets are checked and executed with ONNX's reference runtime, without
disabling optimization or patching `ONNXProgram`.

### Opset 29 is not a generic packed-weight decoder

Schemas were checked against pinned ONNX main
`622adf94d4e720142950aa8178f7357d52e63db0`.
`DequantizeLinear` uses the latest compatible schema (version 28 at opset 29),
including its independent output type. Its output IR type is explicit.

The new `DequantizePacked` version 29 accepts **one 1-D UINT8 input** and the
attributes `format`, `shape`, and optional `output_dtype`. Its format identifiers
are fixed `gguf:*` layouts; output types are FLOAT, FLOAT16, BFLOAT16, DOUBLE.
It does not accept torchao scales/zero points or arbitrary torchao packed layouts.
Consequently this integration does not relabel torchao bytes as GGUF or emit a
fictitious generic packed dequantization operator. A future packed-layout path
must establish exact byte-format equivalence before using it.

## Torchvision matrix

Development-schema CI installs the public Linux wheel
`onnx-weekly==1.24.0.dev20261123` instead of compiling ONNX. It checks required
`Log1p`, `Expm1`, and `DequantizePacked` schemas before running the unchanged
semantic/schema and reference-runtime regressions. The older proxy-available
`1.24.0.dev20261102` lacks `Log1p` and `Expm1` despite reporting maximum opset 29;
the maximum-opset number alone does not establish compatibility.

| Operator | Native ONNX representation | Coverage / limits |
| --- | --- | --- |
| `nms` | `NonMaxSuppression` | Opset 18+, actual-package export/check/ORT/serialization |
| `roi_align` | `RoiAlign` | Opset 18+, actual-package export/check/ORT/serialization |
| `deform_conv2d` | `DeformConv` | Opset 19+, actual-package export/check/reference/serialization |
| `ps_roi_align` | `RoiAlign` plus position-sensitive channel selection | Opset 18+, static channels divisible by pooled area; returns the real INT32 channel mapping as well as pooled values |
| `roi_pool` | Existing separately maintained `MaxRoiPool` path | Not changed or newly validated here |

Position-sensitive alignment includes fixed/adaptive sampling and small,
out-of-bounds, and empty ROI coverage. `ps_roi_pool`, rotated IoU,
quantized `qnms`/`qroi_align`, and backward kernels are not added. No missing
operator is registered with a placeholder implementation.
