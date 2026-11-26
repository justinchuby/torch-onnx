# Reported ONNX issue triage

This inventory tracks reports by `cuiliaomei-beep` in `pytorch/pytorch` against
the standalone exporter. Validation used PyTorch 2.15.0, ONNX 1.23.2,
ONNX IR 1.1.0, ONNX Script 0.7.2 and ONNX Runtime 1.32.0. A report against a
PyTorch nightly does not necessarily reproduce here. Links identify the original
report; this document does not claim that the upstream PyTorch issue is closed.

Directed regressions live in `tests/exporter/test_reported_core_issues.py`,
`test_reported_neural_issues.py` and `test_reported_optimizer_issues.py`.
Tests check ONNX legality and eager execution, with optimization enabled and
disabled. ONNX ReferenceEvaluator covers BF16 kernels and DOUBLE Atan that the
local CPU runtime does not implement.

The FP16-logits/DOUBLE-target CE case with 2049 classes also checks reference
execution exactly. ORT retains a small intermediate FP16 LogSoftmax rounding
deviation even with runtime optimization disabled, so its comparison uses a
half-precision tolerance that still detects the original incorrect class count.

| PyTorch issue | Standalone result |
| --- | --- |
| [206285](https://github.com/pytorch/pytorch/issues/206285), integer division | Integer `div` with floor/trunc rounding now uses integer arithmetic, including values beyond floating-point exact-integer ranges. |
| [206286](https://github.com/pytorch/pytorch/issues/206286), CE smoothing | Implemented label smoothing, weighted/ignored targets, reductions and probability targets. Keep class counts as integers until converting directly to the computation dtype. |
| [206288](https://github.com/pytorch/pytorch/issues/206288), old Softmax | Original opset 11/12 report is outside the native exporter's opset 18+ domain; no old-opset fix claimed. |
| [206401](https://github.com/pytorch/pytorch/issues/206401), RMS epsilon | Export optimization now preserves nonzero epsilon rather than matching it approximately to zero. Default epsilon also follows input dtype. |
| [206402](https://github.com/pytorch/pytorch/issues/206402), SDPA mask | Broadcast the sequence dimension required by Attention while retaining singleton batch/head dimensions. Dynamic sequence lengths are covered. |
| [206403](https://github.com/pytorch/pytorch/issues/206403), old Split | Original opset 12/13 report is outside the native opset 18+ domain; no old-opset fix claimed. |
| [206455](https://github.com/pytorch/pytorch/issues/206455), DOUBLE constants | `full`, `full_like` and `fill` now materialize constants directly in the target dtype. Scalar and linspace cases already preserve DOUBLE precision and have regressions. |
| [206456](https://github.com/pytorch/pytorch/issues/206456), Add zero | Floating Add-zero is retained because it changes negative zero. Exact integer identity rewrites remain available. |
| [206457](https://github.com/pytorch/pytorch/issues/206457), signbit | Exact floating signbit uses BitCast at opset 26+. Older opsets reject clearly rather than returning incorrect negative-zero/negative-NaN results. |
| [206459](https://github.com/pytorch/pytorch/issues/206459), log_ndtr | Confirmed, remains open: float32 inputs -8/-10 produce negative infinity, and -20 produces NaN instead of a finite log probability. A stable tail approximation is still needed. |
| [206460](https://github.com/pytorch/pytorch/issues/206460), BCE logits | Stable native lowering avoids saturated Sigmoid and infinite finite-input losses, preserving mixed-dtype rounding stages. Very small tail losses can still round to zero; avoid a larger compensation graph on the default path. |
| [206462](https://github.com/pytorch/pytorch/issues/206462), mixed stack | Already promotes input dtypes correctly; regression added. |
| [206472](https://github.com/pytorch/pytorch/issues/206472), integer Conv | Already rejected during translation when the target schema does not permit the dtype; regression prevents invalid model emission. |
| [206486](https://github.com/pytorch/pytorch/issues/206486), 8-bit MatMul | Already rejected for invalid schema types; valid int32 MatMul remains supported. |
| [206539](https://github.com/pytorch/pytorch/issues/206539), erfc | Confirmed, remains open: `1 - Erf(x)` cancels to zero at float32 x=5, where eager returns about 1.54e-12. Both optimization modes reproduce it. |
| [206626](https://github.com/pytorch/pytorch/issues/206626), atan2 | Remains open: special-value reproducers are strict expected failures. Retain the compact lowering while seeking an efficient implementation rather than adding many elementwise guards. |
| [206627](https://github.com/pytorch/pytorch/issues/206627), BatchNorm training | Training behavior already correct; directed regressions added. |
| [206790](https://github.com/pytorch/pytorch/issues/206790), squeeze | Known non-unit axes return Identity; symbolic dimensions use a runtime condition. |
| [206791](https://github.com/pytorch/pytorch/issues/206791), BF16 pool | Already rejected for target schemas lacking BF16; regression prevents invalid models. |
| [206813](https://github.com/pytorch/pytorch/issues/206813), einsum ellipsis | Insert singleton dimensions for operands with different ellipsis widths. Cross-review also fixed single-operand BOOL reductions to return INT64 while retaining BOOL for diagonal/permutation operations. |
| [206814](https://github.com/pytorch/pytorch/issues/206814), circular pad | Opset 18 uses modular Gather instead of invalid Pad wrap. Negative padding and empty cropped dimensions are covered. |
| [206815](https://github.com/pytorch/pytorch/issues/206815), unary dtypes | Integer rounding avoids lossy casts; unsigned Neg wraps; unsigned Relu is Identity. BF16 unary lowering consults the target IR schema and computes in float32 when needed. |

## Optimization scope

The export pipeline replaces ONNX Script's approximate arithmetic identity
matching with exact, dtype-aware rules. It reuses the other upstream rules
without changing their process-global configuration. This also preserves scales
such as DOUBLE `1.0000000001`.

`torch_onnx.ONNXProgram` subclasses the PyTorch class and overrides `optimize()`
to use the same standalone optimization pipeline as
`torch_onnx.export(..., optimize=True)`. Existing inference sessions are released
before optimization so subsequent execution uses the updated model. Other
PyTorch program methods are inherited without modifying the upstream class.

## Fuzzing and remaining work

[The differential harness](fuzzing.md) adds bounded, deterministic graph
composition checks and complete replay artifacts. Its initial domain targets
the dtype, scalar precision and signed-zero issues above. Attention, losses,
stateful training and special-function tail approximations still require
directed tests or additional generators.

The independent ORT Where signed-zero observation is documented separately in
the fuzzing guide. Runtime discrepancies remain failures in differential runs;
reference evaluation supplies attribution rather than silently accepting them.

The compact floating `trunc` lowering also retains known negative-zero and
negative-NaN sign limitations, recorded as strict expected failures. Integer
`trunc` still uses Identity to preserve integer precision and avoid unnecessary
work. Special-value fixes must account for execution cost before expanding the
default floating-point graph.
