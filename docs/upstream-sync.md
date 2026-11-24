# Upstream synchronization

The standalone exporter targets **PyTorch 2.15.x**, the latest stable release
(`v2.15.0`, published 2026-10-28), not nightly/main. Python 3.11+ is required.

- Modern upstream main: `de46fbb8532d712ea79ad3826b19715e8e494892`.
- Native migration head (PR 204549, refreshed before port): `36b36c5a02e8a598dfa3b5705b0501aaa5a2bbef`.
- Pre-migration baseline: `b310ef4b8c569397fa27d1150d42c78c927470ad`.
- All 35 modern exporter Python files are synchronized from upstream
  plus the cumulative migration. Main differs from the baseline only in
  `torch/onnx/_internal/fx/_pass.py` and legacy FX promotion, neither used by
  the modern exporter. There are no conflicting modern exporter fixes to reconcile.
- The final port accounts for all 50 exporter source files after migration and
  retains all 593 torchlib function definitions (checked by AST inventory).

## Scope

Native core/factories/sequences/linalg/recurrent/neural/normalization/pooling/
attention/padding/interpolation/FFT/prims/special/quantized/symbolic/higher-order/
optional-vision translations and implementation comments are retained.
Capture strategies, dynamic shapes, reporting, verification, ONNXProgram,
ExportableModule, InputObserver, IR passes, float4 support and optimization are
local implementations, not delegation to the installed PyTorch exporter.
`torch_onnx.ops` reuses PyTorch's symbolic operator definitions: re-registering
their dispatcher schemas would conflict with PyTorch. TorchScript legacy export,
old dynamo_export/ExportOptions, pre-2.15 compatibility and FX graph rerun promotion
are not supported. The legacy ONNX Script *custom translation adapter* remains
for existing export-local custom callables; built-ins are all native.

The upstream exporter unit suites and torchlib OpInfo coverage are included.
OpInfo suites depend on PyTorch's internal test harness and optional torchvision;
availability is distinct from local execution coverage. CUDA/source-build
validation is not implied.

## Standalone verification

The default CPU suite passed 947 tests and 9,476 subtests (one existing skip).
Selected arithmetic/sigmoid/quantization OpInfo coverage passed 16 tests and
991 subtests, with four source skips and four expected failures. Validation
used PyTorch 2.15.0+cpu, Python 3.13, released ONNX 1.23.2, onnx-ir 0.1.16,
ONNX Script 0.7.2 and ONNX Runtime 1.32.0. Nineteen selected small-model
integration tests also passed, covering conditionals, loops, quantization,
Attention, GroupNormalization and RMSNormalization. No unrelated editable
dependency checkout was used. Full OpInfo, Hugging Face integration and CUDA
were not run. [Operator coverage](../tests/torchlib/README.md) lists the retained
families and limitations.

Pyrefly 0.58 checks the modern exporter/native modules. Two legacy ONNX Script
DSL modules (`_building`, `_tensors`) remain outside the configured check because
their tensor-operator/Evaluator overrides conflict with upstream base annotations;
they are covered by runtime adapter tests. Existing upstream type suppressions
are retained, not represented as a clean unsuppressed check.

Fake-tensor exports should use `optimize=False`, then `apply_weights` with real
weights before optimization or saving initializers. The obsolete prototype
TorchScript tests, removed PyTorch PT2E quantization-capture API test, old copied
TorchScript harness and old FX diff debugging utility were removed. Native
quantization boundaries are covered by the retained native and OpInfo tests.
The distributed wheel includes the upstream PyTorch license.

## Cumulative migration accounting

| Source path | Disposition |
| --- | --- |
| `CODEOWNERS` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `OWNERSHIP.yaml` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `docs/source/onnx.md` | Adapted public native APIs and documentation to standalone package |
| `test/onnx/exporter/test_custom_translation.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_ir_builder.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_factory_sequence_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_final_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_leaf_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_neural_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_recurrent_residual_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_registration.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_structural_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_torchlib.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/exporter/test_native_type_promotion.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/test_fx_type_promotion.py` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `test/onnx/torchlib/README.md` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/torchlib/extra_opinfo.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/torchlib/ops_test_common.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/torchlib/ops_test_data.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/onnx/torchlib/test_ops.py` | Ported tests/coverage (PyTorch internal harness dependencies retained) |
| `test/test_public_bindings.py` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `tools/linter/adapters/pyrefly_linter.py` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `torch/onnx/README.md` | Adapted public native APIs and documentation to standalone package |
| `torch/onnx/__init__.py` | Adapted public native APIs and documentation to standalone package |
| `torch/onnx/_internal/exporter/_compat.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_core.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_custom_translation.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_decomp.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_fx_passes.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_ir_builder.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_legacy_promotion.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_registration.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/_tensor_typing.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/_torchlib_registry.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/__init__.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/_promotion.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/common.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/core.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/embedding.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/factories.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/fft.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/hop.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/linalg.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/neural.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/nn.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/normalization.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/prims.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/quantized_decomposed.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/recurrent.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/residual.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/sequences.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/special.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/symbolic.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/symops.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/exporter/_torchlib/ops/vision.py` | Ported with standalone namespace adaptation |
| `torch/onnx/_internal/fx/passes/__init__.py` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
| `torch/onnx/_internal/fx/passes/type_promotion.py` | Not applicable: PyTorch ownership/lint/public-binding or removed legacy FX promotion |
