# Release versioning

- Align the exporter version's major and minor components with the current stable
  PyTorch release used for development. For example, PyTorch 2.15.x corresponds
  to exporter versions 2.15.x.
- Increment the exporter's patch component for every release. Exporter patch
  numbers are independent of PyTorch patch numbers: 2.15.0, 2.15.1, 2.15.2, etc.
- When adopting a new PyTorch major/minor development baseline, start its exporter
  release series at patch zero. Keep package metadata and release documentation
  consistent when preparing a release.

# PyTorch compatibility

- Do not constrain the `torch` dependency to a particular release.
- CI must exercise stable PyTorch releases published during the preceding year.
  Keep the tested-version manifest in `src/torch_onnx/_torch_compat.py` current
  using official PyTorch release dates.
- Keep capability checks and compatibility adapters in that module where useful.
  Prefer capability checks to version branches; do not hide unrelated failures.
- An incompatible installation or unvalidated release series should receive one actionable
  compatibility warning per process, naming tested versions and suggesting an
  upgrade. Stable patch releases in a validated minor series may use the same
  adapters without a version-only warning. Version differences alone must not
  prevent attempting export.

# Translation performance

- Measure runtime before replacing operator implementations for performance.
- Prefer tensor operations over ONNX Sequence operations when they improve
  performance without changing observable outputs or creating excessive memory
  use. Preserve dynamic-shape and empty-input behavior.

# Runtime boundaries

- Target ONNX specification semantics. Do not add graph rewrites, casts, or
  operator decompositions solely to work around ONNX Runtime limitations.
- Keep spec-valid graphs when a runtime lacks support. Test with reference
  execution or explicit graph/schema assertions and document runtime limits.
- Pure downstream FX graph repairs remain with the downstream framework.
