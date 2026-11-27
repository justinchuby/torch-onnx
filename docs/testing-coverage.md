# Project coverage

Coverage measures every Python source file in `src/torch_onnx`, including
namespace directories such as `bin`. Statement and branch coverage are both
reported. Production files are not omitted to raise the percentage.

The default suite includes exporter, compatibility, torchlib, fuzz smoke,
serialization, model regression and benchmark contract tests:

```sh
COVERAGE_FILE=.coverage.default OMP_NUM_THREADS=1 python -m pytest \
  --cov=torch_onnx --cov-branch --cov-report=json:coverage-default.json
```

Optional operator, development-opset and Transformers integration suites run in
their own dependency environments. Follow the pinned installations in
`.github/workflows/test.yaml` and `.github/workflows/transformers.yaml`, then run:

```sh
COVERAGE_FILE=.coverage.optional python -m pytest \
  tests/exporter/test_optional_packages.py -m 'integration or not integration' \
  --cov=torch_onnx --cov-branch --cov-report=json:coverage-optional.json
COVERAGE_FILE=.coverage.opset29 python -m pytest \
  tests/exporter/test_opsets.py -m opset29 \
  --cov=torch_onnx --cov-branch --cov-report=json:coverage-opset29.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  COVERAGE_FILE=.coverage.transformers python -m pytest \
  tests/integration/test_transformers_exporter.py -m integration \
  --cov=torch_onnx --cov-branch --cov-report=json:coverage-transformers.json
```

Use the same source revision for all measurements. Merge the raw data, rather
than averaging percentages or adding per-job counts. Explicit source enumeration
also retains files that no suite imported:

```python
from pathlib import Path

import coverage

cov = coverage.Coverage(data_file='.coverage.combined')
cov.combine([
    '.coverage.default', '.coverage.optional', '.coverage.opset29',
    '.coverage.transformers',
], keep=True)
cov.save()
sources = [str(path) for path in Path('src/torch_onnx').rglob('*.py')]
cov.report(morfs=sources, show_missing=True)
cov.json_report(morfs=sources, outfile='coverage-combined.json')
```

CI uploads the main, optional operator, development-opset, Transformers and
PyTorch-version compatibility results to Codecov. Each job retains XML and JSON
artifacts so partial jobs and their source revision can be inspected separately.
The main job appends the Linux FBGEMM reference cohort even if the default tests
fail, while preserving the failing job status.

A coverage report is not a passing test result. Inspect pytest's exit status and
failure list alongside coverage. Some platform-specific quantized models require
FBGEMM, and some valid newer ONNX operators lack ONNX Runtime kernels. Numerical
checks for those operators can use an explicit supported test opset or ONNX
reference execution; exporter semantics must continue to follow the ONNX spec.
