# Exporter differential fuzzing

The deterministic fuzzer in `tests/fuzz/fuzzer.py` generates small, pure
`torch.nn.Module` DAGs and compares their ONNX execution with PyTorch eager.
It exercises compositions, including reused intermediate values, rather than
running isolated random operators.

## Default smoke budget

Default `pytest` includes eight fixed seeds (0–7), seven generated operations per
DAG, and both `optimize=False` and `optimize=True`: **16 exports total**. The
remaining tests in `tests/fuzz` validate replay, the special-value oracle and the
suffix reducer. Long campaigns are marked `fuzz_long` and excluded by default.

```bash
# The same eight graph cases used by default CI
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/fuzz -m fuzz -q

# Replay one generated seed
TORCH_ONNX_FUZZ_SEED=123 python -m pytest tests/fuzz -m fuzz -q

# Explicit longer budget: 256 graph cases / 512 exports
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TORCH_ONNX_FUZZ_SEED=0 TORCH_ONNX_FUZZ_COUNT=256 \
  python -m pytest tests/fuzz -m fuzz_long -q

# CLI, independent of pytest; fail at the first discrepancy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tests/fuzz/fuzzer.py --seed 0 --count 256
```

The count is a bound on cases, not a wall-clock timeout. Every case checks two
optimization modes; the failure reducer can perform up to eight additional
exports in the failing mode. ORT sessions use one intra/inter-op thread. The
fuzzer does not start background jobs or schedule future campaigns.

## Supported generation domain

The initial domain is CPU float32, float64, int32, int64 and bool at **opset 26**.
That opset permits exact floating-point `signbit` probes through BitCast. It does
not interpret explicit rejection at an older opset as a successful fuzz case.

Operations include arithmetic, integer floor division and round, casts, logical
operators, where, transpose, unsqueeze/squeeze, flatten, concatenate and stack.
The generator tracks intermediate dtype and shape to choose legal inputs;
integer division has a nonzero positive divisor, and floating values containing
NaN/Inf are not converted to integers. Graphs remain small. Zero/one-sized
shapes, noncontiguous inputs, integers above float32/float64 exact-integer ranges,
float64 scalar constants, signed zeros, NaN and infinities receive fixed edge
profiles in addition to randomized compositions.

Risk categories were informed by the exporter issue inventory: precision-losing
integer division, integer round, no-op squeeze, mixed-dtype stack, float64 scalar
materialization, signed-zero rewrites and signbit. This is not coverage of all
22 reported issues: training/state mutation, convolution, attention and losses
need separate model-specific generators or directed regressions.

## What counts as a failure

For both optimization modes, the harness:

1. Exports the model and classifies capture/lowering/optimization exceptions as
   `export` failures.
2. Runs `onnx.checker.check_model(..., full_check=True)` and checks output
   declarations against eager dtype/rank/concrete dimensions (`invalid_graph`).
3. Creates a CPU ONNX Runtime session and executes it (`runtime`).
4. Compares output dtype, shape and values (`semantic`). Only on failure, ONNX
   ReferenceEvaluator is used to attribute the mismatch: `backend` when reference
   agrees with eager, `export_semantic` when reference also disagrees, and
   `unattributed` when the reference evaluator cannot execute it. Every category
   still fails; attribution does not turn a backend issue into a pass.

Integer and bool results must match exactly. Float comparisons separately check
NaN masks and positive/negative infinity masks, then finite values with dtype
specific tolerances. Explicit add-zero probes, and input values carried through
neg/abs or bit-preserving structural operations, also compare exact zero and its
sign. Ordinary arithmetic and Where outputs do not acquire a blanket bitwise
zero requirement through later structural operations. Original floating
inputs have observable Boolean `torch.signbit` outputs, including negative NaNs;
NaN payloads are not required to match bit-for-bit. Thus `allclose(equal_nan=True)`
cannot silently hide a lost NaN, changed infinity sign or the signed-zero probe.

Failures are not skipped, and unsupported randomly generated operators are not
counted as passes. Invalid generated eager programs are separately reported as
`generator` failures, with the same replay payload. Expand the supported domain with a regression when adding
new operations. An optional backend lacking a generated kernel is a visible
`runtime` failure, not an automatic exclusion.

## Stable replay and reduction

Each failure prints the stage, optimization mode, seed and **complete JSON
case**, including all input values/layouts and the DAG. Special values use JSON
strings (`nan`, `negative_nan`, `+inf`, `-inf`), so the file is valid JSON without
nonstandard numeric extensions. Replay uses the stored case, not a freshly
regenerated seed; generator changes do not invalidate a saved reproducer.

The CLI writes a failing case in `torch-onnx-fuzz-failures` under the system
temporary directory by default; python -m pytest writes it in the case's temporary directory and prints its path.
Pass `--failure-dir DIR` to keep a campaign corpus elsewhere.

```bash
python tests/fuzz/fuzzer.py --case /path/to/seed-123-semantic-1.json
TORCH_ONNX_FUZZ_CASE=/path/to/case.json python -m pytest tests/fuzz -m fuzz -q
```

The reducer greedily removes DAG suffixes while preserving the failure category,
reference attribution and optimizer mode. Every intermediate is an output, so a retained prefix is a
legal, inspectable graph. The reducer is deliberately bounded; it does not claim
that the result is globally minimal, does not change input values, and stops
when shortening the prefix loses or changes the failure. Use `--no-reduce` to
retain the original case or to avoid extra exports while debugging a failure.

## Known backend observation

`test_where_backend.py` keeps a separate minimal, exporter-independent observation:
on macOS arm64 with exactly ONNX Runtime 1.32.0, a standard ONNX Where node can
turn selected `-0.0` into `+0.0` even with ORT graph optimization disabled. Eager
and ONNX ReferenceEvaluator preserve the input bit. The test has a **strict**
expected failure only for that exact environment and only for its dedicated
signed-zero mismatch exception; checker/reference failures and other exceptions
are not covered. The random DAG smoke has no seed-level xfails. A generated
Where→Reciprocal composition that changes an infinity sign is still a hard
semantic failure and can be attributed to the backend by the reference oracle.

This exporter-independent observation is marked `integration` and excluded from
the default smoke budget. Run it explicitly when investigating a backend:

```bash
python -m pytest tests/fuzz/test_where_backend.py -m integration -q
```
