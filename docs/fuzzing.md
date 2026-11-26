# Exporter differential fuzzing

The deterministic generator creates small, pure `torch.nn.Module` DAGs and
constrained loss/attention models. It checks ONNX legality, output declarations,
and execution against PyTorch eager in both optimization modes.

## Default CI budget

Default pytest retains the original eight seven-node DAG seeds and adds three
CE/BCE/SDPA templates: **11 cases / 22 exports / 34 input-instance checks**.
Each dynamic template executes its capture shape and two additional shapes
against the same exported model and ORT session. CE smoke explicitly selects
mean reduction; the separate known unreduced CE failure is described below.
Harness tests exercise replay, comparison, reduction, crash and timeout handling;
there are no timing assertions or random-seed skips in this smoke tier.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/fuzz -m fuzz -q
TORCH_ONNX_FUZZ_SEED=123 python -m pytest tests/fuzz -m fuzz -q
```

## Output visibility and generation domains

`output_mode=all` exposes intermediate values and input signbit probes to locate
errors. `output_mode=final` exposes only the final DAG result, allowing optimizers
to eliminate intermediates and fuse operations as they would in a production
model. Integer and Boolean DAGs, casts, broadcasting, shape operations,
noncontiguous layouts, empty dimensions and floating special values are covered.

Generation supports opsets **18, 23 and 26**, with signbit probes only at 26.
The initial data domain is CPU float32, float64, int32, int64 and bool.
CE/BCE/SDPA templates add legal relationships between inputs, randomized loss
reductions, and dynamic shapes. Dynamic SDPA varies batch, query and key sequence
lengths, including a broadcast Boolean mask. Arbitrary DAGs currently use static
shapes; dynamic shapes are limited to these constrained templates.

## Bounded isolated campaigns

The CLI isolates execution by default. The old `fuzzer.py` entry point delegates
to the same campaign CLI, so existing commands also gain isolation.

```sh
# Preserve the original static DAG domain, now with crash-safe execution
python tests/fuzz/fuzzer.py --seed 0 --count 256 --report /tmp/dag-report.json

# Total 256 cases across the 42 supported domain/output/opset/shape configurations
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/fuzz/campaign.py \
  --seed 0 --count 256 --domain all --output-mode both --shape-mode both \
  --opset 18 --opset 23 --opset 26 --timeout 120 --keep-going \
  --report /tmp/fuzz-report.json --failure-dir /tmp/fuzz-artifacts

# Explicit long pytest tier, stopping at the first discrepancy
TORCH_ONNX_FUZZ_COUNT=256 TORCH_ONNX_FUZZ_TIMEOUT=120 \
  python -m pytest tests/fuzz -m fuzz_long -q
```

`--count` is a **total-case budget**, not a multiplier per configuration. At
least one full configuration round is required, and each round is shuffled
deterministically to avoid coupling seed edge profiles to configuration choices.
The CLI stops at the first failure unless `--keep-going` is supplied. That flag
records each classified failure and continues within the budget, then **exits
nonzero** if any case failed. Worker protocol failures remain fatal. Successful
export/input counts in the JSON summary count only completely validated cases;
failed cases are never reported as passes.

Workers are reused and rotated every 64 cases. A complete per-case wall-clock
budget includes worker startup/imports, eager, export, checker, ORT, reference
attribution and bounded reduction. Timeout or abrupt native failure terminates
the owned worker process group on Unix; Windows uses a Job Object with tree-kill
fallback (the Windows branch has not been exercised locally). Native diagnostics
and faulthandler output are retained, and core dumps are disabled. The generator
controller itself imports Torch; generation is not covered by the worker timeout.
`--in-process` is an explicit debugging option with no crash/timeout isolation.

The manual **Differential fuzz campaign** Actions workflow runs the isolated
matrix and uploads the summary and case artifacts even after a failure. It is
separate from required PR checks and has no scheduled execution. A 20-minute
campaign interrupt precedes the job's 30-minute limit, leaving time to close
workers and upload artifacts even when many cases exhaust their budgets.

## Failure artifacts and replay

Before execution, each case receives a unique directory with atomically saved
`case.json` and `config.json`. It also retains `stdout.log`, `stderr.log` and,
when an outcome can be recorded, `result.json`; normal Python failures may
include a minimized `failure-case.json`.
The original case survives even when native code crashes before Python can
raise an exception. Identical seeds with different configurations and repeated
campaigns do not overwrite previous artifacts.

Case JSON version 2 stores actual input values/layouts, the DAG, output mode,
opset, domain, dynamic symbols and additional shape instances. Old version-1
artifacts still load with their original all-output/opset-26 behavior. Replay
uses these values rather than regenerating a seed. NaN/Inf have JSON string
encodings, including negative NaN, so payloads remain valid JSON.

```sh
# Exact case data with the current default execution budget
python tests/fuzz/campaign.py --case /tmp/fuzz-artifacts/seed-.../case.json

# Recorded timeout/reduction settings, including moved artifact directories
python tests/fuzz/runner.py --replay /tmp/fuzz-artifacts/seed-.../config.json
```

Generation flags cannot override `--case`. Config replay uses the current trusted
worker and interpreter, never an executable command embedded in JSON. It reports
recorded environment/cwd differences. Only for trusted own artifacts, explicitly
add `--restore-environment` to restore the recorded working directory and selected
environment keys. Runtime versions and source hashes are diagnostic: use a
matching checkout/environment when reproducing historical failures.

The suffix reducer performs at most eight attempts, preserving stage, optimizer
mode and reference attribution. It validates candidates before retaining them,
so a minimized template remains loadable. It does not minimize input values or
claim a globally minimal result.

## Failure classification and known observations

Capture/lowering/optimization exceptions are `export`; checker/declaration errors
are `invalid_graph`; ORT setup/execution errors are `runtime`; value mismatches
are `semantic`. Invalid generated eager programs are `generator` failures.
Isolation adds `timeout`, `crash` and `worker` protocol/harness failures.

Integer/Boolean results match exactly. Floating comparisons check dtype/shape,
NaN masks, positive/negative infinity masks, and finite tolerances separately.
Explicit add-zero probes and bit-preserving lineages also require signed-zero
agreement; ordinary arithmetic/Where does not acquire blanket bitwise checks.
Only after a numerical mismatch, ONNX ReferenceEvaluator distinguishes `backend`,
`export_semantic` and `unattributed`. Attribution never converts a failure into a
pass. Missing runtime kernels are failures, not automatic exclusions.

Two known observations are intentionally visible:

- On PyTorch 2.15, retracing dynamic CE with `reduction=none` can turn vector
  metadata into scalar metadata and rewrite a later flatten into `view([1])`.
  `tests/exporter/test_dynamic_loss_metadata.py` has an independent, narrowly
  scoped strict expected failure using raw `ExportedProgram.run_decompositions`.
  No exporter workaround or random-case exclusion is installed. Long campaigns
  still generate this configuration and fail normally, saving its full replay.
- `test_where_backend.py` observes ORT Where losing negative zero on macOS arm64
  with exactly ORT 1.32.0. Its dedicated strict expected failure is an optional
  integration test; the default random DAG smoke has no seed-level xfails.
  Generated Where-to-Reciprocal infinity-sign differences remain hard failures.

```sh
python -m pytest tests/fuzz/test_where_backend.py -m integration -q
```
