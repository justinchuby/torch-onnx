# Exporter differential fuzzing

The deterministic generator creates small, pure `torch.nn.Module` DAGs and
constrained loss/attention and model operator recipes. It checks ONNX legality, output declarations,
and execution against PyTorch eager in both optimization modes.

## Default CI budget

The default smoke marker retains the original eight seven-node DAG seeds,
three CE/BCE/SDPA templates and seven model recipes:
**18 cases / 36 exports / 76 input-instance checks**.
Each dynamic template executes its capture shape and two additional shapes
against the same exported model and ORT session. CE smoke explicitly selects
mean reduction; the separate known unreduced CE failure is described below.
Model smoke uses opset 23, including a decode input with one token and an empty
KV cache. Long campaigns continue to include opset 26 and report missing kernels.
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
shapes; dynamic shapes are limited to constrained templates.

## Model architecture recipes

`--domain models` selects all seven families. They are small, independently
implemented operator patterns inspired by the sources below, not complete model
implementations or evidence that those checkpoints export successfully. No model
weights, Transformers installation, downloads or custom CUDA kernels are needed.

| Domain | Patterns exercised | Architecture reference |
| --- | --- | --- |
| `decoder` | RMSNorm, RoPE, grouped-query attention, KV concatenation, causal masks, SwiGLU and residuals | [Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5) |
| `moe` | Softmax/TopK routing, expert-weight gathers, batched matmul, weighted ScatterElements accumulation | [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) |
| `delta` | Causal depthwise Conv1d, L2 normalization, softplus/exp decay and a gated delta state update/read | [Qwen3.5 hybrid attention](https://huggingface.co/docs/transformers/model_doc/qwen3_5) |
| `vision` | Spatial patch Conv2d, bilinear resize, token reshaping, LayerNorm, GELU and pooling | [Qwen3-VL vision patterns](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) |
| `dit` | Adaptive LayerNorm conditioning, chunk/broadcast, GELU MLP and gated residual | [DiT implementation](https://github.com/facebookresearch/DiT/blob/main/models.py) |
| `sparse_attention` | Local masks, TopK selection, GatherElements and weighted value reduction | [DeepSeek-V4 sparse attention](https://huggingface.co/docs/transformers/model_doc/deepseek_v4) |
| `mhc` | Sinkhorn row/column normalization, einsum stream mixing and gated residual streams | [DeepSeek-V4 hyper-connections](https://huggingface.co/docs/transformers/model_doc/deepseek_v4) |

The delta recipe tests a single recurrent state transition, the vision recipe is
spatial, and the sparse recipe does not implement compressed attention. These
limits keep the graphs small and shrinkable without claiming full-model fidelity.

Seeds vary widths, hidden dimensions, expert counts, top-k, normalization epsilon,
GELU approximation, local windows, Sinkhorn iterations and strided layouts. The
non-sparse families rotate zero, small, ordinary and larger finite activations.
MoE includes concentrated routes leaving experts unused and distributed routes;
the gather implementation does not exercise a data-dependent NonZero dispatch.
Routing logits are separated to avoid unspecified TopK ties. Sparse attention
keeps at least top-k visible keys per row and uses nondegenerate random queries.

Parameters are serialized `tensor_const` nodes and become captured initializers;
they are not regenerated during forward/replay. Intermediate-output mode exposes
each recipe node, while final-only mode allows constant folding and fusion.
Version-3 JSON also records explicit dynamic bounds, including zero past length,
batch/token size one, and the spatial lower bounds required by patch convolution.
Each dynamic model reuses the same export for three input instances.

These composed float32 models store their numerical budget in each case:
`rtol=1.3e-6, atol=1e-5`, the float32 defaults of `torch.testing.assert_close`.
This tightens relative error and permits normal absolute roundoff near zero.
Calibration of a LayerNorm intermediate against float64 observed roughly
`3e-7` PyTorch and `2e-6` ONNX error. Existing v1/v2 DAG/loss tolerances are
unchanged, as are dtype/shape, NaN/Inf and signed-zero checks. A saved stricter
case keeps its stricter budget when replayed; no failed seed is rewritten or skipped.

## Bounded isolated campaigns

The CLI isolates execution by default. The old `fuzzer.py` entry point delegates
to the same campaign CLI, so existing commands also gain isolation.

```sh
# Preserve the original static DAG domain, now with crash-safe execution
python tests/fuzz/fuzzer.py --seed 0 --count 256 --report /tmp/dag-report.json

# Total 256 cases across 126 domain/output/opset/shape configurations
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/fuzz/campaign.py \
  --seed 0 --count 256 --domain all --output-mode both --shape-mode both \
  --opset 18 --opset 23 --opset 26 --timeout 120 --keep-going \
  --report /tmp/fuzz-report.json --failure-dir /tmp/fuzz-artifacts

# Model recipes on released kernels: two rounds of 56 configurations
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/fuzz/campaign.py \
  --seed 0 --count 112 --domain models --output-mode both --shape-mode both \
  --opset 18 --opset 23 --timeout 120 --keep-going \
  --report /tmp/model-report.json --failure-dir /tmp/model-artifacts

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

Reports include attempted `domain_counts` and `generated_ops`, plus the union of
captured `aten_ops` and serialized `onnx_ops` from completely validated cases.
ONNX inventory follows referenced functions (including overloads) and control-flow
subgraphs, excluding unused function definitions. This is operator presence,
not branch coverage or proof that every operator variant has been tested.

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
Version 3 adds explicit dynamic bounds and float32 tolerances for model recipes.
V1/v2 serialization omits these new fields, retaining the earlier wire format.
Worker fingerprints include `model_families.py` as well as the runner and core
generator. Multi-dimensional transposed inputs preserve their requested strides.

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

Known observations are intentionally visible:

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
- ORT 1.32 cannot expand Swish at opset 26, although the ONNX graph is legal.
  SiLU-containing model recipes therefore fail normally at that opset, retaining
  their artifacts. This is also covered independently by `test_opsets.py`, which
  uses the ONNX reference backend for Swish above opset 24. The fuzz runner never
  silently switches its runtime or counts those cases as passes. Opsets 18/23
  retain the established Sigmoid/Mul lowering.

```sh
python -m pytest tests/fuzz/test_where_backend.py -m integration -q
```
