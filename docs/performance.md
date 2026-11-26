# Performance baselines

Run the deterministic CPU benchmark from the repository root:

```sh
python benchmarks/performance.py --output /tmp/baseline.json
python benchmarks/performance.py --baseline /tmp/baseline.json --output /tmp/current.json
python -m pytest benchmarks/test_performance.py
```

The four workloads cover matrix multiplication plus activation, BCE logits loss, SDPA, and a large vector `atan2(x, x + 0.5)`. Inputs are float32 `[2, 4, 32, 64]`, except atan2 uses `[131072]` by default (`--atan2-size 1048576` measures a larger vector); the seed is fixed. Defaults are opset 23, optimized export, one intra-op thread and one inter-op thread, ten runtime warmups, 100 runtime samples and three independent process repetitions. CPUExecutionProvider runs sequentially. These are representative microbenchmarks, not a model throughput guarantee.

Each runtime worker also checks its result against eager PyTorch outside the timer. Each repetition exports in a fresh subprocess, then loads and runs its saved model in another fresh subprocess. `export_seconds` measures the first exporter call after imports and model/input construction, including capture, translation and optimization; serialization is outside the timer. `ort_seconds` is the median wall time of synchronous `session.run` calls, including Python dispatch and output allocation, excluding session construction and warmup. The final report takes the median across repetitions and retains individual repetition samples. `nodes` counts serialized top-level graph nodes before ORT optimization, excluding nodes in function bodies and subgraphs.

`export_peak_rss_bytes` and `ort_peak_rss_bytes` are each worker's lifetime process peak resident set size, including imports, input allocation, and (respectively) serialization or session construction. They are **not incremental exporter allocations** or isolated kernel memory. Workers prevent earlier cases contaminating `ru_maxrss`; Linux reports KiB and macOS bytes, normalized to bytes. The script currently supports these Unix platforms.

JSON fingerprints include host identity, OS, CPU, Python/dependency versions and all measurement configuration. Comparison rejects any fingerprint mismatch, including a different host, rather than implying that results from unrelated CI machines are comparable. Use `--source-dir /absolute/path/to/checkout` to compare revisions with the same interpreter and dependency environment: workers start outside either checkout with only the selected `src` on PYTHONPATH, avoiding editable-install source confusion. The source revision is intentionally not part of the compatibility fingerprint. Reports record the resolved source directory and Git HEAD (uncommitted edits are not identified); retain clean checkout commit identifiers alongside archived reports. The harness content hash is part of the fingerprint.

The manual **Performance comparison** Actions workflow checks out the requested baseline and current revision in the same job and runs both with shared dependencies. A baseline does not need to contain the benchmark script: both measurements use the current harness. Dependencies must support both revisions. Artifacts preserve both JSON reports. Default CI contains no timing gate; the manual workflow's optional `max_regression` or CLI `--max-regression 0.2` explicitly fails on an increase over 20% in any metric. This broad threshold includes memory and node count; inspect the per-metric fractional changes before drawing conclusions. Shared runners still have scheduling/frequency noise: repeat measurements on an idle dedicated machine before accepting a small change.

## Embedding bags

The embedding-bag lowering uses tensor scan outputs from ONNX Loop instead of
SequenceEmpty/SequenceInsert/ConcatFromSequence. Each iteration emits its row,
size and argmax directly. The auxiliary index-to-bag map uses ScatterElements
with addition and CumSum, requiring O(N+B) storage instead of an N-by-B mask.
Plain embedding already uses Gather.

Compare an earlier lowering with the current implementation on the same machine:

```sh
git show 6a694439cec4e54678b4b4faa6d908a92b60571b:src/torch_onnx/_torchlib/ops/embedding.py > /tmp/old-embedding.py
python benchmarks/embedding_bag.py --old-source /tmp/old-embedding.py --output /tmp/embedding.json
```

The comparison checks all four outputs against eager ATen before timing, uses
one Torch/ORT thread, five warmups and fifteen alternating old/new samples.
It includes sum, mean and max at three sizes. On the development Mac, median
latencies were:

| Bags × length × width | Sum, old → new | Mean, old → new | Max, old → new |
| --- | --- | --- | --- |
| 4 × 4 × 16 | 63 → 48 µs | 83 → 73 µs | 81 → 64 µs |
| 256 × 16 × 64 | 3.68 → 2.10 ms | 8.83 → 3.14 ms | 8.98 → 2.95 ms |
| 1024 × 32 × 128 | 28.59 → 11.69 ms | 160.08 → 14.91 ms | 162.79 → 16.42 ms |

These graphs retain auxiliary outputs, so they measure the mapping improvement
as well as scans. Models consuming only the primary output can have auxiliary
work pruned and should not be expected to achieve the same large speedups.
The targeted tests separately cover padding, per-sample weights, duplicate
offsets, zero bags, zero width and repeated dynamic inputs.
