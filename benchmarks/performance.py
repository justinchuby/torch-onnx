"""Isolated, reproducible exporter/ORT measurements (see docs/performance.md)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = ("mlp", "bce", "sdpa", "atan2")
METRICS = (
    "export_seconds",
    "ort_seconds",
    "export_peak_rss_bytes",
    "ort_peak_rss_bytes",
    "nodes",
)


def peak_rss():
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("Peak RSS measurement supports only Linux and macOS")
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(args):
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    shape = (args.atan2_size,) if args.case == "atan2" else (2, 4, 32, 64)
    if args.stage == "export":
        import torch_onnx

        if (
            not Path(torch_onnx.__file__)
            .resolve()
            .is_relative_to(args.source_dir.resolve() / "src")
        ):
            raise RuntimeError("Exporter imported from outside --source-dir")

        class Model(torch.nn.Module):
            def forward(self, x):
                if args.case == "atan2":
                    return torch.atan2(x, x + 0.5)
                if args.case == "mlp":
                    return torch.relu(x @ x.transpose(-1, -2))
                if args.case == "bce":
                    return torch.nn.functional.binary_cross_entropy_with_logits(
                        x, torch.sigmoid(x)
                    )
                return torch.nn.functional.scaled_dot_product_attention(x, x, x)

        x = torch.randn(shape)
        start = time.perf_counter()
        program = torch_onnx.export(
            Model().eval(), (x,), opset_version=args.opset, optimize=True, verbose=False
        )
        elapsed = time.perf_counter() - start
        np.save(args.model + ".expected.npy", Model().eval()(x).detach().numpy())
        program.save(args.model)
        program.release()
        result = {"export_seconds": elapsed, "export_peak_rss_bytes": peak_rss()}
    else:
        import onnx
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            args.model, options, providers=["CPUExecutionProvider"]
        )
        # Check semantic equivalence outside the timer, using the export input.
        feed = {session.get_inputs()[0].name: torch.randn(shape).numpy()}
        np.testing.assert_allclose(
            session.run(None, feed)[0],
            np.load(args.model + ".expected.npy"),
            rtol=1e-4,
            atol=1e-5,
        )
        for _ in range(args.warmup):
            session.run(None, feed)
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            session.run(None, feed)
            samples.append(time.perf_counter() - start)
        result = {
            "ort_seconds": statistics.median(samples),
            "ort_peak_rss_bytes": peak_rss(),
            "nodes": len(onnx.load(args.model).graph.node),
        }
    Path(args.result).write_text(json.dumps(result))


def fingerprint(args):
    cpu = platform.processor()
    if sys.platform == "linux":
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    elif sys.platform == "darwin":
        probe = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            cpu = probe.stdout.strip()
    return {
        "schema": 1,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnx-shape-inference",
                "onnxruntime",
                "numpy",
            )
        },
        "config": {
            "opset": args.opset,
            "threads": args.threads,
            "interop_threads": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "timeout": args.timeout,
            "cases": args.cases,
            "input": [2, 4, 32, 64],
            "atan2_size": args.atan2_size,
            "dtype": "float32",
            "optimize": True,
            "provider": "CPUExecutionProvider",
            "seed": 0,
            "python_hash_seed": "0",
        },
    }


def compare(current, baseline):
    if (
        not isinstance(current, dict)
        or not isinstance(baseline, dict)
        or "fingerprint" not in current
        or "fingerprint" not in baseline
    ):
        raise ValueError("Malformed benchmark report: missing fingerprint")
    if current["fingerprint"] != baseline["fingerprint"]:
        raise ValueError(
            "Incompatible hardware, versions, or benchmark configuration; refusing comparison"
        )
    try:
        current_results = current["results"]
        baseline_results = baseline["results"]
        if not current_results or set(current_results) != set(baseline_results):
            raise ValueError("Benchmark cases are missing or do not match")
        for label, results in (
            ("current", current_results),
            ("baseline", baseline_results),
        ):
            for case, metrics in results.items():
                for metric in METRICS:
                    value = metrics[metric]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(
                            f"Invalid {label} metric {case}/{metric}: expected a finite positive number"
                        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Malformed benchmark results: missing cases or metrics"
        ) from exc
    return {
        case: {
            metric: current_results[case][metric] / baseline_results[case][metric] - 1
            for metric in METRICS
        }
        for case in current_results
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="performance.json")
    parser.add_argument("--baseline")
    parser.add_argument(
        "--source-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument(
        "--atan2-size", type=int, choices=(131072, 1048576), default=131072
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--max-regression",
        type=float,
        help="Optional allowed fractional increase in any metric (e.g. 0.2)",
    )
    parser.add_argument("--stage", choices=("export", "ort"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.threads, args.iterations, args.repeats) < 1
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "threads, iterations, repeats and timeout must be positive; warmup nonnegative"
        )
    if args.max_regression is not None and (
        not math.isfinite(args.max_regression)
        or args.max_regression < 0
        or not args.baseline
    ):
        parser.error(
            "--max-regression requires --baseline and a finite nonnegative value"
        )
    if args.stage:
        worker(args)
        return
    if sys.platform not in ("linux", "darwin"):
        parser.error(
            "Benchmark measurements support only Linux and macOS; JSON comparison is portable"
        )
    revision = subprocess.run(
        ["git", "-C", str(args.source_dir.resolve()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "source_revision": revision.stdout.strip()
        if revision.returncode == 0
        else None,
        "fingerprint": fingerprint(args),
        "source_dir": str(args.source_dir.resolve()),
        "results": {},
        "samples": {},
    }
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source_dir.resolve() / "src"),
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        PYTHONHASHSEED="0",
    )
    with tempfile.TemporaryDirectory() as temporary:
        for case in args.cases:
            samples = []
            for _ in range(args.repeats):
                sample = {}
                for stage in ("export", "ort"):
                    result = Path(temporary) / "result.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--atan2-size",
                        str(args.atan2_size),
                        "--source-dir",
                        str(args.source_dir.resolve()),
                        "--stage",
                        stage,
                        "--case",
                        case,
                        "--model",
                        str(Path(temporary) / "model.onnx"),
                        "--result",
                        str(result),
                        "--threads",
                        str(args.threads),
                        "--opset",
                        str(args.opset),
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                    ]
                    completed = subprocess.run(
                        command,
                        env=env,
                        cwd=temporary,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                    if completed.returncode:
                        raise RuntimeError(
                            f"{case}/{stage} failed:\n{completed.stdout}\n{completed.stderr}"
                        )
                    sample.update(json.loads(result.read_text()))
                samples.append(sample)
            report["samples"][case] = samples
            report["results"][case] = {
                metric: statistics.median(sample[metric] for sample in samples)
                for metric in METRICS
            }
    if args.baseline:
        report["changes"] = compare(report, json.loads(Path(args.baseline).read_text()))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.max_regression is not None and any(
        change > args.max_regression
        for changes in report["changes"].values()
        for change in changes.values()
    ):
        raise SystemExit("Benchmark regression exceeds explicit threshold")


if __name__ == "__main__":
    main()
