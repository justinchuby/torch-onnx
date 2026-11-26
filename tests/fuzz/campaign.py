"""Run bounded differential campaigns with crash-safe isolation by default."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

if __package__:
    from . import fuzzer
else:
    import fuzzer


def configurations(domains, output_modes, opsets, shape_modes):
    """Enumerate the supported matrix without counting unsupported cases as passes."""
    variants = [
        (domain, output, opset, dynamic)
        for domain, output, opset, dynamic in itertools.product(
            domains, output_modes, opsets, shape_modes
        )
        if domain != "dag" or not dynamic
    ]
    if not variants:
        raise ValueError("Dynamic shapes require a constrained template domain")
    return list(dict.fromkeys(variants))


def generate_campaign(*, seed, count, variants, steps=7):
    """Keep count a total-case budget while rotating through the selected matrix."""
    if count < len(variants):
        raise ValueError(
            f"--count must be at least {len(variants)} to cover the selected configurations"
        )
    order = list(variants)
    for index in range(count):
        if index % len(order) == 0:
            # Shuffle each round so configuration and seed edge profiles do not
            # remain correlated when the matrix size is divisible by eight.
            order = list(variants)
            random.Random(seed + index // len(order)).shuffle(order)
        domain, output_mode, opset, dynamic = order[index % len(order)]
        yield fuzzer.generate_case(
            seed + index,
            steps=steps,
            domain=domain,
            output_mode=output_mode,
            opset=opset,
            dynamic=dynamic,
        )


def _write_report(path, report):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--count", type=int, default=1, help="Total cases, not cases per configuration"
    )
    parser.add_argument(
        "--steps", type=int, help="DAG node budget, at least 4 (default 7)"
    )
    parser.add_argument("--case", type=Path, help="Replay exact serialized case data")
    parser.add_argument("--domain", choices=(*fuzzer.DOMAINS, "models", "all"))
    parser.add_argument("--output-mode", choices=("all", "final", "both"))
    parser.add_argument("--opset", type=int, choices=(18, 23, 26), action="append")
    parser.add_argument("--shape-mode", choices=("static", "dynamic", "both"))
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Debug without crash/timeout isolation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds per isolated case, including reduction",
    )
    parser.add_argument("--no-reduce", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Collect classified case failures through the budget, then exit nonzero",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write an atomic campaign summary, including failures",
    )
    parser.add_argument(
        "--failure-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "torch-onnx-fuzz-failures",
    )
    args = parser.parse_args(argv)
    if (
        args.count < 1
        or (args.steps is not None and args.steps < 4)
        or not 0 < args.timeout < float("inf")
    ):
        parser.error(
            "--count and finite --timeout must be positive; --steps must be at least 4"
        )
    if args.case:
        if any(
            value is not None
            for value in (
                args.domain,
                args.output_mode,
                args.opset,
                args.shape_mode,
                args.seed,
                args.steps,
            )
        ):
            parser.error(
                "--case preserves its recorded configuration; generation flags cannot override replay"
            )
        if args.count != 1:
            parser.error("--case replays one recorded case; --count must be 1")
        cases = [fuzzer.Case.from_json(args.case.read_text())]
        variants = []
    else:
        domains = (
            fuzzer.DOMAINS
            if args.domain == "all"
            else fuzzer.model_families.DOMAINS
            if args.domain == "models"
            else (args.domain or "dag",)
        )
        outputs = (
            ("all", "final")
            if args.output_mode == "both"
            else (args.output_mode or "all",)
        )
        shapes = (
            (False, True)
            if args.shape_mode == "both"
            else (args.shape_mode == "dynamic",)
        )
        try:
            variants = configurations(domains, outputs, args.opset or [26], shapes)
            if args.count < len(variants):
                raise ValueError(
                    f"--count must be at least {len(variants)} to cover the selected configurations"
                )
        except ValueError as error:
            parser.error(str(error))
        cases = generate_campaign(
            seed=args.seed or 0,
            count=args.count,
            variants=variants,
            steps=args.steps or 7,
        )
    if __package__:
        from .runner import IsolatedRunner, RunnerFailure
    else:
        from runner import IsolatedRunner, RunnerFailure
    context = (
        nullcontext(None)
        if args.in_process
        else IsolatedRunner(
            args.failure_dir, timeout_s=args.timeout, reduce=not args.no_reduce
        )
    )
    start = time.monotonic()
    report = {
        "version": 1,
        "status": "running",
        "requested_cases": args.count,
        "seed": args.seed if args.seed is not None else 0,
        "dag_steps": args.steps if args.steps is not None else 7,
        "replay_case": str(args.case.resolve()) if args.case else None,
        "attempted_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "validated_exports": 0,
        "validated_input_instances": 0,
        "isolated": not args.in_process,
        "timeout_s": None if args.in_process else args.timeout,
        "configurations": variants,
        "elapsed_s": 0.0,
        "failure": None,
        "failures": [],
        "domain_counts": {},
        "generated_ops": [],
        "aten_ops": [],
        "onnx_ops": [],
    }
    _write_report(args.report, report)
    try:
        with context as isolated:
            for case in cases:
                report["attempted_cases"] += 1
                report["domain_counts"][case.domain] = (
                    report["domain_counts"].get(case.domain, 0) + 1
                )
                report["generated_ops"] = sorted(
                    set(report["generated_ops"]) | {node["op"] for node in case.nodes}
                )
                try:
                    if isolated is None:
                        coverage = fuzzer.check_case(
                            case,
                            failure_dir=args.failure_dir,
                            reduce=not args.no_reduce,
                        )
                    else:
                        coverage = isolated.check(case).coverage
                except (fuzzer.FuzzFailure, RunnerFailure) as error:
                    report["failed_cases"] += 1
                    report["failures"].append(
                        {"seed": case.seed, "stage": error.stage, "detail": str(error)}
                    )
                    if not args.keep_going or error.stage == "worker":
                        raise
                else:
                    report["passed_cases"] += 1
                    report["validated_exports"] += 2
                    report["validated_input_instances"] += 2 * (
                        1 + len(case.additional_inputs)
                    )
                    for name in ("aten_ops", "onnx_ops"):
                        report[name] = sorted(
                            set(report[name]) | set((coverage or {}).get(name, []))
                        )
                report["elapsed_s"] = time.monotonic() - start
                _write_report(args.report, report)
        report["status"] = "failed" if report["failed_cases"] else "passed"
    except BaseException as error:
        report["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        report["failure"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - start
        _write_report(args.report, report)
    print(
        f"Passed {report['passed_cases']} cases / {report['validated_exports']} exports / "
        f"{report['validated_input_instances']} input-instance checks in {report['elapsed_s']:.2f}s"
    )
    if report["failed_cases"]:
        raise SystemExit(
            f"{report['failed_cases']} fuzz cases failed; inspect the report and replay artifacts"
        )


if __name__ == "__main__":
    main()
