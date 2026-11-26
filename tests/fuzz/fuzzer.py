"""Generate, execute, replay and shrink small deterministic PyTorch DAGs.

Run directly with ``python tests/fuzz/fuzzer.py --seed 0 --count 100``.
Cases contain their input values and operations; replay does not depend on the
current generator implementation or on global Python/PyTorch random state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx.reference import ReferenceEvaluator

import torch_onnx

_DTYPES = {
    name: getattr(torch, name)
    for name in ("float32", "float64", "int32", "int64", "bool")
}
_FLOATS = {"float32", "float64"}
_INTEGERS = {"int32", "int64"}
_OPS = {
    "add",
    "sub",
    "mul",
    "neg",
    "abs",
    "add_zero",
    "add_scalar",
    "reciprocal",
    "floor_div",
    "round",
    "cast",
    "logical_not",
    "logical_xor",
    "where",
    "transpose",
    "unsqueeze",
    "squeeze",
    "flatten",
    "cat",
    "stack",
}
_ZERO_SIGN_OPS = {
    "neg",
    "abs",
    "add_zero",
    "cast",
    "transpose",
    "unsqueeze",
    "squeeze",
    "flatten",
    "cat",
    "stack",
}


@dataclasses.dataclass
class Case:
    """A complete versioned input-and-graph description."""

    seed: int
    inputs: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self), indent=2, sort_keys=True, allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> Case:
        raw = json.loads(text)
        case = cls(**raw)
        if case.version != 1 or len(case.inputs) != 2:
            raise ValueError("Expected a version-1 case with two inputs")
        for spec in case.inputs:
            if spec["layout"] not in ("contiguous", "transposed"):
                raise ValueError("Invalid input layout")
            if spec["dtype"] not in _DTYPES or any(dim < 0 for dim in spec["shape"]):
                raise ValueError("Invalid input dtype or shape")
            if len(spec["values"]) != math.prod(spec["shape"]):
                raise ValueError("Input shape and value count disagree")
        for index, node in enumerate(case.nodes):
            if node["op"] not in _OPS or any(
                not 0 <= arg < index + 2 for arg in node["args"]
            ):
                raise ValueError("Invalid operation or non-topological argument")
        return case


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _decode(value: Any) -> Any:
    if value == "nan":
        return float("nan")
    if value == "negative_nan":
        return math.copysign(float("nan"), -1)
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return -float("inf")
    return value


def materialize(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild inputs, including their requested strided layout."""
    inputs = []
    for spec in case.inputs:
        tensor = torch.tensor(
            [_decode(v) for v in spec["values"]], dtype=_DTYPES[spec["dtype"]]
        )
        tensor = tensor.reshape(spec["shape"])
        if spec["layout"] == "transposed" and tensor.ndim == 2:
            tensor = tensor.transpose(0, 1).contiguous().transpose(0, 1)
        inputs.append(tensor)
    return inputs[0], inputs[1]


class GraphModule(torch.nn.Module):
    """An unrolled, mutation-free DAG whose topology is fixed by its case."""

    def __init__(self, case: Case):
        super().__init__()
        self.case = case

    def forward(self, x, y):
        values = [x, y]
        for node in self.case.nodes:
            op, attrs = node["op"], node.get("attrs", {})
            args = [values[i] for i in node["args"]]
            a = args[0]
            if op == "add":
                result = a + args[1]
            elif op == "sub":
                result = a - args[1]
            elif op == "mul":
                result = a * args[1]
            elif op == "neg":
                result = -a
            elif op == "abs":
                result = a.abs()
            elif op == "add_zero":
                result = a + 0.0
            elif op == "add_scalar":
                result = a + attrs["value"]
            elif op == "reciprocal":
                result = a.reciprocal()
            elif op == "floor_div":
                result = torch.div(a, attrs["divisor"], rounding_mode="floor")
            elif op == "round":
                result = a.round()
            elif op == "cast":
                result = a.to(_DTYPES[attrs["dtype"]])
            elif op == "logical_not":
                result = torch.logical_not(a)
            elif op == "logical_xor":
                result = torch.logical_xor(a, args[1])
            elif op == "where":
                condition = a if self.case_dtype(node["args"][0]) == "bool" else a >= 0
                result = torch.where(condition, a, args[1])
            elif op == "transpose":
                result = a.transpose(attrs["dim0"], attrs["dim1"])
            elif op == "unsqueeze":
                result = a.unsqueeze(attrs["dim"])
            elif op == "squeeze":
                result = a.squeeze(attrs["dim"])
            elif op == "flatten":
                result = a.flatten()
            elif op == "cat":
                result = torch.cat(args, dim=attrs["dim"])
            elif op == "stack":
                result = torch.stack(args, dim=attrs["dim"])
            else:
                raise ValueError(f"Unknown op {op!r}")
            values.append(result)
        outputs = values[2:]
        # Explicit observable probes: compare signbit as Boolean output, including
        # signed NaNs on original inputs, without demanding NaN payload identity.
        if self.case.inputs[0]["dtype"] in _FLOATS:
            outputs.extend([torch.signbit(x), torch.signbit(y)])
        return tuple(outputs)

    def case_dtype(self, index):
        # This is static Python metadata, not a tensor-dependent capture branch.
        if index < 2:
            return self.case.inputs[index]["dtype"]
        return self.case.nodes[index - 2]["dtype"]


def generate_case(seed: int, *, steps: int = 7) -> Case:
    """Generate supported small multi-op graphs, with rotating edge profiles."""
    rng = random.Random(seed)
    family = seed % 8
    dtype = (
        "float32",
        "float64",
        "int64",
        "int32",
        "bool",
        "float32",
        "int64",
        "float64",
    )[family]
    presets = ([2, 3], [1, 4], [2, 3], [0, 3], [2, 1], [1, 1], [3, 1], [3, 0])
    shape = list(
        presets[family]
        if seed < 8
        else rng.choice(([], [0], [1], [3], [2, 3], [3, 2], [0, 3], [3, 0], [1, 2, 3]))
    )
    profiles = {
        "float32": [-0.0, 0.0, 0.25, -2.0, 3.5, -1e-20],
        "float64": [-0.0, 0.0, 9007199254740994.0, 1e-12, 0.1, -math.pi],
        "int64": [2**53 + 3, -(2**53 + 3), 2**31 + 1, -(2**31 + 1), 0, 1],
        "int32": [2**24 + 3, -(2**24 + 3), 17, -17, 0, 1],
        "bool": [True, False],
    }
    x_values = (
        ["-inf", "+inf", "nan", "negative_nan", -0.0, 0.0]
        if family == 0
        else profiles[dtype]
    )
    y_values = (
        profiles[dtype] if dtype not in _FLOATS else [1.0, -1.0, 2.0, -2.0, 0.0, -0.0]
    )
    inputs = [
        {
            "dtype": dtype,
            "shape": shape,
            "layout": "transposed" if index == 0 else "contiguous",
            "values": [data[i % len(data)] for i in range(math.prod(shape))],
        }
        for index, data in enumerate((x_values, y_values))
    ]
    nodes: list[dict[str, Any]] = []
    specs = [(dtype, tuple(shape)), (dtype, tuple(shape))]

    def append(op, args, attrs=None):
        attrs = attrs or {}
        dt, dims = specs[args[0]]
        out = list(dims)
        if op in ("add", "sub", "mul", "where", "cat", "stack"):
            dt = _dtype_name(
                torch.promote_types(_DTYPES[dt], _DTYPES[specs[args[1]][0]])
            )
        if op == "transpose":
            i, j = attrs["dim0"], attrs["dim1"]
            out[i], out[j] = out[j], out[i]
        elif op == "unsqueeze":
            out.insert(attrs["dim"], 1)
        elif op == "squeeze" and out[attrs["dim"]] == 1:
            out.pop(attrs["dim"])
        elif op == "flatten":
            out = [math.prod(dims)]
        elif op == "cat":
            out[attrs["dim"]] *= 2
        elif op == "stack":
            out.insert(attrs["dim"], 2)
        elif op == "cast":
            dt = attrs["dtype"]
        elif op in ("logical_not", "logical_xor"):
            dt = "bool"
        nodes.append({"op": op, "args": args, "attrs": attrs, "dtype": dt})
        specs.append((dt, tuple(out)))

    if dtype in _FLOATS:
        append("add_zero", [0])
        append("reciprocal", [2])
        append("neg", [0])
        if dtype == "float64":
            append("add_scalar", [1], {"value": 0.1})
    elif dtype in _INTEGERS:
        append("floor_div", [0], {"divisor": 3})
        append("round", [1])
        append("stack", [0, 1], {"dim": 0})
    else:
        append("logical_not", [0])
        append("logical_xor", [0, 1])
        append("unsqueeze", [2], {"dim": 0})

    while len(nodes) < steps:
        arg = rng.randrange(len(specs))
        dt, dims = specs[arg]
        same_shape = [i for i, (_, s) in enumerate(specs) if s == dims]
        numeric_peers = [i for i in same_shape if specs[i][0] != "bool"]
        ops = ["unsqueeze", "flatten", "cast"]
        if dims:
            ops.extend(["squeeze", "cat"])
        if len(dims) > 1:
            ops.append("transpose")
        if len(dims) < 3 and math.prod(dims) <= 64:
            ops.append("stack")
        if dt == "bool":
            ops.extend(["logical_not", "logical_xor"])
        else:
            ops.extend(["neg", "abs", "round", "add", "sub", "where"])
            ops.append("floor_div" if dt in _INTEGERS else "mul")
        op = rng.choice(ops)
        attrs: dict[str, Any] = {}
        args = [arg]
        if op in ("add", "sub", "mul", "where"):
            args.append(rng.choice(numeric_peers))
        elif op in ("logical_xor", "cat", "stack"):
            peers = (
                [i for i in same_shape if specs[i][0] == "bool"]
                if op == "logical_xor"
                else same_shape
            )
            args.append(rng.choice(peers))
        if op in ("squeeze", "cat"):
            attrs["dim"] = rng.randrange(len(dims))
        elif op in ("unsqueeze", "stack"):
            attrs["dim"] = rng.randrange(len(dims) + 1)
        elif op == "transpose":
            attrs = {"dim0": 0, "dim1": len(dims) - 1}
        elif op == "floor_div":
            attrs["divisor"] = rng.choice((2, 3, 7))
        elif op == "cast":
            targets = {
                "bool": ("int64",),
                "int32": ("int64", "float32", "float64"),
                "int64": ("float32", "float64"),
                "float32": ("float64",),
                "float64": ("float32",),
            }
            attrs["dtype"] = rng.choice(targets[dt])
        append(op, args, attrs)
    return Case(seed, inputs, nodes)


class FuzzFailure(AssertionError):
    """A failure categorized by the pipeline stage, with a complete replay case."""

    def __init__(
        self,
        case: Case,
        stage: str,
        optimize: bool,
        detail: str,
        *,
        attribution: str | None = None,
    ):
        self.case, self.stage, self.optimize, self.detail = (
            case,
            stage,
            optimize,
            detail,
        )
        self.attribution = attribution
        category = f" attribution={attribution}" if attribution else ""
        super().__init__(
            f"seed={case.seed} optimize={optimize} stage={stage}{category}: {detail}\nCASE_JSON:\n{case.to_json()}"
        )


def compare_array(
    expected: np.ndarray, actual: np.ndarray, *, zero_sign: bool = False
) -> None:
    """Compare masks and observable special values before finite tolerances."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"shape/dtype: expected {expected.shape}/{expected.dtype}, got {actual.shape}/{actual.dtype}"
        )
    if expected.dtype.kind != "f":
        np.testing.assert_array_equal(actual, expected)
        return
    np.testing.assert_array_equal(
        np.isnan(actual), np.isnan(expected), err_msg="NaN mask"
    )
    np.testing.assert_array_equal(
        np.isposinf(actual), np.isposinf(expected), err_msg="+Inf mask"
    )
    np.testing.assert_array_equal(
        np.isneginf(actual), np.isneginf(expected), err_msg="-Inf mask"
    )
    finite = np.isfinite(expected)
    tolerance = (1e-10, 1e-12) if expected.dtype == np.float64 else (1e-5, 1e-6)
    np.testing.assert_allclose(
        actual[finite], expected[finite], rtol=tolerance[0], atol=tolerance[1]
    )
    if zero_sign:
        zeros = expected == 0
        np.testing.assert_array_equal(
            actual[zeros] == 0,
            np.ones(np.count_nonzero(zeros), dtype=bool),
            err_msg="exact zeros",
        )
        np.testing.assert_array_equal(
            np.signbit(actual[zeros]),
            np.signbit(expected[zeros]),
            err_msg="signed zero",
        )


def _reference_attribution(proto, arrays, expected, zero_sign):
    """Use an independent ONNX evaluator only after an ORT semantic mismatch."""
    try:
        outputs = ReferenceEvaluator(proto).run(
            None, {value.name: arrays[value.name] for value in proto.graph.input}
        )
    except Exception:
        return "unattributed"
    try:
        if len(outputs) != len(expected):
            return "export_semantic"
        for index, (reference, output) in enumerate(
            zip(expected, outputs, strict=True)
        ):
            compare_array(reference, output, zero_sign=zero_sign[index])
    except AssertionError:
        return "export_semantic"
    except Exception:
        return "unattributed"
    return "backend"


def run_case(case: Case, *, optimize_modes=(False, True)) -> None:
    """Check eager against independently checked/executed ONNX in both modes."""
    try:
        model, args = GraphModule(case).eval(), materialize(case)
        with torch.no_grad():
            expected = tuple(value.numpy() for value in model(*args))
    except Exception as error:
        raise FuzzFailure(case, "generator", False, str(error)) from error
    # Track explicit bit-preserving probes, not every numerically equal zero
    # produced by ordinary arithmetic or a backend selection kernel.
    sensitivity = [spec["dtype"] in _FLOATS for spec in case.inputs]
    for node in case.nodes:
        sensitive = node["op"] == "add_zero" or (
            node["op"] in _ZERO_SIGN_OPS
            and all(sensitivity[index] for index in node["args"])
        )
        sensitivity.append(sensitive)
    zero_sign = sensitivity[2:]
    zero_sign.extend([False] * (len(expected) - len(zero_sign)))
    for optimize in optimize_modes:
        try:
            program = torch_onnx.export(
                model,
                args,
                input_names=["x", "y"],
                opset_version=26,
                optimize=optimize,
                verbose=False,
            )
        except Exception as error:
            raise FuzzFailure(case, "export", optimize, str(error)) from error
        try:
            try:
                proto = program.model_proto
                onnx.checker.check_model(proto, full_check=True)
                if len(proto.graph.output) != len(expected):
                    raise AssertionError("Output count differs from eager")
                for info, output in zip(proto.graph.output, expected, strict=True):
                    declared = info.type.tensor_type
                    if declared.elem_type != onnx.helper.np_dtype_to_tensor_dtype(
                        output.dtype
                    ):
                        raise AssertionError(
                            f"Output {info.name} declares the wrong dtype"
                        )
                    if declared.HasField("shape"):
                        if len(declared.shape.dim) != output.ndim:
                            raise AssertionError(
                                f"Output {info.name} declares the wrong rank"
                            )
                        for dim, size in zip(
                            declared.shape.dim, output.shape, strict=True
                        ):
                            if dim.HasField("dim_value") and dim.dim_value != size:
                                raise AssertionError(
                                    f"Output {info.name} declares the wrong dimension"
                                )
            except Exception as error:
                raise FuzzFailure(
                    case, "invalid_graph", optimize, str(error)
                ) from error
            try:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    proto.SerializeToString(),
                    options,
                    providers=["CPUExecutionProvider"],
                )
                arrays = {"x": args[0].numpy(), "y": args[1].numpy()}
                actual = session.run(
                    None,
                    {value.name: arrays[value.name] for value in session.get_inputs()},
                )
            except Exception as error:
                raise FuzzFailure(case, "runtime", optimize, str(error)) from error
            try:
                if len(actual) != len(expected):
                    raise AssertionError("Runtime output count differs from eager")
                for index, (reference, result) in enumerate(
                    zip(expected, actual, strict=True)
                ):
                    try:
                        compare_array(reference, result, zero_sign=zero_sign[index])
                    except AssertionError as error:
                        raise AssertionError(f"output {index}: {error}") from error
            except Exception as error:
                attribution = _reference_attribution(proto, arrays, expected, zero_sign)
                raise FuzzFailure(
                    case, "semantic", optimize, str(error), attribution=attribution
                ) from error
        finally:
            program.release()


def reduce_failure(case: Case, failure: FuzzFailure, *, attempts: int = 8) -> Case:
    """Greedily remove DAG suffixes while preserving failure stage and mode."""
    reduced = case
    for _ in range(attempts):
        if not reduced.nodes:
            break
        candidate = dataclasses.replace(reduced, nodes=reduced.nodes[:-1])
        try:
            run_case(candidate, optimize_modes=(failure.optimize,))
        except FuzzFailure as error:
            if error.stage != failure.stage or error.attribution != failure.attribution:
                break
            reduced = candidate
        else:
            break
    return reduced


def check_case(
    case: Case, *, failure_dir: Path | None = None, reduce: bool = True
) -> None:
    """Run a case and report/save a reduced, directly replayable failure."""
    try:
        run_case(case)
    except FuzzFailure as error:
        reduced = reduce_failure(case, error) if reduce else case
        if failure_dir is not None:
            failure_dir.mkdir(parents=True, exist_ok=True)
            path = (
                failure_dir
                / f"seed-{case.seed}-{error.stage}-{int(error.optimize)}.json"
            )
            path.write_text(reduced.to_json() + "\n")
            detail = f"{error.detail}\nReplay: python tests/fuzz/fuzzer.py --case {shlex.quote(str(path))}"
        else:
            detail = error.detail
        raise FuzzFailure(
            reduced, error.stage, error.optimize, detail, attribution=error.attribution
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--case", type=Path)
    parser.add_argument(
        "--failure-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "torch-onnx-fuzz-failures",
    )
    parser.add_argument("--no-reduce", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    cases = (
        [Case.from_json(args.case.read_text())]
        if args.case
        else (generate_case(seed) for seed in range(args.seed, args.seed + args.count))
    )
    start = time.monotonic()
    count = 0
    for case in cases:
        check_case(case, failure_dir=args.failure_dir, reduce=not args.no_reduce)
        count += 1
    print(
        f"Passed {count} cases / {2 * count} exports in {time.monotonic() - start:.2f}s"
    )


if __name__ == "__main__":
    main()
