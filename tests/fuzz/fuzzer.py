"""Generate, execute, replay and shrink small deterministic PyTorch DAGs.

Run directly with ``python tests/fuzz/fuzzer.py --seed 0 --count 100``.
Cases contain their input values and operations; replay does not depend on the
current generator implementation or on global Python/PyTorch random state.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx.reference import ReferenceEvaluator

import torch_onnx

if __package__:
    from . import model_families
else:
    import model_families

TEMPLATE_DOMAINS = ("ce", "bce", "sdpa", *model_families.DOMAINS)
DOMAINS = ("dag", *TEMPLATE_DOMAINS)

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
    "cross_entropy",
    "binary_cross_entropy_with_logits",
    "scaled_dot_product_attention",
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
_OPS.update(model_families.OPS)


@dataclasses.dataclass
class Case:
    """A complete versioned input-and-graph description."""

    seed: int
    inputs: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    version: int = 2
    output_mode: str = "all"
    opset: int = 26
    domain: str = "dag"
    dynamic_shapes: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    additional_inputs: list[list[dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )
    input_probes: bool = True
    dynamic_bounds: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    float32_tolerance: list[float] | None = None

    def to_json(self) -> str:
        payload = dataclasses.asdict(self)
        if self.version < 3:
            # Preserve the v1/v2 wire format for readers from earlier releases.
            # Non-default v3 settings remain visible so validation rejects them.
            if not self.dynamic_bounds:
                payload.pop("dynamic_bounds")
            if self.float32_tolerance is None:
                payload.pop("float32_tolerance")
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Case:
        raw = json.loads(text)
        case = cls(**raw)
        if case.version not in (1, 2, 3) or not case.inputs:
            raise ValueError(
                "Expected a version-1, version-2 or version-3 case with inputs"
            )
        if case.version == 1 and len(case.inputs) != 2:
            raise ValueError("Expected a version-1 case with two inputs")
        if case.float32_tolerance is not None and (
            case.version < 3
            or not isinstance(case.float32_tolerance, list)
            or len(case.float32_tolerance) != 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in case.float32_tolerance
            )
        ):
            raise ValueError("Invalid explicit float32 tolerance")
        if case.output_mode not in ("all", "final") or case.opset not in (18, 23, 26):
            raise ValueError("Invalid output mode or opset")
        if case.domain not in DOMAINS or not isinstance(case.input_probes, bool):
            raise ValueError("Invalid case domain or probe mode")
        if (
            case.input_probes
            and case.opset < 26
            and any(spec["dtype"] in _FLOATS for spec in case.inputs)
        ):
            raise ValueError("Floating signbit input probes require opset 26")
        for specs in [case.inputs, *case.additional_inputs]:
            if len(specs) != len(case.inputs):
                raise ValueError("Input instance arity differs")
            for original, spec in zip(case.inputs, specs, strict=True):
                if original["dtype"] != spec["dtype"] or len(original["shape"]) != len(
                    spec["shape"]
                ):
                    raise ValueError("Input instance dtype/rank differs")
                cls._validate_input(spec)
        names = (
            ["x", "y"]
            if len(case.inputs) == 2
            else [f"input_{i}" for i in range(len(case.inputs))]
        )
        for name, axes in case.dynamic_shapes.items():
            if name not in names or any(
                not 0 <= int(axis) < len(case.inputs[names.index(name)]["shape"])
                or not isinstance(symbol, str)
                or not symbol.isidentifier()
                for axis, symbol in axes.items()
            ):
                raise ValueError("Invalid dynamic shape declaration")
        symbols = {
            symbol for axes in case.dynamic_shapes.values() for symbol in axes.values()
        }
        if case.version < 3 and case.dynamic_bounds:
            raise ValueError("Explicit dynamic bounds require case version 3")
        if case.version == 3 and (
            set(case.dynamic_bounds) != symbols
            or any(
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not 0 <= bounds[0] < bounds[1] <= 64
                for bounds in case.dynamic_bounds.values()
            )
        ):
            raise ValueError("Invalid explicit dynamic bounds")
        for specs in [case.inputs, *case.additional_inputs]:
            sizes = {}
            for name, original, spec in zip(names, case.inputs, specs, strict=True):
                axes = case.dynamic_shapes.get(name, {})
                for axis, size in enumerate(spec["shape"]):
                    symbol = axes.get(str(axis))
                    if symbol is None:
                        if size != original["shape"][axis]:
                            raise ValueError("Undeclared input dimension changed")
                    else:
                        lower, upper = case.dynamic_bounds.get(symbol, (2, 64))
                        if (
                            not lower <= size <= upper
                            or sizes.setdefault(symbol, size) != size
                        ):
                            raise ValueError(
                                "Dynamic dimension violates bounds or shared symbol"
                            )
            _validate_template_inputs(case.domain, specs)
        for index, node in enumerate(case.nodes):
            if node["op"] not in _OPS or any(
                not 0 <= arg < index + len(case.inputs) for arg in node["args"]
            ):
                raise ValueError("Invalid operation or non-topological argument")
            if (
                node["op"] in model_families.ARITY
                and len(node["args"]) != model_families.ARITY[node["op"]]
            ):
                raise ValueError("Invalid model recipe operation arity")
            if node["op"] == "tensor_const":
                cls._validate_input({"layout": "contiguous", **node["attrs"]})
        return case

    @staticmethod
    def _validate_input(spec):
        if spec["layout"] not in ("contiguous", "transposed"):
            raise ValueError("Invalid input layout")
        if spec["dtype"] not in _DTYPES or any(
            type(dim) is not int or dim < 0 for dim in spec["shape"]
        ):
            raise ValueError("Invalid input dtype or shape")
        if len(spec["values"]) != math.prod(spec["shape"]):
            raise ValueError("Input shape and value count disagree")


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


def materialize(case: Case, *, inputs=None) -> tuple[torch.Tensor, ...]:
    """Rebuild inputs, including their requested strided layout."""
    specs = case.inputs if inputs is None else inputs
    tensors = []
    for spec in specs:
        tensor = torch.tensor(
            [_decode(v) for v in spec["values"]], dtype=_DTYPES[spec["dtype"]]
        )
        tensor = tensor.reshape(spec["shape"])
        if spec["layout"] == "transposed" and tensor.ndim >= 2:
            tensor = tensor.transpose(-2, -1).contiguous().transpose(-2, -1)
        tensors.append(tensor)
    return tuple(tensors)


class GraphModule(torch.nn.Module):
    """An unrolled, mutation-free DAG whose topology is fixed by its case."""

    def __init__(self, case: Case):
        super().__init__()
        self.case = case

    def forward(self, *inputs):
        values = list(inputs)
        for node in self.case.nodes:
            op, attrs = node["op"], node.get("attrs", {})
            args = [values[i] for i in node["args"]]
            a = args[0] if args else None
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
            elif op == "cross_entropy":
                result = torch.nn.functional.cross_entropy(a, args[1], **attrs)
            elif op == "binary_cross_entropy_with_logits":
                result = torch.nn.functional.binary_cross_entropy_with_logits(
                    a, args[1], **attrs
                )
            elif op == "scaled_dot_product_attention":
                result = torch.nn.functional.scaled_dot_product_attention(
                    *args, **attrs
                )
            elif op in model_families.OPS:
                result = model_families.apply(op, args, attrs)
            else:
                raise ValueError(f"Unknown op {op!r}")
            values.append(result)
        outputs = (
            values[len(inputs) :] if self.case.output_mode == "all" else values[-1:]
        )
        # Explicit observable probes: compare signbit as Boolean output, including
        # signed NaNs on original inputs, without demanding NaN payload identity.
        if self.case.input_probes:
            if self.case.version == 1:
                # V1 used the first input's dtype to enable probes for both
                # inputs, including integer peers in hand-authored artifacts.
                probe_inputs = inputs if self.case.inputs[0]["dtype"] in _FLOATS else ()
            else:
                probe_inputs = [
                    value
                    for value, spec in zip(inputs, self.case.inputs, strict=True)
                    if spec["dtype"] in _FLOATS
                ]
            outputs.extend(torch.signbit(value) for value in probe_inputs)
        return tuple(outputs)

    def case_dtype(self, index):
        # This is static Python metadata, not a tensor-dependent capture branch.
        count = len(self.case.inputs)
        if index < count:
            return self.case.inputs[index]["dtype"]
        return self.case.nodes[index - count]["dtype"]


def generate_case(
    seed: int,
    *,
    steps: int = 7,
    domain: str = "dag",
    output_mode: str = "all",
    opset: int = 26,
    dynamic: bool = False,
) -> Case:
    """Generate supported small multi-op graphs, with rotating edge profiles."""
    if domain != "dag":
        return generate_template_case(
            seed, domain=domain, output_mode=output_mode, opset=opset, dynamic=dynamic
        )
    if dynamic:
        raise ValueError(
            "Dynamic DAG generation requires a constrained template domain"
        )
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
    return Case(
        seed,
        inputs,
        nodes,
        output_mode=output_mode,
        opset=opset,
        input_probes=output_mode == "all" and opset >= 26,
    )


def input_names(case: Case) -> list[str]:
    """Stable serialized input ordering used by capture and replay."""
    return (
        ["x", "y"]
        if len(case.inputs) == 2
        else [f"input_{i}" for i in range(len(case.inputs))]
    )


def _validate_template_inputs(domain, specs):
    if domain == "dag":
        return
    if domain in model_families.DOMAINS:
        model_families.validate_inputs(domain, specs)
        return
    if domain in ("ce", "bce"):
        if (
            len(specs) != 2
            or len(specs[0]["shape"]) != 2
            or specs[0]["dtype"] != "float32"
        ):
            raise ValueError("Invalid loss template inputs")
        batch, classes = specs[0]["shape"]
        if not batch or not classes:
            raise ValueError("Loss template dimensions must be positive")
        if domain == "ce":
            if (
                specs[1]["shape"] != [batch]
                or specs[1]["dtype"] != "int64"
                or any(
                    type(value) is not int or not 0 <= value < classes
                    for value in specs[1]["values"]
                )
            ):
                raise ValueError("Invalid class-index targets")
        elif (
            specs[1]["shape"] != [batch, classes]
            or specs[1]["dtype"] != "float32"
            or any(
                not isinstance(value, int | float) or not 0 <= value <= 1
                for value in specs[1]["values"]
            )
        ):
            raise ValueError("Invalid probability targets")
        return
    if len(specs) != 4 or any(len(spec["shape"]) != 4 for spec in specs):
        raise ValueError("Invalid attention template input ranks")
    q, k, v, mask = specs
    batch, heads, length, depth = q["shape"]
    source = k["shape"][2]
    if (
        min(batch, heads, length, depth, source) < 1
        or k["shape"] != [batch, heads, source, depth]
        or v["shape"] != k["shape"]
        or mask["shape"] != [1, 1, length, source]
        or any(spec["dtype"] != "float32" for spec in (q, k, v))
        or mask["dtype"] != "bool"
    ):
        raise ValueError("Invalid attention template dimensions or dtypes")
    if any(
        not any(mask["values"][row : row + source])
        for row in range(0, len(mask["values"]), source)
    ):
        raise ValueError("Attention template requires a visible key per query")


def generate_template_case(
    seed: int,
    *,
    domain: str,
    opset: int = 26,
    dynamic: bool = False,
    output_mode: str = "final",
    reduction: str | None = None,
) -> Case:
    """Constrained loss/attention families with reproducible legal input instances."""
    if domain in model_families.DOMAINS:
        if reduction is not None:
            raise ValueError("Model recipes do not accept loss reductions")
        return model_families.generate(
            seed, domain=domain, opset=opset, dynamic=dynamic, output_mode=output_mode
        )
    if domain not in ("ce", "bce", "sdpa"):
        raise ValueError(f"Unknown template domain {domain!r}")
    rng = random.Random(seed)
    if reduction is not None and (
        domain == "sdpa" or reduction not in ("none", "sum", "mean")
    ):
        raise ValueError("Invalid reduction for template domain")
    reduction = (
        reduction if reduction is not None else rng.choice(("none", "sum", "mean"))
    )
    classes = rng.choice((3, 5, 7))
    heads = rng.choice((1, 2))
    depth = rng.choice((4, 8))

    def specs(batch, length, source):
        if domain == "ce":
            shapes, dtypes = [(batch, classes), (batch,)], ["float32", "int64"]
        elif domain == "bce":
            shapes, dtypes = [(batch, classes)] * 2, ["float32"] * 2
        else:
            shapes = [
                (batch, heads, length, depth),
                (batch, heads, source, depth),
                (batch, heads, source, depth),
                (1, 1, length, source),
            ]
            dtypes = ["float32"] * 3 + ["bool"]
        result = []
        for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
            count = math.prod(shape)
            if domain == "ce" and index == 1:
                values = [rng.randrange(classes) for _ in range(count)]
            elif domain == "bce" and index == 1:
                values = [rng.random() for _ in range(count)]
            elif dtype == "bool":
                # Every query keeps key zero visible, so no all-masked rows.
                values = [
                    i % source == 0 or rng.choice((True, False)) for i in range(count)
                ]
            else:
                values = [rng.uniform(-3.0, 3.0) for _ in range(count)]
            result.append(
                dict(dtype=dtype, shape=list(shape), layout="contiguous", values=values)
            )
        return result

    inputs = specs(2, 3, 4)
    op = {
        "ce": "cross_entropy",
        "bce": "binary_cross_entropy_with_logits",
        "sdpa": "scaled_dot_product_attention",
    }[domain]
    attrs = (
        {"dropout_p": 0.0, "is_causal": False}
        if domain == "sdpa"
        else {"reduction": reduction}
    )
    nodes = [dict(op=op, args=list(range(len(inputs))), attrs=attrs, dtype="float32")]
    case = Case(
        seed,
        inputs,
        nodes,
        output_mode=output_mode,
        opset=opset,
        domain=domain,
        input_probes=output_mode == "all" and opset >= 26,
    )
    if dynamic:
        names = input_names(case)
        if domain == "sdpa":
            case.dynamic_shapes = {
                names[0]: {"0": "batch", "2": "query"},
                names[1]: {"0": "batch", "2": "source"},
                names[2]: {"0": "batch", "2": "source"},
                names[3]: {"2": "query", "3": "source"},
            }
        else:
            case.dynamic_shapes = {name: {"0": "batch"} for name in names}
        case.additional_inputs = [specs(3, 5, 6), specs(4, 2, 3)]
    # The same validation applies to generated cases and externally supplied JSON.
    return Case.from_json(case.to_json())


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
    expected: np.ndarray, actual: np.ndarray, *, zero_sign: bool = False, tolerance=None
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
    if tolerance is None:
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


def _reference_attribution(proto, arrays, expected, zero_sign, float32_tolerance=None):
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
            compare_array(
                reference,
                output,
                zero_sign=zero_sign[index],
                tolerance=float32_tolerance if reference.dtype == np.float32 else None,
            )
    except AssertionError:
        return "export_semantic"
    except Exception:
        return "unattributed"
    return "backend"


def _onnx_ops(proto) -> set[str]:
    """Inventory serialized nodes, including reachable functions and subgraphs."""
    functions = {
        (function.domain, function.name, function.overload): function
        for function in proto.functions
    }
    result = set()
    visited = set()

    def visit(nodes):
        for node in nodes:
            key = node.domain, node.op_type, node.overload
            result.add(
                f"{node.domain}::{node.op_type}" if node.domain else node.op_type
            )
            if key in functions and key not in visited:
                visited.add(key)
                visit(functions[key].node)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        visit(graph.node)

    visit(proto.graph.node)
    return result


def run_case(case: Case, *, optimize_modes=(False, True)) -> dict[str, list[str]]:
    """Export once per mode and execute every serialized legal shape instance."""
    try:
        model = GraphModule(case).eval()
        instances = [
            materialize(case, inputs=specs)
            for specs in [case.inputs, *case.additional_inputs]
        ]
        with torch.no_grad():
            expected_instances = [
                tuple(value.numpy() for value in model(*args)) for args in instances
            ]
        symbols = {
            symbol: torch.export.Dim(
                symbol,
                min=case.dynamic_bounds.get(symbol, (1, 64))[0],
                max=case.dynamic_bounds.get(symbol, (1, 64))[1],
            )
            for axes in case.dynamic_shapes.values()
            for symbol in axes.values()
        }
        names = input_names(case)
        dynamic = tuple(
            {
                int(axis): symbols[symbol]
                for axis, symbol in case.dynamic_shapes.get(name, {}).items()
            }
            for name in names
        )
    except Exception as error:
        raise FuzzFailure(case, "generator", False, str(error)) from error
    count = len(case.inputs)
    sensitivity = [spec["dtype"] in _FLOATS for spec in case.inputs]
    for node in case.nodes:
        sensitive = node["op"] == "add_zero" or (
            node["op"] in _ZERO_SIGN_OPS
            and all(sensitivity[index] for index in node["args"])
        )
        sensitivity.append(sensitive)
    zero_sign = sensitivity[count:] if case.output_mode == "all" else sensitivity[-1:]
    zero_sign = [*zero_sign, *([False] * (len(expected_instances[0]) - len(zero_sign)))]
    aten_ops, onnx_ops = set(), set()
    for optimize in optimize_modes:
        try:
            program = torch_onnx.export(
                model,
                instances[0],
                input_names=names,
                dynamic_shapes=(dynamic,) if case.dynamic_shapes else None,
                opset_version=case.opset,
                optimize=optimize,
                verbose=False,
            )
        except Exception as error:
            raise FuzzFailure(case, "export", optimize, str(error)) from error
        try:
            try:
                proto = program.model_proto
                onnx.checker.check_model(proto, full_check=True)
                onnx_ops.update(_onnx_ops(proto))
                aten_ops.update(
                    str(node.target)
                    for node in program.exported_program.graph_module.graph.nodes
                    if node.op == "call_function"
                    and isinstance(node.target, torch._ops.OpOverload)
                )
                for expected in expected_instances:
                    _validate_outputs(proto, expected)
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
            except Exception as error:
                raise FuzzFailure(case, "runtime", optimize, str(error)) from error
            for instance, (args, expected) in enumerate(
                zip(instances, expected_instances, strict=True)
            ):
                arrays = {
                    name: value.numpy() for name, value in zip(names, args, strict=True)
                }
                try:
                    actual = session.run(
                        None,
                        {
                            value.name: arrays[value.name]
                            for value in session.get_inputs()
                        },
                    )
                except Exception as error:
                    raise FuzzFailure(
                        case, "runtime", optimize, f"instance {instance}: {error}"
                    ) from error
                try:
                    if len(actual) != len(expected):
                        raise AssertionError("Runtime output count differs from eager")
                    for index, (reference, result) in enumerate(
                        zip(expected, actual, strict=True)
                    ):
                        try:
                            compare_array(
                                reference,
                                result,
                                zero_sign=zero_sign[index],
                                tolerance=case.float32_tolerance
                                if reference.dtype == np.float32
                                else None,
                            )
                        except AssertionError as error:
                            raise AssertionError(
                                f"instance {instance} output {index}: {error}"
                            ) from error
                except Exception as error:
                    attribution = _reference_attribution(
                        proto, arrays, expected, zero_sign, case.float32_tolerance
                    )
                    raise FuzzFailure(
                        case, "semantic", optimize, str(error), attribution=attribution
                    ) from error
        finally:
            program.release()
    return {"aten_ops": sorted(aten_ops), "onnx_ops": sorted(onnx_ops)}


def _validate_outputs(proto, expected):
    if len(proto.graph.output) != len(expected):
        raise AssertionError("Output count differs from eager")
    for info, output in zip(proto.graph.output, expected, strict=True):
        declared = info.type.tensor_type
        if declared.elem_type != onnx.helper.np_dtype_to_tensor_dtype(output.dtype):
            raise AssertionError(f"Output {info.name} declares the wrong dtype")
        if declared.HasField("shape"):
            if len(declared.shape.dim) != output.ndim:
                raise AssertionError(f"Output {info.name} declares the wrong rank")
            for dim, size in zip(declared.shape.dim, output.shape, strict=True):
                if dim.HasField("dim_value") and dim.dim_value != size:
                    raise AssertionError(
                        f"Output {info.name} declares the wrong dimension"
                    )


def reduce_failure(case: Case, failure: FuzzFailure, *, attempts: int = 8) -> Case:
    """Greedily remove DAG suffixes while preserving failure stage and mode."""
    reduced = case
    for _ in range(attempts):
        if not reduced.nodes:
            break
        candidate = dataclasses.replace(reduced, nodes=reduced.nodes[:-1])
        try:
            candidate = Case.from_json(candidate.to_json())
        except (ValueError, TypeError):
            # Domain templates can require a loss/attention node. Never emit a
            # reduced artifact that the replay loader would reject.
            break
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
) -> dict[str, list[str]]:
    """Run a case and report/save a reduced, directly replayable failure."""
    try:
        return run_case(case)
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
    # Preserve the original entry point while sharing the isolated campaign CLI.
    if __package__:
        from .campaign import main as campaign_main
    else:
        from campaign import main as campaign_main
    campaign_main()


if __name__ == "__main__":
    main()
