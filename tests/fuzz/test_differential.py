"""Small default smoke tier and explicit longer seeded differential runs."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pytest

from fuzz import fuzzer
from fuzz.runner import IsolatedRunner


def _smoke_cases():
    replay = os.environ.get("TORCH_ONNX_FUZZ_CASE")
    if replay:
        return [fuzzer.Case.from_json(Path(replay).read_text())]
    seed = os.environ.get("TORCH_ONNX_FUZZ_SEED")
    return (
        [fuzzer.generate_case(int(seed))]
        if seed is not None
        else [fuzzer.generate_case(i) for i in range(8)]
        + [
            fuzzer.generate_template_case(
                100 + i,
                domain=domain,
                opset=opset,
                dynamic=True,
                reduction="mean" if domain == "ce" else None,
            )
            for i, (domain, opset) in enumerate((("ce", 18), ("bce", 23), ("sdpa", 26)))
        ]
        + [
            fuzzer.generate_template_case(
                200 + i, domain=domain, opset=23, dynamic=True
            )
            for i, domain in enumerate(fuzzer.model_families.DOMAINS)
        ]
    )


@pytest.mark.fuzz
@pytest.mark.parametrize(
    "case", _smoke_cases(), ids=lambda case: f"{case.domain}-seed-{case.seed}"
)
def test_exporter_differential_smoke(case, tmp_path):
    fuzzer.check_case(case, failure_dir=tmp_path)


@pytest.mark.fuzz_long
def test_exporter_differential_long(tmp_path):
    start = int(os.environ.get("TORCH_ONNX_FUZZ_SEED", "0"))
    count = int(os.environ.get("TORCH_ONNX_FUZZ_COUNT", "256"))
    if count < 1:
        raise ValueError("TORCH_ONNX_FUZZ_COUNT must be positive")
    timeout = float(os.environ.get("TORCH_ONNX_FUZZ_TIMEOUT", "120"))
    with IsolatedRunner(tmp_path, timeout_s=timeout) as isolated:
        for seed in range(start, start + count):
            domain = fuzzer.DOMAINS[seed % len(fuzzer.DOMAINS)]
            case = fuzzer.generate_case(
                seed,
                domain=domain,
                output_mode="final",
                opset=(18, 23, 26)[seed % 3],
                dynamic=domain != "dag",
            )
            isolated.check(case)


@pytest.mark.parametrize("seed", range(8))
def test_case_json_replays_values_layout_and_topology(seed):
    case = fuzzer.generate_case(seed)
    replay = fuzzer.Case.from_json(case.to_json())
    assert replay.to_json() == case.to_json()
    for original, restored in zip(
        fuzzer.materialize(case), fuzzer.materialize(replay), strict=True
    ):
        assert restored.stride() == original.stride()
        np.testing.assert_array_equal(restored.numpy(), original.numpy())
        if original.dtype.is_floating_point:
            np.testing.assert_array_equal(
                np.signbit(restored.numpy()), np.signbit(original.numpy())
            )


@pytest.mark.parametrize(
    "expected,actual,zero_sign",
    [
        ([float("nan")], [0.0], False),
        ([float("inf")], [-float("inf")], False),
        ([-0.0], [0.0], True),
    ],
)
def test_oracle_rejects_special_value_semantic_mismatches(expected, actual, zero_sign):
    with pytest.raises(AssertionError):
        fuzzer.compare_array(np.array(expected), np.array(actual), zero_sign=zero_sign)


def test_reducer_preserves_failure_category_and_optimizer_mode():
    case = fuzzer.generate_case(2)
    failure = fuzzer.FuzzFailure(case, "semantic", True, "injected")
    calls = []

    def check(candidate, *, optimize_modes):
        calls.append((len(candidate.nodes), optimize_modes))
        if len(candidate.nodes) >= 2:
            raise fuzzer.FuzzFailure(candidate, "semantic", True, "same failure")

    with mock.patch.object(fuzzer, "run_case", side_effect=check):
        reduced = fuzzer.reduce_failure(case, failure)
    assert len(reduced.nodes) == 2
    assert all(mode == (True,) for _, mode in calls)
    assert case.nodes == fuzzer.generate_case(2).nodes


@pytest.mark.parametrize(
    "target,stage",
    [
        ("torch_onnx.export", "export"),
        ("onnx.checker.check_model", "invalid_graph"),
        ("onnxruntime.InferenceSession", "runtime"),
        ("fuzz.fuzzer.compare_array", "semantic"),
    ],
)
def test_pipeline_failures_are_not_suppressed_or_reclassified(target, stage):
    case = fuzzer.generate_case(4)
    with (
        mock.patch(target, side_effect=RuntimeError("injected pipeline failure")),
        pytest.raises(fuzzer.FuzzFailure) as captured,
    ):
        fuzzer.run_case(case, optimize_modes=(False,))
    assert captured.value.stage == stage
    assert captured.value.case is case
    assert not captured.value.optimize


@pytest.mark.parametrize("stage", ["export", "invalid_graph", "runtime", "semantic"])
def test_failure_diagnostic_contains_complete_replay_case(stage):
    case = fuzzer.generate_case(2)
    failure = fuzzer.FuzzFailure(case, stage, False, "injected")
    assert f"stage={stage}" in str(failure)
    assert (
        fuzzer.Case.from_json(str(failure).split("CASE_JSON:\n", 1)[1]).to_json()
        == case.to_json()
    )


@pytest.mark.parametrize("reference_matches_eager", [False, True])
def test_semantic_attribution_uses_independent_reference(reference_matches_eager):
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            "reference_attribution",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 26)],
        ir_version=10,
    )
    data = np.array([1.0, 2.0], dtype=np.float32)
    expected = (
        data if reference_matches_eager else np.array([1.0, 3.0], dtype=np.float32)
    )
    assert fuzzer._reference_attribution(model, {"x": data}, [expected], [False]) == (
        "backend" if reference_matches_eager else "export_semantic"
    )
