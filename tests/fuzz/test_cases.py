"""Case configuration, output visibility, and dynamic replay contracts."""

from __future__ import annotations

import dataclasses
import json
from unittest import mock

import numpy as np
import pytest
import torch

from fuzz import fuzzer


def test_reducer_never_emits_an_unreplayable_domain_template():
    case = fuzzer.generate_template_case(9, domain="bce", dynamic=True)
    failure = fuzzer.FuzzFailure(case, "runtime", False, "controlled failure")

    def fail(candidate, **kwargs):
        raise fuzzer.FuzzFailure(candidate, "runtime", False, "same failure")

    with mock.patch.object(fuzzer, "run_case", side_effect=fail):
        reduced = fuzzer.reduce_failure(case, failure)
    assert reduced.nodes == []
    replay = fuzzer.Case.from_json(reduced.to_json())
    assert replay.domain == "bce"
    assert len(fuzzer.GraphModule(replay)(*fuzzer.materialize(replay))) == 1


def test_version_one_json_keeps_legacy_observation_contract():
    case = fuzzer.generate_case(2)
    raw = {
        key: value
        for key, value in dataclasses.asdict(case).items()
        if key in ("seed", "inputs", "nodes", "version")
    }
    raw["version"] = 1
    restored = fuzzer.Case.from_json(json.dumps(raw))
    assert restored.output_mode == "all"
    assert restored.opset == 26
    assert restored.input_probes
    assert len(fuzzer.GraphModule(restored)(*fuzzer.materialize(restored))) == len(
        restored.nodes
    )


@pytest.mark.parametrize(
    "dtypes,expected_count", [(("float32", "int64"), 2), (("int64", "float32"), 0)]
)
def test_version_one_mixed_input_probe_arity_is_preserved(dtypes, expected_count):
    raw = {
        "version": 1,
        "seed": 0,
        "nodes": [],
        "inputs": [
            {"dtype": dtype, "shape": [2], "layout": "contiguous", "values": [-1, 1]}
            for dtype in dtypes
        ],
    }
    case = fuzzer.Case.from_json(json.dumps(raw))
    assert len(fuzzer.GraphModule(case)(*fuzzer.materialize(case))) == expected_count


def test_final_output_mode_does_not_expose_intermediate_or_input_probes():
    case = fuzzer.generate_case(1, output_mode="final")
    assert not case.input_probes
    inputs = fuzzer.materialize(case)
    final = fuzzer.GraphModule(case)(*inputs)
    all_outputs = fuzzer.GraphModule(dataclasses.replace(case, output_mode="all"))(
        *inputs
    )
    assert len(final) == 1
    np.testing.assert_array_equal(final[0].numpy(), all_outputs[-1].numpy())


@pytest.mark.parametrize("domain", ["ce", "bce", "sdpa"])
def test_template_replay_preserves_every_instance_and_configuration(domain):
    case = fuzzer.generate_template_case(17, domain=domain, opset=23, dynamic=True)
    replay = fuzzer.Case.from_json(case.to_json())
    assert replay.to_json() == case.to_json()
    assert len(replay.additional_inputs) == 2
    assert not replay.input_probes
    for specs in [replay.inputs, *replay.additional_inputs]:
        output = fuzzer.GraphModule(replay)(*fuzzer.materialize(replay, inputs=specs))
        assert len(output) == 1
        assert torch.isfinite(output[0]).all()
    if domain == "sdpa":
        shapes = [
            [spec["shape"] for spec in specs]
            for specs in [replay.inputs, *replay.additional_inputs]
        ]
        assert len({tuple(instance[0]) for instance in shapes}) == 3
        assert all(instance[3][:2] == [1, 1] for instance in shapes)
        assert all(
            instance[0][2] == instance[3][2] and instance[1][2] == instance[3][3]
            for instance in shapes
        )


@pytest.mark.parametrize(
    "corruption", ["static", "symbol", "bounds", "arity", "mode", "opset"]
)
def test_invalid_case_configuration_is_rejected(corruption):
    case = fuzzer.generate_template_case(1, domain="bce", dynamic=True)
    raw = dataclasses.asdict(case)
    if corruption == "static":
        raw["additional_inputs"][0][0]["shape"][1] += 1
    elif corruption == "symbol":
        raw["dynamic_shapes"]["y"]["1"] = "batch"
    elif corruption == "bounds":
        raw["dynamic_shapes"]["x"]["0"] = "invalid symbol"
    elif corruption == "arity":
        raw["additional_inputs"][0].pop()
    elif corruption == "mode":
        raw["output_mode"] = "sometimes"
    else:
        raw["opset"] = 19
    with pytest.raises(ValueError):
        fuzzer.Case.from_json(json.dumps(raw))


def test_dynamic_instances_reuse_one_export_and_session_per_mode():
    case = fuzzer.generate_template_case(
        1, domain="ce", dynamic=True, opset=18, reduction="mean"
    )
    with (
        mock.patch.object(
            fuzzer.torch_onnx, "export", wraps=fuzzer.torch_onnx.export
        ) as export,
        mock.patch.object(
            fuzzer.ort, "InferenceSession", wraps=fuzzer.ort.InferenceSession
        ) as session,
        mock.patch.object(
            fuzzer, "compare_array", wraps=fuzzer.compare_array
        ) as compare,
    ):
        fuzzer.run_case(case)
    assert export.call_count == 2
    assert session.call_count == 2
    assert compare.call_count == 6
    assert export.call_args.kwargs["opset_version"] == 18
