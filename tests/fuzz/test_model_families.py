"""Model recipe legality, replay, edge shapes and independent small oracles."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import onnx
import pytest
import torch

from fuzz import campaign, fuzzer, model_families


@pytest.mark.parametrize("domain", model_families.DOMAINS)
def test_model_recipe_is_deterministic_and_replays_every_dynamic_instance(domain):
    case = fuzzer.generate_case(17, domain=domain, dynamic=True, output_mode="all")
    assert case.version == 3
    assert (
        case.to_json()
        == fuzzer.generate_case(
            17, domain=domain, dynamic=True, output_mode="all"
        ).to_json()
    )
    replay = fuzzer.Case.from_json(case.to_json())
    assert replay.to_json() == case.to_json()
    for specs in [case.inputs, *case.additional_inputs]:
        original = fuzzer.GraphModule(case)(*fuzzer.materialize(case, inputs=specs))
        restored = fuzzer.GraphModule(replay)(*fuzzer.materialize(replay, inputs=specs))
        assert len(original) == len(case.nodes)
        for left, right in zip(original, restored, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    if domain == "decoder":
        assert case.additional_inputs[0][2]["shape"][2] == 0
        assert case.additional_inputs[0][0]["shape"][1] == 1
    assert any(node["op"] == "tensor_const" for node in case.nodes) == (
        domain != "sparse_attention"
    )


@pytest.mark.parametrize("domain", model_families.DOMAINS)
def test_multidimensional_noncontiguous_layout_survives_json(domain):
    case = fuzzer.generate_case(2, domain=domain)
    case.inputs[0]["layout"] = "transposed"
    tensor = fuzzer.materialize(fuzzer.Case.from_json(case.to_json()))[0]
    assert not tensor.is_contiguous()
    expected = torch.tensor(case.inputs[0]["values"]).reshape(case.inputs[0]["shape"])
    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption", ["missing", "bounds", "legacy", "constant", "arity", "cache"]
)
def test_malformed_model_replay_is_rejected(corruption):
    raw = json.loads(fuzzer.generate_case(7, domain="decoder", dynamic=True).to_json())
    if corruption == "missing":
        raw["dynamic_bounds"].pop("past")
    elif corruption == "bounds":
        raw["dynamic_bounds"]["batch"] = [-1, 64]
    elif corruption == "legacy":
        raw["version"] = 2
    elif corruption == "constant":
        raw["nodes"][0]["attrs"]["values"].pop()
    elif corruption == "arity":
        raw["nodes"][0]["args"] = [0]
    else:
        raw["inputs"][3]["shape"][1] += 1
    with pytest.raises(ValueError):
        fuzzer.Case.from_json(json.dumps(raw))


def test_v3_bounds_do_not_change_legacy_v2_validation():
    raw = json.loads(
        fuzzer.generate_template_case(1, domain="bce", dynamic=True).to_json()
    )
    raw["additional_inputs"][0][0]["shape"][0] = 1
    raw["additional_inputs"][0][0]["values"] = raw["inputs"][0]["values"][
        : raw["inputs"][0]["shape"][1]
    ]
    with pytest.raises(ValueError, match="bounds"):
        fuzzer.Case.from_json(json.dumps(raw))


def test_legacy_json_does_not_gain_v3_only_fields():
    for case in [
        fuzzer.generate_case(2),
        dataclasses.replace(fuzzer.generate_case(2), version=1),
    ]:
        payload = json.loads(case.to_json())
        assert "dynamic_bounds" not in payload and "float32_tolerance" not in payload
        assert fuzzer.Case.from_json(case.to_json()).to_json() == case.to_json()


def test_concentrated_moe_routes_leave_experts_unused_and_scatter_matches_dense():
    case = fuzzer.generate_case(3, domain="moe", output_mode="all")
    args = fuzzer.materialize(case)
    outputs = fuzzer.GraphModule(case)(*args)
    indices = outputs[
        next(i for i, node in enumerate(case.nodes) if node["op"] == "topk_indices")
    ]
    assert len(indices.unique()) < args[1].shape[-1]
    constants = [
        outputs[i] for i, node in enumerate(case.nodes) if node["op"] == "tensor_const"
    ]
    up, down = constants
    probabilities = torch.softmax(args[1], -1)
    k = indices.shape[-1]
    weights = torch.topk(probabilities, k, -1).values
    weights = weights / weights.sum(-1, keepdim=True)
    expected = torch.zeros_like(args[0])
    for batch in range(expected.shape[0]):
        for token in range(expected.shape[1]):
            for slot in range(k):
                expert = indices[batch, token, slot]
                hidden = torch.nn.functional.silu(args[0][batch, token] @ up[expert])
                expected[batch, token] += weights[batch, token, slot] * (
                    hidden @ down[expert]
                )
    torch.testing.assert_close(outputs[-1], expected)


def test_delta_update_matches_outer_product_equation():
    state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    key = torch.tensor([[[0.5, -0.25]]])
    value = torch.tensor([[[1.0, -1.0]]])
    beta = torch.zeros(1, 1, 1)
    decay = torch.full((1, 1, 1, 1), 0.75)
    result = model_families.apply("delta_update", [state, key, value, beta, decay], {})
    decayed = state * 0.75
    prediction = torch.einsum("bhi,bhij->bhj", key, decayed)
    expected = decayed + torch.einsum("bhi,bhj->bhij", key, 0.5 * (value - prediction))
    torch.testing.assert_close(result, expected)


def test_model_matrix_covers_every_family_without_multiplying_case_budget():
    variants = campaign.configurations(
        model_families.DOMAINS, ["all", "final"], [18, 23, 26], [False, True]
    )
    assert len(variants) == 84
    cases = list(campaign.generate_campaign(seed=0, count=84, variants=variants))
    assert {
        (case.domain, case.output_mode, case.opset, bool(case.dynamic_shapes))
        for case in cases
    } == set(variants)
    assert (
        len(
            campaign.configurations(
                fuzzer.DOMAINS, ["all", "final"], [18, 23, 26], [False, True]
            )
        )
        == 126
    )


def test_coverage_recurses_into_used_functions_and_control_flow_only():
    helper = onnx.helper

    def branch(op):
        return helper.make_graph(
            [helper.make_node(op, ["x"], ["z"])],
            op,
            [],
            [helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [2])],
        )

    graph = helper.make_graph(
        [
            helper.make_node("Used", ["x"], ["intermediate"], domain="local"),
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=branch("Abs"),
                else_branch=branch("Relu"),
            ),
        ],
        "coverage",
        [],
        [],
    )

    def function(name, op):
        return helper.make_function(
            "local",
            name,
            ["x"],
            ["y"],
            [helper.make_node(op, ["x"], ["y"])],
            [helper.make_opsetid("", 18)],
        )

    unused = function("Used", "Exp")
    unused.overload = "unused"
    proto = helper.make_model(graph, functions=[function("Used", "Tanh"), unused])
    assert fuzzer._onnx_ops(proto) == {"local::Used", "Tanh", "If", "Abs", "Relu"}


def test_model_all_and_final_observe_the_same_terminal_value():
    case = fuzzer.generate_case(11, domain="mhc", output_mode="all")
    args = fuzzer.materialize(case)
    all_outputs = fuzzer.GraphModule(case)(*args)
    final = fuzzer.GraphModule(dataclasses.replace(case, output_mode="final"))(*args)
    assert len(final) == 1
    np.testing.assert_array_equal(final[0].numpy(), all_outputs[-1].numpy())


def test_model_float32_budget_is_explicit_and_does_not_weaken_the_old_oracle():
    case = fuzzer.generate_case(7, domain="vision")
    replay = fuzzer.Case.from_json(case.to_json())
    assert replay.float32_tolerance == [1.3e-6, 1e-5]
    assert fuzzer.generate_case(7).float32_tolerance is None
    expected = np.array([0.07], dtype=np.float32)
    actual = expected + np.float32(2e-6)
    with pytest.raises(AssertionError):
        fuzzer.compare_array(expected, actual)
    fuzzer.compare_array(expected, actual, tolerance=case.float32_tolerance)
    # The model budget also tightens relative error: it is not a blanket relaxation.
    expected = np.array([1e6], dtype=np.float32)
    actual = expected + np.float32(5)
    fuzzer.compare_array(expected, actual)
    with pytest.raises(AssertionError):
        fuzzer.compare_array(expected, actual, tolerance=case.float32_tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        fuzzer.Case.from_json(dataclasses.replace(case, version=2).to_json())


def test_constant_dtype_is_part_of_the_serialized_recipe():
    value = model_families.apply(
        "tensor_const", [], {"shape": [2], "values": [1, 2], "dtype": "int64"}
    )
    assert value.dtype == torch.int64
    torch.testing.assert_close(value, torch.tensor([1, 2], dtype=torch.int64))


def test_models_cli_reports_validated_operator_inventory(tmp_path, monkeypatch):
    seen = []

    def check(case, **kwargs):
        seen.append(case)
        return {"aten_ops": ["aten.matmul.default"], "onnx_ops": ["MatMul"]}

    monkeypatch.setattr(fuzzer, "check_case", check)
    report = tmp_path / "report.json"
    campaign.main(
        [
            "--domain",
            "models",
            "--count",
            "7",
            "--opset",
            "23",
            "--shape-mode",
            "dynamic",
            "--in-process",
            "--report",
            str(report),
        ]
    )
    result = json.loads(report.read_text())
    assert {case.domain for case in seen} == set(model_families.DOMAINS)
    assert result["passed_cases"] == 7 and result["validated_input_instances"] == 42
    assert result["aten_ops"] == ["aten.matmul.default"] and result["onnx_ops"] == [
        "MatMul"
    ]
    assert result["domain_counts"] == dict.fromkeys(model_families.DOMAINS, 1)
