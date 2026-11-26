"""Check campaign coverage budgets and exact replay without timing assertions."""

from __future__ import annotations

import json

import pytest

from fuzz import campaign, fuzzer


def test_count_bounds_total_cases_and_covers_supported_matrix():
    variants = campaign.configurations(
        ["dag", "ce", "bce", "sdpa"], ["all", "final"], [18, 23, 26], [False, True]
    )
    assert len(variants) == 42
    cases = list(campaign.generate_campaign(seed=0, count=84, variants=variants))
    assert len(cases) == 84
    for start in (0, 42):
        actual = {
            (case.domain, case.output_mode, case.opset, bool(case.dynamic_shapes))
            for case in cases[start : start + 42]
        }
        assert actual == set(variants)
    assert [case.to_json() for case in cases] == [
        case.to_json()
        for case in campaign.generate_campaign(seed=0, count=84, variants=variants)
    ]


def test_budget_cannot_silently_miss_requested_configurations():
    variants = campaign.configurations(["ce", "bce"], ["final"], [18], [True])
    with pytest.raises(ValueError, match="at least 2"):
        list(campaign.generate_campaign(seed=0, count=1, variants=variants))
    with pytest.raises(ValueError, match="Dynamic shapes require"):
        campaign.configurations(["dag"], ["final"], [18], [True])


@pytest.mark.parametrize(
    "flag",
    ["--domain", "--output-mode", "--opset", "--shape-mode", "--seed", "--steps"],
)
def test_replay_rejects_generation_overrides(tmp_path, flag):
    values = {
        "--domain": "bce",
        "--output-mode": "final",
        "--opset": "18",
        "--shape-mode": "dynamic",
        "--seed": "0",
        "--steps": "4",
    }
    case = tmp_path / "case.json"
    case.write_text(fuzzer.generate_case(2).to_json())
    with pytest.raises(SystemExit) as error:
        campaign.main(["--case", str(case), flag, values[flag]])
    assert error.value.code == 2


def test_campaign_failure_writes_summary_and_preserves_exception(tmp_path, monkeypatch):
    report = tmp_path / "report.json"

    def fail(case, **kwargs):
        raise fuzzer.FuzzFailure(case, "runtime", False, "controlled failure")

    monkeypatch.setattr(fuzzer, "check_case", fail)
    with pytest.raises(fuzzer.FuzzFailure, match="controlled failure"):
        campaign.main(
            ["--in-process", "--report", str(report), "--failure-dir", str(tmp_path)]
        )
    result = json.loads(report.read_text())
    assert result["status"] == "failed"
    assert result["attempted_cases"] == 1 and result["passed_cases"] == 0
    assert result["validated_exports"] == 0
    assert result["timeout_s"] is None
    assert result["failure"]["type"] == "FuzzFailure"


def test_keep_going_records_every_failure_and_still_exits_nonzero(
    tmp_path, monkeypatch
):
    report = tmp_path / "report.json"
    seen = []

    def fail(case, **kwargs):
        seen.append(case.seed)
        raise fuzzer.FuzzFailure(case, "runtime", False, "controlled failure")

    monkeypatch.setattr(fuzzer, "check_case", fail)
    with pytest.raises(SystemExit, match="3 fuzz cases failed"):
        campaign.main(
            ["--in-process", "--count", "3", "--keep-going", "--report", str(report)]
        )
    result = json.loads(report.read_text())
    assert seen == [0, 1, 2]
    assert result["status"] == "failed" and result["failed_cases"] == 3
    assert result["passed_cases"] == 0 and result["validated_exports"] == 0
    assert [failure["seed"] for failure in result["failures"]] == seen


def test_controller_interrupt_retains_partial_campaign_summary(tmp_path, monkeypatch):
    report = tmp_path / "report.json"

    def interrupt(case, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fuzzer, "check_case", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.main(["--in-process", "--report", str(report)])
    result = json.loads(report.read_text())
    assert result["status"] == "interrupted"
    assert result["attempted_cases"] == 1 and result["validated_exports"] == 0
    assert result["failure"]["type"] == "KeyboardInterrupt"
