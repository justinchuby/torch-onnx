"""Comparison policy tests; deliberately no noisy timing assertions."""

import builtins
import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "performance", Path(__file__).with_name("performance.py")
)
performance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(performance)


def report():
    return {
        "fingerprint": {
            "host": "same-host",
            "versions": {"torch": "2.15.0"},
            "config": {"threads": 1},
        },
        "results": {"bce": dict.fromkeys(performance.METRICS, 10)},
    }


def test_comparison_reports_fractional_change():
    baseline = report()
    current = copy.deepcopy(baseline)
    current["results"]["bce"]["export_seconds"] = 12
    assert performance.compare(current, baseline)["bce"][
        "export_seconds"
    ] == pytest.approx(0.2)
    assert performance.compare(current, baseline)["bce"]["nodes"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("host", "different-host"),
        ("versions", {"torch": "2.16.0"}),
        ("config", {"threads": 2}),
    ],
)
def test_reject_incompatible_reports(key, value):
    baseline = report()
    current = copy.deepcopy(baseline)
    current["fingerprint"][key] = value
    with pytest.raises(ValueError, match="Incompatible"):
        performance.compare(current, baseline)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "10", None, True])
@pytest.mark.parametrize("side", ["current", "baseline"])
def test_reject_invalid_metric(value, side):
    reports = {"current": report(), "baseline": report()}
    reports[side]["results"]["bce"]["export_seconds"] = value
    with pytest.raises(ValueError, match="finite positive"):
        performance.compare(reports["current"], reports["baseline"])


@pytest.mark.parametrize("missing", ["case", "metric", "results"])
def test_reject_missing_results(missing):
    baseline = report()
    if missing == "case":
        baseline["results"].clear()
    elif missing == "metric":
        del baseline["results"]["bce"]["nodes"]
    else:
        del baseline["results"]
    with pytest.raises(ValueError, match="missing"):
        performance.compare(report(), baseline)


@pytest.mark.parametrize("option", ["--timeout", "--max-regression"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_reject_nonfinite_cli_controls(monkeypatch, option, value):
    monkeypatch.setattr(
        performance.sys,
        "argv",
        ["performance.py", "--baseline", "unused.json", f"{option}={value}"],
    )
    with pytest.raises(SystemExit) as exc:
        performance.main()
    assert exc.value.code == 2


def test_comparison_module_does_not_require_resource(monkeypatch):
    real_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ModuleNotFoundError("resource unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(module)
    assert module.compare(report(), report())["bce"]["nodes"] == 0
    monkeypatch.setattr(module.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="only Linux and macOS"):
        module.peak_rss()
