"""Exercise isolation with owned controlled workers, never external processes."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from fuzz import runner

# This subprocess only reads this test's own case/config files. No PyTorch
# import is needed to exercise crash, timeout and process-tree cleanup contracts.
_CONTROLLED_WORKER = r"""
import json, os, pathlib, signal, subprocess, sys, time
print(json.dumps({"kind": "ready", "pid": os.getpid()}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    config_path = pathlib.Path(request["config_path"])
    config = json.loads(config_path.read_text())
    case = json.loads(pathlib.Path(config["case_path"]).read_text())
    directory = config_path.parent
    assert config["config_version"] == 1
    assert case["pre_saved"] == "complete"
    pathlib.Path(config["stdout_path"]).write_text("owned worker stdout\n")
    pathlib.Path(config["stderr_path"]).write_text("owned worker stderr\n")
    (directory / "verified-before-execution").write_text("yes")
    behavior = case["behavior"]
    if behavior.startswith("child_"):
        heartbeat = directory / "heartbeat"
        child_code = (
            "import pathlib,time; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
            "exec('while True:\\n with p.open(\"a\") as f: f.write(\"x\"); f.flush()\\n time.sleep(.02)')"
        )
        child = subprocess.Popen([sys.executable, "-u", "-c", child_code])
        (directory / "child.pid").write_text(str(child.pid))
        deadline = time.monotonic() + 2
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(.01)
    if behavior in ("hang", "child_hang"):
        time.sleep(60)
    if behavior in ("crash", "child_crash"):
        os._exit(23)
    if behavior == "segfault":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.raise_signal(signal.SIGSEGV)
    response = {"kind": "result", "job_id": request["job_id"], "status": "passed", "stage": "passed"}
    if behavior == "error":
        response.update({"status": "failed", "stage": case["stage"], "optimize": True,
                         "attribution": "backend", "detail": "classified Python failure",
                         "failure_case_json": json.dumps(case)})
    print(json.dumps(response), flush=True)
"""


class _Case:
    seed = 7

    def __init__(self, behavior="pass", *, stage="semantic"):
        self.behavior, self.stage = behavior, stage

    def to_json(self):
        return json.dumps(
            {
                "seed": self.seed,
                "behavior": self.behavior,
                "stage": self.stage,
                "pre_saved": "complete",
            }
        )


def _runner(path, *, timeout_s=5, max_cases_per_worker=64):
    return runner.IsolatedRunner(
        path,
        timeout_s=timeout_s,
        max_cases_per_worker=max_cases_per_worker,
        worker_command=[sys.executable, "-u", "-c", _CONTROLLED_WORKER],
    )


def test_success_reuses_worker_and_keeps_unique_atomic_artifacts(tmp_path):
    with _runner(tmp_path) as isolated:
        first = isolated.check(_Case())
        second = isolated.check(_Case())
    assert first.passed and first.stage == "passed"
    assert first.worker_pid == second.worker_pid
    assert first.config_path.parent != second.config_path.parent
    for result in (first, second):
        assert (
            result.config_path.parent / "verified-before-execution"
        ).read_text() == "yes"
        assert json.loads(result.case_path.read_text())["pre_saved"] == "complete"
        config = json.loads(result.config_path.read_text())
        assert config["case_path"] == str(result.case_path)
        assert config["timeout_s"] == 5
        assert "source_sha256" in config["runtime_info"]
        assert json.loads(result.result_path.read_text())["stage"] == "passed"
        assert result.stdout_path.read_text() == "owned worker stdout\n"
        assert result.stderr_path.read_text() == "owned worker stderr\n"
        assert "--replay" in result.replay_command
        assert "--restore-environment" not in result.replay_command
        assert not list(result.config_path.parent.glob(".*"))


@pytest.mark.parametrize(
    "stage", ["export", "invalid_graph", "runtime", "semantic", "generator"]
)
def test_normal_python_failure_preserves_category_and_replay(tmp_path, stage):
    with _runner(tmp_path) as isolated:
        with pytest.raises(runner.RunnerFailure) as captured:
            isolated.check(_Case("error", stage=stage))
        failure = captured.value
        result = failure.result
        next_result = isolated.check(_Case())
    assert failure.stage == stage
    assert result.optimize is True and result.attribution == "backend"
    assert result.minimized_case_path is not None
    assert json.loads(result.minimized_case_path.read_text())["stage"] == stage
    assert str(result.config_path) in str(failure)
    assert f"stage={stage}" in str(failure)
    assert result.stdout_path.read_text() and result.stderr_path.read_text()
    assert next_result.worker_pid != result.worker_pid


@pytest.mark.parametrize("behavior,stage", [("hang", "timeout"), ("crash", "crash")])
def test_hang_and_abrupt_exit_keep_original_case_logs_and_restart(
    tmp_path, behavior, stage
):
    with _runner(tmp_path, timeout_s=0.75) as isolated:
        failed = isolated.run(_Case(behavior))
        passed = isolated.check(_Case())
    assert not failed.passed and failed.stage == stage
    assert failed.elapsed_s < 5
    assert failed.returncode is not None
    assert failed.case_path.exists() and failed.config_path.exists()
    assert failed.stdout_path.read_text() == "owned worker stdout\n"
    assert failed.stderr_path.read_text() == "owned worker stderr\n"
    assert passed.passed and passed.worker_pid != failed.worker_pid


@pytest.mark.skipif(
    os.name == "nt", reason="Unix process-group contract; Windows uses a Job Object"
)
@pytest.mark.parametrize("behavior", ["child_hang", "child_crash"])
def test_timeout_or_crash_stops_owned_descendant_heartbeat(tmp_path, behavior):
    with _runner(tmp_path, timeout_s=1.0) as isolated:
        result = isolated.run(_Case(behavior))
    heartbeat = result.config_path.parent / "heartbeat"
    assert (result.config_path.parent / "child.pid").exists()
    assert heartbeat.exists(), result.stderr_path.read_text()
    assert heartbeat.stat().st_size > 0
    size = heartbeat.stat().st_size
    time.sleep(0.15)
    assert heartbeat.stat().st_size == size
    assert result.stage == ("timeout" if behavior == "child_hang" else "crash")


def test_worker_rotation_bounds_import_cache_lifetime(tmp_path):
    with _runner(tmp_path, max_cases_per_worker=1) as isolated:
        first = isolated.check(_Case())
        second = isolated.check(_Case())
    assert first.worker_pid != second.worker_pid


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_is_rejected(tmp_path, timeout_s):
    with pytest.raises(ValueError, match="finite and positive"):
        runner.IsolatedRunner(tmp_path, timeout_s=timeout_s)


def test_replay_never_executes_a_command_from_json(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps({"seed": 7, "pre_saved": "complete"}))
    config_path.write_text(
        json.dumps(
            {
                "config_version": 1,
                "case_path": str(case_path),
                "timeout_s": 5,
                "reduce": False,
                "worker_command": ["untrusted-executable"],
                "environment": {"PYTHONPATH": "/untrusted"},
                "working_directory": "/untrusted",
            }
        )
    )
    calls = []

    class TrustedRunner:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def check(self, case):
            return "trusted result"

    monkeypatch.setattr(runner, "IsolatedRunner", TrustedRunner)
    assert runner.replay(config_path) == "trusted result"
    assert "worker_command" not in calls[0]
    assert calls[0]["env"] is None and calls[0]["cwd"] is None
    diagnostic = capsys.readouterr().err
    assert "working directory" in diagnostic and "PYTHONPATH" in diagnostic
    assert "/untrusted" not in diagnostic


@pytest.mark.parametrize("domain,opset", [("dag", 26), ("moe", 23)])
def test_real_worker_runs_complete_fuzzer_without_changing_case_schema(
    tmp_path, domain, opset
):
    from fuzz import fuzzer

    case = fuzzer.generate_case(4, domain=domain, dynamic=domain != "dag", opset=opset)
    with runner.IsolatedRunner(tmp_path, timeout_s=30, reduce=False) as isolated:
        result = isolated.check(case)
    assert result.passed
    assert json.loads(result.case_path.read_text()) == json.loads(case.to_json())
    assert result.worker_pid is not None
    assert result.coverage["aten_ops"] and result.coverage["onnx_ops"]
    assert "model_families.py" in result.runtime_info["source_sha256"]


@pytest.mark.skipif(
    os.name == "nt", reason="Controlled Unix SIGSEGV without a core dump"
)
def test_native_signal_crash_retains_case_and_logs(tmp_path):
    with _runner(tmp_path) as isolated:
        result = isolated.run(_Case("segfault"))
    assert result.stage == "crash"
    assert result.returncode == -11
    assert result.stdout_path.read_text() == "owned worker stdout\n"
    assert result.stderr_path.read_text() == "owned worker stderr\n"
    assert json.loads(result.case_path.read_text())["behavior"] == "segfault"


def test_serialized_case_and_replay_survive_moving_artifacts(tmp_path, monkeypatch):
    directory = tmp_path / "moved"
    directory.mkdir()
    case_path = directory / "case.json"
    case_path.write_text(json.dumps({"seed": 19, "inputs": [], "version": 2}))
    config_path = directory / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "config_version": 1,
                "case_path": "/old/host/missing/case.json",
                "timeout_s": 2,
                "reduce": True,
            }
        )
    )
    captured = []

    class TrustedRunner:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def check(self, case):
            captured.append(case)
            return "replayed"

    monkeypatch.setattr(runner, "IsolatedRunner", TrustedRunner)
    assert runner.replay(config_path) == "replayed"
    assert isinstance(captured[0], runner.SerializedCase)
    assert captured[0].seed == 19
    assert json.loads(captured[0].to_json())["version"] == 2


def test_deadline_includes_worker_startup_before_ready(tmp_path):
    with runner.IsolatedRunner(
        tmp_path,
        timeout_s=0.25,
        worker_command=[sys.executable, "-u", "-c", "import time; time.sleep(60)"],
    ) as isolated:
        result = isolated.run(_Case())
    assert result.stage == "timeout"
    assert result.elapsed_s < 5
    assert json.loads(result.case_path.read_text())["pre_saved"] == "complete"
    assert result.stdout_path.exists() and result.stderr_path.exists()
