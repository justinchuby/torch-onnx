"""Bounded reusable workers and pre-execution artifacts for long fuzz campaigns.

This module deliberately does not import PyTorch in the controller process.
The saved configuration is diagnostic data, not an executable worker command.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, ClassVar, Self

_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "TORCH_LOGS",
    "TORCHDYNAMO_VERBOSE",
    "TORCH_COMPILE_DEBUG",
    "GITHUB_SHA",
)
_PACKAGES = (
    "torch",
    "onnx",
    "onnx-ir",
    "onnxscript",
    "onnxruntime",
    "onnx-shape-inference",
    "torch-onnx",
)


def _atomic_write(path: Path, text: str) -> None:
    """Commit a complete file before a worker can observe its pathname."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary).unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _runtime_info() -> dict[str, Any]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = (Path(__file__), Path(__file__).with_name("fuzzer.py"))
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }


class SerializedCase:
    """Carry a saved case without importing PyTorch in the controller."""

    def __init__(self, text: str):
        payload = json.loads(text)
        if not isinstance(payload, dict) or type(payload.get("seed")) is not int:
            raise ValueError("A saved case must contain an integer seed")
        self.seed = payload["seed"]
        self._text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)

    def to_json(self) -> str:
        return self._text


@dataclasses.dataclass
class CaseResult:
    """Outcome and immutable artifact locations for one complete case budget."""

    status: str
    stage: str
    optimize: bool | None
    attribution: str | None
    detail: str
    case_path: Path
    config_path: Path
    result_path: Path
    stdout_path: Path
    stderr_path: Path
    minimized_case_path: Path | None
    elapsed_s: float
    returncode: int | None
    worker_pid: int | None
    replay_command: str
    runtime_info: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataclasses.asdict(self).items()
        }


class RunnerFailure(AssertionError):
    """A classified case failure with replayable pre-execution artifacts."""

    def __init__(self, result: CaseResult):
        self.result = result
        self.stage = result.stage
        self.optimize = result.optimize
        self.attribution = result.attribution
        super().__init__(
            f"stage={result.stage} optimize={result.optimize} attribution={result.attribution} "
            f"after {result.elapsed_s:.3f}s: {result.detail}\n"
            f"Case: {result.case_path}\nConfig: {result.config_path}\n"
            f"Stdout: {result.stdout_path}\nStderr: {result.stderr_path}\n"
            f"Replay: {result.replay_command}"
        )


class _DeadlineExceeded(Exception):
    pass


class _WorkerExited(Exception):
    pass


class _ProtocolError(Exception):
    pass


def _windows_job(process: subprocess.Popen) -> Any:
    """Use kill-on-close job ownership when available; otherwise tree-kill."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_: ClassVar = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD),
            ("min_ws", ctypes.c_size_t),
            ("max_ws", ctypes.c_size_t),
            ("active", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_: ClassVar = [
            (name, ctypes.c_ulonglong)
            for name in (
                "read_ops",
                "write_ops",
                "other_ops",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_: ClassVar = [
            ("basic", BasicLimits),
            ("io", IoCounters),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(
        handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
    ) or not kernel.AssignProcessToJobObject(handle, int(process._handle)):
        kernel.CloseHandle(handle)
        return None
    return kernel, handle


class IsolatedRunner:
    """Reuse a trusted worker while bounding each case's entire execution.

    The deadline includes worker startup/imports, eager, export, validation,
    runtimes, reference attribution, and reduction. A failed worker is discarded.
    ``worker_command`` is an injection hook for controlled lifecycle tests only;
    replay never executes a command taken from a JSON configuration.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        timeout_s: float = 60.0,
        reduce: bool = True,
        max_cases_per_worker: int = 64,
        worker_command: list[str] | None = None,
        env: dict[str, str | None] | None = None,
        cwd: Path | None = None,
    ):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if max_cases_per_worker < 1:
            raise ValueError("max_cases_per_worker must be positive")
        self.artifact_dir = Path(artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.reduce = timeout_s, reduce
        self.max_cases_per_worker = max_cases_per_worker
        self.command = (
            list(worker_command)
            if worker_command is not None
            else [
                sys.executable,
                "-X",
                "faulthandler",
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
            ]
        )
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.environment = {key: os.environ.get(key) for key in _ENV_KEYS}
        if self.environment["PYTHONHASHSEED"] is None:
            self.environment["PYTHONHASHSEED"] = "0"
        if env:
            if set(env) - set(_ENV_KEYS):
                raise ValueError(
                    "Only documented fuzz environment overrides are supported"
                )
            self.environment.update(env)
        self.runtime_info = _runtime_info()
        self._process: subprocess.Popen | None = None
        self._events: queue.Queue | None = None
        self._reader: threading.Thread | None = None
        self._bootstrap_path: Path | None = None
        self._job = None
        self._completed = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @staticmethod
    def _read_output(stream, events) -> None:
        try:
            for line in stream:
                events.put(line)
        finally:
            events.put(None)

    def _receive(self, deadline: float) -> dict[str, Any]:
        assert self._process is not None and self._events is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineExceeded
            try:
                line = self._events.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if self._process.poll() is not None:
                    raise _WorkerExited(
                        f"worker exited with return code {self._process.returncode}"
                    ) from None
                continue
            if line is None:
                self._process.poll()
                raise _WorkerExited(
                    f"worker exited with return code {self._process.returncode}"
                )
            try:
                message = json.loads(line)
            except (TypeError, ValueError) as error:
                raise _ProtocolError(
                    f"Invalid worker control message: {line[:200]!r}"
                ) from error
            if not isinstance(message, dict):
                raise _ProtocolError("Worker message is not an object")
            return message

    def _start(self, deadline: float) -> None:
        if self._process is not None:
            if (
                self._process.poll() is None
                and self._completed < self.max_cases_per_worker
            ):
                return
            self.close()
        fd, path = tempfile.mkstemp(
            prefix="worker-", suffix=".stderr", dir=self.artifact_dir
        )
        self._bootstrap_path = Path(path)
        child_env = os.environ.copy()
        for key, value in self.environment.items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = value
        with os.fdopen(fd, "wb") as bootstrap:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=bootstrap,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                env=child_env,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
            )
        self._job = _windows_job(self._process)
        self._events = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_output,
            args=(self._process.stdout, self._events),
            daemon=True,
        )
        self._reader.start()
        self._completed = 0
        ready = self._receive(deadline)
        if ready.get("kind") != "ready":
            raise _ProtocolError("Worker did not acknowledge startup")

    def close(self) -> None:
        """Kill the owned worker group/tree, including on crash and idle close."""
        process = self._process
        if process is None:
            return
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif self._job is not None:
            kernel, handle = self._job
            kernel.CloseHandle(handle)
        else:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if process.poll() is None:
                process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        if self._reader is not None:
            self._reader.join(timeout=1)
        self._process, self._events, self._reader, self._job = None, None, None, None

    def run(self, case: Any) -> CaseResult:
        """Pre-save one unique case/config and return its classified outcome."""
        start = time.monotonic()
        directory = Path(
            tempfile.mkdtemp(prefix=f"seed-{int(case.seed)}-", dir=self.artifact_dir)
        )
        case_path, config_path = directory / "case.json", directory / "config.json"
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        result_path = directory / "result.json"
        replay_command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--replay",
                str(config_path),
            ]
        )
        # Both records are atomically complete before any worker is launched or
        # sent a command. This survives abrupt native crashes and controller exit.
        _atomic_write(case_path, case.to_json() + "\n")
        _write_json(
            config_path,
            {
                "config_version": 1,
                "case_file": "case.json",
                "case_path": str(case_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "timeout_s": self.timeout_s,
                "reduce": self.reduce,
                "max_cases_per_worker": self.max_cases_per_worker,
                "working_directory": str(self.cwd),
                "environment": self.environment,
                "runtime_info": self.runtime_info,
                "worker_command": self.command,
            },
        )
        stdout_path.touch()
        stderr_path.touch()
        deadline = start + self.timeout_s
        response: dict[str, Any]
        returncode = None
        pid = None
        try:
            self._start(deadline)
            assert self._process is not None and self._process.stdin is not None
            pid = self._process.pid
            self._process.stdin.write(
                json.dumps(
                    {
                        "command": "run",
                        "job_id": directory.name,
                        "config_path": str(config_path),
                    }
                )
                + "\n"
            )
            self._process.stdin.flush()
            response = self._receive(deadline)
            if (
                response.get("kind") != "result"
                or response.get("job_id") != directory.name
            ):
                raise _ProtocolError("Worker returned a mismatched case response")
            if response.get("status") not in ("passed", "failed"):
                raise _ProtocolError("Worker returned an invalid outcome")
            if response["status"] == "passed" and response.get("stage") != "passed":
                raise _ProtocolError("Worker success has a failure stage")
            if response["status"] == "failed" and (
                not isinstance(response.get("stage"), str)
                or response["stage"] == "passed"
            ):
                raise _ProtocolError("Worker failure is missing its classification")
            if response.get("failure_case_json") is not None and not isinstance(
                response["failure_case_json"], str
            ):
                raise _ProtocolError("Worker failure case is not serialized JSON")
            self._completed += 1
        except _DeadlineExceeded:
            response = {
                "status": "failed",
                "stage": "timeout",
                "detail": f"Case exceeded its complete {self.timeout_s:g}s budget",
            }
        except (_WorkerExited, BrokenPipeError, OSError) as error:
            response = {"status": "failed", "stage": "crash", "detail": str(error)}
        except _ProtocolError as error:
            response = {"status": "failed", "stage": "worker", "detail": str(error)}
        if self._process is not None:
            pid = self._process.pid
            returncode = self._process.poll()
        if response["status"] != "passed":
            owned_process = self._process
            self.close()
            if returncode is None and owned_process is not None:
                returncode = owned_process.returncode
            if self._bootstrap_path is not None and self._bootstrap_path.exists():
                with stderr_path.open("ab") as stream:
                    stream.write(self._bootstrap_path.read_bytes())
        minimized = None
        if response.get("failure_case_json") is not None:
            minimized = directory / "failure-case.json"
            _atomic_write(minimized, response["failure_case_json"] + "\n")
        result = CaseResult(
            status=response["status"],
            stage=response.get("stage", "worker"),
            optimize=response.get("optimize"),
            attribution=response.get("attribution"),
            detail=response.get("detail", ""),
            case_path=case_path,
            config_path=config_path,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            minimized_case_path=minimized,
            elapsed_s=time.monotonic() - start,
            returncode=returncode,
            worker_pid=pid,
            replay_command=replay_command,
            runtime_info=self.runtime_info,
        )
        _write_json(result_path, result.to_dict())
        return result

    def check(self, case: Any) -> CaseResult:
        result = self.run(case)
        if not result.passed:
            raise RunnerFailure(result)
        return result


@contextlib.contextmanager
def _case_streams(stdout_path: str, stderr_path: str):
    """Redirect file descriptors too, so C/C++ diagnostics survive a crash."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with (
            open(stdout_path, "ab", buffering=0) as output,
            open(stderr_path, "ab", buffering=0) as errors,
        ):
            os.dup2(output.fileno(), 1)
            os.dup2(errors.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _worker() -> None:
    if os.name != "nt":
        import resource

        # A native failure should retain its case and faulthandler log, not
        # consume the campaign disk budget with an unbounded process core.
        with contextlib.suppress(OSError, ValueError):
            _, hard = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    control_fd = os.dup(1)

    def send(message):
        payload = memoryview(
            (json.dumps(message, ensure_ascii=True) + "\n").encode("utf-8")
        )
        while payload:
            written = os.write(control_fd, payload)
            payload = payload[written:]

    send({"kind": "ready", "pid": os.getpid()})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") != "run":
            raise ValueError("Unknown worker command")
        config = json.loads(Path(request["config_path"]).read_text())
        response = {
            "kind": "result",
            "job_id": request["job_id"],
            "status": "passed",
            "stage": "passed",
        }
        with _case_streams(config["stdout_path"], config["stderr_path"]):
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from fuzz import fuzzer

                case = fuzzer.Case.from_json(Path(config["case_path"]).read_text())
                fuzzer.check_case(
                    case,
                    failure_dir=Path(config["case_path"]).parent,
                    reduce=config["reduce"],
                )
            except Exception as error:
                traceback.print_exc()
                response.update(
                    {
                        "status": "failed",
                        "stage": "worker",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                if "fuzzer" in locals() and isinstance(error, fuzzer.FuzzFailure):
                    response.update(
                        {
                            "stage": error.stage,
                            "optimize": error.optimize,
                            "attribution": error.attribution,
                            "detail": error.detail,
                            "failure_case_json": error.case.to_json(),
                        }
                    )
        send(response)
    os.close(control_fd)


def replay(
    config_path: Path,
    *,
    artifact_dir: Path | None = None,
    restore_environment: bool = False,
) -> CaseResult:
    """Replay saved data with this trusted worker, never a saved executable.

    Environment/cwd restoration requires an explicit opt-in because PYTHONPATH
    and a working directory can affect Python imports. Only trust your own
    configurations when enabling it. The recorded interpreter is diagnostic;
    choose a different interpreter by explicitly invoking this script with it.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("config_version") != 1:
        raise ValueError("Unknown runner configuration version")
    if not restore_environment:
        differences = []
        saved_cwd = config.get("working_directory")
        if saved_cwd is not None and Path(saved_cwd).resolve() != Path.cwd().resolve():
            differences.append("working directory")
        saved_environment = config.get("environment", {})
        for key in _ENV_KEYS:
            current = os.environ.get(key)
            if key == "PYTHONHASHSEED" and current is None:
                current = "0"
            if key in saved_environment and saved_environment[key] != current:
                differences.append(key)
        recorded_python = config.get("runtime_info", {}).get("python_executable")
        if recorded_python is not None and recorded_python != sys.executable:
            differences.append("interpreter")
        if differences:
            print(
                "Replay uses the current trusted worker and environment; recorded "
                f"configuration differs in: {', '.join(differences)}. "
                "Restore saved cwd/environment only with an explicit "
                "--restore-environment for a trusted archive.",
                file=sys.stderr,
            )
    # Artifact directories can be downloaded or moved. Prefer the co-located
    # immutable case instead of trusting a stale absolute path from another host.
    case_path = config_path.parent / "case.json"
    if not case_path.is_file():
        case_path = Path(config["case_path"])
    case = SerializedCase(case_path.read_text())
    with IsolatedRunner(
        artifact_dir or config_path.parent / "replays",
        timeout_s=config["timeout_s"],
        reduce=config["reduce"],
        max_cases_per_worker=config.get("max_cases_per_worker", 64),
        env=config.get("environment") if restore_environment else None,
        cwd=Path(config["working_directory"]) if restore_environment else None,
    ) as runner:
        return runner.check(case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--restore-environment",
        action="store_true",
        help="Restore recorded cwd/environment; only use with trusted configs",
    )
    args = parser.parse_args()
    if args.worker:
        _worker()
    elif args.replay:
        result = replay(
            args.replay,
            artifact_dir=args.artifact_dir,
            restore_environment=args.restore_environment,
        )
        print(
            f"Passed replay in {result.elapsed_s:.3f}s; artifacts: {result.result_path.parent}"
        )
    else:
        parser.error("Specify --replay CONFIG")


if __name__ == "__main__":
    main()
