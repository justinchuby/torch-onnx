"""Process isolation and lazy import contracts across platform paths."""

import importlib
import multiprocessing
import os
from unittest import mock

import pytest

from torch_onnx import _isolated, _lazy_import


def _raise_in_worker():
    raise ValueError("worker failure")


def _terminate_worker():
    os._exit(17)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX fork path; Windows fallback is tested separately"
)
def test_isolation_executes_in_a_different_process_and_propagates_errors():
    assert _isolated.safe_call(os.getpid) != os.getpid()
    with pytest.raises(ValueError, match="worker failure"):
        _isolated.safe_call(_raise_in_worker)
    assert _isolated.safe_call(pow, 2, exp=3) == 8


@pytest.mark.skipif(os.name == "nt", reason="POSIX worker termination")
def test_worker_termination_times_out_without_terminating_the_parent():
    with pytest.raises(multiprocessing.TimeoutError):
        _isolated.safe_call(_terminate_worker)
    assert _isolated.safe_call(os.getpid) != os.getpid()


def test_windows_fallback_warns_and_preserves_exceptions(monkeypatch):
    monkeypatch.setattr(_isolated, "_IS_WINDOWS", True)
    with pytest.warns(UserWarning, match="Windows"):
        assert _isolated.safe_call(pow, 2, exp=3) == 8
    with (
        pytest.warns(UserWarning, match="Windows"),
        pytest.raises(ValueError, match="worker failure"),
    ):
        _isolated.safe_call(_raise_in_worker)


def test_worker_wrapper_preserves_an_exception_as_its_result():
    result = _isolated._call_function_and_return_exception(_raise_in_worker, (), {})
    assert isinstance(result, ValueError)
    assert str(result) == "worker failure"
    assert _isolated._call_function_and_return_exception(pow, (2,), {"exp": 3}) == 8


def test_lazy_import_repr_does_not_import_and_attribute_access_is_cached():
    module = _lazy_import._LazyModule("math")
    with mock.patch.object(
        _lazy_import.importlib, "import_module", wraps=importlib.import_module
    ) as imported:
        assert repr(module) == "<lazy module 'math'>"
        imported.assert_not_called()
        assert module.sqrt(9) == 3
        assert module.sqrt(16) == 4
        imported.assert_called_once()
        with pytest.raises(AttributeError):
            _ = module.no_such_attribute


def test_lazy_import_does_not_hide_a_missing_dependency():
    module = _lazy_import._LazyModule("no_such_exporter_dependency")
    with pytest.raises(ModuleNotFoundError, match="no_such_exporter_dependency"):
        _ = module.value
