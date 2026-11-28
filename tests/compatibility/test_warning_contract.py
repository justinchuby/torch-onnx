"""Compatibility diagnostics are explicit, once per process and non-gating."""

from __future__ import annotations

import warnings

import pytest
import torch

from torch_onnx import _torch_compat

pytestmark = pytest.mark.compatibility


def test_unvalidated_version_warns_once_and_suggests_upgrade(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.0.0")
    monkeypatch.setattr(_torch_compat, "_warned", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _torch_compat.warn_if_unvalidated()
        _torch_compat.warn_if_unvalidated()
        _torch_compat.warn_once("A second unavailable capability")
    assert len(captured) == 1
    assert issubclass(captured[0].category, _torch_compat.TorchCompatibilityWarning)
    message = str(captured[0].message)
    assert "2.0.0" in message and "upgrade" in message.lower()
    for version in _torch_compat.SUPPORTED_TORCH_VERSIONS:
        assert version in message


def test_other_patch_of_validated_minor_is_not_rejected(monkeypatch):
    minor = _torch_compat.SUPPORTED_TORCH_VERSIONS[0].rsplit(".", 1)[0]
    monkeypatch.setattr(torch, "__version__", minor + ".999")
    monkeypatch.setattr(_torch_compat, "_warned", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _torch_compat.warn_if_unvalidated()
    assert not captured


def test_missing_dynamic_rnn_helper_is_explicit_without_blocking_ordinary_models(
    monkeypatch,
):
    monkeypatch.setattr(_torch_compat, "_dynamic_rnn_helpers", lambda: None)
    monkeypatch.setattr(_torch_compat, "_warned", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with _torch_compat.dynamic_rnn_decompositions(torch.nn.Identity()):
            assert not captured
        for _ in range(2):
            with _torch_compat.dynamic_rnn_decompositions(torch.nn.LSTM(2, 2)):
                pass
    assert len(captured) == 1
    message = str(captured[0].message)
    assert "LSTM/GRU" in message and "unavailable" in message
    assert "upgrade" in message.lower()


@pytest.mark.parametrize("suffix", [".dev20261126", "a0", "b1", "rc1"])
def test_unvalidated_prerelease_of_a_supported_minor_warns(monkeypatch, suffix):
    monkeypatch.setattr(
        torch, "__version__", _torch_compat.SUPPORTED_TORCH_VERSIONS[-1] + suffix
    )
    monkeypatch.setattr(_torch_compat, "_warned", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _torch_compat.warn_if_unvalidated()
        _torch_compat.warn_if_unvalidated()
    assert len(captured) == 1
    assert issubclass(captured[0].category, _torch_compat.TorchCompatibilityWarning)


@pytest.mark.parametrize("suffix", ["+cpu", "+cu128"])
def test_supported_stable_local_build_does_not_warn(monkeypatch, suffix):
    monkeypatch.setattr(
        torch, "__version__", _torch_compat.SUPPORTED_TORCH_VERSIONS[-1] + suffix
    )
    monkeypatch.setattr(_torch_compat, "_warned", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _torch_compat.warn_if_unvalidated()
    assert not captured
