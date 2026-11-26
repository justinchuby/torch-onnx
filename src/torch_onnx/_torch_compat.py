"""Centralized PyTorch capability checks and release compatibility adapters.

Version information controls diagnostics only. Export remains available whenever
its required capabilities exist, including on releases outside the CI matrix.
"""

from __future__ import annotations

import contextlib
import importlib
import re
import warnings
from collections.abc import Iterator

# Keep this list in sync with the PyTorch compatibility CI matrix.
SUPPORTED_TORCH_VERSIONS = (
    "2.10.0",
    "2.11.0",
    "2.12.0",
    "2.12.1",
    "2.13.0",
    "2.14.0",
    "2.14.1",
    "2.15.0",
)
_warned = False


class TorchCompatibilityWarning(UserWarning):
    """The installed PyTorch lacks a capability or has not been validated."""


def torch_minor(version: str | None = None) -> tuple[int, int] | None:
    import torch

    match = re.match(
        r"^(\d+)\.(\d+)(?:\.|$)", str(torch.__version__) if version is None else version
    )
    return (int(match[1]), int(match[2])) if match else None


def warn_once(reason: str) -> None:
    """Emit at most one compatibility diagnostic per process, without rejecting export."""
    import torch

    global _warned  # noqa: PLW0603
    if _warned:
        return
    _warned = True
    versions = ", ".join(SUPPORTED_TORCH_VERSIONS)
    warnings.warn(
        f"torch-onnx: {reason} Installed PyTorch: {torch.__version__}. "
        f"Validated PyTorch versions: {versions}. "
        "Upgrade PyTorch to the newest validated version if export fails.",
        TorchCompatibilityWarning,
        stacklevel=3,
    )


def warn_if_unvalidated() -> None:
    """Run before importing modules that consume private PyTorch APIs."""
    import torch

    stable = re.fullmatch(
        r"\d+\.\d+\.\d+(?:\+[A-Za-z0-9_.-]+)?", str(torch.__version__)
    )
    if stable is None or torch_minor() not in {
        torch_minor(version) for version in SUPPORTED_TORCH_VERSIONS
    }:
        warn_once("This PyTorch release has not been validated with this exporter.")


def is_fake_tensor(value: object) -> bool:
    """Recognize fake-backed tensors using the API available in this build."""
    from torch._subclasses import fake_tensor

    check = getattr(fake_tensor, "is_fake_tensor", None)
    if check is None:
        check = fake_tensor.is_fake
    return check(value)


def _contains_rnn(model: object) -> bool:
    import torch

    if not isinstance(model, torch.nn.Module):
        return False
    targets = {"aten.lstm.input", "aten.lstm.data", "aten.gru.input", "aten.gru.data"}
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM | torch.nn.GRU):
            return True
        if isinstance(module, torch.fx.GraphModule) and any(
            node.op == "call_function" and str(node.target) in targets
            for node in module.graph.nodes
        ):
            return True
    return False


def _dynamic_rnn_helpers():
    try:
        patches = importlib.import_module("torch.export._patches")
    except ModuleNotFoundError as error:
        if error.name != "torch.export._patches":
            raise
        return None
    lstm = getattr(patches, "register_lstm_while_loop_decomposition", None)
    gru = getattr(patches, "register_gru_while_loop_decomposition", None)
    return (lstm, gru) if lstm is not None and gru is not None else None


@contextlib.contextmanager
def dynamic_rnn_decompositions(model: object = None) -> Iterator[None]:
    """Use available dynamic RNN helpers without blocking ordinary dynamic graphs."""
    helpers = _dynamic_rnn_helpers()
    if helpers is None:
        if _contains_rnn(model):
            warn_once(
                "Dynamic LSTM/GRU loop decomposition helpers are unavailable; "
                "sequence lengths may be specialized or rejected by torch.export."
            )
        yield
        return
    lstm, gru = helpers
    with lstm(), gru():
        yield


# CI reads the version tuple with runpy.run_path, without importing PyTorch or
# producing runtime diagnostics. Normal package import bootstraps diagnostics.
if __name__ == "torch_onnx._torch_compat":
    warn_if_unvalidated()
