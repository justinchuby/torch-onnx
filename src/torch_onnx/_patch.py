"""Opt-in patching of the modern torch.onnx.export API."""

from __future__ import annotations

import functools
import os

import torch

from torch_onnx._compat import export_compat

_original_export = torch.onnx.export
_original_is_in_export = torch.onnx.is_in_onnx_export


def patch_torch(
    *,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    optimize: bool = True,
    verbose: bool | None = None,
) -> None:
    """Route torch.onnx.export to this package, with optional export defaults.

    Only the modern exporter is supported. Call unpatch_torch to restore PyTorch.
    """
    from torch_onnx import export, is_in_onnx_export

    patched_export = functools.partial(
        export,
        report=report,
        verify=verify,
        profile=profile,
        dump_exported_program=dump_exported_program,
        artifacts_dir=artifacts_dir,
        optimize=optimize,
        verbose=verbose,
    )
    torch.onnx.__dict__["export"] = patched_export
    torch.onnx.is_in_onnx_export = is_in_onnx_export


def unpatch_torch() -> None:
    """Restore the original PyTorch exporter and export-state query."""
    torch.onnx.export = _original_export
    torch.onnx.is_in_onnx_export = _original_is_in_export


_torch_onnx_export = export_compat
