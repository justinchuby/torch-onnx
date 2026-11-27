"""Export-local adapters for downstream frameworks using patched torch.onnx.

Transformers installs hooks on PyTorch's private preparation and optimizer entry
points. The standalone exporter owns those stages, so forward their policies
when its actual custom translation table identifies a Transformers export.
No optional framework is imported, and no process-global policy is changed.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
from collections.abc import Callable, Generator
from typing import Any

_POST_DECOMPOSITION: contextvars.ContextVar[Callable[[Any], None] | None] = (
    contextvars.ContextVar("downstream_post_decomposition", default=None)
)
_SHOULD_FOLD: contextvars.ContextVar[Callable[[Any], bool | None] | None] = (
    contextvars.ContextVar("downstream_should_fold", default=None)
)


def _skip_resize(node: Any) -> bool | None:
    # Matches Transformers' optimizer patch without routing through optimize_ir,
    # whose approximate identity rewrites are deliberately not used here.
    return (
        False if node.op_type == "Resize" and node.domain in ("", "ai.onnx") else None
    )


@contextlib.contextmanager
def downstream_export(custom_translation_table: Any) -> Generator[None, None, None]:
    module = sys.modules.get("transformers.exporters.exporter_onnx")
    is_transformers = (
        module is not None
        and custom_translation_table is not None
        and custom_translation_table is getattr(module, "_ONNX_TRANSLATION_TABLE", None)
    )
    postprocess = (
        functools.partial(module.apply_fx_node_fixes, "onnx")
        if is_transformers
        else None
    )
    post_token = _POST_DECOMPOSITION.set(postprocess)
    fold_token = _SHOULD_FOLD.set(_skip_resize if is_transformers else None)
    try:
        yield
    finally:
        _SHOULD_FOLD.reset(fold_token)
        _POST_DECOMPOSITION.reset(post_token)


def postprocess_decomposed_graph(graph_module: Any) -> None:
    callback = _POST_DECOMPOSITION.get()
    if callback is not None:
        callback(graph_module)


def _default_fold_policy(_node: Any) -> None:
    return None


def constant_fold_policy() -> Callable[[Any], bool | None]:
    return _SHOULD_FOLD.get() or _default_fold_policy
