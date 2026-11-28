"""Keep the public-API compatibility suite independent of private test imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    # Marker deselection normally happens *after* importing every test module.
    # Older PyTorch installations cannot import the baseline's private test APIs.
    # Only the dedicated compatibility selection needs pre-import filtering.
    if (
        config.getoption("markexpr") != "compatibility"
        or collection_path.suffix != ".py"
    ):
        return None
    try:
        tree = ast.parse(collection_path.read_text(encoding="utf-8"))
    except SyntaxError:
        # Let pytest report malformed modules rather than hiding them.
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "compatibility"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest"
        ):
            return None
    return True
