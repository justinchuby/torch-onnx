# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Export-time optimization with exact arithmetic identity matching."""

from __future__ import annotations

import math
from typing import Any

import onnx_ir as ir


def _scalar(value: ir.Value, expected: int) -> bool:
    tensor = value.const_value
    if tensor is None or len(tensor.shape) != 0:
        return False
    try:
        return bool(tensor.numpy().item() == expected)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError):
        return False


def _numeric(value: ir.Value) -> bool:
    return value.dtype is not None and (
        value.dtype.is_integer() or value.dtype.is_floating_point()
    )


def _add(op: Any, x: Any, c: Any) -> Any:
    return op.Add(x, c)


def _sub(op: Any, x: Any, c: Any) -> Any:
    return op.Sub(x, c)


def _mul(op: Any, x: Any, c: Any) -> Any:
    return op.Mul(x, c)


def _div(op: Any, x: Any, c: Any) -> Any:
    return op.Div(x, c)


def _identity(op: Any, x: Any, **_: Any) -> Any:
    return op.Identity(x)


def _add_zero(context: Any, x: ir.Value, c: ir.Value) -> bool:
    # Floating x + +0 changes -0 to +0. Keep it even for an exact zero.
    return x.dtype is not None and x.dtype.is_integer() and _scalar(c, 0)


def _sub_zero(context: Any, x: ir.Value, c: ir.Value) -> bool:
    if not _numeric(x) or not _scalar(c, 0):
        return False
    dtype = x.dtype
    tensor = c.const_value
    if dtype is None or tensor is None:
        return False
    if dtype.is_integer():
        return True
    # x - -0 has the same signed-zero problem as x + +0.
    return math.copysign(1.0, float(tensor.numpy().item())) > 0


def _one(context: Any, x: ir.Value, c: ir.Value) -> bool:
    return _numeric(x) and _scalar(c, 1)


def _rules() -> Any:
    from onnxscript import rewriter
    from onnxscript.rewriter.rules.common import _no_op

    # Reuse the upstream rule suite without mutating its process-global policy.
    # The no-op rules use tolerant literal matching, which can erase a nonzero
    # epsilon or a near-one scale. Replace them with exact, dtype-aware rules.
    excluded = {id(rule) for rule in _no_op.rules}
    rules = [
        rule for rule in rewriter._DEFAULT_REWRITE_RULES if id(rule) not in excluded
    ]
    add = rewriter.RewriteRule(_add, _identity, _add_zero, name="ExactIntegerAddZero")
    mul = rewriter.RewriteRule(_mul, _identity, _one, name="ExactMultiplyOne")
    rules.extend(add.commute())
    rules.extend(mul.commute())
    rules.append(
        rewriter.RewriteRule(_sub, _identity, _sub_zero, name="ExactSubtractZero")
    )
    rules.append(rewriter.RewriteRule(_div, _identity, _one, name="ExactDivideOne"))
    return rules


def optimize(model: ir.Model) -> ir.Model:
    """Use upstream constant/structural optimization with safe no-op rewrites."""
    from onnxscript import optimizer, rewriter
    from onnxscript.rewriter import onnx_fusions

    common = ir.passes.common
    common.InlinePass()(model)
    rules = _rules()
    for _ in range(2):
        optimizer.fold_constants_ir(model, onnx_shape_inference=True)
        rewriter.rewrite(model, rules)
    ir.passes.Sequential(
        common.RemoveUnusedNodesPass(),
        common.RemoveUnusedFunctionsPass(),
        common.RemoveUnusedOpsetsPass(),
        common.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0),
        common.LiftSubgraphInitializersToMainGraphPass(),
        common.DeduplicateInitializersPass(),
        common.CommonSubexpressionEliminationPass(),
        common.OutputFixPass(),
        common.NameFixPass(),
    )(model)
    onnx_fusions.fuse(model)
    return model
