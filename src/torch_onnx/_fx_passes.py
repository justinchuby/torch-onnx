from __future__ import annotations

import torch
import torch.export
from torch.onnx._internal.fx import diagnostics, passes

_ATEN_ASSERTION_TARGETS = frozenset(
    {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
    }
)


def insert_type_promotion_nodes(exported_program: torch.export.ExportedProgram) -> None:
    """Inplace pass to insert explicit type promotion nodes."""
    diagnostic_context = diagnostics.DiagnosticContext(
        "torch.onnx.export",
        torch.__version__,
    )
    passes.InsertTypePromotion(diagnostic_context, exported_program.graph_module).run()


def remove_assertion_nodes(exported_program: torch.export.ExportedProgram) -> None:
    """Remove all assertion and check nodes from the FX graph"""
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and node.target in _ATEN_ASSERTION_TARGETS:
            exported_program.graph.erase_node(node)
