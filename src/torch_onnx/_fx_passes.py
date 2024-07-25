from __future__ import annotations

import torch
import torch.export
from torch.onnx._internal.fx import diagnostics, passes
import torch.fx

_ATEN_ASSERTION_TARGETS = frozenset(
    {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
    }
)


def insert_type_promotion_nodes(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Inplace pass to insert explicit type promotion nodes."""
    diagnostic_context = diagnostics.DiagnosticContext(
        "torch.onnx.export",
        torch.__version__,
    )
    return passes.InsertTypePromotion(diagnostic_context, graph_module).run()


def remove_assertion_nodes(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove all assertion and check nodes from the FX graph"""
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target in _ATEN_ASSERTION_TARGETS:
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    return graph_module
