from __future__ import annotations

import packaging.version
import torch
import torch.export
import torch.fx
from torch.onnx._internal.fx import diagnostics, passes

from torch_onnx import _decomp, _registration


def _torch_older_than(version: str) -> bool:
    """Returns True if the torch version is older than the given version."""
    import torch  # pylint: disable=import-outside-toplevel

    return (
        packaging.version.parse(torch.__version__).release
        < packaging.version.parse(version).release
    )


# The _preserve_ops argument is only available in torch>=2.5
_TORCH_EXPORT_HAS_PRESERVE_OPS_PARAM = not _torch_older_than("2.5")


def decompose_with_registry(
    exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> torch.export.ExportedProgram:
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
    onnx_registered_ops = set(_decomp.get_onnx_implemented_overloads(registry))
    decomp_table = _decomp.create_onnx_friendly_decomposition_table(onnx_registered_ops)
    if not _TORCH_EXPORT_HAS_PRESERVE_OPS_PARAM:
        # torch 2.4 or older
        return exported_program.run_decompositions(decomp_table)

    # Try to preserve some known CompositeImplicitAutograd ops
    to_preserve = _decomp.get_preserve_ops()
    # We can only preserve implemented ops
    can_preserve = tuple(to_preserve.intersection(onnx_registered_ops))
    return exported_program.run_decompositions(decomp_table, _preserve_ops=can_preserve)


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
    aten_assertion_targets = {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
    }
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target in aten_assertion_targets:
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    return graph_module
