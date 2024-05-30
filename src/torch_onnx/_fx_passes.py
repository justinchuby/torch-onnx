import torch
import torch.export
from torch.onnx._internal.fx import diagnostics, passes


def insert_type_promotion_nodes(exported_program: torch.export.ExportedProgram):
    """Inplace pass to insert explicit type promotion nodes."""
    diagnostic_context = diagnostics.DiagnosticContext(
        "torch.onnx.export",
        torch.__version__,
    )
    passes.InsertTypePromotion(diagnostic_context, exported_program.graph_module).run()
