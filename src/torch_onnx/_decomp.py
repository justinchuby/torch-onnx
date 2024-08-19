"""Build decomp table from PyTorch."""
# mypy: allow-untyped-defs

from __future__ import annotations

from typing import Callable

import torch
import torch._ops

from torch_onnx import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[torch._ops.OperatorBase]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[torch._ops.OperatorBase] = []
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        op_names = dir(op_namespace)
        for op_name in op_names:
            op_overload_packet = getattr(op_namespace, op_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                if registry.is_registered(op_overload):
                    registered_ops.append(op_overload)
    return registered_ops


def get_preserve_ops() -> set[torch._ops.OpOverload]:
    """Return a set of CompositeImplicitAutograd ops that should be preserved."""
    aten = torch.ops.aten
    # NOTE: Keep this list sorted
    # NOTE: Do _not_ retain aten.linear as its decomposition is addmm, which is Gemm and is preferable for accuracy
    return {
        aten._upsample_bilinear2d_aa.default,
        aten._upsample_nearest_exact1d.vec,
        aten._upsample_nearest_exact2d.vec,
        aten._upsample_nearest_exact3d.vec,
        aten.group_norm.default,
        aten.instance_norm.default,
        aten.upsample_bilinear2d.default,
        aten.upsample_bilinear2d.vec,
        aten.upsample_linear1d.default,
        aten.upsample_linear1d.vec,
        aten.upsample_nearest1d.default,
        aten.upsample_nearest1d.vec,
        aten.upsample_nearest2d.default,
        aten.upsample_nearest2d.vec,
        aten.upsample_nearest3d.default,
        aten.upsample_nearest3d.vec,
        aten.upsample_trilinear3d.default,
        aten.upsample_trilinear3d.vec,
    }


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[torch._ops.OperatorBase],
) -> dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[torch._ops.OperatorBase, Callable] = {}

    # NOTE: If we import torch._decomp, we will get RuntimeError: Only a single
    # TORCH_LIBRARY can be used to register the namespace nvprims; please put all of your
    # definitions in a single TORCH_LIBRARY block.
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():  # type: ignore[attr-defined]
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip torch._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        decomposition_table[op_overload] = decomp_fn

    return decomposition_table
