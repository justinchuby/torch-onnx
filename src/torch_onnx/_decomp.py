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


def full_decomposition_table() -> dict[torch._ops.OperatorBase, Callable]:
    """Return the full decomposition table."""
    # NOTE: If we import torch._decomp, we will get RuntimeError: Only a single
    # TORCH_LIBRARY can be used to register the namespace nvprims; please put all of your
    # definitions in a single TORCH_LIBRARY block.
    return torch._decomp.decomposition_table
