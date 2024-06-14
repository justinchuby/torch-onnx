"""Build decomp table from PyTorch."""

from __future__ import annotations
import torch
import torch._ops

from torch_onnx import _registration

from typing import Callable


def _get_registered_ops(
    registry: _registration.OnnxRegistry,
) -> set[torch._ops.OperatorBase | Callable]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: set[torch._ops.OperatorBase | Callable] = set()
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        op_names = dir(op_namespace)
        for op_name in op_names:
            op_overload_packet = getattr(op_namespace, op_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                if registry.is_registered(op_overload):
                    registered_ops.add(op_overload)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    registry,
) -> dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[torch._ops.OperatorBase, Callable] = {}
    onnx_registered_ops = _get_registered_ops(registry)

    # NOTE: If we import torch._decomp, we will get RuntimeError: Only a single
    # TORCH_LIBRARY can be used to register the namespace nvprims; please put all of your
    # definitions in a single TORCH_LIBRARY block.
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():  # type: ignore[attr-defined]
        # Skip decomposition into "prim::*" ops (defined in 'torch._refs'), because they
        # are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        if "torch._refs" in decomp_fn.__module__ or op_overload in onnx_registered_ops:
            continue
        decomposition_table[op_overload] = decomp_fn

    # NOTE: There are ops in core ATen and under torch._refs,
    # that are not decomposed to prim::ops. We need to pick them
    # back
    for op_overload, decomp_fn in torch._decomp.core_aten_decompositions().items():
        if op_overload in onnx_registered_ops:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
