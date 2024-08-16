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
    onnx_registered_ops = set(get_onnx_implemented_overloads(registry))

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


def valid_to_preserve(op_overload: torch._ops.OperatorBase) -> bool:
    # Adapted from https://github.com/pytorch/pytorch/blob/611c1043709dc04ed500c551aeb40f69e56a1a4f/torch/export/exported_program.py#L177
    # PyTorch License
    # TODO(justinchuby): Update when the source changes
    from torch._subclasses.functional_tensor import FunctionalTensor

    if op_overload in FunctionalTensor.maybe_aliasing_or_mutating_ops:
        return False
    if op_overload in FunctionalTensor.metadata_fns:
        return False

    if not isinstance(op_overload, torch._ops.OpOverload):
        return False

    alias_info = len(
        [i for i in op_overload._schema.arguments if i.alias_info is not None]
    )

    is_mutating_or_aliasing = alias_info != 0 or op_overload._schema.is_mutable

    if is_mutating_or_aliasing:
        return False

    if not torch._C._dispatch_has_kernel(op_overload.name()):
        return False

    return True
