"""Module for handling ATen to ONNX functions registration.

https://github.com/pytorch/pytorch/blob/6aa5bb1a76dee8112f1a9e7c194c790b5cdc6462/torch/onnx/_internal/fx/registration.py
"""

# NOTE: Why do we need a different registry than the one in torchlib?
# The registry in torchlib is used to register functions that are already implemented in
# torchlib, and is designed to be a static singleton. It does not take into account custom ops or different
# opsets etc. The registry implemented for the exporter is designed to be modifiable at
# export time by users, and is designed with dispatching in mind.

from __future__ import annotations

import dataclasses
import math
import types
from typing import Callable, Mapping, TypeAlias, Union
import operator

import onnxscript
import torch
import torch._ops
from onnxscript.function_libs.torch_lib import (
    registration as torchlib_registration,
)

from torch_onnx import _schemas

_DEFAULT_OPSET_VERSION = 18


TorchOp: TypeAlias = Union[torch._ops.OpOverload, types.BuiltinFunctionType, Callable]


@dataclasses.dataclass(frozen=True)
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    signature: The signature of the function.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.

    """

    onnx_function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction
    fx_target: TorchOp
    is_custom: bool = False
    is_complex: bool = False
    device: str | None = None


def _get_overload(qualified_name: str) -> torch._ops.OpOverload:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""
    namespace, opname_overload = qualified_name.split("::")
    op_name, *overload = opname_overload.split(".", 1)
    if namespace == "_operator":
        # Builtin functions
        return getattr(operator, op_name)
    if namespace == "math":
        return getattr(math, op_name)
    if namespace == "torchvision":
        import torchvision.ops
        return getattr(torchvision.ops, op_name)

    op_packet = getattr(getattr(torch.ops, namespace), op_name)
    if overload:
        overload = overload[0]
    elif "default" in op_packet._overload_names:
            overload = "default"
    else:
        # Use the first overload as the default overload
        overload = op_packet._overload_names[0]

    return getattr(op_packet, overload)


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""

        # TODO: Design multi-opset version support
        self._opset_version = _DEFAULT_OPSET_VERSION

        self.functions: dict[TorchOp, list[OnnxDecompMeta]] = {}

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target.

        Defaults to the latest supported ONNX opset version: 18.
        The default version will increment over time as ONNX continues to evolve.
        """

        return self._opset_version

    @classmethod
    def from_torchlib(
        cls, torchlib_registry: Mapping[str, torchlib_registration.OverloadedFunction]
    ):
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        registry = cls()
        for qualified_name, aten_overloads_func in torchlib_registry.items():
            if qualified_name.startswith("internal::"):
                # Skip the custom defined internal functions
                continue
            target = _get_overload(qualified_name)
            for overload_func in aten_overloads_func.overloads:
                overload_func.signature = _schemas.OpSignature.from_function(
                    overload_func, overload_func.function_ir.domain, overload_func.name
                )
                onnx_decomposition = OnnxDecompMeta(
                    onnx_function=overload_func,
                    fx_target=target,
                    is_custom=False,
                    is_complex=False,
                )
                registry._register(target, onnx_decomposition)

            for complex_func in aten_overloads_func.complex:
                overload_func.signature = _schemas.OpSignature.from_function(
                    overload_func, overload_func.function_ir.domain, overload_func.name
                )
                onnx_decomposition = OnnxDecompMeta(
                    onnx_function=complex_func,
                    fx_target=target,
                    is_custom=False,
                    is_complex=True,
                )
                registry._register(target, onnx_decomposition)
        return registry

    def _register(
        self,
        target: TorchOp,
        onnx_decomposition: OnnxDecompMeta,
    ) -> None:
        """Registers a OnnxDecompMeta to an operator.

        Args:
            target: The PyTorch node callable target.
            onnx_decomposition: The OnnxDecompMeta to register.
        """
        if onnx_decomposition.is_custom:
            self.functions.setdefault(target, []).insert(0, onnx_decomposition)
        else:
            self.functions.setdefault(target, []).append(onnx_decomposition)

    def register_op(
        self,
        target: TorchOp,
        function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-script function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
            is_complex: Whether the function is a function that handles complex valued inputs.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
        onnx_decomposition = OnnxDecompMeta(
            onnx_function=function,
            fx_target=target,
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(target, onnx_decomposition)

    def get_decomps(self, target: TorchOp) -> list[OnnxDecompMeta]:
        """Returns a list of OnnxDecompMeta for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of OnnxDecompMeta corresponding to the given name, or None if
            the name is not in the registry.
        """
        return self.functions.get(target, [])

    def is_registered(self, target: TorchOp) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        return bool(self.get_decomps(target))
