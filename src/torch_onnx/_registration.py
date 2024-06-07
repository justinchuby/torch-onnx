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
import types
from collections import defaultdict
from typing import Callable, List, Mapping, Optional, TypeAlias, Union

import onnxscript
import torch._ops
import torch
from onnxscript.function_libs.torch_lib import (
    registration as torchlib_registration,
)

from torch_onnx import _schemas

_DEFAULT_OPSET_VERSION = 18


TorchOp: TypeAlias = Union[torch._ops.OpOverload, types.BuiltinFunctionType]

@dataclasses.dataclass(frozen=True)
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    signature: The signature of the function.
    op_id: The identifier of the PyTorch operator.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.

    """

    onnx_function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction
    op: TorchOp
    is_custom: bool = False
    is_complex: bool = False


def _get_overload(qualified_name: str) -> torch._ops.OpOverload:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""
    namespace, opname_overload = qualified_name.split("::")
    op_name, *overload = opname_overload.split(".", 1)
    overload = overload[0] if overload else "default"
    if namespace == "torchvision":
        import torchvision.ops
        return getattr(torchvision.ops, op_name)
    # TODO: Builtin functions
    return getattr(getattr(getattr(torch.ops, namespace), op_name), overload)

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

        self._functions: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}
        self._complex: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}
        self._customs: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}

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
        for aten_name, aten_overloads_func in torchlib_registry.items():
            op_id = TorchOpIdentifier.from_qualified_name(aten_name)
            for overload_func in aten_overloads_func.overloads:
                overload_func.signature = _schemas.OpSignature.from_function(
                    overload_func, overload_func.function_ir.domain, overload_func.name
                )
                onnx_decomposition = OnnxDecompMeta(
                    onnx_function=overload_func,
                    op_id=op_id,
                    is_custom=False,
                    is_complex=False,
                )
                registry._register(op_id, onnx_decomposition)

            for complex_func in aten_overloads_func.complex:
                overload_func.signature = _schemas.OpSignature.from_function(
                    overload_func, overload_func.function_ir.domain, overload_func.name
                )
                onnx_decomposition = OnnxDecompMeta(
                    onnx_function=complex_func,
                    op_id=op_id,
                    is_custom=False,
                    is_complex=True,
                )
                registry._register(op_id, onnx_decomposition)
        return registry

    def _register(
        self,
        target: TorchOpIdentifier,
        onnx_decomposition: OnnxDecompMeta,
    ) -> None:
        """Registers a OnnxDecompMeta to an operator.

        Args:
            op_identifier: The qualified name of the operator to register: OpName.
            onnx_decomposition: The OnnxDecompMeta to register.
        """
        if onnx_decomposition.is_complex:
            self._complex[op_identifier].append(onnx_decomposition)
        elif onnx_decomposition.is_custom:
            self._customs[op_identifier].append(onnx_decomposition)
        else:
            self._functions[op_identifier].append(onnx_decomposition)

    def register_op(
        self,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        target: Callable,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
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
            target=target,
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(target, onnx_decomposition)

    def get_op_functions(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[List[OnnxDecompMeta]]:
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
        op_id = TorchOpIdentifier.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return self._registry.get(op_id)

    def is_registered_op(
        self, namespace: str, op_name: str, overload: str | None = None
    ) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_op_functions(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return functions is not None
