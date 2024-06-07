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
from typing import List, Mapping, Optional, Union

import onnxscript
import torch._ops
from onnxscript.function_libs.torch_lib import (
    registration as torchlib_registration,
)

from torch_onnx import _schemas

_DEFAULT_OPSET_VERSION = 18


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
    op_id: TorchOpIdentifier
    is_custom: bool = False
    is_complex: bool = False


@dataclasses.dataclass(frozen=True, eq=True)
class TorchOpIdentifier:
    """A class representing an operator in the ONNX converter.

    Note that this is not the same thing as the ONNX op identifier. A Torch op
    ``aten.add.Tensor`` have namespace=aten, op_name=add, and overload=Tensor, but
    it may correspond to an ONNX function that has domain="pkg.onnxscript.torchlib`,
    name="aten_add_Tensor", for example.
    """

    namespace: str
    op_name: str
    overload: str

    @classmethod
    def from_name_parts(
        cls, namespace: str, op_name: str, overload: str | None = None
    ) -> TorchOpIdentifier:
        # NOTE: in PyTorch, the overload could be unprovided to indicate the
        # default overload
        if overload is None or overload == "":
            overload = "default"
        return cls(namespace, op_name, overload)

    @classmethod
    def from_qualified_name(cls, qualified_name: str) -> TorchOpIdentifier:
        """Create TorchOpIdentifier from <namespace>::<op_name>[.<overload>]"""
        namespace, opname_overload = qualified_name.split("::")
        op_name, *overload = opname_overload.split(".", 1)
        overload = overload[0] if overload else "default"
        return cls(namespace, op_name, overload)

    @classmethod
    def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> TorchOpIdentifier:
        return cls.from_qualified_name(op_overload.name())

    @classmethod
    def from_builtin_function(
        cls, builtin_function: types.BuiltinFunctionType
    ) -> TorchOpIdentifier:
        """From a builtin function, e.g. operator.add, math.ceil, etc, get the OpName.

        FX graph uses built-in functions to caculate sympy expression. This function
        is used to get the OpName from a builtin function.

        Args:
            builtin_function (types.BuiltinFunctionType): operator.add, math.ceil, etc.

        Returns:
            OpName: _description_
        """
        op = builtin_function.__name__  # add, sub, etc.
        module = builtin_function.__module__  # _operators or math
        return cls.from_name_parts(module, op)

    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.op_name}.{self.overload}"


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""

        self._functions: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}
        self._complex: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}
        self._customs: dict[TorchOpIdentifier, list[OnnxDecompMeta]] = {}

        # TODO: Design multi-opset version support
        self._opset_version = _DEFAULT_OPSET_VERSION

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target.

        Defaults to the latest supported ONNX opset version: 18.
        The default version will increment over time as ONNX continues to evolve.
        """

        return self._opset_version

    def from_torchlib(
        self, torchlib_registry: Mapping[str, torchlib_registration.OverloadedFunction]
    ):
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
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
                self._register(op_id, onnx_decomposition)

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
                self._register(op_id, onnx_decomposition)

    def _register(
        self,
        internal_qualified_name: TorchOpIdentifier,
        onnx_decomposition: OnnxDecompMeta,
    ) -> None:
        """Registers a ONNXFunction to an operator.

        Args:
            internal_qualified_name: The qualified name of the operator to register: OpName.
            onnx_decomposition: The ONNXFunction to register.
        """
        self._registry[internal_qualified_name].append(onnx_decomposition)

    def register_op(
        self,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        namespace: str,
        op_name: str,
        overload: Optional[str] = None,
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
        op_id = TorchOpIdentifier.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        onnx_decomposition = OnnxDecompMeta(
            onnx_function=function,
            op_full_name=op_id.qualified_name(),
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(op_id, onnx_decomposition)

    def get_op_functions(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[List[OnnxDecompMeta]]:
        """Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of ONNXFunctions corresponding to the given name, or None if
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

    def _all_registered_ops(self) -> set[str]:
        """Returns the set of all registered function names."""
        return {
            op_name_class.qualified_name() for op_name_class in self._registry.keys()
        }
