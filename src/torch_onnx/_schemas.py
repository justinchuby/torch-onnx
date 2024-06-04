from __future__ import annotations


import inspect
import dataclasses
from typing import Any, Callable, Iterator, Mapping, Sequence, Type

from onnxscript import ir
from onnxscript.ir import convenience as ir_convenience
import onnx
import logging
import typing

logger = logging.getLogger(__name__)

# A special value to indicate that the default value is not specified
_EMPTY_DEFAULT = object()


@dataclasses.dataclass(frozen=True)
class TypeConstraintParam:
    """Type constraint for a parameter.

    Attributes:
        name: Name of the parameter. E.g. "TFloat"
        allowed_types: Allowed types for the parameter.
    """

    name: str
    allowed_types: set[ir.TypeProtocol]
    description: str = ""


@dataclasses.dataclass(frozen=True)
class Parameter:
    """A formal parameter of an operator."""

    name: str
    type_constraint: TypeConstraintParam
    required: bool
    variadic: bool
    default: Any = _EMPTY_DEFAULT
    # TODO: Add other properties too

    def has_default(self) -> bool:
        return self.default is not _EMPTY_DEFAULT


@dataclasses.dataclass(frozen=True)
class AttributeParameter:
    name: str
    type: ir.AttributeType
    required: bool
    default: ir.Attr | None = None


def _get_type_from_str(
    type_str: str,
) -> ir.TensorType | ir.SequenceType | ir.OptionalType:
    """Converter a type_str from ONNX Opschema to ir.TypeProtocol.

    A type str has the form of "tensor(float)" or composite type like "seq(tensor(float))".
    """

    # TODO: Upstream this to IR

    # Split the type_str a sequence types and dtypes
    # 1. Remove the ending ")"
    type_str = type_str.rstrip(")")
    # 2. Split the type_str by "("
    type_parts = type_str.split("(")

    # Convert the dtype to ir.DataType
    dtype = ir.DataType[type_parts[-1].upper()]

    # Create a place holder type first
    type_: ir.TypeProtocol = ir.TensorType(ir.DataType.UNDEFINED)

    # Construct the type
    for type_part in reversed(type_parts[:-1]):
        if type_part == "tensor":
            type_ = ir.TensorType(dtype)
        if type_part == "seq":
            type_ = ir.SequenceType(type_)
        if type_part == "optional":
            type_ = ir.OptionalType(type_)
        else:
            raise ValueError(f"Unknown type part: '{type_part}' in type '{type_str}'")
    return type_


def _convert_formal_parameter(
    param: onnx.defs.OpSchema.FormalParameter,
    type_constraints: Mapping[str, TypeConstraintParam],
) -> Parameter:
    """Convert a formal parameter from ONNX Opschema to Parameter."""
    if param.type_str in type_constraints:
        type_constraint = type_constraints[param.type_str]
    else:
        # param.type_str can be a plain type like 'int64'.
        type_constraint = TypeConstraintParam(
            name=param.name,
            allowed_types={_get_type_from_str(param.type_str)},  # type: ignore
        )
    return Parameter(
        name=param.name,
        type_constraint=type_constraint,
        required=param.option != onnx.defs.OpSchema.FormalParameterOption.Optional,
        variadic=param.option == onnx.defs.OpSchema.FormalParameterOption.Variadic,
    )


def _get_attr_type(type_: Type) -> ir.AttributeType:
    """Obtain the type of the attribute from a Python class."""
    try:
        if type_ is int:
            return ir.AttributeType.INT
        if type_ is float:
            return ir.AttributeType.FLOAT
        if type_ is str:
            return ir.AttributeType.STRING
        if type_ is bool:
            return ir.AttributeType.INT
        if type_ is ir.TensorProtocol:
            return ir.AttributeType.TENSOR
        if issubclass(type_, ir.Tensor):
            return ir.AttributeType.TENSOR
        if issubclass(type_, Sequence):
            if typing.get_args(type_) == (int,):
                return ir.AttributeType.INTS
            if typing.get_args(type_) == (float,):
                return ir.AttributeType.FLOATS
            if typing.get_args(type_) == (str,):
                return ir.AttributeType.STRINGS
            if typing.get_args(type_) == (bool,):
                return ir.AttributeType.INTS
            if typing.get_args(type_) == (ir.Tensor,) or typing.get_args(type_) == (ir.TensorProtocol,):
                return ir.AttributeType.TENSORS
    except TypeError:
        logger.warning("TypeError when checking %s.", type_, exc_info=True)
    return ir.AttributeType.UNDEFINED

@dataclasses.dataclass
class OpSignature:
    """Schema for an operator.

    Attributes:
        domain: Domain of the operator. E.g. "".
        name: Name of the operator. E.g. "Add".
        overload: Overload name of the operator.
        params: Input parameters. When the op is an ONNX function definition,
          the order is according to the function signature. This mean we can
          interleave ONNX inputs and ONNX attributes in the list.
        outputs: Output parameters.
    """

    domain: str
    name: str
    overload: str
    params: Sequence[Parameter | AttributeParameter]
    outputs: Sequence[Parameter]
    params_map: Mapping[str, Parameter | AttributeParameter] = dataclasses.field(
        init=False, repr=False
    )

    def __post_init__(self):
        self.params_map = {param.name: param for param in self.params}

    def get(self, name: str) -> Parameter | AttributeParameter:
        return self.params_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self.params_map

    def __iter__(self) -> Iterator[Parameter | AttributeParameter]:
        return iter(self.params)

    @classmethod
    def from_opschema(cls, opschema: onnx.defs.OpSchema) -> OpSignature:
        """Produce an OpSignature from an ONNX Opschema."""
        type_constraints = {
            constraint.type_param_str: TypeConstraintParam(
                name=constraint.type_param_str,
                allowed_types={
                    _get_type_from_str(type_str)
                    for type_str in constraint.allowed_type_strs
                },
                description=constraint.description,
            )
            for constraint in opschema.type_constraints
        }

        params = [
            _convert_formal_parameter(param, type_constraints)
            for param in opschema.inputs
        ]

        for param in opschema.attributes.values():
            params.append(
                AttributeParameter(
                    name=param.name,
                    type=ir.AttributeType(param.type),
                    required=param.required,
                    default=ir.serde.deserialize_attribute(param.default_value)
                    if param.default_value is not None
                    else None,  # type: ignore
                )
            )

        outputs = [
            _convert_formal_parameter(param, type_constraints)
            for param in opschema.outputs
        ]

        return cls(
            domain=opschema.domain,
            name=opschema.name,
            overload="",
            params=params,
            outputs=outputs,
        )

    @classmethod
    def from_function(cls, func) -> OpSignature:
        """Produce an OpSignature from a function using type annotation."""

        signature = inspect.signature(func)
        # Not using inspect.get_annotations because typing.get_type_hints seems to handle more cases
        # https://github.com/python/cpython/issues/102405
        type_hints = typing.get_type_hints(func)

        params = []
        type_constraints = {}

        for param in signature.parameters.values():
            if param.name not in type_hints:
                logger.warning(f"Missing annotation for parameter '{param.name}'. Treating as an Input.")
            else:
                type_ = type_hints[param.name]
                if (attr_type := _get_attr_type(type_)) != ir.AttributeType.UNDEFINED:
                    params.append(
                        AttributeParameter(
                            name=param.name,
                            type=attr_type,
                            required=param.default is inspect.Parameter.empty,
                            default=param.default,
                        )
                    )
                else:
                    # Obtain the type constraint from the type annotation
                    # Handle Sequence[Tensor] and Optional[Tensor] as well
                    # TODO(justinchuby): Pick up from here