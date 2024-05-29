from __future__ import annotations


import dataclasses
from typing import Any, Iterator, Mapping, Sequence

from onnxscript import ir
import onnx

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


@dataclasses.dataclass
class OpSchema:
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

    def __post_init__(self):
        self._params_map = {param.name: param for param in self.params}

    def get(self, name: str) -> Parameter | AttributeParameter:
        return self._params_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._params_map

    def __iter__(self) -> Iterator[Parameter | AttributeParameter]:
        return iter(self.params)

    @classmethod
    def from_opschema(cls, opschema: onnx.defs.OpSchema) -> OpSchema:
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
