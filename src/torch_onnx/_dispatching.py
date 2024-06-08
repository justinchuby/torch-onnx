from __future__ import annotations

from typing import Sequence

import onnxscript
import torch
import torch.fx
from onnxscript import ir

from torch_onnx import _schemas, _registration

# Define utilities to convert PyTorch data types so users do not need to specify manually
_TORCH_DTYPE_TO_ONNX_COMPATIBLE: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.DOUBLE,
    torch.complex64: ir.DataType.FLOAT,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
}


def _torch_dtype_to_onnx_compatible_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX_COMPATIBLE[dtype]


def _attribute_type_compatible_with_arg(
    attr: _schemas.AttributeParameter,
    value: ir.Value | int | float | bool | Sequence[int] | Sequence[float],
) -> bool:
    """Check if the attribute type is compatible with the argument."""
    if isinstance(value, bool):
        return attr.type is ir.AttributeType.INT
    if isinstance(value, str):
        return attr.type is ir.AttributeType.STRING
    if isinstance(value, int):
        return attr.type is ir.AttributeType.INT
    if isinstance(value, float):
        return attr.type is ir.AttributeType.FLOAT
    if isinstance(value, complex):
        return False
    if isinstance(value, Sequence):
        if attr.type is ir.AttributeType.INTS:
            return all(isinstance(i, int) for i in value)
        if attr.type is ir.AttributeType.FLOATS:
            return all(isinstance(i, float) for i in value)
    return False


def _param_type_compatible_with_arg(
    param: _schemas.Parameter,
    value: ir.TypeProtocol
    | str
    | int
    | float
    | complex
    | Sequence[int]
    | Sequence[float],
    assigned_types: dict[_schemas.TypeConstraintParam, ir.TypeProtocol],
) -> bool:
    # Handle Python types first
    if isinstance(value, bool):
        if param.type_constraint.allowed_types & {ir.DataType.BOOL}:
            return True
    if isinstance(value, int):
        if param.type_constraint.allowed_types & {
            ir.DataType.INT4,
            ir.DataType.INT8,
            ir.DataType.INT16,
            ir.DataType.INT32,
            ir.DataType.INT64,
        }:
            return True
    if isinstance(value, float):
        if param.type_constraint.allowed_types & {
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E5M2FNUZ,
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
        }:
            return True
    if isinstance(value, complex):
        if param.type_constraint.allowed_types & {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.COMPLEX64,
            ir.DataType.COMPLEX128,
        }:
            return True
    if isinstance(value, str):
        if param.type_constraint.allowed_types & {ir.DataType.STRING}:
            return True
    if isinstance(value, Sequence):
        if param.type_constraint.allowed_types & {ir.DataType.INT32, ir.DataType.INT64}:
            if all(isinstance(i, int) for i in value):
                return True
        if param.type_constraint.allowed_types & {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
        }:
            if all(isinstance(i, float) for i in value):
                return True

    if not isinstance(value, ir.TypeProtocol):
        return False

    # Then check tensor types
    if param.type_constraint.name in assigned_types:
        # If a typevar is already bound, check if the value has the same type
        assigned_type = assigned_types[param.type_constraint]
        return assigned_type == value
    # If the typevar is not bound, bind it to the value type
    if value in param.type_constraint.allowed_types:
        # TODO: Maybe just check dtype? Being more strict here for now
        assigned_types[param.type_constraint] = value
        return True
    return False


def _get_type_from_tensor(
    tensor: torch.Tensor | Sequence[torch.Tensor],
) -> ir.TypeProtocol:
    if isinstance(tensor, torch.Tensor):
        return ir.TensorType(_torch_dtype_to_onnx_compatible_dtype(tensor.dtype))
    first_tensor = next((item for item in tensor if item is not None), None)
    if first_tensor is None:
        return ir.SequenceType(ir.TensorType(ir.DataType.UNDEFINED))
    return ir.SequenceType(
        ir.TensorType(_torch_dtype_to_onnx_compatible_dtype(first_tensor.dtype))
    )


def _get_named_fx_node_args(node: torch.fx.Node) -> dict[str, torch.fx.node.Argument]:
    torch_schema: torch.FunctionSchema = node.target._schema
    node_args = {}
    for arg, schema_arg in zip(node.args, torch_schema.arguments):
        node_args[schema_arg.name] = arg
    for name, arg in node.kwargs.items():
        node_args[name] = arg
    return node_args


def get_matching_overload(
    node: torch.fx.Node,
    overload_and_signatures: Sequence[tuple[onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction, _schemas.OpSignature]],
):
    named_args = _get_named_fx_node_args(node)
    schema_args = {arg.name: arg for arg in node.target._schema.arguments}
    for overload, schema in overload_and_signatures:
        assigned_types: dict[_schemas.TypeConstraintParam, ir.TypeProtocol] = {}
        matched = True
        for param in schema:
            if param.name not in schema_args and param.required:
                # We don't need to handle variadic inputs as there is none.
                # A required parameter is not supplied.
                matched = False
                break

            # Get the argument
            if param.name in named_args:
                # Provided in Node args
                arg = named_args[param.name]
            elif (
                param.name in schema_args
                and schema_args[param.name].default is not None
            ):
                # Provided in schema args
                arg = schema_args[param.name].default
            elif param.has_default():
                # Provided in the ONNX op definition
                arg = param.default
            else:
                matched = False
                break

            if isinstance(param, _schemas.Parameter):
                if isinstance(arg, torch.Tensor) or (
                    isinstance(arg, Sequence)
                    and any(isinstance(t, torch.Tensor) for t in arg)
                ):
                    arg = _get_type_from_tensor(arg)
                # TODO: Handle None attributes
                # Handle tensors and Python values
                if not _param_type_compatible_with_arg(param, arg, assigned_types):
                    matched = False
                    break
            elif isinstance(param, _schemas.AttributeParameter):
                if not _attribute_type_compatible_with_arg(param, arg):
                    matched = False
                    break
        if matched:
            return overload, schema
    return None, None


def dispatch(node: torch.fx.Node, registry: _registration.OnnxRegistry) -> onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction | None:
    # TODO: Handle when node does not have a target
    # Determine if the node has complex inputs.
    # TODO: Complex can have customs too
    overload, schema = get_matching_overload(node, [(decomp.onnx_function, decomp.onnx_function.signature) for decomp in decomp_metas])
    if overload is None:
        return None
    return overload