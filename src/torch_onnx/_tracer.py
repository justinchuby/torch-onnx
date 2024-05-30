"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from onnxscript._internal import param_manipulation
from onnxscript import evaluator
from onnxscript import ir
from onnxscript.ir import _convenience as ir_convenience
import onnxscript
import onnx
import torch
from torch_onnx import _core, _schemas


AllowedArgType = ir.Value | ir.TensorProtocol | torch.Tensor | int | float | bool | str | Sequence[int] | Sequence[float]


# Logic for adapting inputs from general Python or PyTorch inputs to ONNX ir.Value
# 1. Construct the (named_inputs, named_attrs) mapping based on (args, kwargs) and the signature.
#   a. Loop over all parameters in the signature and args together
#   b. Depending on param.is_input, Record named_inputs[param.name] = arg or named_attrs[param.name] = arg
#   c. Handle kwargs as well
#   d. Fill in None if the input is not provided
# 2. Determine which parameter takes which dtype
#   check: If there are required inputs or attributes that are not provided, raise an error.
#   a. Create a to_resolve_type: set[ArgName]; create type_binding: dict[Constraint, ir.DataType]
#   b. Iterate over all named_inputs
#   b0. Find the corresponding parameter in the signature
#   b1. If the argument is a Python constant and the corresponding parameter
#       is an input, _and_ it's type constraint is not bound yet, add it to to_resolve_type.
#   b2. If the argument is a ir.Value, the corresponding parameter must be an input.
#       Bind {constraint: arg.dtype}.
# 3. Convert Python constants to Constant nodes based on the dtype information;
#    construct sequences
#   a. Iterate over all parameters in the signature the second time
#   b. If the parameter is in to_resolve_type:
#       - If param.constraint in type_binding, set named_args[param.name] = Constant(value, dtype=type_binding[param.constraint])
#       - Otherwise, set named_args[param.name] = Constant(value)
# 4. Construct the node with the inputs and attributes
#    a. Iterate over all parameters in the signature the third time
#    b. If the parameter is an input, set inputs.append(named_args[param.name])
#    c. If the parameter is an attribute, set attributes[param.name] = named_args[param.name]


def _convert_to_input(value: AllowedArgType, param: _schemas.Parameter) -> ir.Value:
    if isinstance(value, ir.Value):
        return value


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, constant_farm: dict[Any, ir.Value]):
        self.nodes = []
        self.functions: dict[ir.OperatorIdentifier, onnxscript.OnnxFunction] = {}
        self.constant_farm = constant_farm

    def _call_op(self, opschema: _schemas.OpSchema, name_args: Mapping[str, AllowedArgType]):
        """Add a node to the graph for the given opschema and arguments.

        Args:
            opschema: The OpSchema containing the node signature.
            name_args: A mapping of argument names to their values.
                Valid values are ir.Value representing dynamic inputs, or
                Python constants representing constant inputs or attributes.
        """
        inputs = []
        attributes = {}
        for parameter in opschema.params:
            pass

        for name, value in name_args.items():
            if isinstance(value, ir.Value):
                inputs.append(value)
            else:
                # Convert Python constants to Constant nodes
                if isinstance(value, (bool, float)):
                    # Be sure to put bool before int, because bool is a subclass of int
                    constant_tensor = ir.tensor(value)
                elif isinstance(value, int):
                    constant_tensor = ir.tensor(value, dtype=ir.DataType.INT64)
                elif isinstance(value, (tuple, list)) and all(isinstance(val, int) for val in value):
                    constant_tensor = ir.tensor(value, dtype=ir.DataType.INT64)
                elif isinstance(value, (tuple, list)) and all(isinstance(val, float) for val in value):
                    constant_tensor = ir.tensor(value)
                elif isinstance(value, str):
                    constant_tensor = ir.tensor(value, dtype=ir.DataType.STRING)
                else:
                    raise TypeError(f"Constant input '{value}' of type '{type(value)}' is not supported")
                self.nodes.append(
                    node := ir.Node(
                        "",
                        "Constant",
                        (),
                        attributes=ir_convenience.convert_attributes({"value": constant_tensor}),
                    )
                )
                inputs.append(node.outputs[0])
        self.nodes.append(
            node := ir.Node(
                opschema.domain,
                opschema.name,
                inputs,
                attributes=ir_convenience.convert_attributes(attributes),
                num_outputs=len(opschema.outputs),
            )
        )
        return node.outputs

    def eval(self, schema, inputs, attributes):
        attributes = {k: v for k, v in attributes.items() if v is not None}
        if schema.name == "CastLike":
            assert len(inputs) == 2
            # Skip CastLike if the input and output types are the same
            src_input = inputs[0]
            target_input = inputs[1]
            dtypes_available = (
                isinstance(src_input, ir.Value)
                and isinstance(target_input, ir.Value)
                and src_input.dtype is not None
                and target_input.dtype is not None
            )
            if dtypes_available:
                if src_input.dtype == target_input.dtype:
                    # Same type. No cast needed
                    return src_input
                else:
                    # Create a Cast node
                    self.nodes.append(
                        node := ir.Node(
                            "",
                            "Cast",
                            (src_input,),
                            attributes=ir_convenience.convert_attributes(
                                {"to": target_input.dtype}
                            ),
                        )
                    )
                    return node.outputs[0]

        onnx_inputs = []
        for input in inputs:
            if isinstance(input, ir.Value):
                onnx_inputs.append(input)
                continue
            elif input in self.constant_farm:
                onnx_inputs.append(self.constant_farm[input])
                continue

            if isinstance(input, (bool, float)):
                # Be sure to put bool before int, because bool is a subclass of int
                constant_tensor = ir.tensor(input)
            elif isinstance(input, int):
                constant_tensor = ir.tensor(input, dtype=ir.DataType.INT64)
            elif isinstance(input, (tuple, list)) and all(
                isinstance(val, int) for val in input
            ):
                constant_tensor = ir.tensor(input, dtype=ir.DataType.INT64)
            elif isinstance(input, (tuple, list)) and all(
                isinstance(val, float) for val in input
            ):
                constant_tensor = ir.tensor(input)
            elif isinstance(input, complex):
                # NOTE: ONNX doesn't support tensor of complex64/complex128, so we
                # convert them to float32/float64 with real representation.
                constant_tensor = _core.TorchTensor(torch.view_as_real(torch.tensor(input).resolve_conj()))
            else:
                raise TypeError(
                    f"Constant input '{input}' of type '{type(input)}' is not supported"
                )
            self.nodes.append(
                node := ir.Node(
                    "",
                    "Constant",
                    (),
                    attributes=ir_convenience.convert_attributes(
                        {"value": constant_tensor}
                    ),
                )
            )
            self.constant_farm[input] = node.outputs[0]
        self.nodes.append(
            node := ir.Node(
                schema.domain,
                schema.name,
                onnx_inputs,
                attributes=ir_convenience.convert_attributes(attributes),
                num_outputs=len(schema.outputs),
            )
        )
        if len(schema.outputs) == 1:
            return node.outputs[0]
        return node.outputs

    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args,
        kwargs,
    ):
        # Special cases for handling IsScalar and Rank
        if function.name == "IsScalar":
            if len(args) != 1:
                raise TypeError(
                    f"Expected 1 positional argument for function '{function}', got {len(args)}."
                )
            if isinstance(args[0], ir.Value):
                if args[0].shape is not None:
                    return args[0].rank() == 0
                else:
                    # Fall to call add_function_call
                    pass
            elif isinstance(args[0], Sequence):  # noqa: SIM103
                return False
            else:
                # Python constants are scalars
                return True
        if function.name == "Rank":
            if len(args) != 1:
                raise TypeError(
                    f"Expected 1 positional argument for function '{function}', got {len(args)}."
                )
            if isinstance(args[0], ir.Value):
                if args[0].shape is not None:
                    return args[0].rank()
                else:
                    # Fall to call add_function_call
                    pass
            elif isinstance(args[0], Sequence):
                if all(isinstance(arg, (int, float)) for arg in args[0]):
                    return 1
                else:
                    # Fall to call add_function_call
                    pass
            else:
                # Python constants are scalars
                return 0
        elif function.experimental_traceable:
            # Trace the function call instead of adding the function as a node
            return function.function(*args, **kwargs)

        # args/kwargs are ir.Value/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )

        # Cast attributes to the correct type based on function signature
        op_schema = function.op_schema
        assert op_schema is not None
        for name, value in attributes.items():
            attribute = op_schema.attributes[name]
            if attribute.type == onnx.defs.OpSchema.AttrType.FLOAT:
                # Cast int to float if the attribute is FLOAT
                attributes[name] = float(value)

            # In PyTorch, an attribute annotated as `int[1]?` accepts an integer
            # or a sequence. When the attribute is an integer, it is treated as
            # a single element sequence. ONNX requires an attribute to either be
            # an integer or a sequence. So we promote the value to a sequence here.
            if attribute.type == onnx.defs.OpSchema.AttrType.INTS and isinstance(
                value, int
            ):
                attributes[name] = (value,)
            if attribute.type == onnx.defs.OpSchema.AttrType.FLOATS and isinstance(
                value, float
            ):
                attributes[name] = (value,)
        self.functions[
            (
                function.function_ir.domain,
                function.name,
                "",
            )
        ] = function
        self.nodes.append(
            node := ir.Node(
                function.function_ir.domain,
                function.name,
                inputs,
                attributes=ir_convenience.convert_attributes(attributes),
                num_outputs=len(function.function_ir.outputs),
            )
        )
        if len(function.function_ir.outputs) == 1:
            return node.outputs[0]
        return node.outputs
