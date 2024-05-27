from typing import Sequence
from onnxscript._internal import param_manipulation
from onnxscript import evaluator
from onnxscript import ir
from onnxscript.ir import _convenience as ir_convenience
import onnxscript
import onnx


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self):
        self.nodes = []
        self.functions: dict[ir.OperatorIdentifier, onnxscript.OnnxFunction] = {}

    def eval(self, schema, inputs, attributes):
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
        self.nodes.append(
            node := ir.Node(
                schema.domain,
                schema.name,
                inputs,
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
