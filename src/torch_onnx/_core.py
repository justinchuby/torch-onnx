from __future__ import annotations
import ctypes
import itertools
from typing import Any

import numpy as np
import torch
from onnxscript import ir
from torch.export import graph_signature

import logging


logger = logging.getLogger(__name__)
# Define utilities to convert PyTorch data types so users do not need to specify manually
_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
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


def _torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX[dtype]


class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None):
        # Pass the tensor as the raw data to ir.Tensor's constructor
        super().__init__(tensor, dtype=_torch_dtype_to_onnx_dtype(tensor.dtype), name=name)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        # numpy() calls __array__ in ir.Tensor
        if self.dtype == ir.DataType.BFLOAT16:
            return self.raw.view(torch.uint16).__array__(dtype)
        if self.dtype in {
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E5M2FNUZ,
        }:
            # TODO: Use ml_dtypes
            return self.raw.view(torch.uint8).__array__(dtype)
        return self.raw.__array__(dtype)

    def tobytes(self) -> bytes:
        # Implement tobytes to support native PyTorch types so we can use types like bloat16
        # Reading from memory directly is also more efficient because
        # it avoids copying to a NumPy array
        tensor = self.raw.detach().cpu().contiguous()
        return bytes(
            (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                tensor.data_ptr()
            )
        )


# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L56C1-L62C19
# class InputKind(Enum):
#     USER_INPUT = auto()
#     PARAMETER = auto()
#     BUFFER = auto()
#     CONSTANT_TENSOR = auto()
#     CUSTOM_OBJ = auto()
#     TOKEN = auto()

# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L89C1-L96C19
# class OutputKind(Enum):
#     USER_OUTPUT = auto()
#     LOSS_OUTPUT = auto()
#     BUFFER_MUTATION = auto()
#     GRADIENT_TO_PARAMETER = auto()
#     GRADIENT_TO_USER_INPUT = auto()
#     USER_INPUT_MUTATION = auto()
#     TOKEN = auto()


def _add_nodes(
    exported_program: torch.export.ExportedProgram, graph: ir.Graph
) -> dict[str, ir.Value]:
    values: dict[str, ir.Value] = {}
    for node in exported_program.graph.nodes:
        print(node.name, node.args, node.target, node.op, node.type, node.kwargs)
        if node.op == "placeholder":
            # Placeholder nodes are user inputs
            # We need to create a new tensor for each user input
            # and add it to the graph's inputs
            name = node.name
            # shape = node.kwargs["shape"]
            # dtype = node.kwargs["dtype"]
            input_ = ir.Input(name)
            input_.meta["node"] = node
            values[name] = input_
            # The inputs will be added to the graph later
        elif node.op == "call_function":
            # Add op to the graph
            op = str(node.target)
            inputs = [values[name] for name in node.args]
            kwargs = node.kwargs
            output_name = node.name
            output = ir.Value(name=output_name)

            values[output_name] = output
            node = ir.Node(
                "pkg.torch.ops",
                op,
                inputs,
                attributes=kwargs,
                outputs=[output],
                name=f"{node.op}_{node.name}",
            )
            node.meta["node"] = node
            graph.append(node)

            # TODO: Set dtype and shape
            # shape = node.kwargs["shape"]
            # dtype = node.kwargs["dtype"]
    return values


def _torch_version_integer() -> int:
    return int(torch.__version__.replace(".", ""))


def exported_program_to_ir_graph(exported_program: torch.export.ExportedProgram):
    # TODO: Make it an Interpreter
    graph = ir.Graph(
        [],
        [],
        nodes=[],
        opset_imports={"": 20, "pkg.torch.ops": _torch_version_integer()},
        name="main_graph",
    )

    # 1. Add all nodes to the graph and create a dictionary of values
    values = _add_nodes(exported_program, graph)

    # 2. Add user inputs and all parameters/buffers to the graph.
    # Since the node names and the tensor names are different, we need to rename
    # the nodes to match the tensor names later. For now we will just use the node names.
    user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == graph_signature.InputKind.USER_INPUT
    ]
    non_user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind != graph_signature.InputKind.USER_INPUT
    ]

    for spec in itertools.chain(user_inputs, non_user_inputs):
        # Put the user inputs first and then the parameters/buffers
        value_name = spec.arg.name
        input_kind = spec.kind
        persistent = spec.persistent
        value = values[value_name]

        value.metadata_props[
            "pkg.torch.export.graph_signature.InputSpec.kind"
        ] = input_kind.name
        value.metadata_props[
            "pkg.torch.export.graph_signature.InputSpec.persistent"
        ] = str(persistent)

        if input_kind == graph_signature.InputKind.USER_INPUT:
            graph.inputs.append(value)  # type: ignore
        else:
            graph.initializers[value_name] = value

    # 3. Add outputs to the graph. Keep the order of the outputs.
    for spec in exported_program.graph_signature.output_specs:
        value_name = spec.arg.name
        output_kind = spec.kind
        value = values[value_name]

        value.metadata_props[
            "pkg.torch.export.graph_signature.OutputSpec.kind"
        ] = output_kind.name

        if output_kind == graph_signature.OutputKind.USER_OUTPUT:
            graph.outputs.append(value)

    # 4. Rename the initializers to match the tensor names
    for name, param_name in itertools.chain(
        exported_program.graph_signature.inputs_to_parameters.items(),
        exported_program.graph_signature.inputs_to_buffers.items(),
    ):
        initializer = graph.initializers.pop(name)
        initializer.name = param_name
        graph.initializers[param_name] = initializer

    # 5. Add initializers to the graph
    for name, value in graph.initializers.items():
        torch_tensor = exported_program.state_dict.get(name)
        if torch_tensor is None:
            logger.warning("Tensor '%s' not found in state_dict", name)
            continue
        tensor = TorchTensor(torch_tensor, name=name)
        graph.initializers[name].const_value = tensor

    # TODO: Decide if we should keep mutated buffers as inputs/outputs

    return graph


def exported_program_to_ir(exported_program: torch.export.ExportedProgram) -> ir.Model:
    return ir.Model(
        exported_program_to_ir_graph(exported_program),
        ir_version=9,
        producer_name="torch",
        producer_version=torch.__version__,
    )
