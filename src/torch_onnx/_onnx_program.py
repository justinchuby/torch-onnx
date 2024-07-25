from __future__ import annotations

import logging
import os
import pathlib
import textwrap
from typing import IO, Sequence

import onnx
import torch
from onnxscript import ir
from torch.utils import _pytree as pytree

logger = logging.getLogger(__name__)


class ONNXProgram:
    """A substitute class for `torch.onnx.ONNXProgram`."""

    def __init__(self, model: ir.Model, exported_program: torch.export.ExportedProgram):
        self.model: ir.Model = model
        self.exported_program = exported_program

    def __repr__(self) -> str:
        return f"""\
ONNXProgram(
    model=
{textwrap.indent(str(self.model), ' ' * 8)}
    ,
    exported_program=
{textwrap.indent(str(self.exported_program), ' ' * 8)}
)
"""

    @property
    def model_proto(self) -> onnx.ModelProto:
        """Compatibility property for `torch.onnx.ONNXProgram.model_proto`."""
        return ir.serde.serialize_model(self.model)

    def save(
        self,
        destination: str | os.PathLike | IO[bytes],
        *,
        include_initializers: bool = True,
        external_data: bool | None = None,
        all_tensors_to_one_file: bool = True,
        **_,
    ):
        """Save the ONNX model to the specified destination.

        When `external_data` is `True` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            external_data: Whether to save the weights as external data in a separate file.
            all_tensors_to_one_file: Whether to save all tensors to one file when saving as external data.

        Raises:
            TypeError: If `external_data` is `True` and `destination` is not a file path.
        """
        if not include_initializers:
            self.model.graph.initializers.clear()
            logger.warning(
                "The initializers have been removed from the model. This is destructive. "
                "Developers: Please implement ir.Model copy() and remove initializers on the copied model."
            )
        proto = ir.serde.serialize_model(self.model)
        byte_size = proto.ByteSize()
        model_too_large = (byte_size) >= 1 << 31
        if external_data or model_too_large:
            # TODO: Create an IR pass to handle external tensors conversion
            if model_too_large:
                logger.warning(
                    "The serialized ONNX model is larger than 2GB (%s). "
                    "Saving the weights as external data in a separate file.",
                    byte_size,
                )
            if not isinstance(destination, (str, os.PathLike)):
                raise TypeError(
                    "Saving the weights as external data is only supported when destination is a file path"
                )
            destination_path = pathlib.Path(destination)
            # Create the directory if it does not exist
            data_path = f"{destination_path.name}.data"
            onnx.save_model(
                proto,
                destination,
                save_as_external_data=True,
                all_tensors_to_one_file=all_tensors_to_one_file,
                location=data_path,
            )
        else:
            onnx.save_model(proto, destination)

    def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        import onnxruntime as ort

        onnx_model = self.model_proto.SerializeToString()
        providers = ("CPUExecutionProvider",)
        args = _process_args(args, kwargs)
        ort_session = ort.InferenceSession(onnx_model, providers=providers)

        onnxruntime_input = {
            k.name: v.numpy(force=True)  # type: ignore[union-attr]
            for k, v in zip(self.model.graph.inputs, args)
        }

        # TODO: Turn off optimization
        # TODO: Isolate the run in a separate process
        outputs = ort_session.run(None, onnxruntime_input)
        return tuple(torch.from_numpy(output) for output in outputs)


def _process_args(args, kwargs) -> tuple[torch.Tensor, ...]:
    """Process input arguments for the ONNX model."""
    args = _flatten_inputs(args, kwargs)
    args = _remove_none_from_inputs(args)
    args = _remove_non_tensor(args)
    args = _convert_complex_to_real_representation(args)
    return args


def _flatten_inputs(model_args, model_kwargs):
    flattened_args, _ = pytree.tree_flatten((model_args, model_kwargs))
    return flattened_args


def _remove_none_from_inputs(model_args):
    return tuple(arg for arg in model_args if arg is not None)


def _remove_non_tensor(model_args):
    """Remove the non-tensor input arguments.

    Dynamo does not support non-tensor input arguments (https://github.com/pytorch/pytorch/issues/99534).

    Specifically, it does put the input into graph with an empty node, but consumed by no ones.
    The concrete value is embedded into the graph as a constant arg of a target node. Meta
    suggests in this case that one should rewrite the model code to make it tensor if the
    input value is supposed to change at runtime. We might need to further investigate
    the feasibility of that suggestion.

    For example,

        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        x = torch.randn(1, 1, 2, dtype=torch.float32)
        gm_fun, _ = dynamo.export(func, x, b=8.0, aten_graph=True, tracing_mode="real")

        # class GraphModule(torch.nn.Module):
        #     def forward(self, x, b):
        #         arg0: f32[1, 1, 2], arg1, = fx_pytree.tree_flatten_spec(([x, b], {}), self._in_spec)
        #         # File: path/to/pytorch/test_constant_input.py:5, code: y = x + b
        #         add_tensor: f32[1, 1, 2] = torch.ops.aten.add.Tensor(arg0, 8.0);  arg0 = None

        #         # File: path/to/pytorch/test_constant_input.py:6, code: z = y.relu()
        #         relu_default: f32[1, 1, 2] = torch.ops.aten.relu.default(add_tensor)
        #         return pytree.tree_unflatten([add_tensor, relu_default], self._out_spec)

    Empty torch.fx.Node input leading to a mismatched number of input with PyTorch, as
    it's ignored in ONNX graph. Thus, we delete the useless input here.

    """

    return tuple(
        arg for arg in model_args if not isinstance(arg, (int, float, bool, str))
    )


def _convert_complex_to_real_representation(model_args):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).
    """
    return tuple(
        torch.view_as_real(arg.resolve_conj())
        if isinstance(arg, torch.Tensor) and arg.is_complex()
        else arg
        for arg in model_args
    )
