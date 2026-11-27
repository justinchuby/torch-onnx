"""PyTorch-compatible ONNXProgram with the standalone optimization pipeline."""

import torch.onnx
from torch.export.graph_signature import ConstantArgument, InputKind
from torch.onnx._internal.exporter._onnx_program import _ort_session_initializer
from torch.utils import _pytree

from torch_onnx import _optimization


class ONNXProgram(torch.onnx.ONNXProgram):
    """An ONNX program using the standalone exporter's optimizer."""

    def _runtime_args(self, args, kwargs):
        if self.exported_program is None:
            return args, kwargs
        specs = [
            spec
            for spec in self.exported_program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        ]
        if not any(isinstance(spec.arg, ConstantArgument) for spec in specs):
            return args, kwargs
        # Match the captured keyword order, even when callers reorder keywords.
        example_inputs = self.exported_program.example_inputs
        order = example_inputs[1] if example_inputs is not None else kwargs
        ordered = {key: kwargs[key] for key in order if key in kwargs}
        ordered.update(
            {key: value for key, value in kwargs.items() if key not in ordered}
        )
        flat, _ = _pytree.tree_flatten((args, ordered))
        if len(flat) != len(specs):
            raise ValueError(
                "Arguments do not match the captured export input structure"
            )
        runtime = []
        for value, spec in zip(flat, specs, strict=True):
            if isinstance(spec.arg, ConstantArgument):
                if type(value) is not type(spec.arg.value) or value != spec.arg.value:
                    raise ValueError(
                        f"Specialized argument {spec.arg.name!r} must equal {spec.arg.value!r}"
                    )
            else:
                runtime.append(value)
        return tuple(runtime), {}

    def __call__(self, *args, **kwargs):
        args, kwargs = self._runtime_args(args, kwargs)
        return super().__call__(*args, **kwargs)

    def call_reference(self, *args, **kwargs):
        args, kwargs = self._runtime_args(args, kwargs)
        return super().call_reference(*args, **kwargs)

    def optimize(self) -> None:
        """Optimize the model in place and invalidate any cached runtime session."""
        if self._inference_session is not None or self._tempdir is not None:
            self.release()
        self.model = _optimization.optimize(self.model)


__all__ = ["ONNXProgram", "_ort_session_initializer"]
