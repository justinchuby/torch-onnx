"""Retained ONNX Script callbacks support public scripted and traced functions."""

import onnx
import onnxscript
import pytest
import torch
from onnxscript import FLOAT
from onnxscript import opset18 as op

import torch_onnx


@onnxscript.script(onnxscript.values.Opset("coverage.functions", 1))
def scripted_negation(x: FLOAT) -> FLOAT:
    return op.Neg(x)


@onnxscript.script(onnxscript.values.Opset("coverage.functions", 1))
def scripted_pair(x: FLOAT) -> tuple[FLOAT, FLOAT]:
    return op.Neg(x), op.Identity(x)


def traced_negation(x: FLOAT, scale: float = 1.0) -> FLOAT:
    return op.Mul(op.Neg(x), scale)


traced_function = onnxscript.TracedOnnxFunction(
    onnxscript.values.Opset("coverage.functions", 1), traced_negation
)


@pytest.mark.parametrize("kind", ["scripted", "multiple_outputs", "traced"])
def test_legacy_callback_function_is_numerically_equivalent(kind):
    def lowering(x):
        if kind == "scripted":
            return scripted_negation(x)
        if kind == "multiple_outputs":
            negative, _ = scripted_pair(x)
            return negative
        return traced_function(x, scale=1.0)

    class Model(torch.nn.Module):
        def forward(self, x):
            return -x

    x = torch.tensor([-2.0, 0.0, 3.0])
    program = torch_onnx.export(
        Model().eval(),
        (x,),
        custom_translation_table={torch.ops.aten.neg.default: lowering},
        opset_version=18,
        optimize=False,
        verbose=False,
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(x)[0], -x)
        if kind != "traced":
            assert any(
                f.domain == "coverage.functions" for f in program.model_proto.functions
            )
    finally:
        program.release()
