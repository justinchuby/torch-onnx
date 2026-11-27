"""Legacy callbacks retain eager ATen promotion at the adapter boundary."""

import onnx
import pytest
import torch
from onnxscript import opset18 as op

import torch_onnx


def _sin(x):
    return op.Sin(x)


def _square(x):
    return op.Mul(x, x)


def _sum(x):
    return op.ReduceSum(x, keepdims=0)


def _div(x, y, rounding_mode=None):
    return op.Div(x, y)


def _cat(tensors, dim=0):
    return op.Concat(*tensors, axis=dim)


@pytest.mark.parametrize(
    "case",
    [
        "integer_sin",
        "boolean_square",
        "half_sum",
        "integer_div",
        "floor_div",
        "mixed_cat",
    ],
)
def test_legacy_callback_inputs_and_results_follow_pytorch_dtypes(case):
    if case == "integer_sin":
        target, callback, args = (
            torch.ops.aten.sin.default,
            _sin,
            (torch.tensor([0, 1, 2]),),
        )
    elif case == "boolean_square":
        target, callback, args = (
            torch.ops.aten.square.default,
            _square,
            (torch.tensor([True, False]),),
        )
    elif case == "half_sum":
        target, callback, args = (
            torch.ops.aten.sum.default,
            _sum,
            (torch.tensor([1.5, 2.5], dtype=torch.float16),),
        )
    elif case in ("integer_div", "floor_div"):
        target = (
            torch.ops.aten.div.Tensor
            if case == "integer_div"
            else torch.ops.aten.div.Tensor_mode
        )
        callback, args = _div, (torch.tensor([3, 8]), torch.tensor([2, 3]))
    else:
        target, callback, args = (
            torch.ops.aten.cat.default,
            _cat,
            ([torch.ones(2, dtype=torch.float16), torch.ones(2, dtype=torch.float32)],),
        )

    class Model(torch.nn.Module):
        def forward(self, *values):
            if case == "floor_div":
                return target(*values, rounding_mode="floor")
            return target(*values)

    model = Model().eval()
    expected = model(*args)
    program = torch_onnx.export(
        model,
        args,
        custom_translation_table={target: callback},
        opset_version=18,
        verbose=False,
    )
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(*args)[0], expected)
    finally:
        program.release()
