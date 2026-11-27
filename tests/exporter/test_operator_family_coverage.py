"""Numerical native FFT coverage and retained ONNX Script callback semantics."""

import operator

import onnx
import pytest
import torch

import torch_onnx


@pytest.mark.parametrize("kind", ["rfft", "fft", "ifft"])
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("opset", [18, 28])
def test_fft_families_multiple_axes_and_normalization(kind, norm, opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            function = {
                "rfft": torch.fft.rfftn,
                "fft": torch.fft.fftn,
                "ifft": torch.fft.ifftn,
            }[kind]
            return torch.view_as_real(function(x, dim=(0, 2), norm=norm))

    x = torch.randn(2, 3, 4)
    if kind != "rfft":
        x = torch.complex(x, torch.randn_like(x))
    model = Model().eval()
    program = torch_onnx.export(model, (x,), opset_version=opset, verbose=False)
    try:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch.testing.assert_close(program(x)[0], model(x), rtol=1e-4, atol=1e-5)
    finally:
        program.release()


LEGACY_CASES = [
    ("add", operator.add, operator.add, torch.float32),
    ("sub", operator.sub, operator.sub, torch.float32),
    ("mul", operator.mul, operator.mul, torch.float32),
    ("div", operator.truediv, operator.truediv, torch.float32),
    ("matmul", operator.matmul, operator.matmul, torch.float32),
    ("pow", operator.pow, operator.pow, torch.float32),
    ("neg", lambda x, y: -x, lambda x, y: -x, torch.float32),
    ("reverse_add", lambda x, y: 2.0 + x, lambda x, y: 2.0 + x, torch.float32),
    ("reverse_sub", lambda x, y: 2.0 - x, lambda x, y: 2.0 - x, torch.float32),
    ("reverse_mul", lambda x, y: 2.0 * x, lambda x, y: 2.0 * x, torch.float32),
    ("ne", operator.ne, operator.ne, torch.float32),
    ("lt", operator.lt, operator.lt, torch.float32),
    ("le", operator.le, operator.le, torch.float32),
    ("ge", operator.ge, operator.ge, torch.float32),
    ("gt", operator.gt, operator.gt, torch.float32),
    ("float_mod", torch.fmod, operator.mod, torch.float32),
    ("integer_mod", torch.remainder, operator.mod, torch.int64),
    ("reverse_and", lambda x, y: True & x, lambda x, y: True & x, torch.bool),
]


@pytest.mark.parametrize(
    "name,eager,lowering,dtype", LEGACY_CASES, ids=[case[0] for case in LEGACY_CASES]
)
def test_legacy_custom_callbacks_keep_python_operator_support(
    name, eager, lowering, dtype
):
    with torch.library._scoped_library("coverage_legacy", "FRAGMENT") as library:
        qualified = "coverage_legacy::operation"
        torch.library.define(qualified, "(Tensor x, Tensor y) -> Tensor", lib=library)
        torch.library.impl(qualified, "CompositeExplicitAutograd", lib=library)(eager)
        result_dtype = (
            torch.bool
            if name in {"ne", "lt", "le", "ge", "gt", "reverse_and"}
            else dtype
        )
        torch.library.register_fake(qualified, lib=library)(
            lambda x, y: torch.empty_like(x, dtype=result_dtype)
        )
        target = torch.ops.coverage_legacy.operation.default

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return target(x, y)

        x = torch.tensor([[0.5, 1.25], [2.0, 3.0]]).to(dtype)
        y = torch.tensor([[1.5, 2.0], [2.5, 4.0]]).to(dtype)
        model = Model().eval()

        def callback(x, y):
            return lowering(x, y)

        program = torch_onnx.export(
            model,
            (x, y),
            custom_translation_table={target: callback},
            opset_version=20,
            verbose=False,
        )
        try:
            onnx.checker.check_model(program.model_proto, full_check=True)
            torch.testing.assert_close(
                program(x, y)[0], model(x, y), rtol=1e-4, atol=1e-5
            )
        finally:
            program.release()
