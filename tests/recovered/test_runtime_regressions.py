# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Numerical regressions recovered from PyTorch's removed ONNX runtime suite.

Source parent: e2c2673d00500d268fcb1cc63fa7cc3173876505.
Removal: 6dca9f63bc4975ac323044a440c2c7e5f8512408 (PR 203621).
Original bodies/comments are retained; the harness exports with torch_onnx,
checks ONNX validity, saves/reloads and compares numerical values/shapes/dtypes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch
from torch import Tensor
from torch.testing._internal import common_utils

import torch_onnx


@pytest.mark.parametrize("mixed_weights", [False, True])
def test_batch_norm_no_stats_half_precision(mixed_weights):
    model = torch.nn.BatchNorm1d(3, track_running_stats=False).eval()
    if not mixed_weights:
        model = model.half()
    program = torch_onnx.export(
        model, (torch.randn(4, 3, dtype=torch.float16),), verbose=False
    )
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("empty", [False, True])
def test_masked_select_broadcast_and_empty(dtype, empty):
    class Model(torch.nn.Module):
        def forward(self, x, mask):
            return torch.masked_select(x, mask)

    x = torch.tensor([[0], [1]], dtype=dtype)
    mask = torch.tensor([[False, not empty, False]])
    program = torch_onnx.export(Model().eval(), (x, mask), verbose=False)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


class RecoveredRuntimeHarness(common_utils.TestCase):
    @torch.no_grad()
    def run_test(
        self,
        model,
        input_args=None,
        input_kwargs=None,
        *,
        rtol=1e-3,
        atol=1e-7,
        additional_test_inputs=None,
        input_names=None,
        dynamic_axes=None,
        dynamic_shapes=None,
        remained_onnx_input_idx=None,
    ):
        args = input_args if isinstance(input_args, tuple) else (input_args,)
        # A trailing dictionary is legacy kwargs syntax unless followed by {}.
        export_args = (
            (*args, {})
            if args and isinstance(args[-1], dict) and input_kwargs is None
            else args
        )
        program = torch_onnx.export(
            model.eval(),
            export_args,
            optimize=True,
            verbose=False,
            kwargs={} if input_kwargs is None else input_kwargs,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
        )
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(
            program, strategy=None, rtol=rtol, atol=atol
        )
        if remained_onnx_input_idx is not None:
            # Modern export preserves the call signature, even for constant outputs.
            self.assertEqual(
                sum(bool(value.uses()) for value in program.model.graph.inputs),
                len(remained_onnx_input_idx),
            )
        with common_utils.TemporaryDirectoryName() as directory:
            path = Path(directory) / "recovered.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            torch_onnx.testing.assert_onnx_program(
                reloaded, strategy=None, rtol=rtol, atol=atol
            )
        if additional_test_inputs is not None:
            for other_input in additional_test_inputs:
                other_args = (
                    other_input if isinstance(other_input, tuple) else (other_input,)
                )
                torch_onnx.testing.assert_onnx_program(
                    program,
                    args=other_args,
                    kwargs={},
                    strategy=None,
                    rtol=rtol,
                    atol=atol,
                )


class TestRecoveredRuntime(RecoveredRuntimeHarness):
    def test_index_1d(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:-1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_mask(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_mask_nd(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[input > 0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, (b, c), d

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, ((b,), (c, d))

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: tuple[Tensor, Tensor]):
                return a

        x = (torch.randn(3, 4), torch.randn(4, 3))
        self.run_test(TupleModel(), input_args=(x,))

    def test_nested_tuple_input(self):
        class NestedTupleModel(torch.nn.Module):
            def forward(self, a, b: tuple[Tensor, tuple[Tensor, Tensor]]):
                return a + b[0] + b[1][0] + b[1][1]

        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(1, 5), torch.randn(4, 1)))
        self.run_test(NestedTupleModel(), input_args=(x, y))

    def test_hardtanh(self):
        model = torch.nn.Hardtanh(-1.5, 2.5)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardswish(self):
        model = torch.nn.Hardswish()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardsigmoid(self):
        model = torch.nn.Hardsigmoid()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # corner cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_tanhshrink(self):
        model = torch.nn.Tanhshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardshrink(self):
        model = torch.nn.Hardshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_softshrink(self):
        model = torch.nn.Softshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_clamp(self):
        class ClampModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_default_stride(self):
        class MaxPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_default_stride(self):
        class AvgPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_arithmetic(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_floor_div(self):
        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x // 3,
                    x // 2.0,
                    x.to(dtype=torch.float64) // 3,
                    x.to(dtype=torch.float64) // 2.0,
                    x.to(dtype=torch.int64) // 3,
                    x.to(dtype=torch.int64) // 2.0,
                    x // (y + 1.0).to(dtype=torch.int64),
                    x // y,
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.float64) // y.to(dtype=torch.float64),
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.int64) // y,
                )

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_floordiv(self):
        class FloordivModule(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloordivModule(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    def test_slice_neg(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1:]

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_square(self):
        class Square(torch.nn.Module):
            def forward(self, x):
                return torch.square(x)

        x = torch.randn(2, 3, 4)
        self.run_test(Square(), x)

    def test_index_put_accumulate(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                return x.index_put((ind,), update, accumulate=True)

        x = torch.randn(3, 4)
        ind = torch.tensor([2], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    def test_flip(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    def test_reshape_different_rank(self):
        class ReshapeModel(torch.nn.Module):
            def forward(self, x):
                x = x.reshape(-1, 2, 4, 4, 5, 5)
                return x

        x = torch.randn(1, 32, 5, 5)
        self.run_test(ReshapeModel(), x)

    def test_groupnorm(self):
        model = torch.nn.GroupNorm(3, 6, 0.002)
        x = torch.randn(4, 6, 36, 36, 18)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_groupnorm_noaffine(self):
        model = torch.nn.GroupNorm(4, 8, 0.002, affine=False)
        x = torch.randn(3, 8, 224, 224)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_pow(self):
        class PowModule(torch.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        x = torch.randn(2, 3, 4).to(dtype=torch.float64)
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        class PowModule2(torch.nn.Module):
            def forward(self, x):
                return torch.pow(2, x)

        x = torch.randn(1, 10)
        self.run_test(PowModule2(), (x,))

        x = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule2(), (x,))

        x = torch.randn(1, 10).to(dtype=torch.float64)
        self.run_test(PowModule2(), (x,))

        class PowModule3(torch.nn.Module):
            def forward(self, x, y):
                return y[torch.pow(2, x)]

        x = torch.randint(5, (2, 3, 4))
        y = torch.rand(100)
        self.run_test(PowModule3(), (x, y))

    def test_mul_bool(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.mul(x, y)

        x_t = torch.tensor([True, False, True, False])
        y_t = torch.tensor([True, True, False, False])
        z_t = torch.tensor([1.0, 2.0, 3.0, 0.0])
        self.run_test(MyModel(), (x_t, y_t))
        self.run_test(MyModel(), (x_t, z_t))
        self.run_test(MyModel(), (z_t, y_t))

    def test_std(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_along_dims(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_keepdim(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_var(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

    @common_utils.unittest.skip(
        "Known ONNX Script optimizer bug: approximate-zero rewrite drops variance epsilon"
    )
    def test_var_sqrt_epsilon(self):
        # TODO: Enable when ONNX Script preserves stabilizing epsilon additions.
        class VarianceSqrt(torch.nn.Module):
            def forward(self, input):
                y = torch.var(input, 1)
                return torch.sqrt(y + 1e-8)

        x = torch.randn(1, 2, 3, 300, 300)
        model = VarianceSqrt()
        self.run_test(model, x)

    def test_var_along_dims(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_keepdim(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        class ReverseDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(2, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = ReverseDims()
        self.run_test(model, x)

        class SkipDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = SkipDims()
        self.run_test(model, x)

        class NonZeroDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(1, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = NonZeroDims()
        self.run_test(model, x)

    def test_var_mean_keepdim(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_narrow(self):
        class NarrowModel(torch.nn.Module):
            def forward(self, input):
                return torch.narrow(input, 0, 0, 2)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    def test_index_fill(self):
        class IndexFillModel(torch.nn.Module):
            def forward(self, input):
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexFillModel(), x)

    def test_select(self):
        class Select(torch.nn.Module):
            def forward(self, x):
                return x[:, 1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        class Select(torch.nn.Module):
            def forward(self, x):
                return x[:, -1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_index_select_constant_scaler_index(self):
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def forward(self, x):
                index = 2
                return torch.index_select(x, 1, torch.tensor(index))

        x = torch.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_take(self):
        class TakeModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.take(x, y)

        x = torch.randn(6, 4, 3, 3)
        y = torch.tensor([4, 1, 7, 15, 63])
        self.run_test(TakeModel(), (x, y))

    def test_topk(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 3)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        self.run_test(MyModule(), x)

    def test_clip_int(self):
        class MyClipInt(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, 0, 1)

        self.run_test(MyClipInt(), torch.randn(3, 3).to(torch.int64))

    def test_relu_int(self):
        self.run_test(torch.nn.ReLU(), torch.randn(3, 3).to(torch.int32))

    def test_pad_int(self):
        class MyPadInt(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, (1, 1))

        self.run_test(MyPadInt(), torch.randn(3, 3).to(torch.int32))

    def test_min_int(self):
        class MyMinInt(torch.nn.Module):
            def forward(self, x):
                return torch.min(x, x + 1)

        self.run_test(MyMinInt(), torch.randn(3, 3).to(torch.int32))

    def test_max_int(self):
        class MyMaxnInt(torch.nn.Module):
            def forward(self, x):
                return torch.max(x, x + 1)

        self.run_test(MyMaxnInt(), torch.randn(3, 3).to(torch.int32))

    def test_normalize(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.normalize(x)

        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_norm_with_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                # TODO(bowbao): There is a slight gap in today's test infrastructure
                # to directly test aten ops. OpInfo `torch.norm`` in `common_methods_invocations.py`
                # will not decompose to below aten op.
                return torch.ops.aten.norm(
                    x, p=2, dim=[1], keepdim=True, dtype=torch.float64
                )

        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_layer_norm(self):
        # As layer_norm works on the last D dimension, please keep
        # this test case at least three dimension to prevent the
        # situation of axis=2 mapping to the same axis as axis=-2
        for elementwise_affine in (True, False):
            for bias in (True, False):
                model = torch.nn.LayerNorm(
                    [10, 10, 10], elementwise_affine=elementwise_affine, bias=bias
                )
                x = torch.randn(20, 5, 10, 10, 10)
                self.run_test(model, x)

    def test_batchnorm1d(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_noaffine(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_norunningstats(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, track_running_stats=False)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm2d(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm2d_noaffine(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=False)
        self.run_test(model, x)

    def test_batchnorm2d_norunningstats(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, track_running_stats=False)
        self.run_test(model, x)

    def test_batchnorm3d(self):
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm3d_noaffine(self):
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=False)
        self.run_test(model, x)

    def test_instancenorm1d_runningstats(self):
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm1d_norunningstats(self):
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=False)
        self.run_test(model, x)

    def test_instancenorm2d_runningstats(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm2d_norunningstats(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    def test_scatter_with_scalar(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    def test_scatter(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 5, 6)
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 2)
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        self.run_test(ScatterModel(), (input, indices, values))

    def test_scatter_add_different_size_index_src(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, src):
                return input.scatter_add(0, indices, src)

        src = torch.ones((2, 5))
        input = torch.zeros(3, 5, dtype=src.dtype)
        indices = torch.tensor([[0, 1, 2, 0, 0]])
        self.run_test(ScatterModel(), input_args=(input, indices, src))

    def test_bucketize(self):
        class BucketModel(torch.nn.Module):
            def forward(self, input, boundaries):
                return torch.bucketize(input, boundaries), torch.bucketize(
                    input, boundaries, right=True
                )

        input = torch.tensor([[2, 5, 10], [6, 8, 3]])
        boundaries = torch.tensor([1, 5, 7, 8, 10])
        self.run_test(BucketModel(), (input, boundaries))

    def test_gather(self):
        class GatherModel(torch.nn.Module):
            def forward(self, input, indices):
                return input.gather(1, indices)

        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(GatherModel(), input_args=(input, indices))

    def test_expand(self):
        class ExpandModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(2, 3, -1)

        input = torch.randn(2, 1, 4)
        self.run_test(ExpandModel(), input_args=(input))

        class ExpandInferDimModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(-1, input.size(0))

        input = torch.randn(3, 1)
        self.run_test(ExpandInferDimModel(), input_args=(input))

        class ExpandTensorSizeModel(torch.nn.Module):
            def forward(self, input, size):
                return input.expand(size)

        input = torch.randn(
            3,
        )
        size = torch.tensor(-1)
        self.run_test(ExpandTensorSizeModel(), input_args=(input, size))

    def test_reduced_min_max(self):
        class ReducedMinMaxModule(torch.nn.Module):
            def forward(self, input):
                return torch.min(input, dim=-1)[0], torch.max(input, dim=0)[0]

        x = torch.randint(10, (4, 4), dtype=torch.int32)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randint(10, (4, 4), dtype=torch.int64)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedMinMaxModule(), x)

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(torch.nn.Module):
            def forward(self, input):
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_logsoftmax(self):
        for i in range(7)[2:]:
            model = torch.nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_test(model, input)

    def test_logsoftmax_dim(self):
        for i in range(-4, 3):
            model = torch.nn.LogSoftmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

    def test_logsoftmax_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float64)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(Model(), x)

    def test_softplus(self):
        class BetaOneModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaOneModel(), x)

        class BetaModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x, beta=2)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaModel(), x)

        class BetaFloatModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x, beta=1.7)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaFloatModel(), x)

    def test_max_tensors(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.max(input, other)

        model = MaxModel()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    def test_amax_amin(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.amax(x, dim=0, keepdim=True), torch.amin(
                    x, dim=[0, 1], keepdim=False
                )

        model = Model()
        x = torch.randn(4, 4)
        self.run_test(model, x)

    def test_aminmax(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.aminmax(x, dim=1, keepdim=True), torch.aminmax(
                    x, keepdim=False
                )

        model = Model()
        x = torch.randn(3, 4)
        self.run_test(model, x)

    def test_logical_and(self):
        class AndModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_and(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(AndModel(), input_args=(x, y))

    def test_logical_or(self):
        class OrModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(OrModel(), input_args=(x, y))

    def test_logical_xor(self):
        class XorModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(XorModel(), input_args=(x, y))

    def test_logical_not(self):
        class NotModel(torch.nn.Module):
            def forward(self, x):
                return torch.logical_not(x)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        self.run_test(NotModel(), input_args=(x,))

    def test_matmul(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_repeat(self):
        class RepeatModel(torch.nn.Module):
            def forward(self, x, y):
                x2 = x.repeat(y.shape[0], 1)
                y1 = y.view(-1, 1)
                return x2 + y1

        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 8, 9])
        self.run_test(RepeatModel(), (x, y))

    def test_repeat_interleave(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(2)

        for shape in ([3], [3, 4], [2, 3, 4]):
            x = torch.randn(shape)
            self.run_test(FlattenModel(), (x,))

        class DimsModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(4, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel(), (x,))

        class DimsModel2(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel2(), (x,))

        class RepeatsDimsModel(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel(), (x,))

        class RepeatsDimsModel2(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel2(), (x,))

    def test_repeat_interleave_noop(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(1, dim=1)

        x = torch.randn(4, 1, 8)
        self.run_test(Model(), (x,))

    def test_view(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                return input.view(4, 24)

        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_view_as(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input, other):
                return input.view_as(other)

        x = torch.randn(2, 3, 4)
        y = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, y))

    def test_flatten(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input)

        model = FlattenModel()

        # flatten with 4d input
        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(model, x)

        # flatten with 0d input
        x = torch.randn([])
        self.run_test(model, x)

        # flatten with 1d input
        x = torch.randn(4)
        self.run_test(model, x)

    def test_flatten2d(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input, 1)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d_neg(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.flatten(x, 1, -1),
                    torch.flatten(x, 0, -2),
                    torch.flatten(x, 1, -2),
                )

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_unbind(self):
        class UnbindModel(torch.nn.Module):
            def forward(self, input):
                _, out, _ = input.unbind()
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(1)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

        class UnbindModel3(torch.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(-2)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel3(), x)

    def test_split(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    def test_split_size_as_list(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input, split_sizes: list[int]):
                out = []
                split_list: list[Tensor] = input.split(split_sizes)

                for ob in split_list:
                    out.append(ob)  # noqa: PERF402
                return torch.cat(out, dim=0)

        x = torch.randn(6, 4, 3)
        split_sizes = [torch.tensor(2), torch.tensor(4)]
        self.run_test(SplitModel(), (x, split_sizes))

    def test_concat(self):
        class ConcatModel(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.cat((x, y, z))

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        self.run_test(ConcatModel(), (x, y, z))

    def test_stack(self):
        class StackModel(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.stack((x, y, z), 1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        z = torch.randn(3, 4, 5)
        self.run_test(StackModel(), (x, y, z))

    def test_eye(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.eye(x.size()[1], 3),
                    torch.eye(4, 4, dtype=torch.long),
                    torch.eye(x.size()[1], 2, dtype=torch.long),
                    torch.eye(x.shape[0]),
                    torch.eye(x.shape[0], dtype=torch.float64),
                )

        x = torch.randn(2, 3, 4)
        another_x = torch.randn(5, 6, 7)
        self.run_test(
            TensorFactory(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_diagonal(self):
        class DiagonalModel(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelNegOffset(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x, offset=-1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelNegOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelPosOffset(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x, offset=1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelPosOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithDims(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x, offset=-1, dim1=1, dim2=2)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithNegativeDims(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x, offset=0, dim1=-2, dim2=-1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithNegativeDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelOffsetOverrun(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x, offset=-2), torch.diagonal(x, offset=5)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelOffsetOverrun(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    def test_new_zeros(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1:2]), x.new_zeros(
                    x.shape[2:], dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    def test_new_ones(self):
        class OnesModel(torch.nn.Module):
            def forward(self, x):
                return x.new_ones(x.shape[1:2]), x.new_ones(
                    x.shape[2:], dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(OnesModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(OnesModel(), x, remained_onnx_input_idx=[])

    def test_new_full(self):
        class Full(torch.nn.Module):
            def forward(self, x):
                return x.new_full(x.shape[1:2], 5), x.new_full(
                    x.shape[0:1], 1.3, dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Full(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Full(), x, remained_onnx_input_idx=[])

    def test_sort(self):
        class SortModel(torch.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=True))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    def test_sort_ascending(self):
        class SortModel(torch.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=False))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    def test_argsort(self):
        class ArgSortModel(torch.nn.Module):
            def forward(self, x):
                return torch.argsort(x, dim=1, descending=False)

        x = torch.randn(3, 4)
        self.run_test(ArgSortModel(), x)

    def test_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, x):
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                return x.masked_fill(mask, 2)

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):
            def forward(self, x):
                return x.masked_fill(x > 3, -1)

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    def test_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, x):
                return torch.masked_scatter(x, x.ge(0.5), torch.ones(100, 100) * 5)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedScatterModel(), x)

    def test_masked_select(self):
        class MaskedSelectModel(torch.nn.Module):
            def forward(self, x):
                return torch.masked_select(x, x.ge(0.5))

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedSelectModel(), x)

    def test_pixel_shuffle(self):
        class PixelShuffle(torch.nn.Module):
            def forward(self, x):
                return torch.pixel_shuffle(x, upscale_factor=2)

        x = torch.randn(2, 16, 4, 3, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelShuffle(), x)
        self.run_test(
            PixelShuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    def test_pixel_unshuffle(self):
        class PixelUnshuffle(torch.nn.Module):
            def forward(self, x):
                return torch.pixel_unshuffle(x, downscale_factor=2)

        x = torch.randn(2, 16, 4, 6, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelUnshuffle(), x)
        self.run_test(
            PixelUnshuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    def test_reciprocal(self):
        class ReciprocalModel(torch.nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        model = ReciprocalModel()
        x = torch.tensor([2, 4])
        self.run_test(model, x.to(torch.long))
        self.run_test(model, x.to(torch.float))
        self.run_test(model, x.to(torch.double))

    def test_full_like(self):
        class FullLikeModel(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, 1.3, dtype=torch.int)

        x = torch.tensor(12)
        self.run_test(FullLikeModel(), x)

    def test_full_like_value(self):
        class FullLikeModel(torch.nn.Module):
            def forward(self, x, y):
                out = y + 2
                return torch.full_like(x, out)

        x = torch.tensor(12)
        y = torch.tensor(2)
        self.run_test(FullLikeModel(), (x, y))

    def test_l1_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p=1, dim=-1, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_l2_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p=2, dim=-2, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p="fro", dim=0, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm_keepdim(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p="fro", dim=(0, 1), keepdim=True)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_mv(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.mv(input, other)

        x = torch.randn(4, 5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (4, 5))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    def test_dot(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.dot(input, other)

        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (5,))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    def test_prelu_scalar(self):
        x = torch.scalar_tensor(1.0)
        self.run_test(torch.nn.PReLU(), x, input_names=["x"])

    def test_tril(self):
        class trilModel(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModel(), (x))

        class trilModelwithDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x, diagonal=1)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithDiagonal(), (x))

        class trilModelwithNegDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x, diagonal=-1)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithNegDiagonal(), (x))

        class trilModelWithDiagonalInput(torch.nn.Module):
            def forward(self, x, diagnonal: int):
                return torch.tril(x, diagonal=diagnonal)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelWithDiagonalInput(), (x, 5))

    def test_triu(self):
        class triuModel(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModel(), (x))

        class triuModelwithDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x, diagonal=1)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithDiagonal(), (x))

        class triuModelwithNegDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x, diagonal=-1)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithNegDiagonal(), (x))

        class triuModelWithDiagonalInput(torch.nn.Module):
            def forward(self, x, diagnonal: int):
                return torch.triu(x, diagonal=diagnonal)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelWithDiagonalInput(), (x, 5))

    def test_remainder(self):
        class RemainderModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.remainder(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(RemainderModel(), (x, y))

        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        y = torch.tensor([2], dtype=torch.long)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        y = y.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(torch.int32)
        self.run_test(RemainderModel(), (x, y))

    def test_fmod(self):
        class FModModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.fmod(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(FModModel(), (x, y))

    def test_fmod_scalar(self):
        class FModModel(torch.nn.Module):
            def forward(self, input):
                return torch.fmod(input, 2.55)

        x = torch.randint(10, (2, 3))
        self.run_test(FModModel(), x)

    def test_glu(self):
        class GluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.glu(x)

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GluModel(), x)

    def test_gelu(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="none")

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    def test_tanh_gelu(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)
