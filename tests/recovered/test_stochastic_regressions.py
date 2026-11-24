# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Stochastic regressions from PyTorch's deleted ONNX runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve source sample counts,
distribution parameters, broadcasting cases and statistical tolerances.
Original and serialized programs must retain actual randomness; distribution
argument validation remains enabled.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnx_ir as ir
import torch
from test_runtime_regressions import RecoveredRuntimeHarness

import torch_onnx


class _NormalSamples(torch.nn.Module):
    def forward(self, x, y):
        return torch.distributions.Normal(x, y).sample([20000])


class _UniformSamples(torch.nn.Module):
    def forward(self, x, y):
        return torch.distributions.Uniform(x, y).sample([10000])


class _InitializedNormal(torch.nn.Module):
    def forward(self):
        x = torch.ones([]).new_empty(1, 400, 50)
        torch.nn.init.normal_(x, 5.0, 10.0)
        return x


class TestRecoveredStochastic(RecoveredRuntimeHarness):
    def _samples(self, model, args, shape, *, check_randomness=True):
        program = torch_onnx.export(model, args, verbose=False)
        onnx.checker.check_model(program.model_proto, full_check=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stochastic.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        results = []
        for candidate in [program, reloaded]:
            first = candidate(*args)[0]
            second = candidate(*args)[0]
            for value in [first, second]:
                self.assertEqual(value.shape, torch.Size(shape))
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all().item())
            if check_randomness:
                self.assertFalse(torch.equal(first, second))
            results.extend([first, second])
        return results

    def test_dist_normal_correctness(self):
        args = (torch.tensor([5.0]), torch.tensor([10.0]))
        for values in self._samples(_NormalSamples().eval(), args, (20000, 1)):
            self.assertLessEqual(abs(values.mean().item() - 5.0), 0.5)
            self.assertLessEqual(abs(values.std(correction=0).item() - 10.0), 1.0)

    def test_nn_init_normal_correctness(self):
        for values in self._samples(_InitializedNormal().eval(), (), (1, 400, 50)):
            self.assertLessEqual(abs(values.mean().item() - 5.0), 0.5)
            self.assertLessEqual(abs(values.std(correction=0).item() - 10.0), 1.0)

    def test_dist_uniform_correctness(self):
        args = (torch.tensor([5.0]), torch.tensor([10.0]))
        for values in self._samples(_UniformSamples().eval(), args, (10000, 1)):
            self.assertGreaterEqual(values.min().item(), 5.0)
            self.assertLessEqual(values.max().item(), 10.0)
            self.assertLessEqual(abs(values.mean().item() - 7.5), 7.5 * 0.05)

    def test_dist_normal(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.distributions.Normal(x, y).sample().size(0), x, y

        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([[1.0], [2.0]])))
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([1.0])))
        self.run_test(
            M(),
            (
                torch.tensor([[[0.0], [10.0]], [[2.0], [8.0]], [[2.0], [8.0]]]),
                torch.tensor([[1.0], [3.0]]),
            ),
        )

    def test_dist_uniform(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.distributions.Uniform(x, y).sample().size(0), x, y

        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([10.0])))
        self.run_test(M(), (torch.tensor([[0.0], [6.0]]), torch.tensor([[1.0], [7.0]])))
        self.run_test(
            M(), (torch.tensor([1.0]), torch.tensor([[10.0], [7.0], [9.0], [20.0]]))
        )

    def test_dropout_training(self):
        model = torch.nn.Sequential(torch.nn.Dropout(0.4)).train()
        x = torch.randn(10)
        for values in self._samples(model, (x,), x.shape, check_randomness=False):
            self.assertFalse(torch.equal(values, x))
            kept = values != 0
            torch.testing.assert_close(values[kept], x[kept] / 0.6)

    def test_dropout_training_zero(self):
        model = torch.nn.Sequential(torch.nn.Dropout(0.5)).train()
        # Ensure there are no zeros in the input.
        x = torch.randn(10, 3, 128, 128)
        x = x.masked_fill(x == 0, 1)
        eager_ratio = (model(x) != 0).sum().item() / x.numel()
        for values in self._samples(model, (x,), x.shape):
            ratio = (values != 0).sum().item() / x.numel()
            torch.testing.assert_close(ratio, eager_ratio, rtol=0.01, atol=0.01)
            kept = values != 0
            torch.testing.assert_close(values[kept], x[kept] * 2, rtol=0, atol=0)
