# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Helper-driven regressions from e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Original helpers
and comments are retained. The historical LessModel typo is corrected.
"""

import torch
from _common.runtime import RecoveredRuntimeHarness


class TestRecoveredHelpers(RecoveredRuntimeHarness):
    def _test_reduced_ops(self, op):
        class ReducedOpModule(torch.nn.Module):
            def forward(self, input):
                return op(input, dim=-1)

        if op != torch.mean:  # torch.mean only supports float types
            x = torch.randint(10, (4, 4), dtype=torch.uint8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int16)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int32)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int64)
            self.run_test(ReducedOpModule(), x)

        # The historical suite excluded double mean/prod and half prod.
        # Current PyTorch and ORT support all three floating-point combinations.
        x = torch.randn(4, 5, dtype=torch.double)
        self.run_test(ReducedOpModule(), x)

        x = torch.randn(4, 4, dtype=torch.half)
        # Half prod can differ by one ULP with PyTorch's CPU accumulation.
        self.run_test(ReducedOpModule(), x, atol=1e-5 if op is torch.prod else 1e-7)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedOpModule(), x)

    def test_reduced_sum(self):
        return self._test_reduced_ops(op=torch.sum)

    def test_reduced_mean(self):
        return self._test_reduced_ops(op=torch.mean)

    def test_reduced_prod(self):
        return self._test_reduced_ops(op=torch.prod)

    def _test_compare_ops(self, model, num_inputs):
        x_float = torch.randn(1, 2, 3, 4, requires_grad=True)
        x_int = torch.randint(10, (3, 4), dtype=torch.int32)
        if num_inputs > 1:
            y_float = torch.randn(1, 2, 3, 4, requires_grad=True)
            y_int = torch.randint(10, (3, 4), dtype=torch.int32)
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    def test_eq(self):
        class EqualModel(torch.nn.Module):
            def forward(self, input, other):
                return input == other

        self._test_compare_ops(EqualModel(), 2)

    def test_gt(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other

        self._test_compare_ops(GreaterModel(), 2)

    def test_ge(self):
        class GreaterOrEqualModel(torch.nn.Module):
            def forward(self, input, other):
                return input >= other

        self._test_compare_ops(GreaterOrEqualModel(), 2)

    def test_gt_scalar(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input):
                return input > 1

        self._test_compare_ops(GreaterModel(), 1)

    def test_ge_scalar(self):
        class GreaterOrEqualModel(torch.nn.Module):
            def forward(self, input):
                return input >= 1

        self._test_compare_ops(GreaterOrEqualModel(), 1)

    def test_lt(self):
        class LessModel(torch.nn.Module):
            def forward(self, input, other):
                return input < other

        self._test_compare_ops(LessModel(), 2)

    def test_le(self):
        class LessOrEqualModel(torch.nn.Module):
            def forward(self, input, other):
                return input <= other

        self._test_compare_ops(LessOrEqualModel(), 2)

    def test_lt_scalar(self):
        class LessModel(torch.nn.Module):
            def forward(self, input):
                return input < 1

        self._test_compare_ops(LessModel(), 1)

    def test_le_scalar(self):
        class LessOrEqualModel(torch.nn.Module):
            def forward(self, input):
                return input <= 1

        self._test_compare_ops(LessOrEqualModel(), 1)
