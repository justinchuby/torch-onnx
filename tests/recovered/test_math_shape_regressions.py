# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Math/shape cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
order/dimension matrices and inputs with the standalone runtime harness.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class _Norm(torch.nn.Module):
    def __init__(self, variant, ord_val=None, dim=None, keepdim=False):
        super().__init__()
        self.variant = variant
        self.ord = ord_val
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.variant == "vector":
            return torch.linalg.vector_norm(
                x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
            )
        if self.variant == "matrix":
            return torch.linalg.matrix_norm(
                x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
            )
        return torch.linalg.norm(x, ord=self.ord, dim=self.dim)


class TestRecoveredMathShapes(RecoveredRuntimeHarness):
    def test_linalg_norm(self):
        x = torch.randn(2, 3, 5, 5)
        vector_orders = [None, 2, float("inf"), -float("inf"), -4, 1.5]
        matrix_orders = ["fro", float("inf"), -float("inf"), 1, -1]
        for order in vector_orders:
            self.run_test(_Norm("norm", order, 1), x)
        for order in matrix_orders:
            self.run_test(_Norm("norm", order, (0, 2)), x)
        for shape in [(2, 3, 5, 5), (2, 3), (2,)]:
            self.run_test(_Norm("norm"), torch.randn(shape))
        x = torch.randn(2)
        for order in vector_orders:
            self.run_test(_Norm("norm", order), x)
        x = torch.randn(2, 3)
        for order in matrix_orders:
            self.run_test(_Norm("norm", order), x)

    def test_linalg_vector_norm_zero(self):
        self.run_test(_Norm("vector", 0), torch.randn(2, 3, 5, 5))

    def test_linalg_vector_norm(self):
        x = torch.randn(2, 3, 5, 5)
        ord_options = [2, float("inf"), -float("inf"), -4, 1.5]
        dim_options = [(None, False), (1, False), ((1, 2), False), ((1, 2), True)]
        for order in ord_options:
            for dim, keepdim in dim_options:
                self.run_test(_Norm("vector", order, dim, keepdim), x)

    def test_linalg_matrix_norm(self):
        x = torch.randn(2, 3, 5, 5)
        for order in ["fro", float("inf"), -float("inf"), 1, -1]:
            self.run_test(_Norm("matrix", order, (-2, -1)), x)
            self.run_test(_Norm("matrix", order, (0, 2)), x)
            self.run_test(_Norm("matrix", order, (0, 2), True), x)

    def _argmin_argmax_model(self, input):
        class ArgminArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.argmin(input),
                    torch.argmax(input),
                    torch.argmin(input, keepdim=True),
                    torch.argmax(input, keepdim=True),
                    torch.argmin(input, dim=0, keepdim=True),
                    torch.argmax(input, dim=1, keepdim=True),
                )

        self.run_test(ArgminArgmaxModel(), input)

    def test_argmin_argmax(self):
        self._argmin_argmax_model(torch.randn(7, 3, 5))

        class AllDims(torch.nn.Module):
            def forward(self, x):
                return torch.argmin(x, keepdim=True), torch.argmax(x, keepdim=True)

        for shape in [(), (4,)]:
            self.run_test(AllDims(), torch.randn(shape))

    # "select_last_index" was added in opset 12 to deal with the corner case
    # where the same value appears multiple times in the tensor.
    def test_argmin_argmax_select_last_index(self):
        self._argmin_argmax_model(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]]))
        self._argmin_argmax_model(torch.ones(7, 3, 5))

    def test_linear(self):
        class LinearModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(16, 16)

            def forward(self, x):
                out = self.fc(x)
                return self.fc(out)

        self.run_test(LinearModel(), torch.randn(3, 16))

        class FunctionalLinearModel(torch.nn.Module):
            def forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

        # Input of rank 2.
        self.run_test(
            FunctionalLinearModel(),
            (torch.randn(2, 2), torch.randn(2, 2), torch.randn(1)),
        )
        # Input of rank 3.
        self.run_test(
            FunctionalLinearModel(),
            (torch.randn(3, 3, 3), torch.randn(3, 3), torch.randn(1)),
        )

    def test_meshgrid_indexing(self):
        class Meshgrid(torch.nn.Module):
            def __init__(self, indexing):
                super().__init__()
                self.indexing = indexing

            def forward(self, x, y, z):
                return torch.meshgrid(x, y, z, indexing=self.indexing)

        x = torch.randn(5, requires_grad=True)
        y = torch.zeros(6, requires_grad=True)
        z = torch.randn(7, requires_grad=True)
        for indexing in ("xy", "ij"):
            self.run_test(Meshgrid(indexing), (x, y, z))

    def test_round_with_decimals(self):
        class Round(torch.nn.Module):
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                return torch.round(x, decimals=self.decimals)

        x = torch.tensor([0.9920, -1234.0362, -1.58960, 3.5000])
        for decimals in (0, -2, 3):
            self.run_test(Round(decimals), x)

    def test_new_zeros_with_dtype(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(50, 64)

            def forward(self, x):
                inp = x.new_zeros(x.shape)
                return self.emb(inp)

        x = torch.tensor([[2, 5, 6], [3, 2, 5]], dtype=torch.int64)
        self.run_test(
            MyModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.ones(3, 5, dtype=torch.int64)],
        )
