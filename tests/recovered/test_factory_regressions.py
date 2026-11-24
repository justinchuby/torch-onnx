# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Factory and deterministic sampling regressions from removed PyTorch tests.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Original random-factory
tests deliberately cancel random values or inspect their shapes/dtypes.
Multinomial inputs have exactly one positive weight per row.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredFactories(RecoveredRuntimeHarness):
    def test_randint(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                randint = torch.randint(1, 10, x.shape)
                x = 0 * randint + x
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randint_value(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return torch.randint(3, 4, x.shape) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randint_like(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return torch.randint_like(x, 3, 4) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randn(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand_dtype(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dynamic_size(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(
            RandN(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[torch.randn(4, 2, 5)],
        )

    def test_rand_dynamic_size(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(
            Rand(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[torch.randn(4, 2, 5)],
        )

    def test_randn_like(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like(self):
        class RandLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_randn_like_dtype(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        class RandLike(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        class Bernoulli(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x).size(0))

        x = torch.empty(3, 3).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

        x = torch.empty(2, 3, 3, dtype=torch.double).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

    def test_bernoulli_p(self):
        class Bernoulli_float(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x, 0.2).size(0))

        class Bernoulli_tensor(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).bernoulli_(x).size(0))

        x = torch.rand(3, 3)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

        x = torch.rand(2, 3, 3, dtype=torch.double)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

    def test_multinomial(self):
        class Multinomial(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 1)

        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        self.run_test(Multinomial(), (weight,))
        self.run_test(MultinomialNoReplacement(), (weight,))

    def test_new_empty(self):
        class Empty(torch.nn.Module):
            def forward(self, x):
                return (
                    x.new_empty(x.shape[0]).fill_(0),
                    x.new_empty(x.shape[0], dtype=torch.long) * 0,
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Empty(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Empty(), x, remained_onnx_input_idx=[])

    def test_dropout(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        x = torch.randn(10, 3, 53)
        self.run_test(M(), (x))
