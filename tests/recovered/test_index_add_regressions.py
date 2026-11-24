# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Index-add/roll cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
index/update tensors and in-place computations; use torch.cond for control flow.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class _IndexAdd(torch.nn.Module):
    def __init__(self, dim, index, updates, loop_count=1):
        super().__init__()
        self.dim = dim
        self.index = index
        self.updates = updates
        self.loop_count = loop_count

    def forward(self, x):
        for _ in range(self.loop_count):
            x.index_add_(self.dim, self.index, self.updates)
        return x


class TestRecoveredIndexAdd(RecoveredRuntimeHarness):
    def test_index_add_normal(self):
        x = torch.ones(5, 1)
        updates = torch.tensor([[1], [4], [7], [3], [2]], dtype=torch.float)
        self.run_test(_IndexAdd(0, torch.tensor([0, 2, 3, 1, 4]), updates), x)
        x = torch.ones(1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        self.run_test(_IndexAdd(1, torch.tensor([0, 2, 3, 1]), updates), x)
        updates = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]], dtype=torch.float
        )
        self.run_test(_IndexAdd(2, torch.tensor([0, 2, 1]), updates), x)

    def test_index_add_dim_size_differ(self):
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6]]], dtype=torch.float)
        self.run_test(
            _IndexAdd(1, torch.tensor([0, 2, 1]), updates), torch.ones(1, 4, 3)
        )

    def test_index_add_in_loop(self):
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        loop_count = torch.randint(20, (1,))[0].item()
        self.run_test(
            _IndexAdd(1, torch.tensor([0, 2, 3, 1]), updates, loop_count),
            torch.ones(1, 4, 3),
        )

    def test_index_add_if(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.updates = torch.tensor(
                    [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]],
                    dtype=torch.float,
                )
                self.index_true = torch.tensor([0, 2, 3, 1])
                self.index_false = torch.tensor([1, 0, 2, 3])

            def forward(self, x, cond):
                def true(x):
                    result = x.clone()
                    result.index_add_(1, self.index_true, self.updates)
                    return result

                def false(x):
                    result = x.clone()
                    result.index_add_(1, self.index_false, self.updates)
                    return result

                return torch.cond(cond, true, false, (x,))

        program = self.run_test(
            M(),
            (torch.ones(1, 4, 3), torch.tensor(True)),
            additional_test_inputs=[(torch.ones(1, 4, 3), torch.tensor(False))],
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})

    def test_index_add_dynamic_axes(self):
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        self.run_test(
            _IndexAdd(1, torch.tensor([0, 2, 3, 1]), updates),
            torch.ones(1, 4, 3),
            input_names=["input_1"],
            dynamic_axes={"input_1": [1]},
            additional_test_inputs=[torch.ones(1, 6, 3)],
        )

    def test_roll(self):
        class M(torch.nn.Module):
            def __init__(self, shifts, dims):
                super().__init__()
                self.shifts = shifts
                self.dims = dims

            def forward(self, x):
                return torch.roll(x, self.shifts, self.dims)

        x = torch.randn(2, 3, 4)
        for shifts, dims in [
            ([1, 1], [1, 0]),
            ([0, 1, 2], [1, 0, 2]),
            (2, 1),
            ([-1, 3], [-2, -1]),
        ]:
            self.run_test(M(shifts, dims), x)
