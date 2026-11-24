# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Branch/alias regressions from PyTorch's deleted ONNX runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. torch.cond preserves data/shape
dependent branches; fresh contiguous outputs meet its nonaliasing contract.
Original fixed-loop bodies/comments are retained, without claiming dynamic
trip counts. Original and changed inputs run through serialization checks.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredBranchesAndAliases(RecoveredRuntimeHarness):
    def test_if_transpose(self):
        class IfModel(torch.nn.Module):
            def forward(self, x):
                x = x.transpose(0, 1)
                return torch.cond(
                    x.size(0) == 2,
                    lambda value: value.transpose(0, 1).clone(
                        memory_format=torch.contiguous_format
                    ),
                    lambda value: value.clone(memory_format=torch.contiguous_format),
                    (x,),
                )

        model = IfModel()
        original = torch.randn(2, 3)
        alternates = [torch.randn(3, 2), torch.randn(4, 5), torch.randn(5, 2)]
        for value in [original, *alternates]:
            expected = value.transpose(0, 1)
            if expected.size(0) == 2:
                expected = expected.transpose(0, 1)
            self.assertEqual(model(value), expected)
        program = self.run_test(
            model,
            original,
            input_names=["x"],
            output_names=["output_1"],
            dynamic_axes={"x": [0, 1], "output_1": [0, 1]},
            additional_test_inputs=alternates,
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})

    def test_if_list(self):
        class IfModel(torch.nn.Module):
            def forward(self, x, y, cond):
                return torch.cond(
                    cond,
                    lambda a, b: [a.clone()],
                    lambda a, b: [b.clone()],
                    (x, y),
                )

        x = torch.randn(2, 3)
        y = torch.randn(3, 3)
        model = IfModel()
        for flag in [True, False]:
            self.assertEqual(model(x, y, torch.tensor(flag)), [x] if flag else [y])
        program = self.run_test(
            model,
            (x, y, torch.tensor(True)),
            additional_test_inputs=[
                (x, y, torch.tensor(False)),
                (x + 1, y - 1, torch.tensor(True)),
                (x - 1, y + 1, torch.tensor(False)),
            ],
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})

    def test_empty_branch(self):
        class EmptyBranchModel(torch.nn.Module):
            def forward(self, input):
                out = input + 1
                if out.dim() > 2:
                    if out.dim() > 3:
                        out += 3
                    else:
                        pass
                else:
                    pass
                return out

        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(EmptyBranchModel(), x)
        for shape in [(2,), (2, 3), (1, 2, 3, 4)]:
            self.run_test(EmptyBranchModel(), torch.randn(shape))

    def test_inplace_with_loop(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.ones(
                    12,
                )
                for _ in range(10):
                    a.add_(
                        torch.ones(
                            12,
                        )
                    )
                return a + x

        m = M()
        x = torch.randn(
            12,
        )
        self.run_test(m, x, additional_test_inputs=[torch.randn(12)])

    def test_inplace_with_loop_2(self):
        class M(torch.nn.Module):
            def forward(self, x):
                _bias = torch.ones(
                    12,
                )
                a = torch.ones(
                    12,
                )  # used in loop, altered.
                _a_ref = a  # not used in loop, should be altered.
                b = x.clone()  # used in loop, not be altered.
                b_ref = b  # not used in loop, should not be altered.
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            a += _bias
                            _bias.add_(
                                torch.ones(
                                    12,
                                )
                            )
                            b = b + torch.ones(
                                12,
                            )

                    _bias.add_(
                        torch.ones(
                            12,
                        )
                    )
                    a += _bias
                # TODO: value for a_ref is incorrect.
                # a_ref += torch.ones(12,)
                b_ref += torch.ones(
                    12,
                )
                return _bias + x, a, b, b_ref

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(m, x, additional_test_inputs=[torch.randn(12)])
