# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Historical control-flow intent expressed through torch.export higher-order ops.

Source: test_pytorch_onnx_onnxruntime.py at
e2c2673d00500d268fcb1cc63fa7cc3173876505, test_loop_dynamic,
test_loop_nested, test_loop_transpose and test_if_view.
"""

import pytest
import torch
from _common.assertions import assert_program_outputs, export_and_assert


@pytest.mark.parametrize("length", [1, 3, 7])
def test_loop_dynamic(length):
    class Model(torch.nn.Module):
        def forward(self, x, steps):
            def condition(i, value):
                return i < steps

            def body(i, value):
                return i + 1, value + i

            return torch.while_loop(condition, body, (torch.tensor(0), x))[1]

    program = export_and_assert(
        Model().eval(),
        (torch.zeros(1, 2, 3, dtype=torch.int64), torch.tensor(3)),
        additional_test_inputs=[
            (torch.zeros(1, 2, 3, dtype=torch.int64), torch.tensor(length))
        ],
    )
    assert any(node.op_type == "Loop" for node in program.model.graph)


def test_loop_nested():
    class Model(torch.nn.Module):
        def forward(self, x):
            def outer_body(i, value):
                def inner_condition(a):
                    return a < 4

                def inner_body(a):
                    return (a + 1,)

                (a,) = torch.while_loop(inner_condition, inner_body, (i - i,))
                return i + 1, value + a

            return torch.while_loop(
                lambda i, value: i < 5, outer_body, (torch.tensor(0), x)
            )[1]

    x = torch.zeros(1, 2, 3, dtype=torch.int64)
    program = export_and_assert(Model().eval(), (x,))
    assert_program_outputs(program, (x,), expected=x + 20, rtol=0, atol=0)


def test_loop_transpose():
    class Model(torch.nn.Module):
        def forward(self, x, steps):
            def body(i, result):
                return i + 1, result + x[0].transpose(0, 1)

            return torch.while_loop(
                lambda i, result: i < steps,
                body,
                (torch.tensor(0), torch.zeros_like(x[0])),
            )[1]

    x = torch.randn(5, 3, 3)
    export_and_assert(
        Model().eval(),
        (x, torch.tensor(5)),
        additional_test_inputs=[(x, torch.tensor(count)) for count in (0, 2, 5)],
    )


@pytest.mark.parametrize("predicate", [False, True])
def test_if_view(predicate):
    class Model(torch.nn.Module):
        def forward(self, x, y, cond):
            def then_branch(x, y):
                bs, seq = y.shape[:2]
                return x.view(bs, seq, -1).clone()

            def else_branch(x, y):
                return y.clone()

            return torch.cond(cond, then_branch, else_branch, (x, y)).transpose(1, 2)

    x, y = torch.randn(2, 16, 2, 4), torch.randn(2, 16, 8)
    export_and_assert(
        Model().eval(),
        (x, y, torch.tensor(True)),
        dynamic_shapes=(
            {0: torch.export.Dim("batch")},
            {0: torch.export.Dim("batch")},
            None,
        ),
        additional_test_inputs=[
            (x, y, torch.tensor(predicate)),
            (
                torch.randn(3, 16, 2, 4),
                torch.randn(3, 16, 8),
                torch.tensor(predicate),
            ),
        ],
    )
