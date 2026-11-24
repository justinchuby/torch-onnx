# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Branch/alias regressions from PyTorch's deleted ONNX runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. torch.cond preserves data/shape
dependent branches; fresh contiguous outputs meet its nonaliasing contract.
Original fixed-loop bodies/comments are retained, without claiming dynamic
trip counts. Original and changed inputs run through serialization checks.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnx_ir as ir
import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor

import torch_onnx


class TestRecoveredBranchesAndAliases(RecoveredRuntimeHarness):
    def test_uninitialized_dynamic(self):
        class UninitializedModel(torch.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        class DynamicModel(torch.nn.Module):
            def forward(self, y):
                rows = torch.scalar_tensor(y.shape[0], dtype=torch.int64)
                columns = torch.scalar_tensor(y.shape[1], dtype=torch.int64)
                return torch.cond(
                    (rows == 1) & (columns < 5),
                    lambda value: value + 4,
                    lambda value: value.clone(),
                    (y,),
                )

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((6, 7), dtype=torch.int)
        program = self.run_test(
            DynamicModel(),
            x,
            input_names=["input_1"],
            dynamic_shapes=(
                {
                    0: torch.export.Dim("rows", min=2),
                    1: torch.export.Dim("columns", min=2),
                },
            ),
            additional_test_inputs=[y],
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic-return.onnx"
            program.save(path, external_data=False)
            loaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        original = UninitializedModel()
        # As in the conditional-state cases, ONNX execution covers 0/1 even
        # though the attached ExportedProgram's capture guards exclude them.
        for value in (
            x,
            y,
            torch.arange(4, dtype=torch.int).reshape(1, 4),
            torch.ones(1, 6, dtype=torch.int),
            torch.empty(0, 4, dtype=torch.int),
        ):
            for candidate in (program, loaded):
                output = candidate(value)
                self.assertEqual(len(output), 1)
                torch.testing.assert_close(output[0], original(value), rtol=0, atol=0)

    def test_uninitialized(self):
        class UninitializedModel(torch.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(
            UninitializedModel(),
            x,
            additional_test_inputs=[torch.arange(12, dtype=torch.int).reshape(3, 4)],
        )
        for shape in [(1, 4), (2, 6)]:
            self.run_test(
                UninitializedModel(), torch.randint(-5, 5, shape, dtype=torch.int)
            )

    def test_uninitialized_tensorList(self):
        class UninitializedTensorListModel(torch.nn.Module):
            def forward(self, x):
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        return [x]
                return [x]

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(
            UninitializedTensorListModel(),
            x,
            additional_test_inputs=[torch.arange(12, dtype=torch.int).reshape(3, 4)],
        )
        for shape in [(1, 4), (2, 6)]:
            self.run_test(
                UninitializedTensorListModel(),
                torch.randint(-5, 5, shape, dtype=torch.int),
            )

    def _check_conditional_state(self, original, hidden_size, *, threshold=1):
        class ConditionalExample(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.original = original

            def forward(self, data, previous):
                count = torch.scalar_tensor(previous.shape[0], dtype=torch.int64)

                def selected(data):
                    if threshold == 1:
                        return self.original(data, data.new_empty((0, 0, 0, 0, 0)))
                    return torch.cond(
                        count == 0,
                        lambda value: self.original(
                            value, value.new_empty((0, 0, 0, 0, 0))
                        ),
                        lambda value: self.original(
                            value, value.new_empty((1, 0, 0, 0, 0))
                        ),
                        (data,),
                    )

                def other(data):
                    return self.original(data, data.new_empty((threshold, 0, 0, 0, 0)))

                return torch.cond(count < threshold, selected, other, (data,))

        # These source bodies inspect only previous.shape[0], not its values.
        # Shape-only sentinels reuse their original mutations and aliasing setup.
        model = ConditionalExample().eval()
        data = torch.rand(2, 5, 4, 5)
        previous = torch.rand(3, 2, hidden_size, 4, 5)
        dimensions = (
            {
                0: torch.export.Dim("data_batch", min=2),
                1: torch.export.Dim("data_channels", min=2),
                2: torch.export.Dim("height", min=2),
                3: torch.export.Dim("width", min=2),
            },
            {i: torch.export.Dim(f"previous_{i}", min=2) for i in range(5)},
        )
        program = torch_onnx.export(
            model, (data, previous), dynamic_shapes=dimensions, verbose=False
        )
        onnx.checker.check_model(program.model_proto, full_check=True)
        self.assertIn("If", {node.op_type for node in program.model.graph})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "states.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        original_extent = 30 if hidden_size == 10 else 4
        examples = [
            (
                torch.rand(1, 5, original_extent, original_extent),
                torch.empty(0, 0, 0, 0, 0),
            ),
            (data, previous),
            (torch.rand(3, 3, 3, 6), torch.rand(1, 3, hidden_size, 3, 6)),
            (torch.rand(2, 7, 5, 3), torch.rand(2, 2, hidden_size, 5, 3)),
        ]
        # Capture uses nonsingleton examples because torch.export specializes 0/1.
        # Check those original cases against the source module directly: the
        # attached ExportedProgram's guards remain narrower than ONNX execution.
        for value, state in examples:
            expected = original(value, state)
            expected = expected if isinstance(expected, tuple) else (expected,)
            for candidate in [program, reloaded]:
                actual = candidate(value, state)
                self.assertEqual(len(actual), len(expected))
                for actual_tensor, expected_tensor in zip(
                    actual, expected, strict=True
                ):
                    torch.testing.assert_close(
                        actual_tensor, expected_tensor, rtol=0, atol=0
                    )
                    self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                    self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)

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

    def test_index_put_if(self):
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    + state[:]
                )
                state_copy[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state_copy[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
            else:
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            return state, state_copy

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        self._check_conditional_state(Example(10), 10, threshold=1)

    def test_index_put_if_3(self):
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                state = state * 3
                if prev_state.size(0) == 0:
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * 3
                    )
                else:
                    state = state + 2

            return state

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        self._check_conditional_state(Example(4), 4, threshold=2)

    def test_index_put_if_4(self):
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        self._check_conditional_state(Example(4), 4, threshold=1)

    def test_index_put_if_5(self):
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_ref = state
            if prev_state.size(0) == 0:
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state, state_ref

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state, state_ref = check_init(
                    input_data, self.hidden_size, prev_state
                )
                return prev_state, state_ref

        self._check_conditional_state(Example(4), 4, threshold=1)

    def test_inplace_attr_with_loop(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("_bias", torch.arange(12))

            def forward(self, x):
                self._bias = torch.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias += torch.arange(
                                12,
                            )
                return self._bias + x

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(
            m, x, additional_test_inputs=[torch.randn(12), -torch.arange(12.0)]
        )

    def test_inplace_attr_copy_with_loop(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("_bias", torch.arange(12))

            def forward(self, x):
                self._bias = torch.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias.copy_(
                                torch.arange(
                                    12,
                                )
                            )
                        self._bias.copy_(
                            self._bias
                            + torch.arange(
                                12,
                            )
                        )

                    self._bias.copy_(
                        self._bias
                        + torch.arange(
                            12,
                        )
                    )
                return self._bias + x

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(
            m, x, additional_test_inputs=[torch.randn(12), -torch.arange(12.0)]
        )
