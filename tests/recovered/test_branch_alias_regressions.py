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
    def test_index_put_loop(self):
        def ngram_attention_bias(sequence_length, ngram, device, dtype):
            bias = torch.ones(
                (ngram, sequence_length), device=device, dtype=dtype
            ) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class Original(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                seq_length = hidden_states.shape[0]
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions,
                    self.ngram,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                return predict_causal_mask[:, :seq_length]

        class LoopModel(Original):
            def forward(self, hidden_states):
                bias = torch.full(
                    (self.ngram, self.max_target_positions),
                    -float("inf"),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                zero = torch.tensor(0)
                five = torch.tensor(5.0)

                def condition(stream, bias):
                    return stream < self.ngram

                def first(stream, bias):
                    def inner_condition(i, value):
                        return i < self.max_target_positions

                    def body(i, value):
                        value = value * 2
                        value = value.index_put(
                            (stream.unsqueeze(0), i.unsqueeze(0)), five
                        )
                        value = value * 5
                        value = value.index_put(
                            (zero.unsqueeze(0), zero.unsqueeze(0)), five
                        )
                        return i + 1, value

                    return stream + 1, torch.while_loop(
                        inner_condition, body, (zero.clone(), bias)
                    )[1]

                def second(stream, bias):
                    def inner_condition(i, value):
                        return i < self.max_target_positions

                    def body(i, value):
                        value = value.index_put(
                            (stream.unsqueeze(0), i.unsqueeze(0)), five
                        )
                        value = value.index_put(
                            (zero.unsqueeze(0), i.unsqueeze(0)), five
                        )
                        return i + 1, value

                    return stream + 1, torch.while_loop(
                        inner_condition, body, (zero.clone(), bias)
                    )[1]

                bias = torch.while_loop(condition, first, (zero.clone(), bias))[1]
                bias = torch.while_loop(condition, second, (zero.clone(), bias))[1]
                return bias[:, : hidden_states.shape[0]]

        x = torch.randn(6, 2)
        program = self.run_test(
            LoopModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": {0: "seq_length", 1: "batch_size"}},
        )
        self.assertEqual(
            sum(node.op_type == "Loop" for node in program.model.graph.all_nodes()), 4
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested-index-put.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            for value in (x, torch.randn(4, 1), torch.randn(2, 3), torch.randn(512, 2)):
                expected = Original()(value)
                for candidate in (program, reloaded):
                    self.assertEqual(candidate(value)[0], expected)

    def test_set_attr_modules(self):
        class InnerModule2(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule2.get_embedding(embedding_dim)
                self._float_tensor = torch.nn.Buffer(torch.FloatTensor(1))
                self.const = 2

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                return torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )

            def forward(self, input, incremental_state: Tensor | None = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 3
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.weights = self.weights.to(self._float_tensor)
                self.weights = self.weights * self.const
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                return self.weights.index_select(
                    0, torch.ones((bsz * seq_len), dtype=torch.int64)
                ).view(bsz, seq_len, -1)

        class InnerModule(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule.get_embedding(embedding_dim)
                self.module = InnerModule2(embedding_dim=8)

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                return torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )

            def forward(self, x):
                return self.module(x) + self.weights

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        class FunctionalModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("outer_weights", InnerModule.get_embedding(8))

            def forward(self, x, weights):
                bsz, seq_len = x.shape
                weights = weights.to(self.outer_weights) * 3
                result = weights.index_select(
                    0, torch.ones((bsz * seq_len), dtype=torch.int64)
                ).view(bsz, seq_len, -1)
                return result + self.outer_weights, weights

        x = torch.randn(3, 256)
        program = self.run_test(
            FunctionalModule(),
            (x, InnerModule2.get_embedding(8)),
            dynamic_shapes=(
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
                {},
            ),
        )
        fixed = self.run_test(
            FunctionalModule(),
            (x, InnerModule2.get_embedding(8)),
            remained_onnx_input_idx=[1],
        )
        self.assertFalse(fixed.model.graph.inputs[0].uses())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested-embedding-state.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            for candidate in (program, reloaded):
                reference = Module()
                state = reference.module.module.weights.clone()
                # Persistent updates are explicit inputs/outputs, not capture-time writes.
                for value in (x, torch.randn(4, 17), torch.randn(2, 2)):
                    expected = reference(value)
                    actual, state = candidate(value, state)
                    self.assertEqual(actual, expected)
                    self.assertEqual(state, reference.module.module.weights)

    def test_set_attr_modules_2(self):
        class InnerModule(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.const = 2.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                self._float_tensor = torch.nn.Buffer(torch.FloatTensor(1))

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            def forward(self, input, incremental_state: Tensor | None = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 1.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                return (
                    self.weights.index_select(
                        0, torch.ones((bsz * seq_len), dtype=torch.int64)
                    ).view(bsz, seq_len, -1)
                ) * self.const

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        x = torch.randn(3, 256)
        self.run_test(
            Module(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.randn(4, 17), torch.randn(2, 2)],
        )
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

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

    def _check_conditional_state(
        self, original, hidden_size, *, threshold=1, reads_previous=False
    ):
        class ConditionalExample(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.original = original

            def forward(self, data, previous):
                count = torch.scalar_tensor(previous.shape[0], dtype=torch.int64)

                def selected(data):
                    if threshold == 1:
                        return self.original(data, data.new_empty((0, 0, 0, 0, 0)))

                    def nonempty(value):
                        if reads_previous:
                            return torch.cond(
                                count == 1,
                                lambda data, previous: self.original(
                                    data, previous[:1]
                                ),
                                lambda data, previous: self.original(
                                    data, data.new_empty((2, 0, 0, 0, 0))
                                ),
                                (value, previous),
                            )
                        return self.original(value, value.new_empty((1, 0, 0, 0, 0)))

                    return torch.cond(
                        count == 0,
                        lambda value: self.original(
                            value, value.new_empty((0, 0, 0, 0, 0))
                        ),
                        nonempty,
                        (data,),
                    )

                def other(data):
                    return self.original(data, data.new_empty((threshold, 0, 0, 0, 0)))

                return torch.cond(count < threshold, selected, other, (data,))

        # Value-reading branches keep the actual state; other bodies need only
        # shape sentinels to retain their original mutations and aliasing setup.
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
        if reads_previous:
            dimensions = (
                dimensions[0],
                {
                    0: torch.export.Dim("previous_count", min=2),
                    1: dimensions[0][0],
                    3: dimensions[0][2],
                    4: dimensions[0][3],
                },
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
        if reads_previous:
            examples.append(
                (
                    torch.rand(1, 5, 30, 30),
                    torch.randn(1, 1, hidden_size, 30, 30),
                )
            )
        # Capture uses nonsingleton examples because torch.export specializes 0/1.
        # Check those original cases against the source module directly: the
        # attached ExportedProgram's guards remain narrower than ONNX execution.
        for value, state in examples:
            expected = original(value, state)
            expected = expected if isinstance(expected, tuple) else (expected,)
            adapted_state = state
            if reads_previous and state.shape[0] == 0:
                # Empty state carries no values; retain its count while supplying
                # the trailing dimensions required by the value-reading branch.
                adapted_state = state.new_empty(
                    (0, value.shape[0], hidden_size, *value.shape[2:])
                )
            for candidate in [program, reloaded]:
                actual = candidate(value, adapted_state)
                self.assertEqual(len(actual), len(expected))
                for actual_tensor, expected_tensor in zip(
                    actual, expected, strict=True
                ):
                    torch.testing.assert_close(
                        actual_tensor, expected_tensor, rtol=0, atol=0
                    )
                    self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                    self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
        return model

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

    def test_index_put_if_2(self):
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
                for i in range(2):
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
                    state_copy[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
            elif prev_state.size(0) == 1:
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
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

        original = Example(10)
        model = self._check_conditional_state(
            original, 10, threshold=3, reads_previous=True
        )
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.empty(0, 0, 0, 0, 0)
        self.run_test(original, (random_data, empty_tensor), remained_onnx_input_idx=[])
        captured_state = torch.randn(3, 1, 10, 30, 30)
        program = self.run_test(
            model,
            (random_data, captured_state),
            dynamic_shapes=(None, {0: torch.export.Dim("state_count", min=2)}),
            remained_onnx_input_idx=[1],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "value-state.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            for state in (
                empty_tensor,
                torch.rand(1, 1, 10, 30, 30),
                -torch.rand(1, 1, 10, 30, 30),
                torch.rand(2, 1, 10, 30, 30),
            ):
                expected = original(random_data, state)
                adapted_state = (
                    state.new_empty((0, 1, 10, 30, 30))
                    if state.shape[0] == 0
                    else state
                )
                for candidate in (program, reloaded):
                    self.assertEqual(
                        list(candidate(random_data, adapted_state)), list(expected)
                    )

    def test_index_put_inplace_ops(self):
        def check_init(input_data: Tensor, hidden_size: int) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if input_data.size(0) == 1:
                state[1] += (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
            for i in range(input_data.size(0)):
                state[1] += torch.ones(
                    batch_size, hidden_size, spatial_size_0, spatial_size_1
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * i
                )
            return state

        class Example(torch.nn.Module):
            def forward(self, input_data):
                return check_init(input_data, 10)

        class FunctionalExample(torch.nn.Module):
            def forward(self, data):
                state = torch.zeros(
                    (2, data.shape[0], 10, data.shape[2], data.shape[3]),
                    device=data.device,
                )

                def initial(value):
                    result = value.clone()
                    result[1] += torch.ones_like(result[1]) * 2
                    result[1] /= torch.ones_like(result[1]) * 3
                    return result

                count = torch.scalar_tensor(data.shape[0], dtype=torch.int64)
                state = torch.cond(
                    count == 1, initial, lambda value: value.clone(), (state,)
                )

                def condition(index, value):
                    return index < count

                def body(index, value):
                    result = value.clone()
                    result[1] += torch.ones_like(result[1])
                    result[1] /= torch.ones_like(result[1]) * index
                    return index + 1, result

                _, state = torch.while_loop(condition, body, (torch.tensor(0), state))
                return state

        data = torch.rand(3, 5, 4, 5)
        program = self.run_test(
            FunctionalExample(),
            data,
            dynamic_shapes=(
                {
                    0: torch.export.Dim("state_batch", min=2),
                    1: torch.export.Dim("data_channels", min=2),
                    2: torch.export.Dim("state_height", min=2),
                    3: torch.export.Dim("state_width", min=2),
                },
            ),
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})
        loop = next(
            node for node in program.model_proto.graph.node if node.op_type == "Loop"
        )
        loop_body = next(attr.g for attr in loop.attribute if attr.name == "body")
        self.assertIn("Div", {node.op_type for node in loop_body.node})
        original_data = torch.rand(1, 5, 30, 30)
        self.run_test(Example(), original_data, remained_onnx_input_idx=[])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inplace-state-loop.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            for candidate in (program, reloaded):
                for values in (
                    data,
                    original_data,
                    torch.rand(2, 7, 9, 11),
                    torch.empty(0, 5, 4, 5),
                ):
                    self.assertEqual(list(candidate(values)), [Example()(values)])

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
