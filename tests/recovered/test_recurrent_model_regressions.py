# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recurrent cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Dimensions come from
the same parent's pytorch_test_common.py. Native optimization replaces the
retired do_constant_folding flag.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx_ir as ir
import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor

import torch_onnx

BATCH_SIZE = 2
RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3


class _LSTMWithInitialState(torch.nn.Module):
    def __init__(
        self, *, batch_derived=False, bias=True, layers=1, bidirectional=False
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            bias=bias,
            bidirectional=bidirectional,
        )
        self.batch_derived = batch_derived

    def forward(self, input, initial_state: tuple[Tensor, Tensor] | None = None):
        if self.batch_derived:
            batch_size = input.size()[1]
            h0 = torch.ones([1, batch_size, RNN_HIDDEN_SIZE])
            c0 = torch.ones([1, batch_size, RNN_HIDDEN_SIZE])
            return self.lstm(input, (h0, c0))
        return self.lstm(input, initial_state)


class TestRecoveredRecurrentModels(RecoveredRuntimeHarness):
    def test_symbolic_shape_inference_time(self):
        class DynamicRNN(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input, initial):
                steps = torch.scalar_tensor(input.shape[0], dtype=torch.int64)
                output = input.new_zeros(
                    (input.shape[0], input.shape[1], RNN_HIDDEN_SIZE)
                )

                def condition(i, hidden, output):
                    return i < steps

                def body(i, hidden, output):
                    row = input.index_select(0, i.reshape(1)).squeeze(0)
                    hidden = torch.tanh(
                        torch.nn.functional.linear(row, self.model.weight_ih_l0)
                        + torch.nn.functional.linear(hidden, self.model.weight_hh_l0)
                    )
                    output = output.index_copy(0, i.reshape(1), hidden.unsqueeze(0))
                    return i + 1, hidden, output

                _, hidden, output = torch.while_loop(
                    condition,
                    body,
                    (torch.tensor(0), initial.squeeze(0), output),
                )
                return output, hidden.unsqueeze(0)

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        batch = torch.export.Dim("rnn_batch", min=2)
        time = torch.export.Dim("rnn_time", min=2)
        for kind in ("lstm", "gru", "rnn"):
            with self.subTest(kind=kind):
                cls = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU, "rnn": torch.nn.RNN}[
                    kind
                ]
                model = cls(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    1,
                    bidirectional=False,
                    bias=kind == "lstm",
                )
                initial = (h0, c0) if kind == "lstm" else h0
                changed_input = torch.randn(7, 4, RNN_INPUT_SIZE)
                changed_h = torch.randn(1, 4, RNN_HIDDEN_SIZE)
                changed_initial = (
                    (changed_h, torch.randn_like(changed_h))
                    if kind == "lstm"
                    else changed_h
                )
                state_shapes = (
                    ({1: batch}, {1: batch}) if kind == "lstm" else {1: batch}
                )
                exported_model = DynamicRNN(model) if kind == "rnn" else model
                program = self.run_test(
                    exported_model,
                    (input, initial),
                    input_names=["x", "y"],
                    dynamic_shapes=({0: time, 1: batch}, state_shapes),
                    additional_test_inputs=[(changed_input, changed_initial)],
                )
                if kind == "rnn":
                    self.assertIn(
                        "Loop", {node.op_type for node in program.model.graph}
                    )
                    for args in (
                        (input, initial),
                        (changed_input, changed_initial),
                    ):
                        self.assertEqual(program(*args), model(*args))
                self.assertNotIsInstance(program.model.graph.inputs[0].shape[0], int)
                self.assertNotIsInstance(program.model.graph.inputs[0].shape[1], int)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / f"dynamic-{kind}.onnx"
                    program.save(path, external_data=False)
                    reloaded = torch_onnx.ONNXProgram(
                        ir.load(path), program.exported_program
                    )
                    torch_onnx.testing.assert_onnx_program(
                        reloaded,
                        args=(changed_input, changed_initial),
                        kwargs={},
                        strategy=None,
                    )
                    if kind == "rnn":
                        self.assertEqual(
                            reloaded(changed_input, changed_initial),
                            model(changed_input, changed_initial),
                        )

    def test_rnn_no_bias(self):
        class RnnModelWithPackedSequence(torch.nn.Module):
            def __init__(self, model, batch_first):
                super().__init__()
                self.model, self.batch_first = model, batch_first

            def forward(self, input, *args):
                args, seq_lengths = args[:-1], args[-1]
                input = torch.nn.utils.rnn.pack_padded_sequence(
                    input, seq_lengths, self.batch_first
                )
                rets = self.model(input, *args)
                ret, rets = rets[0], rets[1:]
                ret, _ = torch.nn.utils.rnn.pad_packed_sequence(ret, self.batch_first)
                return (ret, *rets)

        class MaskedPackedRNN(torch.nn.Module):
            def __init__(self, model, batch_first):
                super().__init__()
                self.model, self.batch_first = model, batch_first

            def forward(self, input, initial, lengths):
                values = input.transpose(0, 1) if self.batch_first else input
                states = []
                for layer in range(self.model.num_layers):
                    hidden = initial[layer]
                    weight_ih = getattr(self.model, f"weight_ih_l{layer}")
                    weight_hh = getattr(self.model, f"weight_hh_l{layer}")
                    outputs = []
                    for step in range(values.shape[0]):
                        active = (step < lengths).unsqueeze(1)
                        candidate = torch.tanh(
                            torch.nn.functional.linear(values[step], weight_ih)
                            + torch.nn.functional.linear(hidden, weight_hh)
                        )
                        hidden = torch.where(active, candidate, hidden)
                        outputs.append(
                            torch.where(active, hidden, torch.zeros_like(hidden))
                        )
                    values = torch.stack(outputs)
                    states.append(hidden)
                output = values.transpose(0, 1) if self.batch_first else values
                return output, torch.stack(states)

        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        for layer_count, packed in zip(layers, packed_sequence, strict=True):
            with self.subTest(layers=layer_count, packed_sequence=packed):
                batch_first = packed == 2
                model = torch.nn.RNN(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    layer_count,
                    bidirectional=False,
                    batch_first=batch_first,
                    bias=False,
                )
                seq_lengths = np.random.randint(
                    1, RNN_SEQUENCE_LENGTH + 1, size=RNN_BATCH_SIZE
                )
                seq_lengths = sorted(map(int, seq_lengths), reverse=True)
                input = torch.nn.utils.rnn.pad_sequence(
                    [torch.randn(length, RNN_INPUT_SIZE) for length in seq_lengths],
                    batch_first=batch_first,
                )
                initial = torch.randn(layer_count, RNN_BATCH_SIZE, RNN_HIDDEN_SIZE)
                if packed == 0:
                    self.run_test(
                        model,
                        (input, initial),
                        additional_test_inputs=[(-input, -initial)],
                    )
                    continue
                original = RnnModelWithPackedSequence(model, batch_first)
                adapter = MaskedPackedRNN(model, batch_first)
                maximum = input.shape[1 if batch_first else 0]
                changed_lengths = (
                    torch.linspace(maximum, 1, RNN_BATCH_SIZE).round().to(torch.int32)
                )
                changed = torch.randn_like(input)
                time_major = changed.transpose(0, 1) if batch_first else changed
                # Packed execution must ignore even NaNs outside each sequence.
                for batch, length in enumerate(changed_lengths.tolist()):
                    time_major[length:, batch] = float("nan")
                examples = [
                    (input, initial, torch.IntTensor(seq_lengths)),
                    (changed, -initial, changed_lengths),
                ]
                for args in examples:
                    self.assertEqual(
                        adapter(*args), original(*args), rtol=1e-3, atol=1e-7
                    )
                program = self.run_test(
                    adapter, examples[0], additional_test_inputs=examples[1:]
                )
                self.assertIn("Where", {node.op_type for node in program.model.graph})
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "packed-rnn.onnx"
                    program.save(path, external_data=False)
                    reloaded = torch_onnx.ONNXProgram(
                        ir.load(path), program.exported_program
                    )
                    for args in examples:
                        expected = original(*args)
                        for candidate in (program, reloaded):
                            self.assertEqual(
                                candidate(*args), expected, rtol=1e-3, atol=1e-7
                            )

    def test_lstm_no_hidden(self):
        self.run_test(torch.nn.LSTM(16, 16), torch.randn(10, 16, 16))

    def test_lstm_proj_no_hidden(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                return self.rnn(x)

        with self.assertRaisesRegex(
            torch_onnx.errors.ConversionError,
            "LSTM projections are not supported.*proj_size=0",
        ):
            self.run_test(LSTMModel(), torch.randn(10, 16, 16))

    def test_lstm(self):
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(_LSTMWithInitialState(), (input, (h0, c0)))

    def test_lstm_cell(self):
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        for bias in [True, False]:
            self.run_test(
                torch.nn.LSTMCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, bias=bias),
                (input, (h0, c0)),
            )

    def test_lstm_default_init_state(self):
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(_LSTMWithInitialState(), input)

    def test_lstm_fixed_batch_size(self):
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # Verify with different input of the same batch size.
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(
            _LSTMWithInitialState(batch_derived=True),
            input,
            additional_test_inputs=[input2],
        )

    def test_lstm_post_fix_init_state(self):
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # Verify with different input of a different batch size.
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # Capture with batch two so torch.export does not specialize singleton one.
        self.run_test(
            _LSTMWithInitialState(batch_derived=True),
            input2,
            input_names=["input.1"],
            dynamic_axes={"input.1": {1: "batch"}},
            additional_test_inputs=[input],
        )

    def test_lstm_constant_folding(self):
        for i, h, n, b, s, bi in [(7, 3, 2, 3, 5, True), (5, 4, 3, 4, 7, False)]:
            model = torch.nn.LSTM(i, h, n, bidirectional=bi)
            input = torch.randn(s, b, i)
            h0 = torch.randn(n * (2 if bi else 1), b, h)
            c0 = torch.randn(n * (2 if bi else 1), b, h)
            self.run_test(model, (input, (h0, c0)))

    def test_lstm_no_bias(self):
        for layers, bidirectional in [(1, True), (1, False), (2, True), (3, False)]:
            input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            directions = 2 if bidirectional else 1
            h0 = torch.randn(layers * directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = torch.randn(layers * directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            model = _LSTMWithInitialState(
                bias=False, layers=layers, bidirectional=bidirectional
            )
            self.run_test(model, (input, (h0, c0)))

    def test_lstm_sequence(self):
        class LstmNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn1 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear1 = torch.nn.Linear(8 * 2, 8)
                self.rnn2 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear2 = torch.nn.Linear(8 * 2, 8)

            def forward(self, input):
                rnn_output1, _ = self.rnn1(input)
                linear_output1 = self.linear1(rnn_output1)
                rnn_output2, _ = self.rnn2(linear_output1)
                return self.linear2(rnn_output2)

        self.run_test(
            LstmNet(),
            torch.zeros((2, 100, 8)),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size", 1: "w"}},
            additional_test_inputs=[torch.zeros((1, 100, 8)), torch.randn(3, 73, 8)],
        )

    def test_gru_no_bias(self):
        for i, h, n, b, s, bi in [(7, 3, 2, 3, 5, True), (5, 4, 3, 4, 7, False)]:
            model = torch.nn.GRU(i, h, n, bidirectional=bi, bias=False)
            input = torch.randn(s, b, i)
            h0 = torch.randn(n * (2 if bi else 1), b, h)
            self.run_test(model, (input, h0))

    def test_gru_constant_folding(self):
        for i, h, n, b, s, bi in [(7, 3, 2, 3, 5, True), (5, 4, 3, 4, 7, False)]:
            model = torch.nn.GRU(i, h, n, bidirectional=bi)
            input = torch.randn(s, b, i)
            h0 = torch.randn(n * (2 if bi else 1), b, h)
            self.run_test(model, (input, h0))
