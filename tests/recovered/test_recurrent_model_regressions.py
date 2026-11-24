# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recurrent cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Dimensions come from
the same parent's pytorch_test_common.py. Native optimization replaces the
retired do_constant_folding flag.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor

BATCH_SIZE = 2
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
    def test_lstm_no_hidden(self):
        self.run_test(torch.nn.LSTM(16, 16), torch.randn(10, 16, 16))

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
