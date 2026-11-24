# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recover test_rnn at line 13974 of PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

The original packed-sequence wrappers are represented by a masked cell recurrence,
not by a claim that torch.export captures PackedSequence. Time and batch remain
symbolic, including reverse directions and every layer's initial/final states.
"""

import numpy as np
import torch
from test_recurrent_model_regressions import (
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
)
from test_runtime_regressions import RecoveredRuntimeHarness
from torch.testing._internal import common_utils
from torch.utils import _pytree


class _MaskedRecurrence(torch.nn.Module):
    def __init__(self, source, kind, packed):
        super().__init__()
        self.kind, self.packed = kind, packed
        self.layers, self.hidden = source.num_layers, source.hidden_size
        self.directions = 2 if source.bidirectional else 1
        self.batch_first = source.batch_first
        self.cells = torch.nn.ModuleList()
        for layer in range(self.layers):
            size = source.input_size if layer == 0 else self.hidden * self.directions
            for direction in range(self.directions):
                if kind in ("relu", "tanh"):
                    cell = torch.nn.RNNCell(size, self.hidden, nonlinearity=kind)
                elif kind == "lstm":
                    cell = torch.nn.LSTMCell(size, self.hidden)
                else:
                    cell = torch.nn.GRUCell(size, self.hidden)
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                with torch.no_grad():
                    for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                        getattr(cell, field).copy_(getattr(source, field + suffix))
                self.cells.append(cell)

    def _run_direction(self, sequence, hidden, state, cell, reverse, lengths):
        steps = torch.scalar_tensor(sequence.shape[0], dtype=torch.int64)
        output = sequence.new_zeros((sequence.shape[0], sequence.shape[1], self.hidden))

        def condition(index, hidden, state, result):
            return index < steps

        def body(index, hidden, state, result):
            position = steps - index - 1 if reverse else index
            row = sequence.index_select(0, position.reshape(1)).squeeze(0)
            if self.kind == "lstm":
                candidate_h, candidate_c = cell(row, (hidden, state))
            else:
                candidate_h = cell(row, hidden)
                candidate_c = state.clone()
            active = (position < lengths).unsqueeze(1)
            hidden = torch.where(active, candidate_h, hidden)
            state = torch.where(active, candidate_c, state)
            row_output = torch.where(active, hidden, torch.zeros_like(hidden))
            result = result.index_copy(0, position.reshape(1), row_output.unsqueeze(0))
            return index + 1, hidden, state, result

        _, hidden, state, output = torch.while_loop(
            condition, body, (torch.tensor(0), hidden, state, output)
        )
        return hidden, state, output

    def forward(self, data, initial, lengths):
        values = data.transpose(0, 1) if self.batch_first else data
        if initial is None:
            h0 = data.new_zeros(
                (self.layers * self.directions, values.shape[1], self.hidden)
            )
            c0 = torch.zeros_like(h0)
        elif self.kind == "lstm":
            h0, c0 = initial
        else:
            h0 = initial
            c0 = torch.zeros_like(h0)
        if not self.packed:
            lengths = torch.full(
                (values.shape[1],),
                values.shape[0],
                dtype=torch.int64,
                device=data.device,
            )
        final_h, final_c = [], []
        for layer in range(self.layers):
            directions = []
            for direction in range(self.directions):
                cell_index = layer * self.directions + direction
                hidden, state, output = self._run_direction(
                    values,
                    h0[cell_index],
                    c0[cell_index],
                    self.cells[cell_index],
                    bool(direction),
                    lengths,
                )
                final_h.append(hidden)
                final_c.append(state)
                directions.append(output)
            values = torch.cat(directions, dim=2)
        if self.batch_first:
            values = values.transpose(0, 1)
        if self.kind == "lstm":
            return values, torch.stack(final_h), torch.stack(final_c)
        return values, torch.stack(final_h)


@common_utils.instantiate_parametrized_tests
class TestRecoveredRecurrentMatrix(RecoveredRuntimeHarness):
    @common_utils.parametrize("kind", ("relu", "tanh", "lstm", "gru"))
    @common_utils.parametrize("layers", (1, 3))
    @common_utils.parametrize("bidirectional", (True, False))
    @common_utils.parametrize("initial_state", (True, False))
    @common_utils.parametrize("packed_sequence", (0, 1, 2))
    def test_rnn(self, kind, layers, bidirectional, initial_state, packed_sequence):
        cls = (
            torch.nn.RNN
            if kind in ("relu", "tanh")
            else torch.nn.LSTM
            if kind == "lstm"
            else torch.nn.GRU
        )
        kwargs = {"nonlinearity": kind} if kind in ("relu", "tanh") else {}
        source = cls(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            bidirectional=bidirectional,
            dropout=0.2,
            batch_first=packed_sequence == 2,
            **kwargs,
        ).eval()
        without_dropout = cls(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            bidirectional=bidirectional,
            dropout=0.0,
            batch_first=packed_sequence == 2,
            **kwargs,
        ).eval()
        without_dropout.load_state_dict(source.state_dict())
        directions = 2 if bidirectional else 1

        def make_input(batch_size, time_limit=RNN_SEQUENCE_LENGTH):
            seq_lengths = np.random.randint(1, time_limit + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            data = torch.nn.utils.rnn.pad_sequence(
                [torch.randn(length, RNN_INPUT_SIZE) for length in seq_lengths],
                batch_first=packed_sequence == 2,
            )
            initial = None
            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                initial = (h0, torch.randn_like(h0)) if kind == "lstm" else h0
            lengths = torch.IntTensor(seq_lengths) if packed_sequence else None
            return data, initial, lengths

        def reference(model, args):
            data, initial, lengths = args
            if packed_sequence:
                data = torch.nn.utils.rnn.pack_padded_sequence(
                    data, lengths, batch_first=packed_sequence == 2
                )
            output, state = model(data, initial)
            if packed_sequence:
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=packed_sequence == 2
                )
            return _pytree.tree_flatten((output, state))[0]

        args = make_input(RNN_BATCH_SIZE)
        # test that the model still runs with a different batch size
        changed = make_input(RNN_BATCH_SIZE + 1)
        shorter = make_input(RNN_BATCH_SIZE + 2, time_limit=5)
        examples = (args, changed, shorter)
        batch = torch.export.Dim("recurrent_batch", min=2)
        time = torch.export.Dim("recurrent_time", min=2)
        state_shapes = None
        if initial_state:
            state_shapes = ({1: batch}, {1: batch}) if kind == "lstm" else {1: batch}
        shapes = (
            {0: batch, 1: time} if packed_sequence == 2 else {0: time, 1: batch},
            state_shapes,
            {0: batch} if packed_sequence else None,
        )
        model = _MaskedRecurrence(source, kind, bool(packed_sequence))
        program = self.run_test(
            model,
            args,
            dynamic_shapes=shapes,
            additional_test_inputs=[changed, shorter],
        )
        self.assertEqual(
            sum(node.op_type == "Loop" for node in program.model.graph),
            layers * directions,
        )
        self.assertNotIsInstance(program.model.graph.inputs[0].shape[0], int)
        self.assertNotIsInstance(program.model.graph.inputs[0].shape[1], int)
        reloaded = self.reload_program(program)
        for values in examples:
            expected = reference(source, values)
            # Both original dropout configurations run in eval mode and use
            # identical weights, so share one export without losing a case.
            torch.testing.assert_close(
                reference(without_dropout, values),
                expected,
                rtol=1e-3,
                atol=1e-7,
            )
            for candidate in (program, reloaded):
                self.assert_program_outputs(
                    candidate,
                    values,
                    expected=expected,
                    rtol=1e-3,
                    atol=1e-7,
                )
