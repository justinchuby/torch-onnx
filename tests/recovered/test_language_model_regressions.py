# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recurrent language-model regressions from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408.
"""

from __future__ import annotations

import torch
from model_defs import word_language_model
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredLanguageModels(RecoveredRuntimeHarness):
    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        if model_name == "GRU":
            model = word_language_model.RNNModelWithTensorHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        elif model_name == "LSTM":
            model = word_language_model.RNNModelWithTupleHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        else:
            model = word_language_model.RNNModel(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        hidden = model.hidden
        alternate_hidden = (
            tuple(torch.full_like(state, 0.1) for state in hidden)
            if isinstance(hidden, tuple)
            else torch.full_like(hidden, 0.1)
        )
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(
            model,
            (x, hidden),
            additional_test_inputs=[((x + 1) % ntokens, alternate_hidden)],
        )

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")
