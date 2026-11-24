# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Floating out= arange fixtures recovered from the removed runtime suite.

Source: e2c2673d00500d268fcb1cc63fa7cc3173876505: test_pytorch_onnx_onnxruntime.py
2226, 2325 and 2335. Local output allocation is made functional for export;
the unchanged out= bodies remain independent value/shape/dtype oracles.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class _ArangeEndOut(torch.nn.Module):
    def forward(self, end):
        out_t = torch.tensor([1], dtype=torch.float)
        return torch.arange(end, out=out_t)


class _ArangeStepOut(torch.nn.Module):
    def forward(self, start, end):
        out_t = torch.tensor([1], dtype=torch.float)
        return torch.arange(start.size(0), end, 1.5, out=out_t)


class _ArangeStartOut(torch.nn.Module):
    def forward(self, start, end):
        out_t = torch.tensor([1], dtype=torch.float)
        return torch.arange(start.size(0), end, out=out_t)


class _FunctionalEnd(torch.nn.Module):
    def forward(self, end):
        return torch.arange(end, dtype=torch.float32)


class _FunctionalStart(torch.nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, start, end):
        return torch.arange(start.size(0), end, self.step, dtype=torch.float32)


class TestRecoveredFloatOutRanges(RecoveredRuntimeHarness):
    def _check_original(self, original, adapted, examples, **export_options):
        for args in examples:
            torch.testing.assert_close(adapted(*args), original(*args), rtol=0, atol=0)
        program = self.run_test(
            adapted,
            examples[0],
            additional_test_inputs=examples[1:],
            rtol=0,
            atol=0,
            **export_options,
        )
        try:
            self.assertIn("Range", {node.op_type for node in program.model.graph})
            self.assert_program_roundtrip(
                program,
                examples=examples,
                reference=original,
                rtol=0,
                atol=0,
            )
        finally:
            program.release()

    def _check_end(self):
        examples = [
            (torch.tensor(end, dtype=torch.float),) for end in (8.5, 0, 0.5, 13.25)
        ]
        self._check_original(_ArangeEndOut(), _FunctionalEnd(), examples)

    def _check_start(self, original, step):
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self._check_original(
            original,
            _FunctionalStart(step),
            [
                (x, y),
                (torch.randn(4, 5, 6), torch.tensor(13.25)),
                (torch.randn(3, 4, 5), torch.tensor(3.0)),
            ],
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self._check_original(
            original,
            _FunctionalStart(step),
            [(x, y), (x, torch.tensor(11.25))],
            remained_onnx_input_idx=[1],
        )

    def test_arange_with_floats_out(self):
        self._check_end()
        self._check_start(_ArangeStepOut(), 1.5)

    def test_arange_out(self):
        self._check_end()

    def test_arange_start_out(self):
        self._check_start(_ArangeStartOut(), 1)
