# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Data-scalar cases from the removed PyTorch ONNX runtime suite.

Source: e2c2673d00500d268fcb1cc63fa7cc3173876505, removed by
6dca9f63bc4975ac323044a440c2c7e5f8512408. Functional masks/classes avoid
Python integer specialization. Linspace uses an explicit capture boundary
with the unchanged eager body and the package's existing native translation.
"""

import onnx_ir as ir
import torch
from test_runtime_regressions import RecoveredRuntimeHarness

import torch_onnx
from torch_onnx._torchlib.ops import factories


class _InputIndexSlice(torch.nn.Module):
    def forward(self, x, y, z):
        x[:y:z, 0::z, :] = 1
        return x


class _FunctionalInputStep(torch.nn.Module):
    def forward(self, x, y, z):
        rows = torch.arange(x.shape[0], device=x.device)
        columns = torch.arange(x.shape[1], device=x.device)
        end = torch.where(y < 0, x.shape[0] + y, y)
        selected = ((rows < end) & (rows.remainder(z) == 0))[:, None] & (
            columns.remainder(z) == 0
        )[None, :]
        return torch.where(selected[:, :, None], torch.ones_like(x), x)


class _OneHotFixed(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.nn.functional.one_hot(x, self.num_classes)


class _OneHotTensor(torch.nn.Module):
    def forward(self, x, num_classes):
        num_classes = num_classes.to(torch.int32)
        return torch.nn.functional.one_hot(x, num_classes[0])


class _FunctionalOneHot(torch.nn.Module):
    def forward(self, x, num_classes):
        classes = torch.arange(num_classes.to(torch.int32)[0], device=x.device)
        return (x.unsqueeze(-1) == classes).to(torch.int64)


class _LinspaceModel(torch.nn.Module):
    def forward(self, start, end, steps):
        return torch.linspace(start, end, steps)


class _NativeLinspace(torch.nn.Module):
    def forward(self, start, end, steps):
        return torch.ops.recovered_tensor_linspace.make.default(start, end, steps)


class TestRecoveredDataScalarOperators(RecoveredRuntimeHarness):
    def _check_original_roundtrip(self, program, examples, original, *, mutates=False):
        def reference(*args):
            original_args = (args[0].clone(), *args[1:]) if mutates else args
            return original(*original_args)

        try:
            self.assert_program_roundtrip(
                program,
                examples=examples,
                reference=reference,
                rtol=1e-6,
                atol=1e-7,
            )
        finally:
            program.release()

    def test_slice_with_input_step_size(self):
        examples = [
            (torch.zeros(56, 6, 256), torch.tensor(5), torch.tensor(2)),
            (torch.randn(11, 8, 7), torch.tensor(-1), torch.tensor(3)),
            (torch.randn(7, 5, 6), torch.tensor(0), torch.tensor(1)),
            (torch.randn(6, 4, 3), torch.tensor(70), torch.tensor(7)),
        ]
        program = self.run_test(
            _FunctionalInputStep(),
            examples[0],
            additional_test_inputs=examples[1:],
            dynamic_shapes=(
                {
                    0: torch.export.Dim("rows", min=2),
                    1: torch.export.Dim("columns", min=2),
                    2: torch.export.Dim("depth", min=2),
                },
                None,
                None,
            ),
        )
        self.assertIn("Mod", {node.op_type for node in program.model.graph})
        self._check_original_roundtrip(
            program, examples, _InputIndexSlice(), mutates=True
        )

    def test_one_hot(self):
        x = torch.arange(10)
        fixed = self.run_test(_OneHotFixed(15), (x,))
        fixed.release()
        examples = [
            (x, 15 * torch.ones(1)),
            (x.flip(0), 10 * torch.ones(1)),
            (x.roll(3), 21 * torch.ones(1)),
        ]
        program = self.run_test(
            _FunctionalOneHot(),
            examples[0],
            additional_test_inputs=examples[1:],
            rtol=0,
            atol=0,
        )
        self.assertIn("Range", {node.op_type for node in program.model.graph})
        self.assertNotIsInstance(program.model.graph.outputs[0].shape[1], int)
        self._check_original_roundtrip(program, examples, _OneHotTensor())

    def _linspace(self, start, end, steps):
        def fake(start, end, steps):
            count = torch.library.get_ctx().new_dynamic_size(min=0)
            return start.new_empty((count,), dtype=torch.float32)

        def translation(
            op: torch_onnx.TranslationContext,
            start: ir.Value,
            end: ir.Value,
            steps: ir.Value,
        ) -> ir.Value:
            return factories.aten_linspace(op, start, end, steps)

        examples = [
            (
                torch.tensor(start, dtype=torch.float),
                torch.tensor(end, dtype=torch.float),
                torch.tensor(steps, dtype=torch.int),
            )
        ]
        examples.extend(
            (
                torch.tensor(-3.25),
                torch.tensor(7.75),
                torch.tensor(count, dtype=torch.int),
            )
            for count in (0, 1, 2, 17)
        )
        with torch.library._scoped_library(
            "recovered_tensor_linspace", "FRAGMENT"
        ) as library:
            qualified = "recovered_tensor_linspace::make"
            torch.library.define(
                qualified,
                "(Tensor start, Tensor end, Tensor steps) -> Tensor",
                lib=library,
            )
            torch.library.impl(qualified, "CompositeExplicitAutograd", lib=library)(
                _LinspaceModel().forward
            )
            torch.library.register_fake(qualified, lib=library)(fake)
            target = torch.ops.recovered_tensor_linspace.make.default
            torch_onnx.register_custom_translation(target, translation)
            try:
                program = self.run_test(
                    _NativeLinspace(),
                    examples[0],
                    additional_test_inputs=examples[1:],
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertIn("Range", {node.op_type for node in program.model.graph})
                self.assertNotIsInstance(program.model.graph.outputs[0].shape[0], int)
                self._check_original_roundtrip(program, examples, _LinspaceModel())
            finally:
                torch_onnx.unregister_custom_translation(target)

    def test_linspace(self):
        self._linspace(3, 10, 5)

    def test_linspace_negative_start(self):
        self._linspace(-1, 1, 6)
