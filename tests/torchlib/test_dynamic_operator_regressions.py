# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Operator cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
inputs and model bodies, replacing ScriptModule with modern capture.
"""

import torch
from _common.runtime import RecoveredRuntimeHarness


class TestRecoveredDynamicOperators(RecoveredRuntimeHarness):
    def test_prelu(self):
        class PReluModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                return self.prelu(x)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        self.run_test(
            PReluModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_relu6(self):
        class Relu6Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu6 = torch.nn.ReLU6()

            def forward(self, x):
                return self.relu6(x)

        x = torch.randn(2, 3, 4) * 100.0
        y = torch.randn(2, 4, 5) * 100.0
        self.run_test(
            Relu6Model(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_silu(self):
        class SiLUModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.silu = torch.nn.SiLU()

            def forward(self, x):
                return self.silu(x)

        self.run_test(SiLUModel(), torch.randn(2, 3, 4))

    def test_mish(self):
        class MishModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mish = torch.nn.Mish()

            def forward(self, x):
                return self.mish(x)

        self.run_test(MishModel(), torch.randn(2, 3, 4))

    def test_remainder_scalar(self):
        class RemainderModel(torch.nn.Module):
            def __init__(self, scalar=2.55):
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                return torch.remainder(input, self.scalar)

        self.run_test(RemainderModel(), torch.randint(10, (2, 3)))
        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        self.run_test(RemainderModel(2), x)

    def test_index_copy(self):
        class IndexCopyModel(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, input):
                index = torch.tensor([2, 0])
                source = torch.ones(3, 2, 5)
                return input.index_copy(self.dim, index, source)

        x = torch.randn(3, 4, 5, requires_grad=True)
        for dim in (1, -2):
            self.run_test(IndexCopyModel(dim), x)

    def test_index_select_scaler_index(self):
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def __init__(self, index_base):
                super().__init__()
                self.index_base = torch.tensor(index_base)

            def forward(self, x, index_offset):
                index = self.index_base + index_offset
                return torch.index_select(x, 1, index)

        x = torch.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(1), (x, torch.tensor(2)))

    def test_slice_dynamic(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        self.run_test(
            DynamicSliceExportMod(),
            torch.rand(5, 5, 5),
            additional_test_inputs=[torch.randn(6, 7, 8)],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3", "output_4"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_slice_dynamic_script(self):
        class DynamicSliceModel(torch.nn.Module):
            def forward(self, x):
                return x[1 : x.size(1)]

        self.run_test(DynamicSliceModel(), torch.rand(1, 2))
        self.run_test(
            DynamicSliceModel(),
            torch.rand(3, 2),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.rand(5, 4)],
        )

    def test_arange_dynamic(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        self.run_test(
            ArangeModel(),
            torch.randn(5, 3, 2),
            additional_test_inputs=[torch.randn(8, 3, 2)],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0]},
        )

    def _chunk_test(self, dynamic_count):
        class ChunkModel(torch.nn.Module):
            def __init__(self, dim=1):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.chunk(x, x.size(0) if dynamic_count else 3, dim=self.dim)

        batch = 3 if dynamic_count else 1
        x = torch.randn(batch, 18)
        examples = [x, *(torch.randn(batch, width) for width in range(13, 16))]
        # A batch-derived chunk count fixes batch=3 in the modern output tree.
        axes = {1: "dims"} if dynamic_count else {0: "batch_size", 1: "dims"}
        for dim in (1, -1):
            with self.subTest(dim=dim):
                model = ChunkModel(dim)
                program = self.run_test(
                    model,
                    x,
                    additional_test_inputs=examples,
                    input_names=["x"],
                    dynamic_axes={"x": axes},
                )
                operators = {node.op_type for node in program.model.graph}
                self.assertIn("Split", operators)
                self.assertNotIn("SplitToSequence", operators)
                for example in examples:
                    self.assert_program_outputs(program, (example,), reference=model)
                program.release()

    def test_chunk(self):
        self._chunk_test(False)

    def test_dynamic_chunk(self):
        self._chunk_test(True)
