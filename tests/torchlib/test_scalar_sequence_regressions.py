# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Scalar/sequence cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Ordinary Modules
replace ScriptModules; original examples and scalar/dtype contracts remain.
"""

import onnx_ir as ir
import torch
from _common.runtime import RecoveredRuntimeHarness

import torch_onnx
from torch_onnx._torchlib.ops import sequences


class _Arange(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant

    def forward(self, a):
        if self.variant == "end":
            return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a
        if self.variant == "end_notype":
            return torch.arange(a.size(0))
        if self.variant == "start_end":
            return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a
        if self.variant == "start_end_notype":
            return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a
        if self.variant == "start_end_step":
            return (
                torch.arange(
                    2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                ).view(-1, 1)
                + a
            )
        return torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1) + a


class TestRecoveredScalarSequences(RecoveredRuntimeHarness):
    def test_getitem(self):
        class GetItemModel(torch.nn.Module):
            def forward(self, x, y, z, ind):
                # this will create prim::ListConstruct(x, y, z) + aten::__getitem__
                arr = [x, y, z]
                return arr[ind]

        class NativeGetItem(torch.nn.Module):
            def forward(self, x, y, z, ind):
                return torch.ops.recovered_getitem.select.default(x, y, z, ind)

        def implementation(x, y, z, ind):
            return GetItemModel()(x, y, z, ind).clone()

        def fake(x, y, z, ind):
            count = torch.library.get_ctx().new_dynamic_size(min=1, max=3)
            return x.new_empty((count, *x.shape[1:]))

        def translation(
            op: torch_onnx.TranslationContext,
            x: ir.Value,
            y: ir.Value,
            z: ir.Value,
            ind: ir.Value,
        ) -> ir.Value:
            return sequences.aten_getitem(op, [x, y, z], ind)

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        examples = [
            (x, y, z, torch.tensor(index, dtype=torch.long))
            for index in (1, -2, 0, 2, -1, -3)
        ]
        with torch.library._scoped_library("recovered_getitem", "FRAGMENT") as library:
            qualified = "recovered_getitem::select"
            torch.library.define(
                qualified,
                "(Tensor x, Tensor y, Tensor z, Tensor ind) -> Tensor",
                lib=library,
            )
            torch.library.impl(qualified, "CompositeExplicitAutograd", lib=library)(
                implementation
            )
            torch.library.register_fake(qualified, lib=library)(fake)
            target = torch.ops.recovered_getitem.select.default
            torch_onnx.register_custom_translation(target, translation)
            try:
                program = self.run_test(
                    NativeGetItem(),
                    examples[0],
                    additional_test_inputs=examples[1:],
                )
                self.assertIn(
                    "SequenceAt", {node.op_type for node in program.model.graph}
                )
                self.assertNotIsInstance(program.model.graph.outputs[0].shape[0], int)
                self.assert_program_roundtrip(
                    program,
                    examples=examples,
                    reference=GetItemModel(),
                )
            finally:
                torch_onnx.unregister_custom_translation(target)

    def test_list_set(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                res[y] = x[y]
                return res

        class TensorizedModel(torch.nn.Module):
            def forward(self, x, y):
                rows = [x[i] for i in range(x.size(0))]
                storage = torch.stack(rows)
                normalized = torch.where(y < 0, y + storage.shape[0], y).reshape(1)
                updated = storage.index_copy(
                    0, normalized, x.index_select(0, normalized)
                )
                return list(updated.unbind(0))

        examples = [
            (torch.randn(12, 4), torch.tensor(index, dtype=torch.long))
            for index in (2, 0, 11, -1, -12)
        ]
        for args in examples:
            self.assertEqual(TensorizedModel()(*args), ListModel()(*args))
        self.run_test(
            TensorizedModel(), examples[0], additional_test_inputs=examples[1:]
        )

    def test_list_idx_sum(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                indices = torch.arange(x.size(0))
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                return res[torch.sum(indices[:y])]

        class TensorizedModel(torch.nn.Module):
            def forward(self, x, y):
                indices = torch.arange(x.size(0))
                rows = [x[i] for i in range(x.size(0))]
                storage = torch.stack(rows)
                stop = torch.where(y < 0, y + indices.shape[0], y)
                index = torch.where(indices < stop, indices, 0).sum().reshape(1)
                return storage.index_select(0, index).squeeze(0)

        examples = [
            (torch.randn(12, 4), torch.tensor(index, dtype=torch.long))
            for index in (2, 0, 1, 3, 4, 5, -12, -10, -7, -15)
        ]
        for args in examples:
            self.assertEqual(TensorizedModel()(*args), ListModel()(*args))
        self.run_test(
            TensorizedModel(), examples[0], additional_test_inputs=examples[1:]
        )

    def _arange_test(self, variant):
        self.run_test(_Arange(variant), torch.randn(3, 4, requires_grad=True))

    def test_arange_end(self):
        self._arange_test("end")

    def test_arange_end_notype(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(
            _Arange("end_notype"),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.randn(5, 7)],
        )
        self.run_test(_Arange("end_notype"), x, remained_onnx_input_idx=[])

    def test_arange_start_end(self):
        self._arange_test("start_end")

    def test_arange_start_end_notype(self):
        self._arange_test("start_end_notype")

    def test_arange_start_end_step(self):
        self._arange_test("start_end_step")

    def test_arange_start_end_step_notype(self):
        self._arange_test("start_end_step_notype")

    def test__dim_arange(self):
        class DimArange(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[1])

        x = torch.ones(5, 6)
        self.run_test(
            DimArange(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.ones(7, 9)],
        )
        self.run_test(DimArange(), x, remained_onnx_input_idx=[])

    def test_gt_primitive(self):
        class GreaterModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y: int = 2

            def forward(self, x: int):
                return self.y > x

        self.run_test(GreaterModel(), (3,))

    def test_item(self):
        class M(torch.nn.Module):
            def forward(self, x, y, i: int):
                # Materialize the same truncating conversion as an ONNX scalar.
                return torch.scalar_tensor(x[y[i]].item(), dtype=torch.int64)

        x = torch.arange(6, dtype=torch.float)
        y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        original = (x, y, 3)
        alternate = (-x - 0.75, torch.tensor([0, 1, 2, 4, 5]), 3)
        model = M()
        for values, indices, offset in [original, alternate]:
            expected = int(values[indices[offset]].item())
            self.assertEqual(model(values, indices, offset).item(), expected)
        self.run_test(model, original, additional_test_inputs=[alternate])

    def test_unbind_dynamic(self):
        class UnbindModel(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, input):
                return input.unbind(self.dim)[1]

        for dim, axes, alternate in [(0, [1, 2], (3, 6, 7)), (-1, [0, 1], (5, 6, 5))]:
            self.run_test(
                UnbindModel(dim),
                torch.randn(3, 4, 5),
                input_names=["input"],
                dynamic_axes={"input": axes},
                additional_test_inputs=[torch.randn(alternate)],
            )

    def test_split_dynamic(self):
        class SplitModel(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, input):
                return input.split(2, self.dim)[1]

        for dim in [0, -3]:
            self.run_test(
                SplitModel(dim),
                torch.randn(5, 4, 3),
                input_names=["input"],
                dynamic_axes={"input": [1, 2]},
                additional_test_inputs=[torch.randn(5, 6, 7)],
            )

    def test_concat_dynamic(self):
        class ConcatDynamicModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat(x.unbind())

        self.run_test(
            ConcatDynamicModel(),
            torch.randn(4, 5, 6),
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[torch.randn(4, 7, 8)],
        )

    def test_stack_dynamic(self):
        class StackDynamicModel(torch.nn.Module):
            def forward(self, x):
                return torch.stack(x.unbind(), 1)

        self.run_test(
            StackDynamicModel(),
            torch.randn(4, 5, 6),
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[torch.randn(4, 7, 8)],
        )
