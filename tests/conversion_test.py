"""Unit test for converting into ONNX format."""
# ruff: noqa: F722, F821

from __future__ import annotations

import unittest
import torch
import torch_onnx
from functorch.experimental.control_flow import cond

IS_MAIN = __name__ == "__main__"

torch_onnx.patch_torch(report=IS_MAIN, profile=IS_MAIN, dump_exported_program=IS_MAIN)


class ConversionTest(unittest.TestCase):
    @unittest.expectedFailure  # Conditionals are not supported yet
    def test_conditional(self):
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        class CondBranchClassMethod(torch.nn.Module):
            """
            The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
            - both branches must take the same args, which must also match the branch args passed to cond.
            - both branches must return a single tensor
            - returned tensor must have the same tensor metadata, e.g. shape and dtype
            - branch function can be free function, nested function, lambda, class methods
            - branch function can not have closure variables
            - no inplace mutations on inputs or global variables

            This example demonstrates using class method in cond().

            NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
            """

            def __init__(self):
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

        model = CondBranchClassMethod()
        input = torch.randn(5)
        onnx_program = torch.onnx.dynamo_export(model, input)
        if IS_MAIN:
            onnx_program.save("conditional.onnx")

    def test_list_as_empty_input(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1):
                view = torch.ops.aten.view.default(arg0_1, [])
                return (view,)

        model = GraphModule()
        input = torch.randn(1)
        onnx_program = torch.onnx.dynamo_export(model, input)
        if IS_MAIN:
            onnx_program.save("list_as_empty_input.onnx")

    def test_getting_metadata_should_not_fail_on_none(self):
        class GraphModule(torch.nn.Module):
            def forward(
                self,
                _lifted_tensor_constant0: i64[2, 2],
                _lifted_tensor_constant1: i64[2],
                _lifted_tensor_constant2: i64[2],
                arg0_1: f32[3, 4, 5, 6, 7],
            ):
                lift_fresh_copy: i64[2, 2] = torch.ops.aten.lift_fresh_copy.default(
                    _lifted_tensor_constant0
                )
                lift_fresh_copy_1: i64[2] = torch.ops.aten.lift_fresh_copy.default(
                    _lifted_tensor_constant1
                )
                lift_fresh_copy_2: i64[2] = torch.ops.aten.lift_fresh_copy.default(
                    _lifted_tensor_constant2
                )
                slice_1: f32[3, 4, 5, 6, 7] = torch.ops.aten.slice.Tensor(
                    arg0_1, 0, 0, 9223372036854775807
                )
                slice_2: f32[3, 4, 5, 6, 7] = torch.ops.aten.slice.Tensor(
                    slice_1, 2, 0, 9223372036854775807
                )
                index: f32[2, 2, 3, 5] = torch.ops.aten.index.Tensor(
                    slice_2,
                    [None, lift_fresh_copy, None, lift_fresh_copy_1, lift_fresh_copy_2],
                )
                return (index,)

        model = GraphModule()
        inputs = (
            torch.randint(1, 2, (2, 2)),
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            torch.randn(3, 4, 5, 6, 7),
        )
        onnx_program = torch.onnx.dynamo_export(model, *inputs)
        if IS_MAIN:
            onnx_program.save("getting_metadata_should_not_fail_on_none.onnx")

    def test_iteration_over_0d_tensor(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[]"):
                zeros: "f32[]" = torch.ops.aten.zeros.default([])
                add: "f32[]" = torch.ops.aten.add.Tensor(zeros, arg0_1)
                return (add,)

        model = GraphModule()
        input = torch.tensor(1.0)
        onnx_program = torch.onnx.dynamo_export(model, input)
        assert onnx_program


if __name__ == "__main__":
    unittest.main()
