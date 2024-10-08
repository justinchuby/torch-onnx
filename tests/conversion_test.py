"""Unit test for converting into ONNX format."""
# ruff: noqa: F722, F821

from __future__ import annotations

import unittest

import torch
from functorch.experimental.control_flow import cond

import torch_onnx


class ConversionTest(unittest.TestCase):
    @unittest.skip("Conditionals are not supported yet")
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

        input = torch.randn(5)
        onnx_program = torch_onnx.export(CondBranchClassMethod(), (input,))
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_list_as_empty_input(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1):
                view = torch.ops.aten.view.default(arg0_1, [])
                return (view,)

        input = torch.randn(1)
        onnx_program = torch_onnx.export(GraphModule(), (input,))
        torch_onnx.testing.assert_onnx_program(onnx_program)

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

        inputs = (
            torch.randint(1, 2, (2, 2)),
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            torch.randn(3, 4, 5, 6, 7),
        )
        onnx_program = torch_onnx.export(GraphModule(), inputs)
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_iteration_over_0d_tensor(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[]"):
                zeros: "f32[]" = torch.ops.aten.zeros.default([])
                add: "f32[]" = torch.ops.aten.add.Tensor(zeros, arg0_1)
                return (add,)

        input = torch.tensor(1.0)
        onnx_program = torch_onnx.export(GraphModule(), (input,))
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_builtin_function_and_symbool_support(self):
        # https://github.com/justinchuby/torch-onnx/issues/42
        class GraphModule(torch.nn.Module):
            def forward(
                self,
                arg0_1: f32[10, 3, 5],
                arg1_1: f32[10, 3, 4],
                arg2_1: f32[10, 4, 5],
                arg3_1: "i64[]",
                arg4_1: "f32[]",
            ):
                _local_scalar_dense: Sym(f4) = (
                    torch.ops.aten._local_scalar_dense.default(arg4_1)
                )
                arg4_1 = None
                _local_scalar_dense_1: Sym(u4) = (
                    torch.ops.aten._local_scalar_dense.default(arg3_1)
                )
                arg3_1 = None
                ge: Sym(u4 >= -9223372036854775808) = (
                    _local_scalar_dense_1 >= -9223372036854775808
                )
                scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(ge)
                ge = None
                _assert_async = torch.ops.aten._assert_async.msg(
                    scalar_tensor,
                    "_local_scalar_dense_1 is outside of inline constraint [-9223372036854775808, 9223372036854775807].",
                )
                scalar_tensor = None
                le: Sym(u4 <= 9223372036854775807) = (
                    _local_scalar_dense_1 <= 9223372036854775807
                )
                scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(le)
                le = None
                _assert_async_1 = torch.ops.aten._assert_async.msg(
                    scalar_tensor_1,
                    "_local_scalar_dense_1 is outside of inline constraint [-9223372036854775808, 9223372036854775807].",
                )
                scalar_tensor_1 = None
                baddbmm: f32[10, 3, 5] = torch.ops.aten.baddbmm.default(
                    arg0_1,
                    arg1_1,
                    arg2_1,
                    beta=_local_scalar_dense,
                    alpha=_local_scalar_dense_1,
                )
                arg0_1 = arg1_1 = arg2_1 = _local_scalar_dense = (
                    _local_scalar_dense_1
                ) = None
                return (baddbmm,)

        inputs = (
            torch.randn(10, 3, 5),
            torch.randn(10, 3, 4),
            torch.randn(10, 4, 5),
            torch.tensor(1),
            torch.tensor(1.0),
        )
        onnx_program = torch_onnx.export(GraphModule(), inputs)
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_python_attributes_are_not_turned_into_attr_objects(self):
        class GraphModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.elu(x)

        inputs = (torch.tensor(1.0),)
        onnx_program = torch_onnx.export(GraphModule(), inputs)
        torch_onnx.testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    unittest.main()
