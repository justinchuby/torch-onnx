# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Modern capture adaptations from e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
model bodies/comments; use modern export rather than Script/trace wrappers.
"""

from __future__ import annotations

import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor
from torch.testing._internal import common_utils


class TestRecoveredCaptureIntents(RecoveredRuntimeHarness):
    def test_data(self):
        class Data(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.data.size())

        x = torch.randn(3, 4)
        self.run_test(Data(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    def test_primitive_input_bool(self):
        class Model(torch.nn.Module):
            def forward(self, flag: bool, x, y):
                if flag:
                    return x
                else:
                    return y

        flag = True
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(Model(), (flag, x, y))

    def test_cste_script(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0)), torch.ones(
                    (x.size(1), x.size(0)), dtype=torch.int64
                )

        x = torch.randn(3, 4)
        self.run_test(MyModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    def test_tensor(self):
        class ScalarInputModel(torch.nn.Module):
            def forward(self, input):
                return torch.tensor(input.shape[1])

        x = torch.randn(3, 4)
        self.run_test(
            ScalarInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        class TensorInputModel(torch.nn.Module):
            def forward(self, input):
                return torch.tensor([input.shape[0], input.shape[1]])

        x = torch.randn(3, 4)
        self.run_test(
            TensorInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        class FloatInputModel(torch.nn.Module):
            def forward(self, input):
                return torch.tensor([float(input)])

        x = torch.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(torch.nn.Module):
            def forward(self, input):
                return torch.tensor(input.shape[1], dtype=torch.long)

        x = torch.randn(3, 4)
        self.run_test(
            InputWithDtypeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        class MixedInputModel(torch.nn.Module):
            def forward(self, input):
                return torch.tensor([input.shape[0], int(input)])

        x = torch.randn(1)
        self.run_test(MixedInputModel(), x)

    def test_hardtanh_script_with_default_values(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardtanh(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardswish_script(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardswish(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_clamp_dyn(self):
        class ClampMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(None, x.size(0))

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)

        class ClampMinModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(x.size(0), None)

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

        class ClampTensorModel(torch.nn.Module):
            def forward(self, x, min, max):
                return x.clamp(min, max)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = torch.randn(3, 4)
        self.run_test(ClampTensorModel(), (x, y, z))

        class ClampTensorMinModel(torch.nn.Module):
            def forward(self, x, min):
                return x.clamp(min=min)

        self.run_test(ClampTensorMinModel(), (x, y))

        class ClampTensorMaxModel(torch.nn.Module):
            def forward(self, x, max):
                return x.clamp(max=max)

        self.run_test(ClampTensorMaxModel(), (x, z))

    def test_full_script(self):
        class FullModelScripting(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_floating_point(self):
        class FloatingPoint(torch.nn.Module):
            def forward(self, x):
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    def test_floating_point_infer_dtype(self):
        class FloatingPoint(torch.nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), x)

    def test_prim_min(self):
        def list_append(boxes: list[Tensor]):
            temp = []
            for i, b in enumerate(
                boxes
            ):  # enumerate is creating a prim::min op in torch graph
                temp.append(torch.full_like(b[:, 1], i))
            return temp[0]

        class Min(torch.nn.Module):
            def forward(self, x):
                boxes = [x for _ in range(3)]
                return list_append(boxes)

        x = torch.rand(5, 5)
        self.run_test(Min(), (x,))

        class M(torch.nn.Module):
            def forward(self, x):
                i = 3
                return min(x[i], i)

        x = torch.arange(6, dtype=torch.int64)
        self.run_test(M(), (x,))

    def test_arithmetic_infer_dtype(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3)
        self.run_test(ArithmeticModule(), x)

    def test_floor_div_script(self):
        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return x // 3, x // 2.0, x // y

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_div_promotion_trace(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        with common_utils.set_default_dtype(torch.float):
            self.run_test(DivModule(), (x, y))

        with common_utils.set_default_dtype(torch.double):
            self.run_test(DivModule(), (x, y))

    def test_div_promotion_script(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still available from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.float):
            self.run_test(DivModule(), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.double):
            self.run_test(DivModule(), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(DivModule(), (x, y))

    def test_div_rounding_mode(self):
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode=None),
                    torch.div(x, y, rounding_mode=None),
                )

        class TruncDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="trunc"),
                    torch.div(x, y, rounding_mode="trunc"),
                )

        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="floor"),
                    torch.div(x, y, rounding_mode="floor"),
                )

        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]

        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(module, (x, y))
            self.run_test(module, (x, y))

        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(module, (x, y))
            self.run_test(module, (x, y))

    def test_topk_script(self):
        class MyModuleDynamic(torch.nn.Module):
            def forward(self, x, k):
                return torch.topk(x, k)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModuleDynamic(), (x, k))

    def test_flatten_dynamic_axes(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def test_len(self):
        class LenModel(torch.nn.Module):
            def forward(self, input):
                # Avoid specializing the Python tuple length during modern capture.
                return input.shape[0] + input

        x = torch.randn(4, 5)
        self.run_test(
            LenModel(),
            x,
            input_names=["input"],
            dynamic_axes={"input": {0: "seq"}},
            additional_test_inputs=(torch.randn(5, 5),),
        )

    def test_len_list(self):
        class LenListModel(torch.nn.Module):
            def forward(self, input):
                return torch.ones(len(input.shape))

        x = torch.randn(4, 5)
        self.run_test(LenListModel(), x, remained_onnx_input_idx=[])

    def test_list(self):
        class ListModel(torch.nn.Module):
            def forward(self, x):
                tensors = x.unbind()
                res = []
                res.append(tensors[0])
                res.append(tensors[1])
                res.pop(1)

                res.insert(0, tensors[1])
                res.append(tensors[2])
                res += [tensors[3], tensors[4]]
                res = res + [tensors[5]]  # noqa: RUF005 - Retain the list-concatenation regression.
                return torch.ones(len(res))

        model = ListModel()
        inputs = torch.randn(16, 1)
        self.run_test(model, inputs)

    def test_tensor_factories(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    def test_tensor_factories_script(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(
                    x.shape, dtype=torch.float
                )

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    def test_tensor_like_factories_script(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return zeros + ones

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic(self):
        class Arithmetic(torch.nn.Module):
            def forward(self, x, y):
                x.add_(3)
                y.mul_(x)
                return x, y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(Arithmetic(), (x, y))

    def test_masked_fill_inplace(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, x):
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                x.masked_fill_(mask, 2)
                return x

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):
            def forward(self, x):
                x.masked_fill_(x > 3, -1)
                return x

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    def test_dtype(self):
        class MyModel(torch.nn.Module):
            def forward(self, input, other):
                return input.to(dtype=other.dtype) + other

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_dtype_eq(self):
        class MyModel(torch.nn.Module):
            def forward(self, input, other):
                if input.dtype == other.dtype:
                    return input + other
                return input

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_cast_to(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, other):
                return input.to(other) + other

        x = torch.randn(2, 3, 4)
        y = torch.tensor([1], dtype=torch.int64)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_trace_script(self):
        def center_slice_helper(input, h_offset):
            return input[:, h_offset:]

        class CenterCrop(torch.nn.Module):
            def forward(self, input):
                return center_slice_helper(input, torch.tensor(input.shape[1] - 1))

        x = torch.randn(3, 4)
        self.run_test(CenterCrop(), x)

    def test_dim(self):
        class DimModel(torch.nn.Module):
            def forward(self, input):
                out = input * 2
                out *= out.dim()
                return out

        empty_input = torch.randn(0, requires_grad=True)
        multi_dim_input = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    def test_cdist_dynamic_axes_script(self):
        class CdistModule(torch.nn.Module):
            def forward(self, a, b):
                return torch.cdist(a, b)

        model = CdistModule()
        a = torch.randn(10, 3)
        b = torch.randn(10, 3)
        self.run_test(
            model,
            input_args=(a, b),
            input_names=["a", "b"],
            output_names=["out"],
            dynamic_axes={"a": {0: "size"}, "b": {0: "size"}},
        )

    def test_embedding(self):
        class EmbedModel(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb, padding_idx=1)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = torch.rand(10, 3)
        self.run_test(model, (x, embedding_matrix))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x, embedding_matrix))
        self.run_test(model, (x, embedding_matrix), training=True)

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x, embedding_matrix))

    def test_0d_tensor_broadcast(self):
        class fn(torch.nn.Module):
            def forward(self, x, y):
                a = torch.add(x, y)
                b = torch.mul(y, y)
                return a + b

        x = torch.ones(0)
        y = torch.ones(1)
        self.run_test(fn(), (x, y), input_names=["x", "y"], output_names=["output"])

    def test_device_eq(self):
        class M(torch.nn.Module):
            def forward(self, a):
                # exercise both Tensor.device (prim::device)
                # and torch.device (prim::Constant).
                if a.device != torch.device("cpu"):
                    return a
                return torch.zeros_like(a)

        mod = M()  # preserve control flow

        self.run_test(
            mod,
            # In order for the ONNX model behavior to match the torch model, we
            # need to construct input that has the same device that is checked for
            # in forward(). In ONNX there is no such thing as a device, so the if
            # condition is always false.
            torch.randn(3, 3, device="cpu"),
            # Force dynamic axes so that the output shape depends on the input.
            # Otherwise the entire model will just return a constant and not have
            # any inputs.
            input_names=["a"],
            dynamic_axes={"a": {0: "a0"}},
        )
