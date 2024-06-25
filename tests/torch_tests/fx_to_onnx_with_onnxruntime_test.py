# Owner(s): ["module: onnx"]
from __future__ import annotations

import itertools
import math
import operator
from typing import Any, Mapping

import onnx_test_common
import onnxruntime  # type: ignore[import]
import parameterized  # type: ignore[import]
import pytorch_test_common

import transformers  # type: ignore[import]

import torch
import torch.onnx
from torch import nn

import torchvision
from torch.testing._internal import common_utils

import torch_onnx

torch_onnx.patch_torch(error_report=True)


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (True, False),
            (True, False),
            (
                pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
                pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
            ),
        )
    )
    return {
        "attrs": ["op_level_debug", "dynamic_shapes", "model_type"],
        "input_values": input_values,
    }


def _parameterize_class_name(cls: type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffixes = []
    for k, v in input_dicts.items():
        suffixes.append(f"{k}_{v}")
    return f"{cls.__name__}_{'_'.join(suffixes)}"


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),
    class_name_func=_parameterize_class_name,
)
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    op_level_debug: bool
    dynamic_shapes: bool
    model_type: pytorch_test_common.TorchModelType

    def setUp(self):
        super().setUp()
        self.ort_version = onnxruntime.__version__

    def test_simple_function(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                # TODO(justinchuby): Replicate torch's type casting policy
                # in the exporter for type promotion support
                y = x + 1.0
                z = y.relu()
                return (y, z)

        func = Foo()

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))

    @pytorch_test_common.xfail(
        error_message="Unexpectedly found a <class 'torch.Tensor'> in the inputs.",
        reason="https://github.com/pytorch/pytorch/issues/96379",
    )
    def test_func_with_args_and_tensor_kwargs(self):
        # Non-tensor optional kwargs are always folded into constant and
        # removed from input list in Dynamo-traced graph, if its value is not provided
        # to tracer. So for a function like
        #   def func(x, b=1.0)
        # here. E.g., if you first Dynamo-trace the model with arguments (x,),
        # and then call the traced graph with arguments (x, b=2.0), it will complain
        # somewhere that model is called with extra args because the modified
        # function is traced into
        #   def forward(self, x : torch.Tensor):
        #     add = x + 1.0;  x = None
        #     relu = add.relu()
        #     return (add, relu)
        # To summarize, in order to be traced as graph input, the value of optional kwarg
        # must be provided. Otherwise, they are treated as in-graph constants in Dynamo.
        # Tensor optional kwargs are an exception. It is always traced as input.
        # It is unclear if this behavior is intended or not. But in general it is bad
        # practice to set mutable default values.
        # `DynamoOptimizeExporter` applies a workaround by binding args and kwargs to
        # model signature and fill in the default values of unprovided optional arguments.
        class Foo(torch.nn.Module):
            def forward(self, x, b=torch.tensor(1.0)):  # noqa: B008
                y = x + b
                z = y.relu()
                return (y, z)

        func = Foo()

        tensor_x = torch.randn(1, 2, 3, dtype=torch.float32)

        # Test without providing optional kwarg.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))
        # Test with only positional args.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x, torch.tensor(8.0))
        )
        # Test while specifying optional kwarg.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x,), input_kwargs={"b": torch.tensor(5.0)}
        )

    @pytorch_test_common.skip_dynamic_fx_test(
        "sympy operation tests don't need dynamic shape"
    )
    def test_sympy_operatons_return_numeric(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # TODO: add boolean tests when SymBool is supported
                # to infer types
                return (
                    torch.tensor([operator.add(x.item(), y.item())]),
                    torch.tensor([operator.sub(x.item(), y.item())]),
                    torch.tensor([operator.mul(x.item(), y.item())]),
                    torch.tensor([operator.truediv(x.item(), y.item())]),
                    # This requires torch.sym_float, probably easy to lower to
                    # ONNX but I don't know where to put it
                    # torch.tensor([operator.floordiv(x.item(), y.item())]),
                    # NB: abs so that the base and exponent are provably
                    # non-negative, so we don't generate runtime asserts
                    torch.tensor([operator.pow(abs(x.item()), abs(y.item()))]),
                    torch.tensor([operator.abs(x.item())]),
                    torch.tensor([operator.neg(x.item())]),
                    torch.tensor([math.ceil(x.item())]),
                    torch.tensor([math.floor(x.item())]),
                )

        func = Foo()

        x = torch.randn(1, dtype=torch.float32)
        y = torch.randn(1, dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func,
            (
                x,
                y,
            ),
        )

    @pytorch_test_common.xfail(
        error_message="Model inputs incompatible with the format that was exported",
        reason="https://github.com/pytorch/pytorch/issues/99534",
    )
    def test_xfail_func_with_non_tensor_args(self):
        class Foo(torch.nn.Module):
            def forward(self, x, b=1.0):
                y = x + b
                z = y.relu()
                return (y, z)

        func = Foo()

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        onnx_program = torch.onnx.dynamo_export(
            func,
            tensor_x,
            8.0,
            export_options=torch.onnx.ExportOptions(
                op_level_debug=self.op_level_debug,
                dynamic_shapes=self.dynamic_shapes,
            ),
        )
        onnx_test_common.assert_dynamic_shapes(onnx_program, self.dynamic_shapes)
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, b=8.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 8.0))
        ort_outputs = onnx_test_common.run_ort(onnx_program, onnx_format_args)
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

        # test on different non-tensor input - xfail
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, b=9.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 9.0))
        _ = onnx_test_common.run_ort(onnx_program, onnx_format_args)
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    def test_func_with_nested_input_structure(self):
        class Foo(torch.nn.Module):
            def forward(
                self,
                x_dict: dict[str, torch.Tensor],
                y_tuple: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
                z_list: list[list[torch.Tensor]],
            ):
                if "a" in x_dict:
                    x = x_dict["a"]
                elif "b" in x_dict:
                    x = x_dict["b"]
                else:
                    x = torch.randn(3)

                y1, (y2, y3) = y_tuple

                z = x + y1 + y2 + y3
                for z_sub_list in z_list:
                    z = z + torch.stack(z_sub_list).sum()

                return z

        func = Foo()

        x_dict = {"a": torch.randn(3), "c": torch.randn(3)}
        y_tuple = (torch.randn(3), (torch.randn(3), torch.randn(3)))
        z_list = [
            [torch.randn(3), torch.randn(3)],
            [torch.randn(3), torch.randn(3), torch.randn(3)],
        ]
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (x_dict, y_tuple, z_list)
        )

    def test_func_with_nested_output_structure(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                x = x + y
                y = y + z
                z = x + y
                out1 = (x, (y, z))
                out2 = [[x, y], [y, z]]
                out3 = {"z": z, "x": x}
                return out1, out2, out3

        func = Foo()

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x, y, z))

    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = torch.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = torch.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MNISTModel(), (tensor_x,)
        )

    def test_log_sigmoid(self):
        # This produces op as `torch.ops.aten.log_sigmoid_forward`, instead of the more
        # conventional `torch.ops.aten.log_sigmoid`.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.LogSigmoid()

            def forward(self, x):
                return self.m(x)

        input = torch.randn(2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (input,))

    def test_resnet18(self):
        # TODO(bowbao): Note [training vs eval in dynamo_export]
        # So we are effectively exporting all models in traning mode by
        # default. But for the sake of this export we are only interested in eval mode.
        # The question is, should we call `model.eval()` in `dynamo_export`?
        # This particular test fails 'functionalization' in training mode.
        # So we are explicitly calling `model.eval()` for any model that contains
        # batch norm.
        # Ref: https://github.com/pytorch/pytorch/issues/99662#issuecomment-1528178221
        model = torchvision.models.resnet18(weights=None).eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
        )

    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input"
    )
    def test_shufflenet_v2(self):
        # TODO(bowbao): see Note [training vs eval in dynamo_export]
        model = torchvision.models.shufflenet_v2_x0_5(weights=None).eval()
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=False)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
            additional_test_inputs=[((test_inputs,),)],
            rtol=1e-3,
            atol=1e-5,
        )

    def test_add(self):
        class DynamicAdd(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.add(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        another_x = torch.randn(3, 4)
        another_y = torch.randn(3, 4)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(),
            (x, y),
            additional_test_inputs=[((another_x, another_y),)],
        )

    def test_sigmoid_add(self):
        class DynamicAdd(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, y):
                z = torch.ops.aten.add(x, y)
                return self.sigmoid(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        x = x[1:, :]
        y = y[1:, :]
        input_x = torch.randn(1, 4)
        input_y = torch.randn(1, 4)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )

    def test_matmul(self):
        class DynamicMatMul(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.matmul(x, y)

        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicMatMul(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )

    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="The values for attribute 'shape' do not match: torch.Size([]) != torch.Size([1])"
    )
    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, x):
                return torch.scalar_tensor(x.size(0)), torch.scalar_tensor(
                    x.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            test(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    def test_transpose_infer_shape(self):
        class TransposeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            TransposeModule(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    @pytorch_test_common.xfail(
        error_message=("Unsupported FX nodes: {'call_function': [")
    )
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])  # problematic user code for dynamo
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d1, d4), additional_test_inputs=[((d3, d4),)]
        )
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d3, d4), additional_test_inputs=[((d1, d3),)]
        )

    def test_slice(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])  # noqa: PERF401
                return tuple(results)

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicSliceExportMod(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2",
    )
    def test_mutation(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x):
                x.view(3, 2, -1).add_(2.0)
                return x

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MutationModel(), (torch.randn(12),), has_mutation=True
        )

    def test_arange(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ArangeModel(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node. "
    )
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2"
    )
    def test_expand_as_fill_zero(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = 0
                return x

        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node. "
    )
    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Expected 1 outputs, got 2"
    )
    def test_expand_as_fill_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.xfail_if_model_type_is_not_exportedprogram(
        error_message="at::functionalization::impl::isFunctionalTensor(self_) INTERNAL ASSERT FAILED"
    )
    def test_expand_as_fill_separate_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.skipIfNoCuda
    def test__scaled_dot_product_flash_attention(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                (
                    output,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = torch.ops.aten._scaled_dot_product_flash_attention(x, x, x)
                return output

        func = Foo()

        x = torch.randn(1, 1, 1, 32, device=torch.device("cuda"))
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x,))

    def test_view_dynamic_zero_dim(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        y = torch.empty(0)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ViewModel(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    def test_flatten_dynamic_axes(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model, (x,), additional_test_inputs=[((y,),)]
        )

    def test_none_input(self):
        class NoneInputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor | None, z: torch.Tensor):
                if y is None:
                    return x + z
                return x + y + z

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            NoneInputModel(), (torch.randn(1, 2), None, torch.randn(1, 2))
        )

    def test_operator_with_data_dependent_output(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                # Repro from llama. Emits `torch.ops.aten._local_scalar_dense`.
                return x + torch.full(x.shape, torch.tensor(torch.finfo(x.dtype).min))

        func = Foo()

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.randn(3, 4),)
        )

    def test_operator_with_scalar_output(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x.item() + y

        func = Foo()

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.tensor([1]), torch.randn(3, 4))
        )

    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Unsupported FX nodes: {'call_function': ['aten._assert_async.msg']}",
        reason="https://github.com/pytorch/pytorch/issues/112622",
    )
    def test_operator_with_dynamic_output_shape(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.nonzero()

        func = Foo()

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (torch.randn(3, 4),)
        )

    @pytorch_test_common.xfail_if_model_type_is_exportedprogram(
        error_message="Trying to flatten user inputs with exported input tree spec"
    )
    @pytorch_test_common.xfail_dynamic_fx_test(
        error_message="!(it.GetName().empty())",
        reason="With after onnx==1.16, constant folding in optimizer causes this error.",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    )
    def test_gpt2_tiny_from_config(self):
        # Model
        config = transformers.GPT2Config(
            num_hidden_layers=4,
            vocab_size=8096,
            hidden_size=16,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )
        model = transformers.GPT2Model(config).eval()

        def input_generator(batch: int, seq: int):
            input_ids = torch.randint(0, 8096, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)
            return input_ids, attention_mask, position_ids

        # Encoded inputs
        input_ids, attention_mask, position_ids = input_generator(2, 128)

        # Another encoded inputs to test dynamic shapes
        (
            another_input_ids,
            another_attention_mask,
            another_position_ids,
        ) = input_generator(3, 256)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (input_ids,),
            input_kwargs={
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            additional_test_inputs=[
                (
                    (another_input_ids,),
                    {
                        "attention_mask": another_attention_mask,
                        "position_ids": another_position_ids,
                    },
                )
            ],
        )

    def test_prims_device_put(self):
        class CustomModule(nn.Module):
            def forward(self, x):
                # Assuming x is a tensor on the CPU, move it to the desired device using device_put()
                x = torch.ops.prims.device_put(x, "cpu")
                return x

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            CustomModule(), (torch.randn(1, 2, 3),)
        )


if __name__ == "__main__":
    common_utils.run_tests()
