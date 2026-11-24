# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Additional numerical cases from e2c2673d00500d268fcb1cc63fa7cc3173876505.

Recovered before deletion 6dca9f63bc4975ac323044a440c2c7e5f8512408.
Original method bodies and comments use the standalone modern-export harness.
"""

from collections import OrderedDict

import torch
import torchvision
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor


class TestRecoveredAdditional(RecoveredRuntimeHarness):
    def test_reference_preserves_non_leaf_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.ones(2, requires_grad=True) * 2
        self.run_test(Model(), (x,))

    def test_scalar_output_initializer_name_collision(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("constant_output_0", torch.tensor([2.0, 3.0]))

            def forward(self, x):
                return 42, x + self.constant_output_0

        self.run_test(Model(), (torch.ones(2),))

    def test_cdist_empty_feature_vectors(self):
        class Model(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, x, y):
                return torch.ops.aten._cdist_forward.default(x, y, self.p, None)

        for p in (0.0, 2.0, float("inf")):
            self.run_test(Model(p), (torch.empty(2, 3, 0), torch.empty(2, 5, 0)))

    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    def test_dict_output(self):
        class DictModelOutput(OrderedDict):
            tensor_out: Tensor
            tuple_out: tuple[Tensor] | None = None
            list_out: list[Tensor] | None = None

        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        torch.utils._pytree.register_pytree_node(
            DictModelOutput,
            lambda output: (list(output.values()), list(output.keys())),
            lambda values, keys: DictModelOutput(zip(keys, values, strict=True)),
            serialized_type_name="torch_onnx.recovered.DictModelOutput",
        )
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_primitive_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: tuple[int, Tensor], b):
                return a[0], a[1] + b

        x = (3, torch.randn(4, 3))
        y = torch.randn(4, 3)
        self.run_test(TupleModel(), input_args=(x, y))

    def test_mixed_optional_default_none(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Tensor | None = None,
                z: Tensor | None = None,
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        # Without kwargs dict.
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))
        # With kwargs dict.
        self.run_test(model, (x,), {"y": y, "z": None})
        self.run_test(model, (x,), {"y": None, "z": z})
        self.run_test(model, (x,), {"z": z})
        self.run_test(model, (x,), {"y": y})

    def test_mixed_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Tensor | None = torch.ones(2, 3),  # noqa: B008 - tensor defaults are the regression
                z: Tensor | None = torch.zeros(2, 3),  # noqa: B008
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()

        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))

    def test_all_optional_default_none(self):
        class Model(torch.nn.Module):
            def forward(self, x: Tensor | None = None, y: Tensor | None = None):
                if x is not None:
                    return x
                if y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(
            model,
            (),
            {"x": x, "y": None},
            # y disappears in tracing.
            input_names=("x",),
        )

    def test_all_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x: Tensor | None = torch.ones(2, 3),  # noqa: B008 - tensor defaults are the regression
                y: Tensor | None = torch.zeros(2, 3),  # noqa: B008
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (None, y))
        # Modern ONNXProgram preserves the Python call signature for unused inputs.
        self.run_test(model, (x, y))

    def test_mixed_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: Tensor | None):
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (x, x))

    def test_tuple_of_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: tuple[Tensor | None, Tensor | None]):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    def test_tuple_of_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Tensor | None, Tensor | None] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    def test_primitive_input_integer(self):
        class Model(torch.nn.Module):
            def forward(self, x: int, y):
                return x + y

        x = 3
        y = torch.randint(10, (2, 3, 4))
        self.run_test(Model(), (x, y))

    def test_primitive_input_floating(self):
        class Model(torch.nn.Module):
            def forward(self, x: float, y):
                return x + y

        x = 3.0
        y = torch.randn(2, 3, 4)
        self.run_test(Model(), (x, y))

    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, input):
                return torch.scalar_tensor(input.size(0)), torch.scalar_tensor(
                    input.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_hardshrink_dtype(self):
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Hardshrink(), x)

    def test_softshrink_dtype(self):
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Softshrink(), x)

    def test_full_trace(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_numpy_T(self):
        class NumpyTranspose(torch.nn.Module):
            def forward(self, x):
                return x.T

        self.run_test(NumpyTranspose(), torch.randn(4, 7))

    def test_squeeze(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    def test_squeeze_dynamic_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.squeeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = 1
        self.run_test(Squeeze(), (x, dim))

    def test_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, dim=-2)

        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    def test_unsqueeze_dynamic_dim(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.unsqueeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = -1
        self.run_test(Unsqueeze(), (x, dim))

    def test_maxpool_adaptive(self):
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        y = torch.randn(32, 16, 50, requires_grad=True)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[y],
        )

    def test_maxpool_2d(self):
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    def test_maxpool_1d_ceil_corner(self):
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    def test_maxpool_2d_ceil_corner(self):
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil_corner(self):
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    def test_maxpool_1d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=True
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    def test_maxpool_2d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    def test_maxpool_with_indices(self):
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_dilation(self):
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)
        self.run_test(model, x)

    def test_avgpool_3d_ceil(self):
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        y = torch.randn(32, 8, 50, 44, 31)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[y],
        )

    def test_arithmetic_prim_long(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[0]

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    def test_arithmetic_prim_float(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: float):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[1] / 2

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    def test_arithmetic_prim_bool(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int, z: bool, t: float):
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return x / t, z

        x = torch.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x: int, y: int):
                return x == y

        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    def test_tuple_with_none_outputs(self):
        class TupleModel(torch.nn.Module):
            def forward(self, x):
                return (x, (x, None, (x, None)))

        x = torch.randn(3, 4)
        self.run_test(TupleModel(), (x,))

    def test_div(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    def test_slice_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_with_input_index(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[: y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    def test_slice_with_1d_input_index(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[:y, 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.tensor([5], dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y))

    def test_slice_dynamic_shape_script(self):
        class DynamicSliceModel(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1 : x.size(2)])

        x = torch.rand(1, 2, 3, 4)
        self.run_test(
            DynamicSliceModel(),
            x,
            input_names=["x"],
            # Axis 2 selects the output rank, and singleton axis 0 is specialized.
            dynamic_axes={"x": [1, 3]},
            additional_test_inputs=[torch.rand(1, 4, 3, 7)],
        )
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    def test_slice_dynamic_to_end(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            input_names=["input_1"],
            dynamic_shapes=(
                {
                    0: torch.export.Dim("batch"),
                    1: torch.export.Dim("width", min=4),
                    2: torch.export.Dim("depth", min=5),
                },
            ),
            additional_test_inputs=[torch.rand(6, 8, 7)],
        )

    def test_arange_with_floats(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStepNeg(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(end, start.size(0), -1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStepNeg(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStart(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStart(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])

    def test_arange_with_floats_override(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end, dtype=torch.int64)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    def test_arange_no_type(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end), torch.arange(0, end)

        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), x)

    def test_size(self):
        class SizeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.size(0)),
                    torch.arange(input.size(-1)),
                    torch.ones(input.shape),
                )

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    def test_as_strided(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided(
                    (3, 3, 3), (1, 4, 2), storage_offset=2
                ), x.as_strided(chunk_size, chunk_stride)

        x = torch.randn(5, 8, 7)
        self.run_test(Model(), x)

    def test_tensor_index_advanced_indexing_ellipsis(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    torch.tensor([[0, 2], [1, 1]]),
                    :,
                    torch.tensor([2, 1]),
                    torch.tensor([0, 3]),
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])
                ]

        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    torch.tensor([0, 2]),
                    torch.tensor([1]),
                    2:4,
                    torch.tensor([[1], [4]]),
                ]

        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_index_put(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                x[ind] = update
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor([1], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    def test_index_put_singular(self):
        class IndexPutBoolModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = True
                return mask

        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = torch.tensor(5.5)
                return mask

        mask = torch.rand(100, dtype=torch.float)
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

    def test_index_put_slice_index(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[1:2, 1:3, torch.tensor([1])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), torch.tensor([1, 2])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 1:2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(torch.nn.Module):
            def forward(self, x, update):
                x[1:, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(torch.nn.Module):
            def forward(self, x, update):
                x[:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(3 * 5).to(torch.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

        class IndexPutModel9(torch.nn.Module):
            def forward(self, poses):
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        self.run_test(IndexPutModel9(), (x,))

        class IndexPutModel10(torch.nn.Module):
            def forward(self, x, ind, update):
                x[ind, 1:3] = update.view(1, 1, 1, 5).expand(2, 2, 2, 5)
                return x

        x = torch.randn(3, 4, 5)
        ind = torch.tensor([[0, 2], [1, 1]])
        update = torch.randn(5)
        self.run_test(IndexPutModel10(), (x, ind, update))

    def test_index_put_ellipsis(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    def test_copy_(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        class CopyModel2(torch.nn.Module):
            def forward(self, x, data):
                x[1:3, 0] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(torch.nn.Module):
            def forward(self, x, ind, data):
                x[ind] = data
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(torch.nn.Module):
            def forward(self, x, mask):
                if mask is not None:
                    x.copy_(mask)
                    return x

        x = torch.randn(3, 4)
        mask = torch.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    def test_copy_tracing(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

    def test_copy_ellipsis(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., 1] = update
                return x

        x = torch.randn(2, 3, 4)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

        x = torch.randn(2, 3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    def test_copy_ellipsis_script(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                # Insert reshape node to ensure no shape/type info for
                # x in scripting, without onnx shape inference.
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x

        x = torch.randn(3, 4, 5, 6)

        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    def test_reshape_allowzero(self):
        class ReshapeModel(torch.nn.Module):
            def forward(self, x):
                x = x.reshape(3, 4, 0)
                return x

        x = torch.randn(0, 3, 4)
        self.run_test(ReshapeModel(), x)

    def test_arithmeticOps_with_low_precision(self):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class SubModule(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class MulModule(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y

        class PowModule(torch.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        z = torch.tensor([1], dtype=torch.uint8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        z = torch.tensor([1], dtype=torch.int8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        z = torch.tensor([1], dtype=torch.int16)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float32)
        z = torch.tensor([1], dtype=torch.float64)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int64)
        z = torch.tensor([1], dtype=torch.int32)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

    def test_mod_with_low_precision(self):
        class ModModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.fmod(x, y).to(dtype=torch.long)

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int32)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float64)
        self.run_test(ModModule(), (x, y))

    def test_empty_constant_shape(self):
        class Zeros(torch.nn.Module):
            def forward(self, x):
                y = torch.zeros(())
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Zeros(), x)

        class Ones(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(())
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Ones(), x)

        class Full(torch.nn.Module):
            def forward(self, x):
                y = torch.full((), 1.0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Full(), x)

        class Empty(torch.nn.Module):
            def forward(self, x):
                y = torch.empty(()).fill_(0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Empty(), x)

    def test_std_correction(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_var_correction(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_var_mean_correction(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_std_mean_correction(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_bitshift(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input):
                return (
                    input >> 1,
                    input << 3,
                    input >> torch.tensor([1, 2]),
                    input << 4,
                )

        input = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), input)

    def test_bitwise_and(self):
        class BitwiseAndModel(torch.nn.Module):
            def forward(self, input, other):
                return (
                    input & 20,
                    torch.bitwise_and(input, other),
                    other & torch.tensor([1, 2], dtype=torch.int32),
                )

        input = torch.randint(0, 255, (3, 4, 2), dtype=torch.uint8)
        other = torch.randint(-128, 127, (3, 4, 2), dtype=torch.int8)
        self.run_test(BitwiseAndModel(), (input, other))

    def test_bitshift_uint8(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input, input2):
                return (
                    input >> 1,
                    input << 3,
                    input2 >> torch.tensor([1, 2], dtype=torch.uint8),
                    input2 << 4,
                )

        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow_dynamic(self):
        class NarrowModel(torch.nn.Module):
            def forward(self, input):
                return torch.narrow(input, 0, 0, input.shape[0] - 1)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    def test_topk_int32_k(self):
        class Model(torch.nn.Module):
            def forward(self, x, k):
                return torch.topk(x, k)

        x = torch.arange(1.0, 6.0)
        k = torch.tensor(3, dtype=torch.int32)
        self.run_test(Model(), (x, k))

    def test_topk_smallest_unsorted(self):
        class MyModule(torch.nn.Module):
            def forward(self, x, k):
                # When sorted=False, order of elements in the output tensors
                # are not expected to match between PyTorch and ORT
                topk_unsorted = torch.topk(x, k, largest=False, sorted=False)
                topk_sorted = torch.topk(x, k, largest=False, sorted=True)
                return topk_sorted, torch.sort(topk_unsorted.values).values

        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModule(), (x, k))

    def test_instancenorm3d_runningstats(self):
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm3d_norunningstats(self):
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    def test_scatter_with_scalar_different_types(self):
        # Tests the case when scalar src (updates values) type is different
        # from self type. Happens only with scalar src - PyTorch does not
        # allow this when src is a tensor.
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    def test_scatter_reduce(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, index, input):
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1.0, 2.0, 3.0, 8.0])

        self.run_test(model, (src, index, input))

    def test_scatter_reduce_self_rank_zero(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, index, input):
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        empty_tensor = torch.tensor([])
        empty_idx = torch.tensor([], dtype=torch.int64)

        self.run_test(model, (empty_tensor, empty_idx, empty_tensor))

    def test_dynamic_expand_as(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = 0
                return x

        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [1]},
            additional_test_inputs=[x2],
        )

    def test_reduced_sum_dtypes(self):
        class NoDimModel(torch.nn.Module):
            def forward(self, input):
                return input.sum(dtype=torch.float)

        class DimModel(torch.nn.Module):
            def forward(self, input):
                return input.sum(dim=-1, dtype=torch.float)

        input = torch.randn((4, 4), dtype=torch.half)
        self.run_test(NoDimModel(), input)
        self.run_test(DimModel(), input)

    def test_and_or_xor(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return x ^ y, x | y, x & y, ~x

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(MyModel(), input_args=(x, y))

    def test_dynamic_repeat_interleave(self):
        class SingleDynamicModel(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(
            SingleDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class NegDynamicModel(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=-1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(
            NegDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class SingleDynamicModelFloat(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        another_x = torch.tensor([[7.1, 8.1], [5.1, 6.1]])
        self.run_test(
            SingleDynamicModelFloat(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "h"}},
        )

        class DynamicRepeatsModel(torch.nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(another_x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class DynamicRepeatsModel2(torch.nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            # The length-one repeats vector is broadcast; its values remain dynamic.
        )

        class DynamicFlattenModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(2)

        x = torch.tensor([1, 2, 3])
        self.run_test(
            DynamicFlattenModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "w"}},
        )

    def test_multiple_dynamic_repeat_interleave(self):
        class DynamicRepeatsModel(torch.nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3, 4])
        another_repeats = torch.tensor([4, 3, 2])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"input_1": {1: "r"}, "repeats_1": {0: "r"}},
        )

        class DynamicRepeatsModel2(torch.nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3])
        another_repeats = torch.tensor([4, 3])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"input_1": {0: "r"}, "repeats_1": {0: "r"}},
        )

    def test_view_dynamic(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input, other):
                return input.view(other.shape)

        x = torch.randn(2, 3, 4)
        shape = torch.randn(6, 4)
        self.run_test(
            ViewModel(),
            (x, shape),
            input_names=["x", "shape"],
            dynamic_axes={"x": [0, 1, 2], "shape": [0, 1]},
        )
        self.run_test(ViewModel(), (x, shape), remained_onnx_input_idx=[0])

    def test_view_dynamic_zero_dim(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        another_x = torch.empty((0,))
        self.run_test(
            ViewModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={
                "input_1": [
                    0,
                ]
            },
        )

    def test_weight_norm(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3))
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3), dim=-2)
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 6, 3), name="weight")
        x = torch.randn(3, 3, 5, requires_grad=True)
        self.run_test(model, x)

    def test_weight_norm_nodim(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

    def test_nonzero(self):
        class NonzeroModel(torch.nn.Module):
            def forward(self, x):
                return x.nonzero(), x.nonzero(as_tuple=True)

        x = torch.randn(60).index_fill_(0, torch.randint(0, 60, (20,)), 0).view(3, 4, 5)
        self.run_test(NonzeroModel(), (x,))

    def test_split_script(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    def test_split_size_with_slice(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        self.run_test(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        self.run_test(SplitModule(), (x, y, t), remained_onnx_input_idx=[2])

    def test_split_dynamic_axes(self):
        class Split(torch.nn.Module):
            def forward(self, x):
                return x.split(1, dim=-1)

        x = torch.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            Split(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    def test_tensor_split(self):
        class TensorSplitModel(torch.nn.Module):
            def forward(self, input):
                return (
                    input.tensor_split([1, 3]),
                    # test with output indexing.
                    input.tensor_split([2, 4])[0],
                    # test split on specific dim.
                    input.tensor_split([1, 3, 4], dim=-2),
                    # test split on specific dim and output indexing.
                    input.tensor_split([0, 2], dim=-2)[-1],
                    # test with out of bound end index (5).
                    input.tensor_split([2, 3, 5]),
                )

        self.run_test(TensorSplitModel(), torch.randn(5, 4, 3))

    def test_tensor_split_scalar(self):
        class TensorSplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.tensor_split(x, x.size(1))

        self.run_test(TensorSplitModel(), torch.randn(1, 2, 3))

    def test_tensor_split_dynamic_axes(self):
        class TensorSplitModel(torch.nn.Module):
            def forward(self, x):
                return x.tensor_split(1, dim=-1)

        x = torch.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            TensorSplitModel(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    def test_inplace_zero(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.zero_(), x

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    def test_inplace_zero_qkv(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x[2:4].zero_()

        x = torch.randn(24, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})

    def test_zeros_ones_with_tensor_input(self):
        class ZeroAndOnes(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x, 1), torch.ones(x, 1)

        x = torch.tensor([2])
        self.run_test(ZeroAndOnes(), (x,))

    def test_list_pass(self):
        class Slice(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape[2:] + y.shape[1:])

        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(1, 2, 3, 4)
        self.run_test(
            Slice(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(Slice(), (x, y), remained_onnx_input_idx=[])

        class Size(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            Size(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Size(), (x, y), remained_onnx_input_idx=[])

        class Array(torch.nn.Module):
            def forward(self, x, y):
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            Array(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Array(), (x, y), remained_onnx_input_idx=[])

        class List(torch.nn.Module):
            def forward(self, x, y):
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            List(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(List(), (x, y), remained_onnx_input_idx=[])

    def test_inplace_fill(self):
        class Fill_(torch.nn.Module):
            def forward(self, x):
                return x.fill_(3), x

        x = torch.randn(2, 3, 4)
        self.run_test(Fill_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Fill_(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic_half(self):
        class InplaceAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x.add_(y)

        class InplaceMulModel(torch.nn.Module):
            def forward(self, x, y):
                return x.mul_(y)

        x = torch.randn(2, 2, dtype=torch.half)
        y = torch.randn(2, 2, dtype=torch.float)
        self.run_test(InplaceAddModel(), (x, y), rtol=1e-2, atol=1e-2)
        self.run_test(InplaceMulModel(), (x, y), rtol=1e-2, atol=1e-2)

    def test_index_put_to_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = 1
                mask[mask == some_const] = 0
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedFillModel(), (mask, constant))

    def test_index_put_to_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = torch.ones(8)
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (mask, constant))

    def test_index_put_with_1d_mask_to_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, tensor, mask, some_const):
                tensor[mask] = some_const
                return tensor

        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        tensor = torch.randn(8, 4, 5, requires_grad=True)
        some_const = torch.randn(4, 4, 5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (tensor, mask, some_const))

    def test_scalar_type(self):
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x):
                return x.size(0) * 2 * x, 2 - x

        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ArithmeticModel(), x)

        class ComparisonModel(torch.nn.Module):
            def forward(self, x, y):
                a = torch.tensor([12.0])
                return x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0))

        x = torch.ones(2, 3, dtype=torch.int32)
        y = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ComparisonModel(), (x, y))

        class MatMulModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x + torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(MatMulModel(), x)

        class AddMMModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddMMModel(), x)

        class FullModel(torch.nn.Module):
            # add is used for exporting full
            def forward(self, x):
                return torch.full((3, 4), x)

        x = torch.tensor(12.0)
        self.run_test(FullModel(), x)

        class CatModel(torch.nn.Module):
            def forward(self, fp16, fp32):
                return torch.cat([fp16, fp32])

        fp16 = Tensor([0.5])
        fp16 = fp16.half()
        fp32 = Tensor([1.5])
        self.run_test(CatModel(), (fp16, fp32))

    def test_scalar_type_does_not_trigger_upcast_type_promotion(self):
        class DoNotUpcastModel(torch.nn.Module):
            def forward(self, x):
                scale = x.size()[-1] ** -0.5
                # 'scale' is exported as onnx float32 rank 0 tensor.
                # The following 'Mul' should NOT be promoted to float32.
                return x * scale

        x = torch.ones(2, 3, dtype=torch.float16)
        self.run_test(DoNotUpcastModel(), x)

    def test_scalar_type_promotion_onnx_where_two_prim_const(self):
        class TwoPrimConstCastWhereModel(torch.nn.Module):
            def forward(self, c):
                return torch.where(c, 0, 1.0)

        c = torch.ones(8, dtype=torch.bool)
        self.run_test(TwoPrimConstCastWhereModel(), (c))

    def test_scalar_type_promotion_onnx_where_one_prim_const(self):
        class OnePrimConstCastWhereModel(torch.nn.Module):
            def forward(self, c, x):
                return torch.where(c, x, 1.0)

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        self.run_test(OnePrimConstCastWhereModel(), (c, x))

    def test_scalar_type_promotion_onnx_where_one_tensor_const(self):
        class OneTensorConstCastWhereModel(torch.nn.Module):
            def forward(self, c, x):
                return torch.where(c, x, torch.ones(size=(), dtype=torch.float64))

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        self.run_test(OneTensorConstCastWhereModel(), (c, x))

    def test_scalar_type_upcast_type_promotion_onnx_where_no_const(self):
        class OnnxWhereUpcastModel(torch.nn.Module):
            def forward(self, c, x, y):
                return torch.where(c, x, y)

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        y = torch.ones(8, dtype=torch.float32)

        self.run_test(OnnxWhereUpcastModel(), (c, x, y))

    def test_unfold(self):
        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(4, 2, 3, requires_grad=True)
        y = torch.randn(2, 1, 3, requires_grad=True)
        self.run_test(
            UnfoldModel(),
            x,
            dynamic_axes={"x": [0, 1]},
            input_names=["x"],
            additional_test_inputs=[y],
        )

    def test_unfold_dynamic_inputs(self):
        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    def test_spectral_norm(self):
        m = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 4))

        x = torch.randn(6, 2)
        self.run_test(m, (x,))

    def test_add_inplace(self):
        class InplaceAddModel(torch.nn.Module):
            def forward(self, x):
                x += 12
                return x

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(InplaceAddModel(), x)

    def test_addcmul(self):
        class AddcmulModel(torch.nn.Module):
            def forward(self, x, t1, t2):
                return torch.addcmul(x, t1, t2), torch.addcmul(x, t1, t2, value=2.2)

        x = torch.randn(1, 3)
        t1 = torch.randn(3, 1)
        t2 = torch.randn(1, 3)
        self.run_test(AddcmulModel(), (x, t1, t2))

    def test_rsqrt(self):
        class RsqrtModel(torch.nn.Module):
            def forward(self, x):
                return x.rsqrt()

        x = torch.randn(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        class RsqrtModel(torch.nn.Module):
            def forward(self, x):
                return x.rsqrt()

        x = torch.zeros(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    def test_unique(self):
        class UniqueModel(torch.nn.Module):
            def forward(self, x):
                return torch.unique(
                    x, sorted=True, return_inverse=False, return_counts=True
                )

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    def test_unique_along_dim(self):
        class UniqueModel(torch.nn.Module):
            def forward(self, x):
                return torch.unique(
                    x, dim=0, sorted=True, return_inverse=True, return_counts=False
                )

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    def test_cumsum(self):
        class CumSum(torch.nn.Module):
            def forward(self, input):
                return torch.cumsum(input, dim=0)

        x = torch.randn(2, 3, 4)
        model = CumSum()
        self.run_test(model, x)

    def test_cumsum_with_cast(self):
        class CumSum(torch.nn.Module):
            def forward(self, input):
                return torch.cumsum(input, dim=0, dtype=torch.float32)

        model = CumSum()
        x = torch.tensor([2, 3, 4], dtype=torch.int32)
        self.run_test(model, x)
        x = torch.tensor([False, True, True])
        self.run_test(model, x)

    def test_embedding_bag(self):
        model = torch.nn.EmbeddingBag(10, 5, mode="sum", scale_grad_by_freq=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode="sum", include_last_offset=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode="max")
        input = torch.randint(10, (7, 5))
        self.run_test(model, (input))

    def test_embedding_bag_1d_per_sample_weights(self):
        class EmbeddingModel(torch.nn.Module):
            def forward(self, embedding_matrix, input, offset, weights):
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offset,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel()
        x = torch.randint(7, (6,))
        w = torch.randn(
            6,
        )
        offset = torch.tensor([0, 2, 5])
        embedding_matrix = torch.rand(10, 15)
        self.run_test(model, (embedding_matrix, x, offset, w))

    def test_embedding_bag_dynamic_input(self):
        class EmbeddingModel1D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights, offsets):
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offsets,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel1D()
        x = torch.randint(7, (6,))
        w = torch.randn(
            6,
        )
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (2,))
        w2 = torch.randn(
            2,
        )
        embedding_matrix2 = torch.rand(12, 25)
        offsets2 = torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
        )
        self.run_test(
            model,
            (embedding_matrix, x, w, offsets),
            additional_test_inputs=[(embedding_matrix2, x2, w2, offsets2)],
            input_names=["embedding_matrix", "x", "offsets", "w"],
            dynamic_axes={
                "embedding_matrix": [0, 1],
                "x": [0],
                "offsets": [0],
                "w": [0],
            },
        )

        class EmbeddingModel2D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        model = EmbeddingModel2D()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (3, 5))
        w2 = torch.randn(3, 5)
        embedding_matrix2 = torch.rand(12, 25)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            additional_test_inputs=[(embedding_matrix2, x2, w2)],
            input_names=["embedding_matrix", "x", "w"],
            dynamic_axes={"embedding_matrix": [0, 1], "x": [0, 1], "w": [0, 1]},
        )

    def test_meshgrid(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        x = torch.randn(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    def test_meshgrid_scalar(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.tensor(2.0)
        self.run_test(Meshgrid(), (x, y, z))

    def test_baddbmm(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, batch1, batch2):
                return torch.baddbmm(
                    input, batch1, batch2, alpha=torch.tensor(5), beta=3.5
                )

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2))

    def test_baddbmm_dynamic(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, batch1, batch2, alpha, beta):
                return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = torch.tensor(5)
        beta = torch.tensor(3.5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    def test_numel(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return input.numel() * input

        x = torch.randn(2, 3, 5)
        x2 = torch.randn(4, 5, 6)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[(x2,)],
        )

    def test_numel_empty(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return input.numel() * input

        x = torch.randn(0)
        x2 = torch.randn(4)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[(x2,)],
        )

    def test_cast_to_bool(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, other):
                return torch.cat((input.to(other), other), 0)

        x = torch.randn(2, 3, 4)
        y = torch.zeros([2, 3, 4], dtype=torch.bool)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_cast_type_as_with_bfloat16(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ones((3, 4), dtype=torch.bfloat16)
                x = x.type_as(y)
                return x.to(dtype=torch.float16)

        x = torch.ones(3, 4, dtype=torch.float16)
        model = MyModule()
        self.run_test(model, x)

    def test_type_as(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([1.0])
                return x.type_as(y)

        a = torch.tensor([True, False], dtype=torch.bool)
        b = torch.randn(3, 4, dtype=torch.double)
        c = torch.ones((2, 2), dtype=torch.int64)
        model = MyModule()
        self.run_test(model, a)
        self.run_test(model, b)
        self.run_test(model, c)

    def test_ones_bool(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                true = torch.ones(input.shape, dtype=torch.bool)
                return input.to(true) & true

        x = torch.randn(2, 3, 4)
        model = MyModule()
        self.run_test(model, x)

    def test_log(self):
        class Log(torch.nn.Module):
            def forward(self, input):
                return torch.log(input)

        x = torch.rand(2, 3, 4)
        model = Log()
        self.run_test(model, x)

    def test_log1p(self):
        class Log1p(torch.nn.Module):
            def forward(self, input):
                return torch.log1p(input)

        x = torch.rand(2, 3, 4)
        model = Log1p()
        self.run_test(model, x)

    def test_log10(self):
        class Log10(torch.nn.Module):
            def forward(self, input):
                return torch.log10(input)

        x = torch.rand(2, 3, 4)
        model = Log10()
        self.run_test(model, x)

    def test_log2(self):
        class Log2(torch.nn.Module):
            def forward(self, input):
                return torch.log2(input)

        x = torch.tensor(1.0)
        model = Log2()
        self.run_test(model, x)

    def test_round(self):
        class Round(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        x = torch.tensor([0.9920, -1.0362, -1.5000, 3.5000], requires_grad=True)
        self.run_test(Round(), x)

        int_x = torch.tensor([9920, 1036, -1500, 35], dtype=torch.int32)
        self.run_test(Round(), int_x)

    def test_stft_default(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_hop_length(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                hop_length = 4
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_non_divisible_hop_length(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                hop_length = 5
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_window_int_same_size(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_window_int_different_size(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 9
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_window_custom(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = torch.hann_window(16)
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    window=window,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_wrong_custom_window_size(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = torch.hann_window(10)
                return torch.stft(
                    x, n_fft=n_fft, window=window, center=False, return_complex=False
                )

        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.run_test(STFT(), x)

    def test_stft_wrong_window_length(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_len = 17
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    def test_stft_window_size_with_win_len(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = torch.hann_window(10)
                win_len = 10
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    window=window,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_one_dimension(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_wrong_input_size(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = torch.randn((1, 1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    def test_stft_normalize(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    normalized=True,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_stft_not_onesided(self):
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    onesided=False,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_constant_pad(self):
        model = torch.nn.ConstantPad1d(2, 3.5)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_pad_circular(self):
        class PadModel(torch.nn.Module):
            def forward(self, x):
                out = torch.nn.functional.pad(x, (1, 2, 1, 2), mode="circular")
                return out

        x = torch.randn(2, 3, 3, 4)
        self.run_test(PadModel(), (x))

    def test_pad_circular_negative(self):
        # Test for different pad integer types
        class PadModel(torch.nn.Module):
            def forward(self, x):
                out = torch.nn.functional.pad(x, (-1, -2), mode="circular")
                return out

        x = torch.randn(2, 3, 6)
        self.run_test(PadModel(), (x))

    def test_pad_circular_dynamic_axes(self):
        class PadModel(torch.nn.Module):
            def forward(self, x):
                out = torch.nn.functional.pad(x, (2, 1, 2, 1), mode="circular")
                return out

        x = torch.randn(4, 3, 5, 6)
        self.run_test(
            PadModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    def test_unsupported_pad(self):
        class Pad(torch.nn.Module):
            def forward(self, x, pad: list[int]):
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        y = [2, 4]

        # Modern torch.export specializes Python integer padding arguments.
        self.run_test(Pad(), (x, y))

    def test_if_fold(self):
        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    y = y - 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.numel() > 1:  # noqa: SIM108 - preserve the control-flow regression
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() != 3:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() >= 1:  # noqa: SIM108 - preserve the control-flow regression
                    y = y + 4
                else:
                    y = y - 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() <= 1:  # noqa: SIM108 - preserve the control-flow regression
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() < 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() == 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    y = y + 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.numel() != 0 and y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, x, y):
                if x.numel() == y.numel():  # noqa: SIM108 - preserve the control-flow regression
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

        class IfFoldModel(torch.nn.Module):
            def forward(self, x, y):
                if x.numel() != y.numel():  # noqa: SIM108 - preserve the control-flow regression
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

    def test_reflection_pad(self):
        model = torch.nn.ReflectionPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReflectionPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_replication_pad(self):
        model = torch.nn.ReplicationPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReplicationPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_im2col(self):
        class Unfold(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.nn.functional.unfold(
                        input, kernel_size=(10, 15), dilation=2, padding=5, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(2, 2), dilation=1, padding=0, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(1, 1), dilation=5, padding=2, stride=3
                    ),
                )

        x = torch.rand(1, 1, 200, 100)
        self.run_test(Unfold(), x)

    def test_det(self):
        class Det(torch.nn.Module):
            def forward(self, x):
                return torch.linalg.det(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(Det(), x)

    def test_linalg_cross(self):
        class Cross(torch.nn.Module):
            def forward(self, x, y):
                return torch.linalg.cross(x, y, dim=1), torch.linalg.cross(x, y)

        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(1, 3, 1, 3)
        self.run_test(Cross(), input_args=(x, y))

    def test_logdet(self):
        class LogDet(torch.nn.Module):
            def forward(self, x):
                return torch.logdet(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LogDet(), x)

    def test_outer(self):
        class Outer(torch.nn.Module):
            def forward(self, x, y):
                return torch.outer(x, y)

        x = torch.arange(1, 5)
        y = torch.arange(1, 4)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(1, 6).to(dtype=torch.float32)
        y = torch.arange(1, 4).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(2, 5).to(dtype=torch.float32)
        y = torch.arange(2, 4).to(dtype=torch.float64)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(3, 6).to(dtype=torch.int32)
        y = torch.arange(4, 7).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

    def test_movedim(self):
        class MovedimModel(torch.nn.Module):
            def forward(self, x):
                return (
                    x.movedim(1, 3),
                    x.movedim(2, 0),
                    x.movedim(1, 1),
                    x.movedim((1, 2, 3), (3, 0, 1)),
                    x.movedim((0, 1, 2), (1, 2, 3)),
                    x.movedim((1, 3, 2), (1, 3, 2)),
                )

        x = torch.randn(5, 3, 4, 2)

        self.run_test(MovedimModel(), x)

    def test_moveaxis(self):
        # moveaxis is an alias of movedim; thus, mostly copied from `test_movedim`.
        class MoveaxisModel(torch.nn.Module):
            def forward(self, x):
                return (
                    x.moveaxis(1, 3),
                    x.moveaxis(2, 0),
                    x.moveaxis(1, 1),
                    x.moveaxis((1, 2, 3), (3, 0, 1)),
                    x.moveaxis((0, 1, 2), (1, 2, 3)),
                    x.moveaxis((1, 3, 2), (1, 3, 2)),
                )

        x = torch.randn(5, 3, 4, 2)

        self.run_test(MoveaxisModel(), x)

    def test_einsum(self):
        class EinsumModelBatchDiagonal(torch.nn.Module):
            def forward(self, x):
                eqn = "...ii ->...i"
                return torch.einsum(eqn, x)

        for x in [torch.randn(3, 5, 5), torch.randn(3, 5, 5).to(dtype=torch.bool)]:
            self.run_test(EinsumModelBatchDiagonal(), input_args=(x,))

        class EinsumModelBatchMatmul(torch.nn.Module):
            def forward(self, x, y):
                eqn = "bij, bjk -> bik"
                return torch.einsum(eqn, x, y)

        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input_args=(x, y))

        class EinsumModelInnerProd(torch.nn.Module):
            def forward(self, x, y):
                eqn = "i,i"
                return torch.einsum(eqn, x, y)

        x = torch.randn(5)
        y = torch.randn(5)
        self.run_test(EinsumModelInnerProd(), input_args=(x, y))

        class EinsumModelTranspose(torch.nn.Module):
            def forward(self, x):
                eqn = "ij->ji"
                return torch.einsum(eqn, x)

        for x in [torch.randn(3, 4), torch.randn(3, 4).to(dtype=torch.bool)]:
            self.run_test(EinsumModelTranspose(), input_args=(x,))

    def test_cosine_similarity(self):
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.CosineSimilarity(dim=2), input_args=(x, y))

    def test_pairwise_distance(self):
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.PairwiseDistance(p=2.0), input_args=(x, y))

    def test_cross(self):
        class Cross(torch.nn.Module):
            def forward(self, x, y):
                return torch.cross(x, y, dim=3), torch.cross(x, y)

        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(5, 3, 2, 3)
        self.run_test(Cross(), input_args=(x, y))

    def test_cdist(self):
        class Cdist(torch.nn.Module):
            def forward(self, x, y):
                return torch.cdist(x, y)

        x = torch.randn(5, 3, 3)
        y = torch.randn(5, 2, 3)
        self.run_test(Cdist(), input_args=(x, y))

    def test_cdist_euclid_dist(self):
        class Cdist(torch.nn.Module):
            def forward(self, x, y):
                return torch.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist")

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    def test_cdist_euclid_dist_if_necessary(self):
        class Cdist(torch.nn.Module):
            def forward(self, x, y):
                return torch.cdist(
                    x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"
                )

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    def test_cdist_no_euclid_dist(self):
        class Cdist(torch.nn.Module):
            def forward(self, x, y):
                return torch.cdist(
                    x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
                )

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    def test_torch_mm(self):
        class M(torch.nn.Module):
            def forward(self, mat1, mat2):
                mm = torch.mm(mat1, mat2)
                return mm

        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    def test_where_with_bool_tensor(self):
        class M(torch.nn.Module):
            def forward(self, mat1, mat2):
                out = torch.where(mat1 > 0, mat1, mat2)
                return out

        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    def test_where_with_byte_tensor(self):
        class M(torch.nn.Module):
            def forward(self, cond, mat1, mat2):
                # Byte predicates are deprecated; make their boolean meaning explicit.
                out = torch.where(cond.to(torch.bool), mat1, mat2)
                return out

        cond = torch.ones(2, 3, dtype=torch.uint8)
        cond[1, 2] = 0
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(cond, mat1, mat2))

    def test_isinf(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isinf()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    def test_isfinite(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isfinite()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(M(), (x,))

    def test_isnan(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isnan()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    def test_nan_to_num(self):
        class NoParams(torch.nn.Module):
            def forward(self, x):
                return x.nan_to_num()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        xint = torch.ones((2, 4), dtype=torch.int)
        xhalf = torch.ones((2, 4), dtype=torch.half)
        self.run_test(NoParams(), (x,))
        self.run_test(NoParams(), (xint,))
        self.run_test(NoParams(), (xhalf,))

        class WithParams(torch.nn.Module):
            def forward(self, x):
                return x.nan_to_num(nan=2.3, posinf=4.5, neginf=6.7)

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(WithParams(), (x,))

    def test_maximum_minimum(self):
        class ModelWithNan(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y), torch.minimum(x, y)

        x = torch.tensor([-2, -2, float("nan")])
        y = torch.rand(1, 3)
        self.run_test(ModelWithNan(), (x, y))

    def test_minimum_dtypes(self):
        class MinimumModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.minimum(x, y)

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MinimumModel(), (x, y))

    def test_maximum_dtypes(self):
        class MaximumModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y)

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MaximumModel(), (x, y))

    def test_any(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.any()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):
            def forward(self, x):
                return x.any(dim=1)

        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):
            def forward(self, x):
                return x.any(dim=1, keepdim=True)

        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    def test_all(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.all()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):
            def forward(self, x):
                return x.all(dim=1)

        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):
            def forward(self, x):
                return x.all(dim=1, keepdim=True)

        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    def test_rrelu_eval(self):
        x = torch.tensor([0.5, -0.5])
        self.run_test(torch.nn.RReLU(0.1, 0.3).eval(), x)

    def test_lower_tuple(self):
        class TupleModule(torch.nn.Module):
            def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
                a = (input1, input2)
                b = a
                c = (input1, input2, input3)
                for _ in range(5):
                    d = a[0]
                    for _ in range(2):
                        e, f = a
                        a = (d, f)
                        f = c[2]
                        if f.size(0) != input1.size(-1):
                            g = b[1]
                            b = (g, f)
                        else:
                            k = c[1:]
                            b = (f, k[0])
                    m, n = b
                    c = (input1, n, m)
                p, q, r = c
                return p + q + r

        input1 = torch.randn(2)
        input2 = torch.randn(2)
        input3 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2, input3))

    def test_lower_tuple_2(self):
        class TupleModule(torch.nn.Module):
            def forward(self, input1: Tensor, input2: Tensor) -> tuple[Tensor, Tensor]:
                a = (input1, input2)
                for _ in range(5):
                    c, d = a
                    a = (c, d)
                return a

        input1 = torch.randn(2)
        input2 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2))

    def test_lower_tuple_3(self):
        class TupleModule(torch.nn.Module):
            def forward(
                self,
                input1: tuple[Tensor, Tensor],
                input2: tuple[Tensor, Tensor],
            ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
                a = input1
                b = input2
                for _ in range(5):
                    c, d = a
                    e, f = b
                    if c.shape[0] == e.shape[0]:
                        e = e + c
                    else:
                        f = f + d
                    a = (e, f)
                    b = (c, d)
                return a, b

        input1 = (torch.randn(2), torch.randn(2))
        input2 = (torch.randn(2), torch.randn(2))
        self.run_test(TupleModule(), (input1, input2))

    def test_where(self):
        class Model(torch.nn.Module):
            def forward(self, cond, input, other):
                return torch.where(cond, input, other)

        x = torch.randint(0, 1, (2, 3, 4), dtype=torch.bool)
        y = torch.randn(2, 1, 4)
        z = torch.ones(2, 3, 1)
        self.run_test(Model(), (x, y, z))

    def test_where_condition(self):
        class Model1(torch.nn.Module):
            def forward(self, input):
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            def forward(self, input, other):
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    def test_where_condition_script(self):
        class Model1(torch.nn.Module):
            def forward(self, input):
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            def forward(self, input, other):
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    def test_derive_index_scripting(self):
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    def test_derive_index(self):
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    def test_split_tensor_scalar(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.split(x, x.size(1))

        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(SplitModel(), x)

    def test_split_tensor_multi(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.split(x, torch.ones(3))

        x = torch.randn(1, 2, 3, requires_grad=True)

        def run_model():
            SplitModel(x)

        self.assertRaises(TypeError, run_model)

    def test_embedding_renorm(self):
        n, d = 7, 5
        embedding = torch.nn.Embedding(n, d, max_norm=0.2)
        idx = torch.tensor([2, 1])
        self.run_test(embedding, idx)

        embedding = torch.nn.Embedding(n, d, max_norm=0.5, norm_type=1.0)
        idx = torch.tensor([4, 3, 4, 2])
        self.run_test(embedding, idx)

    def test_fake_quantize_per_channel(self):
        class FakeQuantizePerChannelModel(torch.nn.Module):
            def forward(self, input):
                amax = torch.ones(4)
                scale = amax / 127.0
                zero_point = torch.zeros_like(amax, dtype=torch.int)
                # Quantize twice to test different branches
                y = torch.fake_quantize_per_channel_affine(
                    input, scale, zero_point, 1, 0, 255
                )
                return torch.fake_quantize_per_channel_affine(
                    y, scale, zero_point, 1, -128, 127
                )

        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerChannelModel(), (x))

    def test_nms(self):
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return torchvision.ops.nms(boxes, scores, 0.5)

        self.run_test(Module(), (boxes, scores))

    def test_batched_nms(self):
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)
        idxs = torch.randint(0, 5, size=(num_boxes,))

        class Module(torch.nn.Module):
            def forward(self, boxes, scores, idxs):
                return torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)

        self.run_test(Module(), (boxes, scores, idxs))

    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = torchvision.ops.RoIAlign((5, 5), 1.0, 2)
        self.run_test(model, (x, single_roi))

    def test_roi_align_aligned(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        model1 = torchvision.ops.RoIAlign((5, 5), 1.0, 2, aligned=True)
        self.run_test(model1, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model2 = torchvision.ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model3 = torchvision.ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model4 = torchvision.ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    def test_unsafe_chunk(self):
        class ChunkModel(torch.nn.Module):
            def forward(self, x):
                return torch.unsafe_chunk(x, 3, dim=1)

        model = ChunkModel()
        model.eval()
        x = torch.randn(1, 18)
        self.run_test(model, x, input_names=["x"])

    def test_symbolic_shape_inference(self):
        # ConstantOfShape is tested in test_embedding_bag
        # Tile is tested in test_repeat
        # test Shape, Reshape, Transpose, Gather
        class ShapeModel(torch.nn.Module):
            def forward(self, x, y):
                shape = x.size()[:3] + (-1,)  # shape [4], ("batch", 3, 4, -1)
                y = y.reshape(shape)  # batch, 3, 4, 10/batch
                return y.transpose(1, 2)

        model = ShapeModel()
        model.eval()
        x = torch.ones(2, 3, 4, 5)
        y = torch.ones(3, 4, 5, 2)
        self.run_test(
            model,
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(model, (x, y), remained_onnx_input_idx=[1])

        class ViewModel(torch.nn.Module):
            def forward(self, x):
                return x.view(-1)

        model = ViewModel()
        model.eval()
        x = torch.tensor(2.0)
        self.run_test(model, (x,))

        # test prim::ListConstruct for Reshape input 1
        class ViewModel_2(torch.nn.Module):
            def forward(self, x):
                N, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)

        model = ViewModel_2()
        model.eval()
        x = torch.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    def test_symbolic_shape_inference_arange(self):
        # test Range
        class ArangeModel(torch.nn.Module):
            def forward(self, signal):
                frame_step = 2
                frames, frame_length = signal.size()[-2:]

                subframe_length = signal.size()[0]
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                frame = torch.arange(0, output_subframes)
                return frame

        model = ArangeModel()
        model.eval()
        M, C, K, N = 1, 2, 3, 4
        x = torch.randint(5, (M, C, K, N))
        y = torch.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]})
        self.run_test(model, x, remained_onnx_input_idx=[])
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[(x,), (y,)],
        )

    def test_symbolic_shape_inference_box(self):
        # test NonZero
        class BoxModel(torch.nn.Module):
            def forward(self, boxes):
                min_size = 1e-2
                ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                keep = (ws >= min_size) & (hs >= min_size)
                keep = torch.where(keep)[0]
                return keep

        model = BoxModel()
        model.eval()
        x = torch.ones(2, 4)
        y = torch.ones(3, 5)
        self.run_test(model, x)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[(x,), (y,)],
        )

    def test_symbolic_shape_inference_arange_2(self):
        # test Range
        class ArangeModel(torch.nn.Module):
            def forward(self, start):
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        self.run_test(
            ArangeModel(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel(), (x,), remained_onnx_input_idx=[])

        class ArangeModel2(torch.nn.Module):
            def forward(self, start):
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.double)

        x = torch.randn(2, 3, 4)
        self.run_test(
            ArangeModel2(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel2(), (x,), remained_onnx_input_idx=[])

    def test_symbolic_shape_inference_nonzero(self):
        class OneLikeModel(torch.nn.Module):
            def forward(self, x):
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return torch.nonzero(ones)

        x = torch.randn(2)
        self.run_test(OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(
            OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])

        class ZeroLikeModel(torch.nn.Module):
            def forward(self, x):
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return torch.nonzero(zeros)

        x = torch.randn(2)
        self.run_test(ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(
            ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])

    def test_symbolic_shape_inference_expand_1(self):
        class ExpandModel(torch.nn.Module):
            def forward(self, x):
                return x.expand(4, 6, 2)

        x = torch.randn(6, 1, requires_grad=True)
        self.run_test(ExpandModel(), (x,))

    def test_symbolic_shape_inference_expand_2(self):
        class M(torch.nn.Module):
            def forward(self, x):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                return causal_mask.transpose(0, 1)

        x = torch.randn(3, 16)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(M(), (x,), remained_onnx_input_idx=[])

    def test_symbolic_shape_inference_slice(self):
        class M(torch.nn.Module):
            def forward(self, x, position_bias):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                position_bias = position_bias[:, :, -seq_length:, :]
                return position_bias.transpose(0, 1)

        x = torch.randn(3, 16)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(
            M(),
            (x, position_bias),
            input_names=["x", "position_bias"],
            dynamic_axes={"x": [0, 1], "position_bias": [0, 1, 2, 3]},
        )
        self.run_test(M(), (x, position_bias), remained_onnx_input_idx=[1])

    def test_symbolic_shape_inference_slice_2(self):
        class M(torch.nn.Module):
            def forward(self, position_bias):
                position_bias = position_bias[:, :, -2:, :]
                return position_bias.transpose(0, 1)

        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (position_bias,))

    def test_symbolic_shape_inference_dynamic_axes(self):
        class M(torch.nn.Module):
            def forward(self, input_ids):
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                return input_ids.transpose(0, 1)

        x = torch.randn(3, 16)
        self.run_test(
            M(),
            (x,),
            input_names=["input_ids"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        )

    def test_tensordot_dim_count(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                output = torch.tensordot(x, y, 2)
                return output

        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        self.run_test(M(), (x, y))

    def test_tensordot_dim_list(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                output = torch.tensordot(x, y, ([1, -2, -1], [1, 0, 3]))
                return output

        x = torch.randint(6, (7, 4, 3, 5, 2))
        y = torch.randint(6, (5, 4, 4, 2, 6))

        self.run_test(M(), (x, y))

    def test_tensordot_dynamic_dim(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                output = torch.tensordot(x, y, 2)
                return output

        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        new_x = torch.randint(6, (8, 6, 2, 5))
        new_y = torch.randint(6, (2, 5, 3, 4))

        self.run_test(
            M(),
            (x, y),
            additional_test_inputs=[(new_x, new_y)],
            input_names=["input_x", "input_y"],
            dynamic_axes={"input_x": [0, 1, 2, 3], "input_y": [0, 1, 2, 3]},
        )

    def test_to_device(self):
        class M_ToDevice(torch.nn.Module):
            def forward(self, x, y):
                return x.to(y.device), y

        class M_ToDeviceDtype(torch.nn.Module):
            def forward(self, x, y):
                return x.to(y.device, dtype=torch.long), y

        x = torch.randn(6)
        y = torch.randn(6)

        self.run_test(M_ToDevice(), (x, y))
        self.run_test(M_ToDeviceDtype(), (x, y))

    def test_fill(self):
        class FillModule(torch.nn.Module):
            def forward(self, x, filled_value: int):
                return x.fill_(filled_value)

        x = torch.randn((4, 5, 6))
        filled_value = 7
        self.run_test(FillModule(), (x, filled_value))

        class FillFloatModule(torch.nn.Module):
            def forward(self, x, filled_value: float):
                return x.fill_(filled_value)

        x = torch.randn((4, 5, 6))
        filled_value = 7.5
        self.run_test(FillFloatModule(), (x, filled_value))

        class FillScalarModule(torch.nn.Module):
            def forward(self, x):
                res = x + 2
                res.fill_(2.5)
                return res, x

        x = torch.ones(2, 3, 4, dtype=torch.long)
        self.run_test(FillScalarModule(), x)

    def test_sum(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x)

        x = torch.ones(12, 3)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0]})

    def test_sum_empty_tensor(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x[0:0].sum(), x.sum()

        x = torch.ones(12)
        self.run_test(M(), (x,))

        x = torch.ones(2, 0, 3)
        self.run_test(M(), (x,))

        x = torch.ones(0)
        self.run_test(M(), (x,))

    def test_broad_cast_tensors(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                m = torch.broadcast_tensors(x, y)
                return m

        x = torch.randint(5, (1,))
        y = torch.randint(5, (5,))

        self.run_test(M(), (x, y))

        x = torch.randint(5, (4, 2, 1, 4))
        y = torch.randint(5, (2, 3, 1))

        self.run_test(M(), (x, y))

        x = torch.randn(2, 1, 4)
        y = torch.randn(5, 2, 3, 1)

        self.run_test(M(), (x, y))

    def test_scaled_dot_product_attention(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=1.0
                )

        # Parameters
        batch_size = 2  # Number of samples in the batch
        num_heads = 4  # Number of attention heads
        seq_length = 5  # Sequence length
        head_dim = 8  # Dimensionality of each head

        # Create random query, key, and value tensors
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim)

        self.run_test(M(), (q, k, v))

    def test_sequence_to_int(self):
        class M(torch.nn.Module):
            def forward(self, x):
                result = torch.tensor([2 for i in range(x.size()[0])], dtype=torch.int)
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    def test_sequence_to_float(self):
        class M(torch.nn.Module):
            def forward(self, x):
                result = torch.tensor(
                    [1.1 for i in range(x.size()[0])], dtype=torch.float
                )
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    def test_sequence_to_bool(self):
        class M(torch.nn.Module):
            def forward(self, x):
                result = torch.tensor(
                    [False for i in range(x.size()[0])], dtype=torch.bool
                )
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    def test_lerp(self):
        class LerpModel(torch.nn.Module):
            def forward(self, x):
                return (
                    x.lerp(torch.full_like(x, 10), 0.4),
                    x.lerp(torch.full_like(x, 20), 0.7),
                    x.lerp(torch.full_like(x, 30), torch.tensor(0.4)),
                    x.lerp(torch.full_like(x, 40), x / 10.0),
                    x.lerp(torch.tensor(10.0), x / 10.0),
                    x.lerp(torch.tensor(10.0), 0.4),
                    x.lerp(torch.tensor(10.0), torch.tensor(0.4)),
                )

        self.run_test(LerpModel(), torch.rand(5, 4, 3))
