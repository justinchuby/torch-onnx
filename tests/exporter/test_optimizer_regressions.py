# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Optimizer intents from test_utility_funs.py at e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removal 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve source model
bodies/comments and folded-operator assertions. Native initializers replace
legacy Constant-node accounting; static examples do not claim unrestricted
dynamic dimensions where constant operands or recurrent weights fix sizes.
"""

import onnx
import torch
from _common.assertions import assert_program_roundtrip
from torch.testing._internal import common_utils

import torch_onnx


class TestRecoveredOptimizer(common_utils.TestCase):
    @staticmethod
    def _semantic_node_count(graph):
        return sum(node.op_type not in {"Constant", "Identity"} for node in graph)

    @torch.no_grad()
    def _model_to_graph(self, model, args, *, input_names=None):
        args = args if isinstance(args, tuple) else (args,)
        program = torch_onnx.export(
            model.eval(), args, input_names=input_names, verbose=False
        )
        alternatives = tuple(torch.randn_like(value) for value in args)
        assert_program_roundtrip(
            program,
            examples=[alternatives],
            rtol=1e-3,
            atol=1e-6,
            strategy="TorchExportNonStrictStrategy",
        )
        parameters = {}
        for name, value in program.model.graph.initializers.items():
            assert value.const_value is not None
            parameters[name] = torch.from_numpy(value.const_value.numpy().copy())
        return program.model.graph, parameters, None

    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.transpose(a, 1, 0)
                return b + x

        x = torch.ones(3, 2)
        graph, _, __ = self._model_to_graph(TransposeModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Transpose")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_reduceL2(self):
        class ReduceModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=2, dim=-2, keepdim=False)
                return b + x

        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(ReduceModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::ReduceL2")

    def test_constant_fold_reduceL1(self):
        class NormModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=1, dim=-2)
                return b + x

        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(NormModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::ReduceL1")

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(NarrowModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Slice")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[1:10]  # index exceeds dimension
                return b + x

        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceIndexExceedsDimModule(),
            (x,),
            input_names=["x"],
        )

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Slice")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[0:-1]  # index relative to the end
                c = torch.select(a, dim=-1, index=-2)
                d = torch.select(a, dim=1, index=0)
                return b + x, c + d

        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceNegativeIndexModule(),
            (x,),
            input_names=["x"],
        )

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Slice")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")

    def test_constant_fold_gather(self):
        class GatherModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.select(a, dim=1, index=-2)
                c = torch.index_select(a, dim=-2, index=torch.tensor([0, 1]))
                return b + 1, c + x

        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(GatherModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Gather")

    def test_constant_fold_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.unsqueeze(a, -2)
                return b + x

        x = torch.ones(1, 2, 3)
        graph, _, __ = self._model_to_graph(UnsqueezeModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Unsqueeze")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_squeeze_without_axes(self):
        class SqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a) + x + torch.squeeze(a)

        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(SqueezeModule(), (x,), input_names=["x"])
        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Squeeze")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 2)

    def test_constant_fold_squeeze_with_axes(self):
        class SqueezeAxesModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a, dim=-3) + x

        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeAxesModule(), (x,), input_names=["x"]
        )

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Squeeze")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_concat(self):
        class ConcatModule(torch.nn.Module):
            def forward(self, x):
                # Why did I insert a Cast here?  There appears to be intentional
                # behavior in ONNX constant folding where constant tensors which
                # are not attached to any known to be foldable onnx
                # operations don't get extracted into the initializer graph.  So
                # without these casts, we will actually fail to pull out one of
                # the constants, thus failing constant folding.  I think the
                # test is wrong but I don't have time to write a more correct
                # test (I think the right way to go about the test is to setup
                # a predicate for what invariant graphs should hold after
                # constant folding, and then verify this predicate holds.
                # I think the asserts below are an attempt at this predicate,
                # but it is not right!)
                #
                # More commentary at
                # https://github.com/pytorch/pytorch/pull/18698/files#r340107552
                a = torch.tensor([[1.0, 2.0, 3.0]]).to(torch.float)
                b = torch.tensor([[4.0, 5.0, 6.0]]).to(torch.float)
                c = torch.cat((a, b), 0)
                d = b + c
                return x + d

        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(ConcatModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Concat")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Cast")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = self._model_to_graph(
            GruNet(),
            (input, h0),
            input_names=["input", "h0"],
        )

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Slice")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Concat")
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Unsqueeze")

        # Unsqueeze op parameter "axes" as an input instead of as an attribute when opset version >= 13
        self.assertLessEqual(self._semantic_node_count(graph), 2)

    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        A = torch.randn(2, 3)
        graph, _, __ = self._model_to_graph(MatMulNet(), (A,), input_names=["A"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Transpose")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        x = torch.randn(4, 5)
        graph, _, __ = self._model_to_graph(ReshapeModule(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Reshape")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_div(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                div = self.weight.div(torch.tensor([1, 2, 3, 4, 5]))
                return div * x

        x = torch.randn(2, 5)
        graph, _, __ = self._model_to_graph(Module(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Div")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_mul(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                mul = self.weight.mul(torch.tensor([1, 2, 3, 4, 5]))
                return mul / x

        x = torch.randn(2, 5)
        graph, _, __ = self._model_to_graph(Module(), (x,), input_names=["x"])

        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Mul")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_add(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                add = self.weight + torch.tensor([1, 2, 3, 4, 5])
                return add - x

        x = torch.randn(2, 5)
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            input_names=["x"],
        )
        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Add")
        self.assertLessEqual(self._semantic_node_count(graph), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_constant_fold_sub(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                sub = self.weight - torch.tensor([1, 2, 3, 4, 5])
                return sub + x

        x = torch.randn(2, 5)
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            input_names=["x"],
        )
        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Sub")
        self.assertLessEqual(self._semantic_node_count(graph), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0]))

    def test_constant_fold_sqrt(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                sqrt = torch.sqrt(self.weight)
                return sqrt / x

        x = torch.randn(2, 5)
        graph, _, __ = self._model_to_graph(Module(), (x,), input_names=["x"])
        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Sqrt")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    def test_constant_fold_shape(self):
        class ShapeModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        graph, _, __ = self._model_to_graph(ShapeModule(), (x,), input_names=["x"])
        for node in graph:
            self.assertNotEqual(f"onnx::{node.op_type}", "onnx::Shape")
        self.assertLessEqual(self._semantic_node_count(graph), 1)

    @torch.no_grad()
    def test_constant_fold_unsqueeze_multi_axies(self):
        class PReluModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                a = torch.randn(2, 3, 4, 5, 8, 7)
                return self.prelu(x) + a

        model = PReluModel().eval()
        x = torch.randn(2, 3, 4, 5, 8, 7)
        # ORT 1.32 does not implement RandomNormalLike-22 or later.
        program = torch_onnx.export(model, (x,), opset_version=20, verbose=False)
        onnx.checker.check_model(program.model_proto, full_check=True)
        for node in program.model.graph:
            self.assertNotIn(node.op_type, {"Unsqueeze", "Cast"})
        random_values = [
            node.outputs[0].name
            for node in program.model.graph
            if node.op_type in {"RandomNormal", "RandomNormalLike"}
        ]
        self.assertEqual(len(random_values), 1)
        names = [random_values[0], program.model.graph.outputs[0].name]
        # Observe both values in one execution, rather than equating Torch/ORT RNGs.
        noise, actual = program.compute_values(names, (x,))
        torch.testing.assert_close(actual, model.prelu(x) + noise)

    @torch.no_grad()
    def test_constant_fold_upsample_scale_fold_as_constant(self):
        model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.randn(1, 32, 224, 224)
        program = torch_onnx.export(model.eval(), (x,), verbose=False)
        # Native IR may store constants as initializers; scales must not become parameters.
        self.assertEqual(len(program.model.graph.inputs), 1)
        self.assertEqual(len(program.exported_program.graph_signature.parameters), 0)
        assert_program_roundtrip(
            program,
            rtol=1e-3,
            atol=1e-4,
            strategy="TorchExportNonStrictStrategy",
        )
