# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Export utility intents from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Use native IR,
public ONNXProgram serialization, and modern capture instead of JIT graphs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import onnx
import onnx_ir as ir
import torch
import torchvision
from test_runtime_regressions import RecoveredRuntimeHarness
from torch.testing._internal import common_utils

import torch_onnx


def _custom_gelu(
    op: torch_onnx.TranslationContext, value: ir.Value, approximate: str = "none"
) -> ir.Value:
    if approximate != "none":
        raise ValueError("This custom Gelu is exact. Use approximate='none'.")
    return op.builder.node(
        "Gelu", [value], domain="com.microsoft", version=1, output_types=[value.type]
    )[0]


def _custom_inverse(op: torch_onnx.TranslationContext, value: ir.Value) -> ir.Value:
    return op.builder.node(
        "Inverse", [value], domain="com.microsoft", version=1, output_types=[value.type]
    )[0]


class TestRecoveredExportUtilities(RecoveredRuntimeHarness):
    def test_custom_layer_tuple(self):
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input, input

        formal = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT))
        body = ir.Graph(
            [formal], [], nodes=[], name="custom_tuple", opset_imports={"": 20}
        )
        builder = torch_onnx.GraphBuilder(body, opset=20)
        body.outputs.extend(
            [builder.call("Identity", formal)[0], builder.call("Identity", formal)[0]]
        )
        function = ir.Function(
            "CustomNamespace", "Custom", "", graph=body, attributes=[]
        )

        def custom(
            op: torch_onnx.TranslationContext, value: ir.Value
        ) -> tuple[ir.Value, ir.Value]:
            left, right = op.builder.call_function(function, value, version=1)
            return left, right

        with torch.library._scoped_library("recovered_tuple", "FRAGMENT") as library:
            torch.library.define(
                "recovered_tuple::custom",
                "(Tensor input) -> (Tensor, Tensor)",
                lib=library,
            )

            @torch.library.impl(
                "recovered_tuple::custom", "CompositeExplicitAutograd", lib=library
            )
            def implementation(
                value: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return value.clone(), value.clone()

            @torch.library.register_fake("recovered_tuple::custom", lib=library)
            def fake(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return value.clone(), value.clone()

            class Custom(torch.nn.Module):
                def forward(self, input):
                    return torch.ops.recovered_tuple.custom.default(input)

            target = torch.ops.recovered_tuple.custom.default
            torch_onnx.register_custom_translation(target, custom)
            self.addCleanup(torch_onnx.unregister_custom_translation, target)
            program = torch_onnx.export(
                Custom().eval(),
                (torch.randn(2, 3),),
                opset_version=20,
                optimize=False,
                verbose=False,
                input_names=["batch"],
                dynamic_shapes=(
                    {
                        0: torch.export.Dim("rows", min=2),
                        1: torch.export.Dim("columns", min=2),
                    },
                ),
            )
            nodes = [
                node for node in program.model.graph if node.domain == "CustomNamespace"
            ]
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].op_type, "Custom")
            self.assertEqual(len(nodes[0].outputs), 2)
            self.assertIn(function.identifier(), program.model.functions)
            onnx.checker.check_model(program.model_proto, full_check=True)
            with common_utils.TemporaryDirectoryName() as directory:
                path = Path(directory) / "custom-tuple.onnx"
                program.save(path, external_data=False)
                loaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            # Retain the original singleton fixture despite narrower EP guards.
            for value in (torch.randn(1, 3), torch.randn(2, 3), torch.randn(3, 5)):
                expected = CustomFunction.apply(value)
                for candidate in (program, loaded):
                    actual = candidate(value)
                    self.assertEqual(len(actual), 2)
                    for got, want in zip(actual, expected, strict=True):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def _check_custom_domain(self, model, args, opname):
        for opset in [None, 18, 24]:
            program = torch_onnx.export(model, args, opset_version=opset, verbose=False)
            onnx.checker.check_model(program.model_proto, full_check=True)
            if opset is not None:
                self.assertEqual(program.model.opset_imports[""], opset)
            self.assertEqual(program.model.opset_imports["com.microsoft"], 1)
            self.assertTrue(
                any(
                    node.op_type == opname and node.domain == "com.microsoft"
                    for node in program.model.graph
                )
            )
            torch_onnx.testing.assert_onnx_program(program, strategy=None)
            with common_utils.TemporaryDirectoryName() as directory:
                path = Path(directory) / "custom-domain.onnx"
                program.save(path, external_data=False)
                loaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            torch_onnx.testing.assert_onnx_program(loaded, strategy=None)

    def test_custom_opsets_gelu(self):
        target = torch.ops.aten.gelu.default
        torch_onnx.register_custom_translation(target)(_custom_gelu)
        self.addCleanup(torch_onnx.unregister_custom_translation, target)
        self._check_custom_domain(
            torch.nn.GELU(approximate="none").eval(), (torch.randn(3, 3),), "Gelu"
        )

    def test_register_aten_custom_op_symbolic(self):
        target = torch.ops.aten.gelu.default
        torch_onnx.register_custom_translation(target, _custom_gelu)
        self.addCleanup(torch_onnx.unregister_custom_translation, target)
        self._check_custom_domain(
            torch.nn.GELU(approximate="none").eval(), (torch.randn(3, 3),), "Gelu"
        )

    def test_custom_opsets_inverse(self):
        class CustomInverse(torch.nn.Module):
            def forward(self, x):
                return torch.inverse(x) + x

        target = torch.ops.aten.linalg_inv.default
        torch_onnx.register_custom_translation(target, _custom_inverse)
        self.addCleanup(torch_onnx.unregister_custom_translation, target)
        x = torch.randn(2, 3, 3)
        x = x @ x.mT + torch.eye(3)
        self._check_custom_domain(CustomInverse().eval(), (x,), "Inverse")

    def test_bad_symbolic_registration(self):
        def bad_cat(op):
            return op.Identity(0)

        class CatModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, x, x), 0)

        target = torch.ops.aten.cat.default
        torch_onnx.register_custom_translation(target, bad_cat)
        self.addCleanup(torch_onnx.unregister_custom_translation, target)
        with self.assertRaisesRegex(
            torch_onnx.errors.ConversionError,
            "Native translation.*bad_cat.*failed",
        ):
            torch_onnx.export(CatModel().eval(), (torch.randn(2, 3),), verbose=False)

    @staticmethod
    def _module_scope(node):
        return ast.literal_eval(node.metadata_props["pkg.torch.onnx.name_scopes"])[-2]

    def test_node_scope(self):
        class N(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=float(i)) for i in range(num_layers)]
                )
                self.gelu1 = torch.nn.GELU()
                self.gelu2 = torch.nn.GELU()
                self.relu = N()

            def forward(self, x, y, z):
                res1 = self.gelu1(x)
                res2 = self.gelu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.relu(z)

        program = self.run_test(
            M(3), tuple(torch.randn(2, 3) for _ in range(3)), atol=1e-6
        )
        scopes = {
            self._module_scope(node)
            for node in program.model.graph
            if node.op_type != "Constant"
        }
        self.assertEqual(
            scopes, {"", "gelu1", "gelu2", "lns.0", "lns.1", "lns.2", "relu.relu"}
        )

    def test_scope_of_constants_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant):
                super().__init__()
                self.constant = constant

            def forward(self, x):
                # 'self.constant' is designed to be the same for all layers,
                # hence it is common sub expression.
                return x + self.constant

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [M(constant=torch.tensor(1.0)) for i in range(layers)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        program = self.run_test(N(), torch.randn(2, 3))
        # Native CSE shares an initializer instead of retaining three JIT Constants.
        self.assertEqual(len(program.model.graph.initializers), 1)
        scopes = {
            self._module_scope(node)
            for node in program.model.graph
            if node.op_type != "Constant"
        }
        self.assertTrue(scopes)
        self.assertTrue(scopes.issubset({f"layers.{i}" for i in range(layer_num)}))

    def test_scope_of_nodes_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant, bias):
                super().__init__()
                self.constant = constant
                self.bias = bias

            def forward(self, x):
                # 'constant' and 'x' is designed to be the same for all layers,
                # hence `x + self.constant` is common sub expression.
                # 'bias' is designed to be different for all layers,
                # hence `* self.bias` is not common sub expression.
                return (x + self.constant) * self.bias

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        M(constant=torch.tensor([1.0]), bias=torch.randn(1))
                        for i in range(layers)
                    ]
                )

            def forward(self, x):
                y = []
                for layer in self.layers:
                    y.append(layer(x))
                return y[0], y[1], y[2]

        program = self.run_test(N(), torch.randn(2, 3))
        adds = [node for node in program.model.graph if node.op_type == "Add"]
        multiplies = [node for node in program.model.graph if node.op_type == "Mul"]
        self.assertEqual([self._module_scope(node) for node in adds], ["layers.0"])
        self.assertEqual(
            [self._module_scope(node) for node in multiplies],
            [f"layers.{i}" for i in range(layer_num)],
        )

    def test_is_in_onnx_export(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                if not (
                    torch_onnx.is_in_onnx_export() and torch.onnx.is_in_onnx_export()
                ):
                    raise AssertionError("Export flags must be set inside capture")
                raise ValueError("recovered-export-failure")

        self.assertFalse(torch_onnx.is_in_onnx_export())
        self.assertFalse(torch.onnx.is_in_onnx_export())
        with self.assertRaisesRegex(
            torch_onnx.errors.TorchExportError, "recovered-export-failure"
        ):
            torch_onnx.export(MyModule(), (torch.randn(3, 4),), verbose=False)
        self.assertFalse(torch_onnx.is_in_onnx_export())
        self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_split_to_slice(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        program = self.run_test(
            SplitModule(),
            (torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 7)),
            input_names=["x", "y", "t"],
            dynamic_shapes=tuple(
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO} for _ in range(3)
            ),
            additional_test_inputs=[
                (torch.randn(3, 2), torch.randn(4, 5), torch.randn(6, 7)),
                (torch.randn(3, 4), torch.randn(4, 5), torch.randn(6, 9)),
            ],
        )
        self.assertNotIn(
            "SplitToSequence", [node.op_type for node in program.model.graph]
        )

    def test_export_mode(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return y

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        for training in (False, True):
            with self.subTest(training=training):
                model.train(training)
                self.run_test(model, x, training=training)
                self.assertEqual(model.training, training)

    def test_error_on_data_parallel(self):
        # Original NB: remove the rejection once DataParallel can be handled.
        # Modern capture handles the CPU wrapper; do not retain the old rejection.
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        # The original height 3 was invalid for its bottom reflection padding 4.
        self.run_test(model, torch.randn(1, 2, 6, 7))

    def test_scripting_param(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model = MyModule().train()
        x = torch.randn(10, 3, 128, 128)
        program = torch_onnx.export(model, (x,), optimize=False, verbose=False)
        torch_onnx.testing.assert_onnx_program(program, rtol=1e-3, atol=1e-5)
        with common_utils.TemporaryDirectoryName() as directory:
            path = Path(directory) / "training-parameters.onnx"
            program.save(path, keep_initializers_as_inputs=True, external_data=False)
            graph = onnx.load(path)
            onnx.checker.check_model(graph, full_check=True)
        names = {value.name for value in graph.graph.input}
        self.assertTrue(set(dict(model.named_parameters())).issubset(names))

    def test_onnx_node_naming(self):
        class MainModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._module_1 = torch.nn.Linear(10, 10)
                self._module_2 = torch.nn.Linear(10, 10)
                self._module_3 = torch.nn.Linear(10, 10)
                self._module_4 = torch.nn.Linear(10, 10)

            def forward(self, x):
                y = self._module_1(x)
                z = self._module_2(y)
                z = self._module_3(y * z)
                z = self._module_4(y * z)
                return z

        program = self.run_test(MainModule(), torch.ones(1, 10), output_names=["y"])
        names = [node.name for node in program.model.graph]
        self.assertTrue(all(names))
        self.assertEqual(len(names), len(set(names)))
        scopes = {
            self._module_scope(node)
            for node in program.model.graph
            if node.op_type == "Gemm"
        }
        self.assertEqual(scopes, {f"_module_{index}" for index in range(1, 5)})

    def test_unused_initializers(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1)
                )
                self.k_proj = torch.nn.Linear(5, 5, bias=True)

            def forward(self, x):
                x = self.conv2(x)
                return x

        program = self.run_test(Model(), torch.randn(20, 16, 50, 100), atol=1e-4)
        self.assertEqual(
            set(program.model.graph.initializers), {"conv2.weight", "conv2.bias"}
        )

    def test_fuse_conv_bn(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        program = self.run_test(
            Fuse(), torch.randn(2, 3, 2, 2, requires_grad=True), atol=1e-6
        )
        self.assertEqual([node.op_type for node in program.model.graph], ["Conv"])

    def test_fuse_resnet18(self):
        model = torchvision.models.resnet18(weights=None)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        program = self.run_test(model, x, atol=1e-5)
        self.assertNotIn(
            "BatchNormalization", [node.op_type for node in program.model.graph]
        )

    def test_onnx_function_substitution_pass(self):
        def f(x: torch.Tensor, y: torch.Tensor):
            z = x - y
            return x + z

        class MyModule(torch.nn.Module):
            def forward(self, x, y):
                return f(x, y)

        program = self.run_test(
            MyModule(),
            (torch.tensor([11]), torch.tensor([12])),
            additional_test_inputs=[(torch.tensor([21]), torch.tensor([-2]))],
        )
        # Check that the prim::Constant node in the graph for representing the
        # scripted function `f` is removed and the following prim::CallFunction
        # is replced by inline graph, with onnx::Sub and onnx::Add nodes.
        self.assertEqual([node.op_type for node in program.model.graph], ["Sub", "Add"])

    def test_onnx_value_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_weight = torch.nn.Parameter(torch.randn(3, 3))
                self.in_bias = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                start = 0
                end = None
                weight = self.in_weight
                bias = self.in_bias
                weight = weight[start:end, :]
                if bias is not None:
                    bias = bias[start:end]
                return torch.nn.functional.linear(x, weight, bias)

        model = MyModule().eval()
        x = torch.randn(3, 3)
        program = torch_onnx.export(model, (x,), optimize=False, verbose=False)
        torch_onnx.testing.assert_onnx_program(program)
        with common_utils.TemporaryDirectoryName() as directory:
            path = Path(directory) / "named-weights.onnx"
            program.save(path, keep_initializers_as_inputs=True, external_data=False)
            graph = onnx.load(path)
            onnx.checker.check_model(graph, full_check=True)
        self.assertEqual(graph.graph.input[1].name, "in_weight")
        self.assertEqual(graph.graph.input[2].name, "in_bias")

    def _test_deduplicate_initializers(self, *, precaptured):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 3)
                self.layer2 = torch.nn.Linear(3, 3)

                # Reusing layers.
                self.layer3 = self.layer1

                # Reusing parameters.
                self.layer2.weight = self.layer1.weight
                self.layer1.bias = self.layer2.bias

                # Parameter with different tensors equal in value.
                self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
                self.param2 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                return (
                    self.layer3(self.layer2(self.layer1(x))) + self.param1 + self.param2
                )

        model = MyModule()
        x = torch.randn(3, 3)
        for training in (True, False):
            with self.subTest(training=training, precaptured=precaptured):
                model.train(training)
                source = (
                    torch.export.export(model, (x,), strict=False)
                    if precaptured
                    else model
                )
                program = torch_onnx.export(source, (x,), optimize=False, verbose=False)
                raw_names = set(program.model.graph.initializers)
                self.assertEqual(len(raw_names), len(list(model.parameters())))
                self.assertTrue(raw_names.issubset(model.state_dict()))
                self.assertTrue({"param1", "param2"}.issubset(raw_names))
                strategy = None if precaptured else "TorchExportNonStrictStrategy"
                torch_onnx.testing.assert_onnx_program(program, strategy=strategy)
                if not training:
                    program.optimize()
                    names = set(program.model.graph.initializers)
                    self.assertEqual(len(names), len(raw_names) - 1)
                    self.assertEqual(len(names & {"param1", "param2"}), 1)
                    torch_onnx.testing.assert_onnx_program(program, strategy=strategy)
                onnx.checker.check_model(program.model_proto, full_check=True)

    def test_deduplicate_initializers(self):
        self._test_deduplicate_initializers(precaptured=False)

    def test_deduplicate_initializers_torchscript(self):
        self._test_deduplicate_initializers(precaptured=True)

    def test_duplicated_output_node(self):
        class DuplicatedOutputNet(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, num_classes)

            def forward(self, input0, input1):
                out1 = self.fc1(input0)
                out2 = self.fc1(input1)
                return out1, out1, out2, out1, out2

        N, D_in, D_out = 64, 784, 10
        model = DuplicatedOutputNet(D_in, D_out)
        x = torch.randn(N, D_in)
        batch = torch.export.Dim("batch")
        output_names = [f"output-{index}" for index in range(5)]
        program = self.run_test(
            model,
            (x, x.clone()),
            input_names=["input0", "input1"],
            output_names=output_names,
            dynamic_shapes=({0: batch}, {0: batch}),
            additional_test_inputs=[(torch.randn(3, D_in), torch.randn(3, D_in))],
        )
        self.assertEqual(
            [value.name for value in program.model.graph.inputs],
            ["input0", "input1"],
        )
        self.assertEqual(
            [value.name for value in program.model.graph.outputs], output_names
        )

    def test_deduplicate_ignore_upsample_scale(self):
        # upsample scale is a constant, not a model parameter,
        # therefore should be ignored by shared weight deduplication.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.upsample_1 = torch.nn.Upsample(scale_factor=2)
                self.upsample_2 = torch.nn.Upsample(scale_factor=2)

            def forward(self, x):
                return self.upsample_1(x), self.upsample_2(x)

        program = self.run_test(Model(), torch.randn(1, 32, 224, 224))
        resize_nodes = [
            node for node in program.model.graph if node.op_type == "Resize"
        ]
        self.assertGreaterEqual(len(resize_nodes), 1)
        for node in resize_nodes:
            scales = node.inputs[2]
            sizes = node.inputs[3] if len(node.inputs) > 3 else None
            value = scales if scales is not None else sizes
            self.assertIsNotNone(value)
            self.assertIsNotNone(value.const_value)
            actual = tuple(value.const_value.numpy().tolist())
            expected = (
                ((2.0, 2.0), (1.0, 1.0, 2.0, 2.0))
                if scales is not None
                else ((448, 448), (1, 32, 448, 448))
            )
            self.assertIn(actual, expected)
