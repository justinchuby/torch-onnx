# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Quantization cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Decomposed operators
replace deprecated QTensor capture with explicit integer storage and Q/DQ.
Untouched eager quantized operations provide independent numerical oracles.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx_ir as ir
import torch
import torch.ao.quantization.fx._decomposed  # Registers quantized_decomposed.
from test_runtime_regressions import RecoveredRuntimeHarness
from torch.utils import _pytree

import torch_onnx

_qd = torch.ops.quantized_decomposed

# NOTE: For quantization tests, choose scale and zero point carefully
# such that inputs and outputs do not always overflow/underflow.
# Otherwise test results could be inaccurate.


def _storage(value):
    return value.int_repr() if value.is_quantized else value


def _quantize(value, scale, zero_point):
    return _qd.quantize_per_tensor.default(
        value, scale, zero_point, 0, 255, torch.uint8
    )


def _dequantize(value, scale, zero_point):
    return _qd.dequantize_per_tensor.default(
        value, scale, zero_point, 0, 255, torch.uint8
    )


class TestRecoveredQuantizedStorage(RecoveredRuntimeHarness):
    def _compare(self, model, original, examples):
        stored = [_pytree.tree_map(_storage, args) for args in examples]
        program = self.run_test(
            model, stored[0], additional_test_inputs=stored[1:], rtol=0, atol=0
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.onnx"
            program.save(path, external_data=False)
            reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
            for args, stored_args in zip(examples, stored, strict=True):
                expected = _pytree.tree_map(_storage, original(*args))
                self.assertEqual(model(*stored_args), expected, rtol=0, atol=0)
                expected_outputs = _pytree.tree_leaves(expected)
                for candidate in (program, reloaded):
                    outputs = candidate(*stored_args)
                    self.assertEqual(len(outputs), len(expected_outputs))
                    for actual, expected_output in zip(
                        outputs, expected_outputs, strict=True
                    ):
                        torch.testing.assert_close(
                            actual, expected_output, rtol=0, atol=0
                        )
        return program

    def test_quantized_unary_ops(self):
        class Model(torch.nn.Module):
            def __init__(self, function_or_module):
                super().__init__()
                self.function_or_module = function_or_module

            def forward(self, x):
                return self.function_or_module(x)

        class StorageModel(torch.nn.Module):
            def __init__(self, function_or_module, scale, zero_point):
                super().__init__()
                self.function_or_module = function_or_module
                self.scale, self.zero_point = scale, zero_point

            def forward(self, x):
                value = _dequantize(x, 0.26, 128)
                result = self.function_or_module(value)
                return _quantize(result, self.scale, self.zero_point)

        group_norm = torch.nn.GroupNorm(2, 4)
        group_norm.weight = torch.nn.Parameter(torch.zeros(4))
        group_norm.bias = torch.nn.Parameter(torch.zeros(4))
        cases = [
            ("relu", torch.nn.ReLU(), torch.nn.ReLU()),
            ("leaky_relu", torch.nn.LeakyReLU(), torch.nn.LeakyReLU()),
            (
                "quantized_leaky_relu",
                torch.ao.nn.quantized.LeakyReLU(2.0, 1),
                torch.nn.LeakyReLU(),
            ),
            (
                "quantized_hardswish",
                torch.ao.nn.quantized.Hardswish(2.0, 1),
                torch.nn.Hardswish(),
            ),
            ("sigmoid", torch.nn.Sigmoid(), torch.nn.Sigmoid()),
            (
                "quantized_sigmoid",
                torch.ao.nn.quantized.Sigmoid(2.0, 1),
                torch.nn.Sigmoid(),
            ),
            ("hardsigmoid", torch.nn.Hardsigmoid(), torch.nn.Hardsigmoid()),
            ("tanh", torch.nn.Tanh(), torch.nn.Tanh()),
            ("hardtanh", torch.nn.Hardtanh(), torch.nn.Hardtanh()),
            ("transpose", lambda x: torch.transpose(x, 0, 1), None),
            ("expand", lambda x: x.expand(2, 4, 2, 3), None),
            ("view", lambda x: x.view(1, 4, 6), None),
            ("select", lambda x: x.select(1, 1), None),
            (
                "layer_norm",
                torch.ao.nn.quantized.LayerNorm(
                    [4, 2, 3],
                    torch.nn.Parameter(torch.ones([4, 2, 3])),
                    torch.nn.Parameter(torch.zeros([4, 2, 3])),
                    2.0,
                    1,
                ),
                torch.nn.LayerNorm([4, 2, 3]),
            ),
            (
                "instance_norm",
                torch.ao.nn.quantized.InstanceNorm1d(
                    2,
                    torch.nn.Parameter(torch.ones(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                # The quantized 1d wrapper normalizes this original 4D fixture.
                torch.nn.InstanceNorm2d(4, affine=True),
            ),
            (
                "group_norm",
                torch.ao.nn.quantized.GroupNorm(
                    2,
                    4,
                    torch.nn.Parameter(torch.zeros(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                group_norm,
            ),
            ("as_strided", lambda x: torch.as_strided(x, (2, 2), (1, 2)), None),
        ]
        for name, function, floating in cases:
            with self.subTest(name=name):
                x = torch.quantize_per_tensor(
                    torch.randn(1, 4, 2, 3), 0.26, 128, torch.quint8
                )
                original = Model(function)
                expected = original(x)
                model = StorageModel(
                    function if floating is None else floating,
                    expected.q_scale(),
                    expected.q_zero_point(),
                )
                examples = [(x,)]
                for start in range(0, 256, 24):
                    raw = (
                        (torch.arange(24, dtype=torch.int64) + start)
                        .clamp(max=255)
                        .to(torch.uint8)
                        .reshape(1, 4, 2, 3)
                    )
                    quantized = torch._make_per_tensor_quantized_tensor(raw, 0.26, 128)
                    result = original(quantized)
                    self.assertEqual(result.q_scale(), expected.q_scale())
                    self.assertEqual(result.q_zero_point(), expected.q_zero_point())
                    examples.append((quantized,))
                self._compare(model, original, examples)

    def test_quantize_per_tensor(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.quantize_per_tensor(x, 0.2, 0, torch.qint8),
                    torch.quantize_per_tensor(x, 0.2, 128, torch.quint8),
                )

        class StorageModel(torch.nn.Module):
            def forward(self, x):
                return (
                    _qd.quantize_per_tensor.default(x, 0.2, 0, -128, 127, torch.int8),
                    _quantize(x, 0.2, 128),
                )

        x = torch.randn(4, 6)
        program = self._compare(
            StorageModel(),
            Module(),
            [(x,), (torch.linspace(-100, 100, 24).reshape(4, 6),)],
        )
        self.assertIn("QuantizeLinear", {node.op_type for node in program.model.graph})

    def test_dequantize(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.dequantize(x)

        class StorageModel(torch.nn.Module):
            def forward(self, x):
                return _qd.dequantize_per_tensor.default(
                    x, 0.2, 0, -128, 127, torch.int8
                )

        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 0, torch.qint8)
        changed = torch.quantize_per_tensor(
            torch.linspace(-100, 100, 12).reshape(3, 4), 0.2, 0, torch.qint8
        )
        program = self._compare(StorageModel(), Module(), [(x,), (changed,)])
        self.assertIn(
            "DequantizeLinear", {node.op_type for node in program.model.graph}
        )

    def test_quantized_flatten(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input)

        x = torch.quantize_per_tensor(torch.randn(1, 2, 3, 4), 1, 0, torch.quint8)
        changed = torch.quantize_per_tensor(
            torch.arange(24, dtype=torch.float).reshape(1, 2, 3, 4),
            1,
            0,
            torch.quint8,
        )
        self._compare(FlattenModel(), FlattenModel(), [(x,), (changed,)])

    def test_quantized_cat_when_concatinating_the_same_tensor(self):
        class QuantizedSelfConcatenationModel(torch.nn.Module):
            def forward(self, x):
                return torch.ao.nn.quantized.QFunctional().cat((x, x), dim=1)

        class StorageModel(torch.nn.Module):
            def forward(self, x):
                value = _dequantize(x, 0.26, 128)
                return _quantize(torch.cat((value, value), dim=1), 1.0, 0)

        x = torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8)
        changed = torch.quantize_per_tensor(
            torch.tensor([[-2.0, 0.0, 0.7], [3.0, 12.0, 40.0]]),
            0.26,
            128,
            torch.quint8,
        )
        self._compare(
            StorageModel(), QuantizedSelfConcatenationModel(), [(x,), (changed,)]
        )

    def test_quantized_cat(self):
        class QuantizedConcatenationModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.ao.nn.quantized.QFunctional().cat((x, y), dim=0)

        class StorageModel(torch.nn.Module):
            def __init__(self, scale, zero_point):
                super().__init__()
                self.scale, self.zero_point = scale, zero_point

            def forward(self, x, y):
                a = _dequantize(x, 0.26, 128)
                b = _dequantize(y, self.scale, self.zero_point)
                return _quantize(torch.cat((a, b), dim=0), 1.0, 0)

        for shape, scale, zero_point in (
            ((1, 3), 0.26, 128),
            ((2, 3), 42.0, 1),
            ((2, 3), 0.26, 63),
            ((2, 3), 0.1, 63),
        ):
            with self.subTest(shape=shape, scale=scale, zero_point=zero_point):
                x = torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8)
                values = torch.zeros(shape) if shape[0] == 1 else torch.ones(shape)
                y = torch.quantize_per_tensor(values, scale, zero_point, torch.quint8)
                changed_x = torch.quantize_per_tensor(
                    torch.arange(6, dtype=torch.float).reshape(2, 3) - 2,
                    0.26,
                    128,
                    torch.quint8,
                )
                changed_y = torch.quantize_per_tensor(
                    torch.arange(torch.Size(shape).numel(), dtype=torch.float).reshape(
                        shape
                    )
                    * 17
                    - 9,
                    scale,
                    zero_point,
                    torch.quint8,
                )
                self._compare(
                    StorageModel(scale, zero_point),
                    QuantizedConcatenationModel(),
                    [(x, y), (changed_x, changed_y)],
                )

    def _arithmetic(self, qfunctional):
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x, y):
                if qfunctional:
                    o = torch.ao.nn.quantized.QFunctional().add(x, y)
                    return torch.ao.nn.quantized.QFunctional().mul(o, x)
                o = torch.ops.quantized.add(x, y, 0.4, 100)
                return torch.ops.quantized.mul(o, x, 0.4, 100)

        class StorageModel(torch.nn.Module):
            def forward(self, x, y):
                a = _dequantize(x, 0.2, 128)
                b = _dequantize(y, 0.2, 128)
                scale, zero_point = (1.0, 0) if qfunctional else (0.4, 100)
                o = _quantize(a + b, scale, zero_point)
                value = _dequantize(o, scale, zero_point)
                return _quantize(value * a, scale, zero_point)

        examples = [
            (
                torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8),
                torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8),
            )
        ]
        generator = torch.Generator().manual_seed(2026)
        for _ in range(100):
            examples.append(
                tuple(
                    torch.quantize_per_tensor(
                        torch.randn(3, 4, generator=generator),
                        0.2,
                        128,
                        torch.quint8,
                    )
                    for _ in range(2)
                )
            )
        self._compare(StorageModel(), ArithmeticModel(), examples)

    def test_quantized_arithmetic_qfunctional(self):
        self._arithmetic(qfunctional=True)

    def test_quantized_arithmetic(self):
        self._arithmetic(qfunctional=False)

    def _qat_pool(self, pool, shape):
        original = torch.nn.Sequential(
            torch.ao.quantization.QuantStub(),
            pool,
            torch.ao.quantization.DeQuantStub(),
        )
        original.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        original = torch.ao.quantization.prepare_qat(original.train())
        original = torch.ao.quantization.convert(original)
        scale = original[0].scale.item()
        zero_point = original[0].zero_point.item()

        class StorageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = pool

            def forward(self, x):
                value = _dequantize(_quantize(x, scale, zero_point), scale, zero_point)
                result = self.pool(value)
                return _dequantize(
                    _quantize(result, scale, zero_point), scale, zero_point
                )

        # Set fixed input to avoid flaky test.
        x = torch.arange(torch.Size(shape).numel(), dtype=torch.float).reshape(shape)
        self._compare(StorageModel(), original, [(x,), ((x - 48) / 2,), (x * 7,)])

    def test_qat_maxpool2d(self):
        self._qat_pool(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1), (4, 4, 3, 2)
        )

    def test_qat_avg_pool2d(self):
        self._qat_pool(
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1), (4, 4, 3, 2)
        )

    def test_qat_upsample_nearest2d(self):
        self._qat_pool(torch.nn.UpsamplingNearest2d(scale_factor=1.5), (4, 3, 2, 2))
