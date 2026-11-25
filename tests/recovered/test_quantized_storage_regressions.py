# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Quantization cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Decomposed operators
replace deprecated QTensor capture with explicit integer storage and Q/DQ.
Untouched eager quantized operations provide independent numerical oracles.
"""

import math

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # Registers quantized_decomposed.
from test_runtime_regressions import RecoveredRuntimeHarness
from torch.utils import _pytree

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


def _construct_tensor_for_quantization_test(shape, offset=None, max_val=None):
    """Generate deterministic weights/inputs to avoid quantization flakiness."""
    tensor = torch.arange(math.prod(shape), dtype=torch.float).view(shape)
    if offset is not None:
        tensor = tensor + offset
    if max_val is not None and tensor.abs().max() > max_val:
        tensor = tensor * max_val / tensor.abs().max()
    return tensor


class _ConvertedOperation(torch.nn.Module):
    def __init__(self, operation, scale, zero_point):
        super().__init__()
        self.operation = operation
        self.scale, self.zero_point = scale, zero_point

    def forward(self, x):
        value = _dequantize(
            _quantize(x, self.scale, self.zero_point), self.scale, self.zero_point
        )
        result = self.operation(value)
        return _dequantize(
            _quantize(result, self.scale, self.zero_point), self.scale, self.zero_point
        )


class _ConvertedLayer(torch.nn.Module):
    def __init__(self, quantizer, layer, *, linear, relu):
        super().__init__()
        weight = layer.weight()
        self.register_buffer("weight", weight.int_repr())
        self.register_buffer("scales", weight.q_per_channel_scales())
        self.register_buffer("zeros", weight.q_per_channel_zero_points())
        self.register_buffer("bias", layer.bias().detach())
        self.axis = weight.q_per_channel_axis()
        self.input_scale = quantizer.scale.item()
        self.input_zero = quantizer.zero_point.item()
        self.output_scale, self.output_zero = layer.scale, layer.zero_point
        self.linear, self.relu = linear, relu
        self.convolution = (
            {}
            if linear
            else dict(
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
            )
        )
        self.pointwise = (
            not linear
            and weight.shape[2:] == (1, 1)
            and layer.groups == 1
            and layer.stride == (1, 1)
            and layer.padding == (0, 0)
            and self.axis == 0
        )
        self.depthwise = (
            not linear
            and weight.shape[2:] == (3, 3)
            and layer.groups == weight.shape[0]
            and weight.shape[1] == 1
            and layer.dilation == (1, 1)
            and layer.padding == (1, 1)
            and self.axis == 0
        )
        if self.pointwise or self.depthwise:
            scale = self.scales.float() * self.input_scale
            self.register_buffer("requantization_scale", scale / self.output_scale)
            self.register_buffer("requantization_bias", self.bias / scale)

    def forward(self, x):
        if self.pointwise or self.depthwise:
            # Accumulate integer products before float32 requantization:
            # dequantizing first can move a result across a rounding tie.
            value = _quantize(x, self.input_scale, self.input_zero).double()
            value = value - self.input_zero
            if self.depthwise:
                height, width = value.shape[-2:]
                stride_h, stride_w = self.convolution["stride"]
                out_h = (height - 1) // stride_h + 1
                out_w = (width - 1) // stride_w + 1
                value = torch.nn.functional.pad(value, (1, 1, 1, 1))
                weight = self.weight.double() - self.zeros.reshape(-1, 1, 1, 1)
                result = value.new_zeros(
                    (value.shape[0], weight.shape[0], out_h, out_w)
                )
                for row in range(3):
                    for column in range(3):
                        result = result + (
                            value[
                                :,
                                :,
                                row : row + out_h * stride_h : stride_h,
                                column : column + out_w * stride_w : stride_w,
                            ]
                            * weight[:, 0, row, column].reshape(1, -1, 1, 1)
                        )
                result = result.permute(0, 2, 3, 1).float()
            else:
                weight = self.weight.double().flatten(1) - self.zeros.reshape(-1, 1)
                result = torch.matmul(value.permute(0, 2, 3, 1), weight.T).float()
            result = (result + self.requantization_bias) * self.requantization_scale
            result = torch.round(result) + self.output_zero
            result = result.clamp(self.output_zero if self.relu else 0, 255)
            return _dequantize(
                result.permute(0, 3, 1, 2).to(torch.uint8),
                self.output_scale,
                self.output_zero,
            )
        value = _dequantize(
            _quantize(x, self.input_scale, self.input_zero),
            self.input_scale,
            self.input_zero,
        )
        weight = _qd.dequantize_per_channel.default(
            self.weight, self.scales, self.zeros, self.axis, -128, 127, torch.int8
        )
        if self.linear:
            result = torch.nn.functional.linear(value, weight, self.bias)
        else:
            result = torch.nn.functional.conv2d(
                value, weight, self.bias, **self.convolution
            )
        if self.relu:
            result = torch.relu(result)
        return _dequantize(
            _quantize(result, self.output_scale, self.output_zero),
            self.output_scale,
            self.output_zero,
        )


class _StoredLayer(torch.nn.Module):
    def __init__(self, layer, input, *, linear, relu, transposed):
        super().__init__()
        weight = layer.weight()
        self.register_buffer("weight", weight.int_repr())
        self.register_buffer("bias", layer.bias().detach())
        self.input_scale, self.input_zero = input.q_scale(), input.q_zero_point()
        self.weight_scale, self.weight_zero = weight.q_scale(), weight.q_zero_point()
        self.output_scale, self.output_zero = layer.scale, layer.zero_point
        self.relu, self.transposed = relu, transposed
        self.dimension = weight.dim() - 2
        self.operation = (
            torch.nn.functional.linear
            if linear
            else getattr(
                torch.nn.functional,
                f"conv{'_transpose' if transposed else ''}{self.dimension}d",
            )
        )
        self.convolution = (
            {}
            if linear
            else dict(
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
            )
        )
        self.output_padding = layer.output_padding if transposed else ()
        if transposed:
            assert not any(
                layer.padding
            ), "These recovered transpose fixtures use zero padding."

    def forward(self, x):
        value = _dequantize(x, self.input_scale, self.input_zero)
        weight = _qd.dequantize_per_tensor.default(
            self.weight,
            self.weight_scale,
            self.weight_zero,
            -128,
            127,
            torch.int8,
        )
        if self.transposed:
            result = self.operation(value, weight, None, **self.convolution)
            # The original 3D output_padding exceeds float ConvTranspose's bound.
            # At zero padding, extra output positions contain only the bias.
            pads = [
                item
                for padding in reversed(self.output_padding)
                for item in (0, padding)
            ]
            result = torch.nn.functional.pad(result, pads)
            result = result + self.bias.reshape(1, -1, *([1] * self.dimension))
        else:
            result = self.operation(value, weight, self.bias, **self.convolution)
        if self.relu:
            result = torch.relu(result)
        return _quantize(result, self.output_scale, self.output_zero)


class TestRecoveredQuantizedStorage(RecoveredRuntimeHarness):
    def _compare(self, model, original, examples):
        stored = [_pytree.tree_map(_storage, args) for args in examples]
        program = self.run_test(
            model, stored[0], additional_test_inputs=stored[1:], rtol=0, atol=0
        )
        reloaded = self.reload_program(program)
        for args, stored_args in zip(examples, stored, strict=True):
            expected = _pytree.tree_map(_storage, original(*args))
            self.assertEqual(model(*stored_args), expected, rtol=0, atol=0)
            for candidate in (program, reloaded):
                self.assert_program_outputs(
                    candidate,
                    stored_args,
                    expected=expected,
                    rtol=0,
                    atol=0,
                )
        return program

    def _stored_layer(
        self, layer, input, *, linear=False, relu=False, transposed=False
    ):
        changed = torch.quantize_per_tensor(
            -input.dequantize(), input.q_scale(), input.q_zero_point(), torch.quint8
        )
        self._compare(
            _StoredLayer(layer, input, linear=linear, relu=relu, transposed=transposed),
            layer,
            [(input,), (changed,)],
        )

    def test_quantized_linear(self):
        model = torch.ao.nn.quantized.Linear(4, 8)
        # Set fixed weight to avoid flaky test.
        weight = torch.quantize_per_tensor(
            torch.arange(32, dtype=torch.float).view(8, 4), 0.5, 0, torch.qint8
        )
        # Set non-zero bias.
        bias = torch.arange(8, dtype=torch.float)
        model.set_weight_bias(weight, bias)
        # Set fixed input to avoid flaky test.
        input = torch.arange(16, dtype=torch.float).view(4, 4) - 8
        input_tensor = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self._stored_layer(model, input_tensor, linear=True)

    def _quantized_conv(self, dimension, *, relu=False, transposed=False):
        kernel = (2, 3, 4) if dimension == 3 else (3,) * dimension
        stride = (3, 1, 2) if dimension == 3 else (2,) * dimension
        shape = (3, 16, *((8,) * 3 if dimension == 3 else (32,) * dimension))
        namespace = torch.ao.nn.intrinsic.quantized if relu else torch.ao.nn.quantized
        name = (
            f"Conv{'Transpose' if transposed else 'ReLU' if relu else ''}{dimension}d"
        )
        constructor = getattr(namespace, name)
        if transposed:
            output_padding = (
                (1,) if dimension == 1 else (0, 1) if dimension == 2 else (0, 1, 2)
            )
            model = constructor(
                16, 33, kernel, output_padding=output_padding, stride=stride
            )
        else:
            model = constructor(16, 33, kernel, stride=stride)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        weight_shape = (16, 33, *kernel) if transposed else (33, 16, *kernel)
        q_weight = torch.quantize_per_tensor(
            torch.randn(weight_shape), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(shape)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self._stored_layer(model, q_input, relu=relu, transposed=transposed)

    def test_quantized_conv1d(self):
        self._quantized_conv(1)

    def test_quantized_conv2d(self):
        self._quantized_conv(2)

    def test_quantized_conv3d(self):
        self._quantized_conv(3)

    def test_quantized_conv1d_relu(self):
        self._quantized_conv(1, relu=True)

    def test_quantized_conv2d_relu(self):
        self._quantized_conv(2, relu=True)

    def test_quantized_conv3d_relu(self):
        self._quantized_conv(3, relu=True)

    def _require_fbgemm(self):
        self.assertIn(
            "fbgemm",
            torch.backends.quantized.supported_engines,
            "These original nonzero-output-padding reference fixtures require a "
            "PyTorch build with FBGEMM; run this backend suite on Linux x86_64.",
        )
        previous = torch.backends.quantized.engine
        self.addCleanup(setattr, torch.backends.quantized, "engine", previous)
        torch.backends.quantized.engine = "fbgemm"

    @pytest.mark.integration
    def test_quantized_conv_transpose1d(self):
        self._require_fbgemm()
        self._quantized_conv(1, transposed=True)

    @pytest.mark.integration
    def test_quantized_conv_transpose2d(self):
        self._require_fbgemm()
        self._quantized_conv(2, transposed=True)

    @pytest.mark.integration
    def test_quantized_conv_transpose3d(self):
        self._require_fbgemm()
        self._quantized_conv(3, transposed=True)

    def test_quantized_adaptive_avg_pool2d(self):
        class StorageModel(torch.nn.Module):
            def forward(self, x):
                # Averaging integer storage avoids dequantization half-step drift.
                return (
                    torch.nn.functional.adaptive_avg_pool2d(x.float(), (5, 7))
                    .round()
                    .to(torch.uint8)
                )

        model = torch.nn.AdaptiveAvgPool2d((5, 7))
        input = torch.randn(4, 3, 10, 14)
        q_input = torch.quantize_per_tensor(input, 0.2, 128, torch.quint8)
        raw = (torch.arange(1680) % 256).to(torch.uint8).reshape(4, 3, 10, 14)
        changed = torch._make_per_tensor_quantized_tensor(raw, 0.2, 128)
        self._compare(StorageModel(), model, [(q_input,), (changed,)])

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

    def _qat_layer(self, model, input, *, linear=False, relu=False):
        layer = model.linear if linear else model.conv
        self._compare(
            _ConvertedLayer(model.quant, layer, linear=linear, relu=relu),
            model,
            [(input,), (-input,), (input * 0.5,), (input.flip(-1),)],
        )

    def test_qat_linear_per_channel(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 3)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                return self.dequant(self.linear(self.quant(x)))

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((3, 4))
        )
        model.linear.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((4, 4), offset=-8)
        self._qat_layer(model, input, linear=True)

    def test_quantized_list_of_inputs_with_cat(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = torch.cat([x, x], 1)
                return self.dequant(x)

        class Concatenate(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, x], 1)

        model = TestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        x = torch.randn(2, 4, 6)
        self._compare(
            _ConvertedOperation(
                Concatenate(), model.quant.scale.item(), model.quant.zero_point.item()
            ),
            model,
            [(x,), (-x,), (x * 0.5,)],
        )

    def test_qat_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                return self.dequant(self.relu(self.quant(x)))

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        input = torch.randn(8, 4)
        self._compare(
            _ConvertedOperation(
                torch.nn.ReLU(), model.quant.scale.item(), model.quant.zero_point.item()
            ),
            model,
            [(input,), (-input,), (input * 0.5,)],
        )

    def test_qat_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                return self.dequant(self.conv(self.quant(x)))

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self._qat_layer(model, input)

    def _qat_conv2d_relu(self, fused):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                return self.dequant(self.relu(self.conv(self.quant(x))))

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        if fused:
            model = torch.ao.quantization.fuse_modules(model.eval(), [["conv", "relu"]])
        model = torch.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self._qat_layer(model, input, relu=True)

    def test_qat_conv2d_relu(self):
        self._qat_conv2d_relu(fused=False)

    def test_qat_conv2d_relu_fused(self):
        self._qat_conv2d_relu(fused=True)

    def test_qat_linear_relu_fused(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                return self.dequant(self.relu(self.linear(self.quant(x))))

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.fuse_modules(model.eval(), [["linear", "relu"]])
        model = torch.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4), max_val=2)
        )
        model.linear.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((3, 4), offset=-384, max_val=12)
        self._qat_layer(model, input, linear=True, relu=True)

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

        # Set fixed input to avoid flaky test.
        x = torch.arange(torch.Size(shape).numel(), dtype=torch.float).reshape(shape)
        self._compare(
            _ConvertedOperation(pool, scale, zero_point),
            original,
            [(x,), ((x - 48) / 2,), (x * 7,)],
        )

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
