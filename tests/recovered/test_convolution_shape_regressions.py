# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Convolution/shape regressions from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
model bodies and dimensions; replace ScriptModule with modern capture.
"""

from __future__ import annotations

import functools

import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor


class TestRecoveredConvolutionShapes(RecoveredRuntimeHarness):
    def test_convolution_allow_tf32(self):
        class Module(torch.nn.Module):
            def __init__(self, allow_tf32):
                super().__init__()
                self.allow_tf32 = allow_tf32
                weight = torch.rand(32, 3, 3, 3)
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                if self.allow_tf32:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                        True,
                    )
                else:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                    )

        x = torch.randn(1, 3, 224, 224)
        self.run_test(Module(False), x, rtol=1e-3, atol=1e-6)
        self.run_test(Module(True), x, rtol=1e-3, atol=1e-6)

    def test_low_level_convolution_overloads(self):
        class Module(torch.nn.Module):
            def __init__(self, target, transposed):
                super().__init__()
                self.target = target
                self.transposed = transposed
                self.weight = torch.nn.Parameter(
                    torch.randn((4, 3, 3) if transposed else (6, 2, 3))
                )
                self.bias = torch.nn.Parameter(torch.randn(6))

            def forward(self, x):
                flags = (True, True, False)
                if self.target is torch.ops.aten._convolution.default:
                    flags = (*flags, False)
                return self.target(
                    x,
                    self.weight,
                    self.bias,
                    [2],
                    [1],
                    [1],
                    self.transposed,
                    [1 if self.transposed else 0],
                    2,
                    *flags,
                )

        for target in (
            torch.ops.aten._convolution.default,
            torch.ops.aten._convolution.deprecated,
        ):
            for transposed in (False, True):
                with self.subTest(target=target, transposed=transposed):
                    self.run_test(
                        Module(target, transposed),
                        torch.randn(2, 4, 9),
                        additional_test_inputs=[torch.randn(2, 4, 9)],
                    )

    def test_fuse_conv_bn1d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,), atol=1e-6)

    def test_fuse_conv_bn2d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False
                )
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

    def test_conv_tbc(self):
        from torch.nn.modules.utils import _single

        class ConvTBC(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)
                self.weight = torch.nn.Parameter(
                    Tensor(self.kernel_size[0], in_channels, out_channels)
                )
                self.bias = torch.nn.Parameter(Tensor(out_channels))
                self.reset_parameters()

            def reset_parameters(self):
                torch.nn.init.xavier_normal_(self.weight)
                torch.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                return torch.conv_tbc(
                    input.contiguous(), self.weight, self.bias, self.padding[0]
                )

            def forward(self, input):
                return self.conv_tbc(input)

        model = ConvTBC(3, 5, 5, padding=0)
        x = torch.randn(10, 7, 3, requires_grad=True)
        self.run_test(model, (x,), atol=1e-5)

    def test_reshape_constant_fold(self):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

    def test_conv(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_str_padding(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, padding="valid")
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=1, padding="valid", dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=1, padding="same"
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_shape_inference(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                return self.conv2(input) + 2

        x = torch.randn(20, 16, 50, 100)
        self.run_test(
            Model(),
            x,
            atol=10e-5,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[torch.randn(3, 16, 50, 100)],
        )

    def test_conv_transpose(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 10)
        x2 = torch.randn(20, 16, 10, 10)
        x3 = torch.randn(20, 16, 10, 10, 10)
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    def test_transpose_infer_shape(self):
        class TransposeModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(
            TransposeModule(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=[y],
        )

    def test_squeeze_without_no_op(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=1)

        self.run_test(Squeeze(), torch.randn(2, 1, 4))

    def test_squeeze_neg_without_no_op(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

        self.run_test(Squeeze(), torch.randn(2, 1, 4))

    def test_maxpool_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        self.run_test(
            Model(8, 16),
            torch.randn(2, 8, 64, 64),
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
            additional_test_inputs=[torch.randn(2, 8, 65, 67)],
        )

    def test_avgpool_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.AvgPool2d(
                    (2, 2), stride=2, ceil_mode=True, count_include_pad=False
                )
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        self.run_test(
            Model(8, 16),
            torch.randn(2, 8, 64, 64),
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
            additional_test_inputs=[torch.randn(2, 8, 65, 67)],
        )
