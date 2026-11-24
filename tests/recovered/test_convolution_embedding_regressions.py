# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Convolution and weighted embedding cases from PyTorch's deleted suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original models,
comments and train/eval cases using the standalone runtime harness.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredConvolutionEmbedding(RecoveredRuntimeHarness):
    def test_fuse_conv_in_block(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=1,
                )
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                def transformed(value):
                    return self.bn(self.conv(value))

                return torch.cond(
                    torch.logical_not(x.sum() > -1),
                    transformed,
                    lambda value: value.clone(),
                    (x,),
                )

        model = Fuse().eval()
        original = torch.randn(2, 5, 9, requires_grad=True)
        alternates = [
            -torch.ones(2, 5, 9),
            torch.ones(2, 5, 9),
            -torch.ones(3, 5, 7),
            torch.ones(3, 5, 7),
        ]
        with torch.no_grad():
            for value in [original, *alternates]:
                expected = value if value.sum() > -1 else model.bn(model.conv(value))
                self.assertEqual(model(value), expected)
        program = self.run_test(
            model,
            original,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=alternates,
            rtol=1e-3,
            atol=1e-6,
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})

    def test_conv_bn(self):
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

        model_export = MyModule()
        x = torch.randn(10, 3, 128, 128)
        self.run_test(model_export, (x,), training=False)
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_multiple_conv_bn(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                self.conv2 = torch.nn.Conv2d(
                    64, 2, kernel_size=1, stride=1, padding=0, bias=False
                )
                self.conv3 = torch.nn.Conv2d(
                    2, 2, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(64)
                self.bn2 = torch.nn.BatchNorm2d(2)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        model_export = MyModule()
        x = torch.randn(2, 3, 224, 224)
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )
        self.run_test(model_export, (x,), training=False)

    def test_unfold_infer_shape(self):
        class UnfoldModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(32, 3, 64)
        self.run_test(UnfoldModule(), x)

    def test_embedding_bag_2d_per_sample_weights(self):
        class EmbeddingModel(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        embedding_matrix = torch.rand(10, 15)
        model = EmbeddingModel()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)

        x2 = torch.randint(7, (4, 3))
        w2 = torch.randn(4, 3)
        batch = torch.export.Dim("embedding_batch", min=2)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            input_names=["embed", "x", "w"],
            dynamic_shapes=({}, {0: batch}, {0: batch}),
            additional_test_inputs=[(embedding_matrix, x2, w2)],
        )
