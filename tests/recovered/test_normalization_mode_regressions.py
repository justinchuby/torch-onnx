# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Normalization train/eval regressions from PyTorch's deleted runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Original models and global
mode transitions are retained; None preserves explicit per-layer modes.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredNormalizationModes(RecoveredRuntimeHarness):
    def test_batchnorm_training(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 20, 20) * 2
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_training_mode_fix_layer(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.eval()

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        modes = [module.training for module in model_export.modules()]
        self.run_test(model_export, (x,), training=None, rtol=1e-3, atol=1e-5)
        self.assertEqual([module.training for module in model_export.modules()], modes)
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_eval_mode_train_layer(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.train()

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        model_export.eval()
        model_export.bn3.train()
        modes = [module.training for module in model_export.modules()]
        self.run_test(model_export, (x,), training=None, rtol=1e-3, atol=1e-5)
        self.assertEqual([module.training for module in model_export.modules()], modes)
        self.run_test(
            model_export,
            (x,),
            training=False,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training_mode_fix_layer(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)
                self.in3.eval()

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        modes = [module.training for module in model_export.modules()]
        self.run_test(model_export, (x,), training=None, rtol=1e-3, atol=1e-5)
        self.assertEqual([module.training for module in model_export.modules()], modes)
        self.run_test(
            model_export,
            (x,),
            training=True,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_eval_mode_train_layer(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(8, affine=True)
                self.cv1 = torch.nn.Conv2d(8, 8, 10)
                self.in2 = torch.nn.InstanceNorm2d(8, affine=False)
                self.cv2 = torch.nn.Conv2d(8, 8, 10)
                self.in3 = torch.nn.InstanceNorm2d(8, affine=True)
                self.in3.train()

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 8, 128, 128)
        model_export = MyModule()
        model_export.eval()
        model_export.in3.train()
        modes = [module.training for module in model_export.modules()]
        self.run_test(model_export, (x,), training=None, rtol=1e-3, atol=1e-5)
        self.assertEqual([module.training for module in model_export.modules()], modes)
        self.run_test(
            model_export,
            (x,),
            training=False,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=None,
            rtol=1e-3,
            atol=1e-5,
        )
