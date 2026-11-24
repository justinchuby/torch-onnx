# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Interpolation cases from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original
inputs/sizes/scales and run the formerly disabled 1D/3D linear cases.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class _Interpolate(torch.nn.Module):
    def __init__(self, mode, *, size=None, scale=None, align_corners=None):
        super().__init__()
        self.mode = mode
        self.size = size
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            mode=self.mode,
            size=self.size,
            scale_factor=self.scale,
            align_corners=self.align_corners,
            recompute_scale_factor=False if self.scale is not None else None,
        )


class TestRecoveredInterpolation(RecoveredRuntimeHarness):
    def _interpolate_tests(self, is_upsample):
        for shape in [(1, 2, 6), (1, 2, 4, 6), (1, 2, 4, 4, 6)]:
            x = torch.randn(shape, requires_grad=True)
            rank = x.ndim - 2
            modes = ["nearest", {1: "linear", 2: "bilinear", 3: "trilinear"}[rank]]
            if rank == 2:
                modes.append("bicubic")
            for mode in modes:
                for align in [None] if mode == "nearest" else [False, True]:
                    for size in [24 if is_upsample else 2, [16, 32, 64][:rank]]:
                        with self.subTest(
                            shape=shape, mode=mode, align=align, size=size
                        ):
                            self.run_test(
                                _Interpolate(mode, size=size, align_corners=align),
                                x,
                                atol=1e-6,
                            )
                    for scale in [2.0 if is_upsample else 0.5, [2.3, 3.1, 4.6][:rank]]:
                        with self.subTest(
                            shape=shape, mode=mode, align=align, scale=scale
                        ):
                            self.run_test(
                                _Interpolate(mode, scale=scale, align_corners=align),
                                x,
                                atol=1e-6,
                            )

    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    def test_interpolate_downsample(self):
        self._interpolate_tests(False)

    def test_interpolate_half_pixel(self):
        # Test "half_pixel" versus "pytorch_half_pixel" when a spatial size is one.
        for shape in [(1, 2, 6), (1, 2, 4, 6), (1, 2, 4, 4, 6)]:
            x = torch.randn(shape, requires_grad=True)
            modes = [{3: "linear", 4: "bilinear", 5: "trilinear"}[x.ndim]]
            if x.ndim == 4:
                modes.append("bicubic")
            for mode in modes:
                for axis in range(x.ndim - 2):
                    size = list(x.shape[2:])
                    size[axis] = 1
                    self.run_test(_Interpolate(mode, size=size), x)

    def test_interpolate_function_substitution(self):
        class Submodule(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, mode="nearest", scale_factor=2.0
                )

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()

            def forward(self, input):
                return self.submodule(input)

        def interpolate(x):
            return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2.0)

        class FunctionalModule(torch.nn.Module):
            def forward(self, x):
                return interpolate(x)

        x = torch.randn(1, 2, 4, 4, 6)
        self.run_test(OuterModule(), x)
        self.run_test(FunctionalModule(), x)

    def test_interpolate_no_shape(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                x = torch.add(x, x)
                out1 = torch.nn.functional.interpolate(
                    x, mode="bilinear", size=(16, 16), align_corners=False
                )
                out2 = torch.nn.functional.interpolate(
                    x, mode="nearest", size=(int(y.size(0)), int(y.size(1)))
                )
                return out1, out2

        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        self.run_test(
            MyModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [2, 3], "y": [0, 1]},
            additional_test_inputs=[(torch.randn(1, 2, 5, 6), torch.randn(7, 9))],
        )
        self.run_test(MyModel(), (x, y), remained_onnx_input_idx=[0])
