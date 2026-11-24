# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Parameterized regressions recovered from PyTorch's removed runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Original bodies, parameters
and comments are retained; legacy frontend/opset selectors are removed.
"""

import itertools

import numpy as np
import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TestRecoveredParameterized(RecoveredRuntimeHarness):
    opset_version = 20

    class Atleast1d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_1d((t, w, x, y, z))

    class Atleast2d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_2d((t, w, x, y, z))

    class Atleast3d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_3d((t, w, x, y, z))

    class Atleast1dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_1d(x)

    class Atleast2dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_2d(x)

    class Atleast3dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_3d(x)

    @common_utils.parametrize("module_class", (Atleast1d, Atleast2d, Atleast3d))
    def test_atleast_nd_list_input(self, module_class: torch.nn.Module):
        inputs = (
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        )
        self.run_test(module_class(), inputs)

    @common_utils.parametrize(
        "module_class", (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor)
    )
    @common_utils.parametrize(
        "inputs",
        [
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        ],
    )
    def test_atleast_nd_single_tensor_input(
        self, module_class: torch.nn.Module, inputs: torch.Tensor
    ):
        self.run_test(module_class(), inputs)

    @common_utils.parametrize(
        "padding",
        (0, 1),
    )
    @common_utils.parametrize(
        "count_include_pad",
        (True, False),
    )
    def test_avgpool_2d(self, padding, count_include_pad):
        model = torch.nn.AvgPool2d(
            3,
            3,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    @common_utils.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64]
    )
    def test_bitshift_signed(self, dtype):
        class RightShiftModel(torch.nn.Module):
            def forward(self, input):
                return input >> 1, input >> 3

        input = torch.tensor([-128, -7, -4, -1, 0, 1, 3, 127], dtype=dtype)
        self.run_test(RightShiftModel(), input)

    @common_utils.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64]
    )
    def test_bitwise_right_shift_signed(self, dtype):
        class BitwiseRightShiftModel(torch.nn.Module):
            def forward(self, input):
                return torch.bitwise_right_shift(input, 2)

        input = torch.tensor([-128, -7, -4, -1, 0, 1, 3, 127], dtype=dtype)
        self.run_test(BitwiseRightShiftModel(), input)

    @common_utils.parametrize(
        "src, indices",
        [
            common_utils.subtest(
                [torch.ones((1, 5)), torch.tensor([[0, 1, 2, 0, 0]])],
                name="src_indices_dynamic_combination1",
            ),
            common_utils.subtest(
                [torch.ones((2, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination2",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination3",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1]])],
                name="src_indices_dynamic_combination4",
            ),
        ],
    )
    def test_scatter_add_dynamic_index(self, src, indices):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, src):
                return input.scatter_add(0, indices, src)

        input = torch.zeros(3, 5, dtype=src.dtype)
        self.run_test(
            ScatterModel(),
            input_args=(input, indices, src),
            input_names=["input", "indices", "src"],
            dynamic_axes={"indices": {0: "a", 1: "b"}, "src": {0: "c", 1: "d"}},
        )

    @common_utils.parametrize(
        "pad",
        [
            common_utils.subtest([2, 4], name="scalar_list"),
            common_utils.subtest(
                [
                    torch.tensor(2, dtype=torch.int64),
                    torch.tensor(4, dtype=torch.int64),
                ],
                name="scalar_tensor_list",
            ),
        ],
    )
    def test_pad_types(self, pad):
        # Test for different pad integer types
        class Pad(torch.nn.Module):
            def forward(self, x, pad: list[int]):
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        self.run_test(Pad(), (x, pad))

    class AffineGridModule(torch.nn.Module):
        def __init__(self, align_corners) -> None:
            super().__init__()
            self.align_corners = align_corners

        def forward(self, theta, size):
            return torch.nn.functional.affine_grid(theta, size, self.align_corners)

    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                10,
                np.array([0.3, -0.5]),
                np.array([1.5, 0.5]),
            ),
            (
                60,
                np.array([-0.5, -0.5]),
                np.array([3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2], [2, 10, 2, 3]),
    )
    def test_affine_grid_2d(self, align_corners, theta_params, size):
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian = (angle / 180.0) * np.pi
            theta = np.append(
                theta,
                [
                    np.cos(angle_radian) * scale[0],
                    -np.sin(angle_radian),
                    translation[0],
                    np.sin(angle_radian),
                    np.cos(angle_radian) * scale[1],
                    translation[1],
                ],
            )
        theta = theta.reshape(size[0], 2, 3)
        theta = torch.Tensor(theta)
        self.run_test(
            TestRecoveredParameterized.AffineGridModule(align_corners), (theta, size)
        )

    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                [10, 20],
                np.array([0.3, -0.5, 1.8]),
                np.array([1.5, 2.0, 0.5]),
            ),
            (
                [60, -30],
                np.array([-0.5, -0.5, 0.3]),
                np.array([0.3, 3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2, 2], [2, 10, 2, 2, 3]),
    )
    def test_affine_grid_3d(self, align_corners, theta_params, size):
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian_x = (angle[0] / 180.0) * np.pi
            angle_radian_y = (angle[1] / 180.0) * np.pi
            rot_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_radian_x), -np.sin(angle_radian_x)],
                    [0, np.sin(angle_radian_x), np.cos(angle_radian_x)],
                ]
            )
            rot_matrix_y = np.array(
                [
                    [np.cos(angle_radian_y), 0, np.sin(angle_radian_y)],
                    [0, 1, 0],
                    [-np.sin(angle_radian_y), 0, np.cos(angle_radian_y)],
                ]
            )
            rot_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
            rot_matrix = rot_matrix * scale.reshape(3, 1)
            rot_matrix = np.append(rot_matrix, np.reshape(translation, (3, 1)), axis=1)
            theta = np.append(theta, rot_matrix.flatten())

        theta = theta.reshape(size[0], 3, 4)
        theta = torch.Tensor(theta)
        self.run_test(
            TestRecoveredParameterized.AffineGridModule(align_corners), (theta, size)
        )

    @common_utils.parametrize(
        "mode",
        ("bilinear", "nearest", "bicubic"),
    )
    @common_utils.parametrize(
        "padding_mode",
        ("zeros", "border", "reflection"),
    )
    @common_utils.parametrize(
        "align_corners",
        (True, False),
        name_fn=lambda align_corners: str(align_corners),
    )
    def test_grid_sample(self, mode, padding_mode, align_corners):
        n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4

        atol_rtol = {}
        if (mode, padding_mode) == ("bicubic", "border"):
            if align_corners:
                atol_rtol.update({"atol": 0.3, "rtol": 0.4})
            else:
                atol_rtol.update({"atol": 0.02, "rtol": 0.02})
        input, grid = torch.randn(n, c, h_in, w_in), torch.randn(n, h_out, w_out, 2)

        class GridSampleModule(torch.nn.Module):
            def __init__(self, mode, padding_mode, align_corners) -> None:
                super().__init__()
                self.mode, self.padding_mode, self.align_corners = (
                    mode,
                    padding_mode,
                    align_corners,
                )

            def forward(self, input, grid):
                return torch.nn.functional.grid_sample(
                    input, grid, self.mode, self.padding_mode, self.align_corners
                )

        self.run_test(
            GridSampleModule(mode, padding_mode, align_corners),
            (input, grid),
            **atol_rtol,
        )

        # ONNX Opset 16 GridSample with 5D volumetric input is not supported.
        volumetric_input_tensor = torch.randn(n, c, d_in, h_in, w_in)
        volumetric_grid_tensor = torch.randn(n, d_out, h_out, w_out, 3)
        for mode, padding_mode, align_corners in itertools.product(
            (
                "bilinear",
                "nearest",
            ),  # PyTorch grid_sample "bicubic" mode does not support 5D volumetric input.
            (
                "zeros",
                "border",
                "reflection",
            ),
            (
                True,
                False,
            ),
        ):
            if self.opset_version < 20:
                with self.assertRaises(
                    torch.onnx.OnnxExporterError,
                ):
                    self.run_test(
                        GridSampleModule(mode, padding_mode, align_corners),
                        (volumetric_input_tensor, volumetric_grid_tensor),
                        **atol_rtol,
                    )
            else:
                self.run_test(
                    GridSampleModule(mode, padding_mode, align_corners),
                    (volumetric_input_tensor, volumetric_grid_tensor),
                    **atol_rtol,
                )
