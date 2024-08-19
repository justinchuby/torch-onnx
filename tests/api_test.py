# Owner(s): ["module: onnx"]
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import os

import torch
import torch_onnx
from torch.testing._internal import common_utils


class SampleModel(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y.relu()
        return (y, z)


class SampleModelTwoInputs(torch.nn.Module):
    def forward(self, x, b):
        y = x + b
        z = y.relu()
        return (y, z)


class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class TestExportAPIDynamo(common_utils.TestCase):
    """Tests for the ONNX exporter API when dynamo=True."""

    def test_args_normalization_with_no_kwargs(self):
        onnx_program = torch_onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        onnx_program = torch_onnx.export_compat(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        onnx_program = torch_onnx.export_compat(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        onnx_program = torch_onnx.export_compat(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch_onnx.export_compat(SampleModel(), (torch.randn(1, 1, 2),), path)
            self.assertTrue(os.path.exists(path))

    def test_export_supports_script_module(self):
        class ScriptModule(torch.nn.Module):
            def forward(self, x):
                return x

        onnx_program = torch_onnx.export(
            torch.jit.script(ScriptModule()), (torch.randn(1, 1, 2),)
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_dynamic_shapes_with_fully_specified_axes(self):
        exported_program = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("customx_dim_0"),
                    1: torch.export.Dim("customx_dim_1"),
                    2: torch.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )

        onnx_program = torch_onnx.export(exported_program)
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)

    def test_partial_dynamic_shapes(self):
        onnx_program = torch_onnx.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": None,
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )
        assert onnx_program
        torch_onnx.testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()