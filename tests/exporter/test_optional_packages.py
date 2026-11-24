"""Actual optional-package exports; run explicitly in the optional integration job."""

import importlib.util
import subprocess
import sys
from unittest import mock

import onnx
import pytest
import torch

import torch_onnx
from torch_onnx import _registration


def test_missing_optional_packages():
    original = importlib.util.find_spec
    with mock.patch(
        "importlib.util.find_spec",
        side_effect=lambda name: (
            None if name in ("torchao", "torchvision") else original(name)
        ),
    ):
        program = torch_onnx.export(
            torch.nn.ReLU().eval(), (torch.randn(2),), verbose=False
        )
    assert type(program) is torch.onnx.ONNXProgram


def test_broken_optional_import_is_actionable():
    with (
        mock.patch("importlib.util.find_spec", return_value=object()),
        mock.patch(
            "importlib.import_module", side_effect=ImportError("incompatible binary")
        ),
        pytest.raises(
            RuntimeError, match="compatible with the installed PyTorch"
        ) as error,
    ):
        _registration.ONNXRegistry.from_torchlib()
    assert isinstance(error.value.__cause__, ImportError)


@pytest.mark.integration
def test_lazy_registration_after_initial_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util
import sys
from unittest import mock
import torch_onnx
from torch_onnx._registration import ONNXRegistry
assert 'torchao' not in sys.modules
assert 'torchvision' not in sys.modules
original = importlib.util.find_spec
with mock.patch('importlib.util.find_spec', side_effect=lambda name: None if name in ('torchao', 'torchvision') else original(name)):
    ONNXRegistry.from_torchlib()
assert 'torchao' not in sys.modules
registry = ONNXRegistry.from_torchlib()
import torch
assert registry.get_decomps(torch.ops.torchao.quantize_affine.default)
assert registry.get_decomps(torch.ops.torchvision.ps_roi_align.default)
""",
        ],
        check=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dtype", "opset"),
    [
        (dtype, opset)
        for dtype in (torch.int8, torch.uint8, torch.int16, torch.int32)
        for opset in (18, 21, 29)
        if dtype != torch.int16 or opset >= 21
    ],
)
@pytest.mark.parametrize("block", [[2, 8], [1, 8], [1, 2], [2, 2]])
def test_torchao_affine_roundtrip(dtype, opset, block, tmp_path):
    import torchao  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, scale, zero):
            quantized = torch.ops.torchao.quantize_affine(
                x,
                block,
                scale,
                zero,
                dtype,
                -7 if dtype != torch.uint8 else 0,
                7,
            )
            return quantized, torch.ops.torchao.dequantize_affine(
                quantized,
                block,
                scale,
                zero,
                dtype,
                -7 if dtype != torch.uint8 else 0,
                7,
            )

    count = (2 // block[0]) * (8 // block[1])
    x = torch.tensor([[-100, -0.75, -0.65, -0.05, 0.05, 0.15, 0.25, 100]]).repeat(2, 1)
    args = (x, torch.full((count,), 0.1), torch.ones(count, dtype=torch.int64))
    program = torch_onnx.export(
        Model().eval(), args, opset_version=opset, verbose=False
    )
    assert type(program) is torch.onnx.ONNXProgram
    assert any(n.op_type == "DequantizeLinear" for n in program.model.graph)
    if dtype != torch.int32:
        assert any(n.op_type == "QuantizeLinear" for n in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    backend = (
        "reference"
        if (dtype in (torch.int16, torch.int32) and opset >= 19) or opset == 29
        else "onnxruntime"
    )
    torch_onnx.testing.assert_onnx_program(program, backend=backend)
    path = tmp_path / "affine.onnx"
    program.save(path)
    onnx.checker.check_model(onnx.load(path), full_check=True)


@pytest.mark.integration
@pytest.mark.parametrize("scale_shape", [[], [2, 1], [1, 2]])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("opset", [19, 21, 29])
def test_torchao_float8(scale_shape, dtype, opset, tmp_path):
    import torchao  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, scale):
            quantized = torch.ops.torchao.quantize_affine_float8_non_decomposed(
                x,
                scale,
                dtype,
            )
            return torch.ops.torchao.dequantize_affine_float8_non_decomposed(
                quantized, scale
            )

    program = torch_onnx.export(
        Model().eval(),
        (torch.randn(2, 8), torch.full(scale_shape, 0.5)),
        opset_version=opset,
        verbose=False,
    )
    assert {"QuantizeLinear", "DequantizeLinear"} <= {
        n.op_type for n in program.model.graph
    }
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, backend="reference")
    program.save(tmp_path / "float8.onnx")


@pytest.mark.integration
@pytest.mark.parametrize("sampling_ratio", [0, 2])
@pytest.mark.parametrize("geometry", ["ordinary", "small", "outside", "empty"])
def test_torchvision_ps_roi_align(sampling_ratio, geometry, tmp_path):
    import torchvision

    class Model(torch.nn.Module):
        def forward(self, x, boxes):
            return torchvision.ops.ps_roi_align(
                x,
                boxes,
                (2, 3),
                spatial_scale=0.5,
                sampling_ratio=sampling_ratio,
            )

    boxes = {
        "ordinary": [[0, 1.0, 2.0, 10.0, 12.0], [1, 0.0, 0.0, 8.0, 9.0]],
        "small": [[0, 1.0, 1.0, 1.2, 1.3]],
        "outside": [[0, -4.0, -4.0, 20.0, 22.0]],
        "empty": [],
    }
    args = (torch.randn(2, 12, 8, 8), torch.tensor(boxes[geometry]).reshape(-1, 5))
    program = torch_onnx.export(Model().eval(), args, verbose=False)
    assert any(n.op_type == "RoiAlign" for n in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)
    program.save(tmp_path / "ps_roi_align.onnx")


@pytest.mark.integration
def test_torchvision_ps_roi_align_channel_mapping():
    import torchvision  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, boxes):
            return torch.ops.torchvision.ps_roi_align(x, boxes, 1.0, 2, 2, 2)

    program = torch_onnx.export(
        Model().eval(),
        (torch.randn(1, 8, 8, 8), torch.tensor([[0, 0.0, 1.0, 6.0, 7.0]])),
        verbose=False,
    )
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.integration
def test_torchao_real_int8_weight_only_linear(tmp_path):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    model = torch.nn.Sequential(torch.nn.Linear(8, 4)).eval()
    quantize_(model, Int8WeightOnlyConfig())
    program = torch_onnx.export(
        model,
        (torch.randn(2, 8),),
        opset_version=21,
        verbose=False,
    )
    assert type(program) is torch.onnx.ONNXProgram
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)
    program.save(tmp_path / "weight_only.onnx")


@pytest.mark.integration
def test_torchao_dequantize_without_zero_point():
    import torchao  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, scale):
            return torch.ops.torchao.dequantize_affine(
                x,
                [1, 2],
                scale,
                None,
                torch.int8,
            )

    program = torch_onnx.export(
        Model().eval(),
        (torch.randint(-8, 8, (2, 8), dtype=torch.int8), torch.rand(2, 4)),
        verbose=False,
    )
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.integration
@pytest.mark.parametrize("unsupported", ["half_input", "floating_zero_point"])
def test_torchao_precision_diagnostics(unsupported):
    import torchao  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, scale, zero):
            return torch.ops.torchao.quantize_affine(
                x,
                [1, 8],
                scale,
                zero,
                torch.int8,
            )

    input_dtype = torch.float16 if unsupported == "half_input" else torch.float32
    zero_dtype = torch.float32 if unsupported == "floating_zero_point" else torch.int64
    message = (
        "float32 input" if unsupported == "half_input" else "integer zero_point storage"
    )
    with pytest.raises(torch_onnx.errors.ConversionError, match=message):
        torch_onnx.export(
            Model().eval(),
            (
                torch.randn(2, 8, dtype=input_dtype),
                torch.ones(2),
                torch.ones(2, dtype=zero_dtype),
            ),
            verbose=False,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dtype", "opset", "message"),
    [(torch.int16, 18, "opset 21"), (torch.float8_e4m3fn, 18, "opset 19")],
)
def test_torchao_lower_opset_diagnostics(dtype, opset, message):
    import torchao  # noqa: F401

    class Model(torch.nn.Module):
        def forward(self, x, scale, zero):
            if dtype == torch.float8_e4m3fn:
                return torch.ops.torchao.quantize_affine_float8_non_decomposed(
                    x, scale, dtype
                )
            return torch.ops.torchao.quantize_affine(x, [1, 8], scale, zero, dtype)

    with pytest.raises(torch_onnx.errors.ConversionError, match=message):
        torch_onnx.export(
            Model().eval(),
            (torch.randn(2, 8), torch.ones(2, 1), torch.zeros(2, 1)),
            opset_version=opset,
            verbose=False,
        )


@pytest.mark.integration
@pytest.mark.parametrize("operator", ["nms", "roi_align", "deform_conv2d"])
def test_torchvision_existing_native_operators(operator, tmp_path):
    import torchvision

    class Model(torch.nn.Module):
        def forward(self, x, auxiliary):
            if operator == "nms":
                return torchvision.ops.nms(x, auxiliary, 0.5)
            if operator == "roi_align":
                return torchvision.ops.roi_align(
                    x,
                    auxiliary,
                    (2, 2),
                    sampling_ratio=2,
                    aligned=True,
                )
            return torchvision.ops.deform_conv2d(
                x,
                auxiliary,
                torch.ones(2, 2, 3, 3),
            )

    if operator == "nms":
        args = (
            torch.tensor([[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]]),
            torch.tensor([0.9, 0.7]),
        )
    elif operator == "roi_align":
        args = (torch.randn(1, 2, 8, 8), torch.tensor([[0, 1.0, 1.0, 6.0, 6.0]]))
    else:
        args = (torch.randn(1, 2, 5, 5), torch.randn(1, 18, 3, 3) * 0.1)
    program = torch_onnx.export(Model().eval(), args, opset_version=19, verbose=False)
    native_name = {
        "nms": "NonMaxSuppression",
        "roi_align": "RoiAlign",
        "deform_conv2d": "DeformConv",
    }[operator]
    assert any(n.op_type == native_name for n in program.model.graph)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(
        program,
        backend="reference" if operator == "deform_conv2d" else "onnxruntime",
    )
    program.save(tmp_path / f"{operator}.onnx")
