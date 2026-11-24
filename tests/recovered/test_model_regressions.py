# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Model regressions recovered from test_models.py at e2c2673d00500d268fcb1cc63fa7cc3173876505.

Original model definitions, including initialization and implementation comments,
are retained in model_defs. No pretrained weights or network downloads are used.
"""

import onnx
import onnx_ir as ir
import pytest
import torch
from model_defs.dcgan import _netD, _netG, bsz, imgsz, nz, weights_init
from model_defs.emb_seq import EmbeddingNetwork1, EmbeddingNetwork2
from model_defs.mnist import MNIST
from model_defs.op_test import ConcatNet, DummyNet, FakeQuantNet, PermuteNet, PReluNet
from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet

import torch_onnx


@pytest.mark.parametrize(
    ("model_type", "shape"),
    [
        (DummyNet, (2, 3, 224, 224)),
        (PReluNet, (2, 3, 224, 224)),
        (PermuteNet, (2, 3, 10, 12)),
        (MNIST, (2, 1, 28, 28)),
        (FakeQuantNet, (2, 3, 224, 224)),
    ],
)
def test_original_small_models(model_type, shape, tmp_path):
    model = model_type().eval()
    x = torch.ones(shape)
    if model_type is FakeQuantNet:
        fake_quant = model.fake_quant
        scale, zero_point = float(fake_quant.scale), int(fake_quant.zero_point)
        quant_min, quant_max = fake_quant.quant_min, fake_quant.quant_max

        class InferenceFakeQuant(torch.nn.Module):
            def forward(self, value):
                return torch.fake_quantize_per_tensor_affine(
                    value, scale, zero_point, quant_min, quant_max
                )

        # Preserve frozen-observer inference without tensor-valued Python flags.
        original = model(x)
        model.fake_quant = InferenceFakeQuant()
        torch.testing.assert_close(model(x), original)
    program = torch_onnx.export(model, (x,), verbose=False)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, rtol=1e-2, atol=1e-6)
    path = tmp_path / "model.onnx"
    program.save(path, external_data=True)
    reloaded = torch.onnx.ONNXProgram(
        ir.serde.deserialize_model(onnx.load(path)), program.exported_program
    )
    torch_onnx.testing.assert_onnx_program(
        reloaded, strategy=None, rtol=1e-2, atol=1e-6
    )


def test_concat_model():
    program = torch_onnx.export(
        ConcatNet().eval(), ((torch.randn(2, 3), torch.randn(2, 3)),), verbose=False
    )
    torch_onnx.testing.assert_onnx_program(program)


@pytest.mark.parametrize("model_type", [EmbeddingNetwork1, EmbeddingNetwork2])
def test_original_embedding_sequence(model_type):
    model = model_type().eval()
    program = torch_onnx.export(
        model,
        (torch.randint(0, 10, (2, 3)),),
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        verbose=False,
    )
    for batch in (2, 5):
        torch_onnx.testing.assert_onnx_program(
            program, args=(torch.randint(0, 10, (batch, 3)),), kwargs={}
        )


def test_super_resolution():
    model = SuperResolutionNet(upscale_factor=3).eval()
    program = torch_onnx.export(model, (torch.ones(2, 1, 224, 224),), verbose=False)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize("generator", [False, True])
def test_dcgan_models(generator):
    model = (_netG(1) if generator else _netD(1)).eval()
    model.apply(weights_init)
    shape = (bsz, nz, 1, 1) if generator else (bsz, 3, imgsz, imgsz)
    program = torch_onnx.export(model, (torch.randn(shape),), verbose=False)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize("version", [1.0, 1.1])
def test_squeezenet_models(version):
    # AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
    # SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters.
    model = SqueezeNet(version=version, ceil_mode=version == 1.0).eval()
    program = torch_onnx.export(model, (torch.ones(2, 3, 224, 224),), verbose=False)
    onnx.checker.check_model(program.model_proto, full_check=True)
    torch_onnx.testing.assert_onnx_program(program, rtol=1e-2, atol=1e-5)
