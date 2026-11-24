# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Model regressions recovered from test_models.py at e2c2673d00500d268fcb1cc63fa7cc3173876505.

Original model definitions, including initialization and implementation comments,
are retained in model_defs. No pretrained weights or network downloads are used.
"""

import pytest
import torch
from _assertions import export_and_assert
from model_defs.dcgan import _netD, _netG, bsz, imgsz, nz, weights_init
from model_defs.emb_seq import EmbeddingNetwork1, EmbeddingNetwork2
from model_defs.mnist import MNIST
from model_defs.op_test import ConcatNet, DummyNet, FakeQuantNet, PermuteNet, PReluNet
from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet


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
def test_original_small_models(model_type, shape):
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
    export_and_assert(
        model,
        (x,),
        rtol=1e-2,
        atol=1e-6,
        check_serialization=True,
        external_data=True,
    )


def test_concat_model():
    export_and_assert(ConcatNet().eval(), ((torch.randn(2, 3), torch.randn(2, 3)),))


@pytest.mark.parametrize("model_type", [EmbeddingNetwork1, EmbeddingNetwork2])
def test_original_embedding_sequence(model_type):
    model = model_type().eval()
    export_and_assert(
        model,
        (torch.randint(0, 10, (2, 3)),),
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        additional_test_inputs=[
            (torch.randint(0, 10, (batch, 3)),) for batch in (2, 5)
        ],
    )


def test_super_resolution():
    model = SuperResolutionNet(upscale_factor=3).eval()
    export_and_assert(model, (torch.ones(2, 1, 224, 224),), rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize("generator", [False, True])
def test_dcgan_models(generator):
    model = (_netG(1) if generator else _netD(1)).eval()
    model.apply(weights_init)
    shape = (bsz, nz, 1, 1) if generator else (bsz, 3, imgsz, imgsz)
    export_and_assert(model, (torch.randn(shape),), rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize("version", [1.0, 1.1])
def test_squeezenet_models(version):
    # AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
    # SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters.
    model = SqueezeNet(version=version, ceil_mode=version == 1.0).eval()
    export_and_assert(model, (torch.ones(2, 3, 224, 224),), rtol=1e-2, atol=1e-5)
