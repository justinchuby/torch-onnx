# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Current vision-model adaptations of the removed test_models.py model cases.

Source: e2c2673d00500d268fcb1cc63fa7cc3173876505, removed by
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve original architectures,
input dimensions, constant-one inputs and inference numerical checks.
"""

import copy

import onnx
import onnxruntime as ort
import pytest
import torch
import torchvision.models as models
from model_defs.srresnet import SRResNet

import torch_onnx


class _RowTiledConv2d(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        assert self.stride == self.dilation == (1, 1) and self.groups == 1
        assert self.padding == tuple(size // 2 for size in self.kernel_size)
        height_padding, width_padding = self.padding
        padded = torch.nn.functional.pad(
            input, (width_padding, width_padding, height_padding, height_padding)
        )
        # Bound the float64 im2col workspace without changing convolution sums.
        rows = []
        for start in range(0, input.shape[-2], 8):
            tile = padded[:, :, start : start + 8 + self.kernel_size[0] - 1, :]
            rows.append(torch.nn.functional.conv2d(tile, self.weight, self.bias))
        return torch.cat(rows, dim=-2)


def test_row_tiled_oracle_convolution():
    original = torch.nn.Conv2d(4, 3, 9, padding=4).double()
    tiled = _RowTiledConv2d(4, 3, 9, padding=4, dtype=torch.float64)
    tiled.load_state_dict(original.state_dict())
    with torch.no_grad():
        x = torch.randn(2, 4, 19, 23, dtype=torch.float64)
        torch.testing.assert_close(tiled(x), original(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "shape", "kwargs", "rtol", "atol"),
    [
        ("resnet50", (2, 3, 224, 224), {}, 1e-2, 1e-6),
        ("densenet121", (2, 3, 224, 224), {}, 1e-2, 1e-5),
        ("inception_v3", (2, 3, 299, 299), {}, 1e-2, 1e-7),
        ("googlenet", (2, 3, 224, 224), {}, 1e-3, 1e-5),
        ("mnasnet1_0", (2, 3, 224, 224), {}, 1e-3, 1e-5),
        ("mobilenet_v2", (2, 3, 224, 224), {}, 1e-3, 1e-5),
        ("shufflenet_v2_x1_0", (2, 3, 224, 224), {}, 1e-3, 1e-5),
        ("alexnet", (2, 3, 224, 224), {}, 1e-2, 1e-7),
        # VGG 16-layer model (configuration "D"), without/with batch normalization.
        ("vgg16", (2, 3, 224, 224), {}, 1e-2, 1e-7),
        ("vgg16_bn", (2, 3, 224, 224), {}, 1e-2, 1e-7),
        # VGG 19-layer model (configuration "E"), without/with batch normalization.
        ("vgg19", (2, 3, 224, 224), {}, 1e-2, 1e-7),
        ("vgg19_bn", (2, 3, 224, 224), {}, 1e-2, 1e-7),
        (
            "srresnet",
            (1, 3, 224, 224),
            {"rescale_factor": 4, "n_filters": 64, "n_blocks": 8},
            1e-2,
            1e-7,
        ),
        (
            "segmentation.fcn_resnet101",
            (2, 3, 224, 224),
            {"weights_backbone": None},
            1e-3,
            1e-5,
        ),
        (
            "segmentation.deeplabv3_resnet101",
            (2, 3, 224, 224),
            {"weights_backbone": None},
            1e-3,
            1e-5,
        ),
        ("video.r3d_18", (1, 3, 4, 112, 112), {}, 1e-3, 1e-5),
        ("video.mc3_18", (1, 3, 4, 112, 112), {}, 1e-3, 1e-5),
        ("video.r2plus1d_18", (1, 3, 4, 112, 112), {}, 1e-3, 1e-5),
    ],
)
def test_original_vision_model(name, shape, kwargs, rtol, atol):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    program = None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            if name == "srresnet":
                model = SRResNet(**kwargs).eval()
            else:
                factory = models
                for component in name.split("."):
                    factory = getattr(factory, component)
                model = factory(weights=None, **kwargs).eval()
            x = torch.randn(shape) if name == "inception_v3" else torch.ones(shape)
            program = torch_onnx.export(model, (x,), verbose=False)
            onnx.checker.check_model(program.model_proto, full_check=True)

            def initialize_session(serialized):
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                return ort.InferenceSession(
                    serialized, options, providers=["CPUExecutionProvider"]
                )

            program.initialize_inference_session(initialize_session)
            if name in {
                "inception_v3",
                "segmentation.fcn_resnet101",
                "segmentation.deeplabv3_resnet101",
                "srresnet",
            }:
                # Near-zero cancellation makes float32 itself an unstable oracle
                # for these deep, untrained networks with large output ranges.
                with torch.no_grad():
                    reference = torch.utils._pytree.tree_leaves(model(x))[0]
                    oracle_model = copy.deepcopy(model).double()
                    if name == "srresnet":
                        conv = oracle_model.output_conv
                        tiled = _RowTiledConv2d(
                            conv.in_channels,
                            conv.out_channels,
                            conv.kernel_size,
                            padding=conv.padding,
                            dtype=torch.float64,
                        )
                        tiled.load_state_dict(conv.state_dict())
                        oracle_model.output_conv = tiled
                    oracle = torch.utils._pytree.tree_leaves(oracle_model(x.double()))[
                        0
                    ]
                    actual = program(x)[0]
                assert actual.shape == reference.shape
                assert actual.dtype == reference.dtype
                difference = actual.double() - oracle
                assert difference.norm() / oracle.norm() <= 1e-5
                assert difference.abs().max() / oracle.abs().max() <= 1e-5
                if name == "inception_v3":
                    # Preserve the source's explicit 1% unstable-element allowance.
                    mismatch = ~torch.isclose(actual, reference, rtol=rtol, atol=atol)
                    assert mismatch.double().mean() <= 0.01
            else:
                torch_onnx.testing.assert_onnx_program(program, rtol=rtol, atol=atol)
    finally:
        if program is not None:
            program.release()
        torch.set_num_threads(previous_threads)
