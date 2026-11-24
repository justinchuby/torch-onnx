# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recover the pretrained ResNet cases from test_models_quantized_onnxruntime.py.

Source: e2c2673d00500d268fcb1cc63fa7cc3173876505, removed by
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve pretrained=True's
V1 checkpoints, original image/preprocessing and independent quantized oracle.
Integer-storage Q/DQ adapters replace unsupported QTensor module capture.
"""

import operator
from pathlib import Path

import pytest
import torch
import torchvision
from PIL import Image
from test_quantized_storage_regressions import (
    _ConvertedLayer,
    _ConvertedOperation,
    _dequantize,
    _quantize,
)
from test_runtime_regressions import RecoveredRuntimeHarness
from torchvision.models import quantization


def _get_test_image_tensor():
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    with Image.open(
        Path(__file__).parent / "assets" / "grace_hopper_517x606.jpg"
    ) as image:
        return preprocess(image).unsqueeze(0)


# Due to precision error from quantization, check only that the top prediction matches.
class _TopPredictor(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        _, topk_id = torch.topk(x[0], 1)
        return topk_id


class _ConvertedAddRelu(torch.nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale, self.zero_point = scale, zero_point

    def forward(self, x, y):
        return _dequantize(
            _quantize(torch.relu(x + y), self.scale, self.zero_point),
            self.scale,
            self.zero_point,
        )


def _adapt_resnet(model):
    adapted = torch.fx.symbolic_trace(model)
    parameters: dict[torch.fx.Node, tuple[float, int] | None] = {}
    for node in adapted.graph.nodes:
        incoming = (
            parameters.get(node.args[0])
            if node.args and isinstance(node.args[0], torch.fx.Node)
            else None
        )
        outgoing = incoming
        if node.op == "call_module":
            assert isinstance(node.target, str)
            module = adapted.get_submodule(node.target)
            if isinstance(module, torch.ao.nn.quantized.Quantize):
                outgoing = (float(module.scale.item()), int(module.zero_point.item()))
                replacement = _ConvertedOperation(torch.nn.Identity(), *outgoing)
            elif isinstance(
                module, torch.ao.nn.quantized.Conv2d | torch.ao.nn.quantized.Linear
            ):
                assert incoming is not None, f"Missing input qparams for {node}"
                outgoing = (module.scale, module.zero_point)
                replacement = _ConvertedLayer(
                    torch.ao.nn.quantized.Quantize(*incoming, torch.quint8),
                    module,
                    linear=isinstance(module, torch.ao.nn.quantized.Linear),
                    relu=isinstance(
                        module,
                        torch.ao.nn.intrinsic.quantized.ConvReLU2d
                        | torch.ao.nn.intrinsic.quantized.LinearReLU,
                    ),
                )
            elif isinstance(module, torch.ao.nn.quantized.DeQuantize):
                replacement = torch.nn.Identity()
                outgoing = None
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                assert incoming is not None, f"Missing pooling qparams for {node}"
                replacement = _ConvertedOperation(module, *incoming)
            else:
                assert isinstance(
                    module, torch.nn.MaxPool2d | torch.nn.Identity
                ), f"Add an explicit Q/DQ adaptation for {node}: {type(module)}"
                replacement = module
            adapted.set_submodule(node.target, replacement)
        elif node.op == "call_function":
            if node.target == torch.ops.quantized.add_relu:
                outgoing = (
                    float(node.kwargs["scale"]),
                    int(node.kwargs["zero_point"]),
                )
                name = f"adapted_{node.name}"
                adapted.add_submodule(name, _ConvertedAddRelu(*outgoing))
                node.op, node.target, node.kwargs = "call_module", name, {}
            else:
                assert node.target in {
                    torch.flatten,
                    torch.topk,
                    operator.getitem,
                }, f"Add an explicit quantization contract for {node}"
        parameters[node] = outgoing
    adapted.graph.lint()
    adapted.recompile()
    return adapted


def _load_pretrained_model(name, weights):
    factory = getattr(quantization, name)
    if "fbgemm" in torch.backends.quantized.supported_engines:
        model = factory(weights=None, quantize=True, backend="fbgemm", progress=False)
    else:
        assert (
            "onednn" in torch.backends.quantized.supported_engines
        ), "These pretrained quantized ResNet fixtures require fbgemm or oneDNN."
        # Windows lacks fbgemm. Pack the same serialized weights with oneDNN,
        # retaining torchvision's fbgemm observer/weight configuration.
        torch.backends.quantized.engine = "onednn"
        model = factory(weights=None, quantize=False).eval()
        model.fuse_model()
        config = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        torch.ao.quantization.propagate_qconfig_(model, {"": config})
        torch.ao.quantization.prepare(model, inplace=True)
        model(torch.rand(1, 3, 299, 299))
        torch.ao.quantization.convert(model, inplace=True)
    model.load_state_dict(
        weights.get_state_dict(progress=False, check_hash=True, weights_only=True)
    )
    return model.eval()


@pytest.mark.parametrize(
    ("name", "weights"),
    [
        ("resnet18", quantization.ResNet18_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
        ("resnet50", quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
    ],
)
@torch.no_grad()
def test_pretrained_quantized_model(name, weights):
    previous_threads = torch.get_num_threads()
    previous_engine = torch.backends.quantized.engine
    torch.set_num_threads(1)
    program = None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            original = _TopPredictor(_load_pretrained_model(name, weights)).eval()
            adapted = _adapt_resnet(original)
            image = _get_test_image_tensor()
            flipped = image.flip(-1)
            for value in (image, flipped):
                torch.testing.assert_close(
                    adapted(value), original(value), rtol=0, atol=0
                )
            program = RecoveredRuntimeHarness().run_test(
                adapted,
                (image,),
                additional_test_inputs=[flipped],
                rtol=0,
                atol=0,
            )
            assert {node.op_type for node in program.model.graph} >= {
                "Conv",
                "QuantizeLinear",
                "DequantizeLinear",
                "TopK",
            }
            for value in (image, flipped):
                torch.testing.assert_close(
                    program(value)[0], original(value), rtol=0, atol=0
                )
    finally:
        if program is not None:
            program.release()
        torch.set_num_threads(previous_threads)
        torch.backends.quantized.engine = previous_engine
