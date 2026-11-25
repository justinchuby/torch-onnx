# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recover pretrained network cases from test_models_quantized_onnxruntime.py.

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


class _ConvertedCat(torch.nn.Module):
    def __init__(self, dim, scale, zero_point):
        super().__init__()
        self.dim, self.scale, self.zero_point = dim, scale, zero_point

    def forward(self, values):
        return _dequantize(
            _quantize(torch.cat(values, dim=self.dim), self.scale, self.zero_point),
            self.scale,
            self.zero_point,
        )


class _ConvertedMean(torch.nn.Module):
    def __init__(self, dims, scale, zero_point):
        super().__init__()
        self.dims, self.scale, self.zero_point = dims, scale, zero_point

    def forward(self, value):
        return _dequantize(
            _quantize(value.mean(self.dims), self.scale, self.zero_point),
            self.scale,
            self.zero_point,
        )


def _adapt_model(model):
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
                    module, torch.nn.MaxPool2d | torch.nn.Identity | torch.nn.Dropout
                ), f"Add an explicit Q/DQ adaptation for {node}: {type(module)}"
                assert not module.training, f"Expected an evaluation module: {node}"
                replacement = (
                    torch.nn.Identity()
                    if isinstance(module, torch.nn.Dropout)
                    else module
                )
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
            elif node.target == torch.ops.quantized.cat:
                assert all(
                    parameters[value] is not None for value in node.args[0]
                ), f"Missing concatenation input qparams for {node}"
                outgoing = (
                    float(node.kwargs["scale"]),
                    int(node.kwargs["zero_point"]),
                )
                name = f"adapted_{node.name}"
                adapted.add_submodule(
                    name, _ConvertedCat(node.kwargs["dim"], *outgoing)
                )
                node.op, node.target, node.kwargs = "call_module", name, {}
            elif node.target in {
                torch.nn.functional.avg_pool2d,
                torch.nn.functional.adaptive_avg_pool2d,
            }:
                assert incoming is not None, f"Missing pooling qparams for {node}"
                pool_type = (
                    torch.nn.AvgPool2d
                    if node.target == torch.nn.functional.avg_pool2d
                    else torch.nn.AdaptiveAvgPool2d
                )
                name = f"adapted_{node.name}"
                adapted.add_submodule(
                    name,
                    _ConvertedOperation(
                        pool_type(*node.args[1:], **node.kwargs), *incoming
                    ),
                )
                node.op, node.target, node.args, node.kwargs = (
                    "call_module",
                    name,
                    (node.args[0],),
                    {},
                )
            elif node.target in {
                torch.cat,
                torch.unsqueeze,
                operator.add,
                operator.mul,
                operator.floordiv,
            }:
                assert (
                    incoming is None
                ), f"Expected an unquantized scalar/input-transform operation: {node}"
            else:
                assert node.target in {
                    torch.flatten,
                    torch.topk,
                    operator.getitem,
                    torch.nn.functional.max_pool2d,
                    torch.chunk,
                    torch.transpose,
                }, f"Add an explicit quantization contract for {node}"
        elif node.op == "call_method":
            if node.target == "size":
                outgoing = None
            elif node.target == "mean":
                assert incoming is not None, f"Missing mean qparams for {node}"
                name = f"adapted_{node.name}"
                adapted.add_submodule(name, _ConvertedMean(node.args[1], *incoming))
                node.op, node.target, node.args = (
                    "call_module",
                    name,
                    (node.args[0],),
                )
            else:
                assert node.target in {
                    "view",
                    "contiguous",
                    "chunk",
                }, f"Add an explicit quantization contract for {node}"
        parameters[node] = outgoing
    adapted.graph.lint()
    adapted.recompile()
    return adapted


def _quantized_reference_diagnostics(original, adapted, value):
    class Recorder(torch.fx.Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.values = {}

        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                self.values[node.name] = result.detach()
            return result

    reference = Recorder(torch.fx.symbolic_trace(original))
    actual = Recorder(adapted)
    reference.run(value)
    actual.run(value)
    differences = [
        f"Quantized reference backend: {torch.backends.quantized.engine}.",
        "First divergent tensor nodes (before ONNX export):",
    ]
    for name, reference_value in reference.values.items():
        result = actual.values.get(name)
        if result is None or result.shape != reference_value.shape:
            continue
        qparams = (
            f" qparams=({reference_value.q_scale()}, {reference_value.q_zero_point()})"
            if reference_value.is_quantized
            else ""
        )
        expected = (
            reference_value.dequantize()
            if reference_value.is_quantized
            else reference_value
        )
        if not torch.equal(result, expected):
            differences.append(
                f"{name}: max_abs={(result - expected).abs().max().item()}"
                f" mismatched={(result != expected).sum().item()}/{result.numel()}"
                f"{qparams}"
            )
            index = tuple((result != expected).nonzero()[0].tolist())
            differences.append(
                f"{name}: first_index={index}"
                f" reference={expected[index].item()} adapted={result[index].item()}"
            )
            node = next(
                node for node in reference.module.graph.nodes if node.name == name
            )
            if node.op == "call_module":
                layer = reference.module.get_submodule(node.target)
                if isinstance(layer, torch.ao.nn.quantized.Conv2d):
                    input_value = reference.values[node.args[0].name]
                    batch, channel, row, column = index
                    stride_h, stride_w = layer.stride
                    pad_h, pad_w = layer.padding
                    dilation_h, dilation_w = layer.dilation
                    weight = layer.weight()
                    channels = weight.shape[1]
                    group = channel // (weight.shape[0] // layer.groups)
                    raw_input = input_value.int_repr()
                    patch = []
                    for local_channel in range(channels):
                        for kernel_row in range(weight.shape[2]):
                            for kernel_column in range(weight.shape[3]):
                                h = row * stride_h - pad_h + kernel_row * dilation_h
                                w = (
                                    column * stride_w
                                    - pad_w
                                    + kernel_column * dilation_w
                                )
                                patch.append(
                                    int(
                                        raw_input[
                                            batch,
                                            group * channels + local_channel,
                                            h,
                                            w,
                                        ]
                                    )
                                    if 0 <= h < raw_input.shape[2]
                                    and 0 <= w < raw_input.shape[3]
                                    else input_value.q_zero_point()
                                )
                    raw_weight = weight.int_repr()[channel].flatten().tolist()
                    zero = int(weight.q_per_channel_zero_points()[channel])
                    accumulator = sum(
                        (value - input_value.q_zero_point()) * (coefficient - zero)
                        for value, coefficient in zip(patch, raw_weight, strict=True)
                    )
                    differences.append(
                        f"{name}: exact_accumulator={accumulator}"
                        f" input_scale={input_value.q_scale()}"
                        f" input_zero={input_value.q_zero_point()}"
                        f" weight_scale={weight.q_per_channel_scales()[channel].item()}"
                        f" weight_zero={zero} bias={layer.bias()[channel].item()}"
                        f" patch={patch} weight={raw_weight}"
                    )
            break
        if len(differences) == 12:
            break
    return "\n".join(differences)


def _load_pretrained_model(name, weights):
    factory = getattr(quantization, name)
    kwargs = {"transform_input": True} if name in {"googlenet", "inception_v3"} else {}
    if name == "inception_v3":
        kwargs["aux_logits"] = False
    if "fbgemm" in torch.backends.quantized.supported_engines:
        model = factory(
            weights=None, quantize=True, backend="fbgemm", progress=False, **kwargs
        )
    else:
        assert (
            "onednn" in torch.backends.quantized.supported_engines
        ), "These pretrained quantized model fixtures require fbgemm or oneDNN."
        # Windows lacks fbgemm. Pack the same serialized weights with oneDNN,
        # retaining torchvision's fbgemm observer/weight configuration.
        torch.backends.quantized.engine = "onednn"
        model = factory(weights=None, quantize=False, **kwargs).eval()
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
    if name == "googlenet":
        model.aux_logits = False
        model.aux1 = model.aux2 = None
    return model.eval()


@pytest.mark.parametrize(
    ("name", "weights"),
    [
        ("resnet18", quantization.ResNet18_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
        ("resnet50", quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
        (
            "resnext101_32x8d",
            quantization.ResNeXt101_32X8D_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
        ),
        ("googlenet", quantization.GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
        (
            "inception_v3",
            quantization.Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
        ),
        (
            "shufflenet_v2_x0_5",
            quantization.ShuffleNet_V2_X0_5_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
        ),
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
            adapted = _adapt_model(original)
            image = _get_test_image_tensor()
            flipped = image.flip(-1)
            for value in (image, flipped):
                torch.testing.assert_close(
                    adapted(value),
                    original(value),
                    rtol=0,
                    atol=0,
                    msg=lambda message, value=value: (
                        f"{message}\n"
                        f"{_quantized_reference_diagnostics(original, adapted, value)}"
                    ),
                )
            harness = RecoveredRuntimeHarness()
            program = harness.run_test(
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
            if name in {"googlenet", "inception_v3", "shufflenet_v2_x0_5"}:
                assert any(node.op_type == "Concat" for node in program.model.graph)
            if name == "shufflenet_v2_x0_5":
                assert any(node.op_type == "Transpose" for node in program.model.graph)
            harness.assert_program_roundtrip(
                program,
                examples=[image, flipped],
                reference=original,
                rtol=0,
                atol=0,
            )
    finally:
        if program is not None:
            program.release()
        torch.set_num_threads(previous_threads)
        torch.backends.quantized.engine = previous_engine
