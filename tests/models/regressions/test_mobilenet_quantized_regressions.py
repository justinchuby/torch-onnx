# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Recover test_models_quantized_onnxruntime.py::test_mobilenet_v3.

Source parent: e2c2673d00500d268fcb1cc63fa7cc3173876505.
Removal: 6dca9f63bc4975ac323044a440c2c7e5f8512408.
Retain the V1 pretrained QAT checkpoint, original photo and top-one oracle.
Explicit integer-storage Q/DQ replaces unsupported QTensor module capture.
"""

import operator

import torch
from _common.runtime import RecoveredRuntimeHarness
from test_pretrained_quantized_model_regressions import (
    _get_test_image_tensor,
    _TopPredictor,
)
from test_quantized_storage_regressions import (
    _ConvertedOperation,
    _dequantize,
    _quantize,
    _StoredLayer,
)
from torchvision.models import quantization


class _ConvertedLayer(torch.nn.Module):
    def __init__(self, layer, incoming):
        super().__init__()
        assert layer.weight().qscheme() == torch.per_tensor_affine
        self.scale, self.zero = incoming
        quantized_input = torch.quantize_per_tensor(
            torch.zeros(1), *incoming, torch.quint8
        )
        self.layer = _StoredLayer(
            layer,
            quantized_input,
            linear=isinstance(layer, torch.ao.nn.quantized.Linear),
            relu=isinstance(layer, torch.ao.nn.intrinsic.quantized.ConvReLU2d),
            transposed=False,
        )
        self.output_scale, self.output_zero = layer.scale, layer.zero_point

    def forward(self, x):
        return _dequantize(
            self.layer(_quantize(x, self.scale, self.zero)),
            self.output_scale,
            self.output_zero,
        )


class _ConvertedUnary(torch.nn.Module):
    def __init__(self, operation, scale, zero):
        super().__init__()
        self.operation, self.scale, self.zero = operation, scale, zero

    def forward(self, x):
        return _dequantize(
            _quantize(self.operation(x), self.scale, self.zero), self.scale, self.zero
        )


class _ConvertedBinary(torch.nn.Module):
    def __init__(self, multiply, scale, zero):
        super().__init__()
        self.multiply, self.scale, self.zero = multiply, scale, zero

    def forward(self, x, y):
        value = x * y if self.multiply else x + y
        return _dequantize(
            _quantize(value, self.scale, self.zero), self.scale, self.zero
        )


def _adapt_mobilenet(model):
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
                replacement = _ConvertedUnary(torch.nn.Identity(), *outgoing)
            elif isinstance(
                module, torch.ao.nn.quantized.Conv2d | torch.ao.nn.quantized.Linear
            ):
                assert incoming is not None, f"Missing input qparams for {node}"
                outgoing = (module.scale, module.zero_point)
                replacement = _ConvertedLayer(module, incoming)
            elif isinstance(module, torch.ao.nn.quantized.Hardswish):
                outgoing = (float(module.scale), int(module.zero_point))
                replacement = _ConvertedUnary(torch.nn.Hardswish(), *outgoing)
            elif isinstance(module, torch.nn.Hardsigmoid):
                # The quantized hard-sigmoid kernel has fixed output qparams.
                outgoing = (1 / 256, 0)
                replacement = _ConvertedUnary(torch.nn.Hardsigmoid(), *outgoing)
            elif isinstance(module, torch.ao.nn.quantized.DeQuantize):
                replacement, outgoing = torch.nn.Identity(), None
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                assert incoming is not None, f"Missing pooling qparams for {node}"
                replacement = _ConvertedOperation(module, *incoming)
            else:
                assert isinstance(
                    module, torch.nn.Identity | torch.nn.Dropout
                ), f"Add an explicit Q/DQ adaptation for {node}: {type(module)}"
                assert not module.training, f"Expected an evaluation module: {node}"
                replacement = torch.nn.Identity()
            adapted.set_submodule(node.target, replacement)
        elif node.op == "call_function":
            if node.target in {torch.ops.quantized.add, torch.ops.quantized.mul}:
                outgoing = (
                    float(node.kwargs["scale"]),
                    int(node.kwargs["zero_point"]),
                )
                name = f"adapted_{node.name}"
                adapted.add_submodule(
                    name,
                    _ConvertedBinary(node.target == torch.ops.quantized.mul, *outgoing),
                )
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


def _load_pretrained_mobilenet():
    weights = quantization.MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
    model = quantization.mobilenet_v3_large(weights=None, quantize=False)
    # torchvision loads this QAT checkpoint before conversion, not into a PTQ
    # model: https://github.com/pytorch/vision/issues/5890.
    model.fuse_model(is_qat=True)
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.load_state_dict(
        weights.get_state_dict(progress=False, check_hash=True, weights_only=True)
    )
    torch.ao.quantization.convert(model, inplace=True)
    return model.eval()


@torch.no_grad()
def test_pretrained_mobilenet_v3():
    previous_threads = torch.get_num_threads()
    previous_engine = torch.backends.quantized.engine
    torch.set_num_threads(1)
    program = None
    try:
        engines = torch.backends.quantized.supported_engines
        assert (
            "qnnpack" in engines or "onednn" in engines
        ), "The original MobileNetV3 checkpoint requires QNNPACK or oneDNN."
        # Windows has no QNNPACK; retain its QAT configuration and checkpoint,
        # but pack the original quantized oracle with the available oneDNN.
        torch.backends.quantized.engine = (
            "qnnpack" if "qnnpack" in engines else "onednn"
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            original = _TopPredictor(_load_pretrained_mobilenet()).eval()
            adapted = _adapt_mobilenet(original)
            image = _get_test_image_tensor()
            examples = [image, image.flip(-1)]
            for value in examples:
                torch.testing.assert_close(
                    adapted(value), original(value), rtol=0, atol=0
                )
            harness = RecoveredRuntimeHarness()
            program = harness.run_test(
                adapted,
                (image,),
                additional_test_inputs=examples[1:],
                rtol=0,
                atol=0,
            )
            assert {node.op_type for node in program.model.graph} >= {
                "Conv",
                "HardSigmoid",
                "HardSwish",
                "Mul",
                "QuantizeLinear",
                "DequantizeLinear",
                "TopK",
            }
            harness.assert_program_roundtrip(
                program,
                examples=examples,
                reference=original,
                rtol=0,
                atol=0,
            )
    finally:
        if program is not None:
            program.release()
        torch.set_num_threads(previous_threads)
        torch.backends.quantized.engine = previous_engine
