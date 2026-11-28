"""pytorch/pytorch#206938: all native window spellings and PyTorch coefficients."""

import onnx
import pytest
import torch

import torch_onnx

pytestmark = pytest.mark.compatibility


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "bartlett", "kaiser"])
@pytest.mark.parametrize("periodic", [None, False, True])
@pytest.mark.parametrize("length", [0, 1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_window_family(name, periodic, length, dtype):
    fn = getattr(torch, name + "_window")
    kwargs = {"dtype": dtype}
    if periodic is not None:
        kwargs["periodic"] = periodic

    class Model(torch.nn.Module):
        def forward(self):
            return fn(length, **kwargs)

    model = Model().eval()
    expected = model()
    p = torch_onnx.export(model, (), opset_version=18, verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        (actual,) = p()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-5 if dtype == torch.float32 else 2e-12,
            atol=2e-6 if dtype == torch.float32 else 2e-13,
        )
    finally:
        p.release()


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "bartlett", "kaiser"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("opset", [18, 28])
def test_native_dynamic_window_length(name, periodic, opset):
    # The ATen factory schema takes int, so torch.export currently specializes
    # shape-derived lengths. Exercise the native dynamic scalar input directly.
    import numpy as np
    import onnx_ir as ir
    import onnxruntime as ort

    from torch_onnx._ir_builder import GraphBuilder, TranslationContext
    from torch_onnx._torchlib.ops import factories

    length = ir.Value(
        name="length", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT64)
    )
    graph = ir.Graph([length], [], nodes=[], name="window", opset_imports={"": opset})
    op = TranslationContext(GraphBuilder(graph, opset=opset))
    result = getattr(factories, "aten_" + name + "_window")(op, length, periodic)
    result.shape = ir.Shape([None])
    result.type = ir.TensorType(ir.DataType.FLOAT)
    graph.outputs.append(result)
    model = ir.Model(graph, ir_version=10)
    ir.passes.common.NameFixPass()(model)
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for size in [0, 1, 3, 12]:
        actual = torch.from_numpy(
            session.run(None, {"length": np.array(size, dtype=np.int64)})[0]
        )
        torch.testing.assert_close(
            actual,
            getattr(torch, name + "_window")(size, periodic=periodic),
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
        )


@pytest.mark.parametrize("parameters", [{"alpha": 0.6}, {"alpha": 0.6, "beta": 0.2}])
def test_hamming_custom_coefficients(parameters):
    class Model(torch.nn.Module):
        def forward(self):
            return torch.hamming_window(8, periodic=False, **parameters)

    expected = Model()()
    p = torch_onnx.export(Model().eval(), (), verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        torch.testing.assert_close(p()[0], expected, rtol=1e-5, atol=1e-6)
    finally:
        p.release()


@pytest.mark.parametrize("beta", [0.0, 2.0, 12.0, 100.0])
def test_kaiser_explicit_beta(beta):
    class Model(torch.nn.Module):
        def forward(self):
            return torch.kaiser_window(8, False, beta)

    p = torch_onnx.export(Model().eval(), (), verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        torch.testing.assert_close(
            p()[0], Model()(), rtol=2e-5, atol=2e-6, equal_nan=True
        )
    finally:
        p.release()
