"""pytorch/pytorch#206940: legal Boolean sort and schema rejection contracts."""

import onnx
import pytest
import torch

import torch_onnx

pytestmark = pytest.mark.compatibility


@pytest.mark.parametrize("opset", [18, 23, 28])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_bool_sort_values_indices_and_ties(opset, descending, stable, dim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sort(x, dim=dim, descending=descending, stable=stable)

    x = torch.rand(3, 19, generator=torch.Generator().manual_seed(940)) > 0.5
    p = torch_onnx.export(Model().eval(), (x,), opset_version=opset, verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        values, indices = p(x)
        expected = Model()(x)
        torch.testing.assert_close(values, expected.values)
        assert values.dtype == torch.bool and indices.dtype == torch.int64
        torch.testing.assert_close(torch.gather(x, dim % x.ndim, indices), values)
        target = torch.arange(x.shape[dim]).expand_as(x if dim != 0 else x.T)
        torch.testing.assert_close(
            torch.sort(indices, dim=dim).values, target if dim != 0 else target.T
        )
        if stable:
            torch.testing.assert_close(indices, expected.indices)
    finally:
        p.release()


@pytest.mark.parametrize("opset", [18, 23, 28])
def test_issue_bool_argsort_and_pow(opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.argsort(x), torch.pow(x, 2)

    x = torch.tensor([True, False, True])
    p = torch_onnx.export(Model().eval(), (x,), opset_version=opset, verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        for actual, expected in zip(p(x), Model()(x), strict=True):
            torch.testing.assert_close(actual, expected)
    finally:
        p.release()


@pytest.mark.parametrize(
    "operation",
    [
        torch.argmax,
        torch.round,
        torch.ceil,
        torch.floor,
        torch.abs,
        torch.relu,
        lambda x: torch.softmax(x, 0),
    ],
)
@pytest.mark.parametrize("opset", [18, 23, 28])
def test_invalid_bool_input_never_writes_an_onnx_artifact(tmp_path, operation, opset):
    class Model(torch.nn.Module):
        def forward(self, x):
            return operation(x)

    x = torch.tensor([True, False, True])
    with pytest.raises((RuntimeError, NotImplementedError)):
        operation(x)
    destination = tmp_path / "invalid.onnx"
    with pytest.raises(torch_onnx.errors.OnnxExporterError):
        torch_onnx.export(
            Model().eval(), (x,), destination, opset_version=opset, verbose=False
        )
    assert not destination.exists()
