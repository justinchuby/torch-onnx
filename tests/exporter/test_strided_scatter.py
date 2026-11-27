"""Native strided writes, including Transformers issue #51305 item 6."""

import math

import pytest
import torch
from _common.assertions import export_and_assert


@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (7, 3), (2, 2, 2), (0, 3), (3, 0)])
@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_fill_diagonal(shape, wrap, dtype):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.clone().fill_diagonal_(7, wrap=wrap)

    x = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    program = export_and_assert(Model().eval(), (x,), check_serialization=True)
    assert not any(node.op_type.startswith("Sequence") for node in program.model.graph)
    program.release()


def test_fill_diagonal_transposed_dense_input():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.clone().fill_diagonal_(-2)

    x = torch.arange(21.0).reshape(3, 7).t()
    program = export_and_assert(Model().eval(), (x,), check_serialization=True)
    program.release()


@pytest.mark.parametrize("opset", [20, 28])
@pytest.mark.parametrize(
    "size,stride,offset", [((2, 3), (6, 2), 1), ((), (), 3), ((0,), (1,), 0)]
)
def test_as_strided_scatter(size, stride, offset, opset):
    class Model(torch.nn.Module):
        def forward(self, x, src):
            return torch.as_strided_scatter(x, src, size, stride, offset)

    args = (
        torch.arange(24.0).reshape(2, 12),
        -torch.arange(math.prod(size), dtype=torch.float32).reshape(size) - 3,
    )
    program = export_and_assert(
        Model(), args, opset_version=opset, check_serialization=True
    )
    program.release()


@pytest.mark.parametrize("offset", [None, 0, 6])
def test_as_strided_scatter_base_storage_offset(offset):
    class Model(torch.nn.Module):
        def forward(self, x, src):
            return torch.as_strided_scatter(x, src, (3,), (4,), offset)

    args = (torch.arange(30.0)[5:25].reshape(4, 5), torch.full((3,), -9.0))
    program = export_and_assert(Model().eval(), args, check_serialization=True)
    program.release()


def test_as_strided_scatter_dynamic_size_and_stride():
    class Model(torch.nn.Module):
        def forward(self, x, src):
            return torch.as_strided_scatter(x, src, (x.shape[0],), (x.shape[1] + 1,), 0)

    rows = torch.export.Dim("rows", min=2, max=4)
    columns = torch.export.Dim("columns", min=5, max=8)
    program = export_and_assert(
        Model(),
        (torch.randn(3, 5), torch.randn(3)),
        dynamic_shapes=({0: rows, 1: columns}, {0: rows}),
        additional_test_inputs=[(torch.randn(4, 7), torch.randn(4))],
        check_serialization=True,
    )
    program.release()


def test_gapped_base_is_rejected_instead_of_flattening_logical_values():
    import torch_onnx

    class Model(torch.nn.Module):
        def forward(self, x, src):
            return torch.as_strided_scatter(x, src, (2,), (2,), 0)

    with pytest.raises(torch_onnx.errors.ConversionError, match="storage holes"):
        torch_onnx.export(
            Model().eval(), (torch.arange(12.0)[::2], torch.ones(2)), verbose=False
        )


def test_as_strided_scatter_noncontiguous_updates():
    class Model(torch.nn.Module):
        def forward(self, x, src):
            return x, torch.as_strided_scatter(x, src, (2, 3), (5, 1), 1)

    args = (torch.arange(20.0).reshape(4, 5), torch.arange(6.0).reshape(3, 2).t())
    program = export_and_assert(Model().eval(), args, check_serialization=True)
    program.release()
