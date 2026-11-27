"""Preserve FX value identity in raw IR and dynamic axis names after serialization."""

import onnx
import onnx_ir as ir
import pytest
import torch

import torch_onnx
from torch_onnx import _dynamic_shapes


class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        values, indices = torch.max(x + self.weight, dim=1)
        return values + 1, indices, 3, True, 1.5


def test_raw_ir_multi_output_getitem_reuses_producer_values(tmp_path):
    ep = torch.export.export(MultiOutputModel(), (torch.randn(2, 3),))
    model = torch_onnx.exported_program_to_ir(ep, lower="none")
    nodes = list(model.graph)
    assert all("getitem" not in node.op_type for node in nodes)
    maximum = next(node for node in nodes if node.op_type == "aten.max.dim")
    assert maximum.outputs[0].shape == ir.Shape([2])
    assert maximum.outputs[1].dtype == ir.DataType.INT64
    assert model.graph.outputs[1] is maximum.outputs[1]
    consumer = next(node for node in nodes if maximum.outputs[0] in node.inputs)
    assert consumer.op_type == "aten.add.Tensor"
    assert "pkg.torch.onnx.fx_node" in maximum.metadata_props
    assert len(model.graph.initializers) == 4  # One weight and three scalar outputs.
    path = tmp_path / "raw.onnx"
    ir.save(model, path)
    loaded = ir.load(path)
    assert [value.dtype for value in loaded.graph.outputs] == [
        ir.DataType.FLOAT,
        ir.DataType.INT64,
        ir.DataType.INT64,
        ir.DataType.BOOL,
        ir.DataType.FLOAT,
    ]
    onnx.checker.check_model(str(path), check_custom_domain=False)


@pytest.mark.parametrize("axes", [["batch", None], ("batch", None)])
@pytest.mark.parametrize("optimize", [False, True])
def test_list_and_tuple_dynamic_axes_run_after_save_reload(tmp_path, axes, optimize):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.sin() + x

    model = Model().eval()
    program = torch_onnx.export(
        model,
        (torch.randn(2, 3),),
        dynamic_shapes=(axes,),
        optimize=optimize,
        verbose=False,
    )
    try:
        assert program.model.graph.inputs[0].shape[0].value == "batch"
        path = tmp_path / "dynamic.onnx"
        program.save(path)
        loaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        try:
            for batch in (2, 5):
                x = torch.randn(batch, 3)
                torch.testing.assert_close(loaded(x)[0], model(x))
            onnx.checker.check_model(str(path), full_check=True)
        finally:
            loaded.release()
    finally:
        program.release()


@pytest.mark.parametrize("axes", [{"bad": "batch"}, ["bad"], "batch"])
def test_malformed_legacy_dynamic_axes_are_rejected(axes):
    with pytest.raises(ValueError, match="dynamic_axes"):
        _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes(
            torch.nn.ReLU(),
            (torch.randn(2, 3),),
            {},
            dynamic_axes={"x": axes},
            input_names=["x"],
            output_names=set(),
        )


def test_legacy_dynamic_axes_warns_about_unknown_input_names():
    x = torch.randn(2, 3)
    with pytest.warns(UserWarning, match="not a valid input/output name"):
        shapes, args, kwargs = _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes(
            torch.nn.ReLU(),
            (x,),
            None,
            dynamic_axes={"typo": [0]},
            input_names=["x"],
            output_names=set(),
        )
    assert shapes == [None]
    assert args == (x,) and kwargs == {}


def test_raw_ir_preserves_tensor_list_inputs_and_factory_attributes():
    class Model(torch.nn.Module):
        def forward(self, x):
            y = x.to(dtype=torch.float64, copy=True)
            z = torch.zeros((2, 3), dtype=torch.float64, device="cpu")
            return torch.cat((y, z), dim=0)

    ep = torch.export.export(Model(), (torch.randn(2, 3),))
    model = torch_onnx.exported_program_to_ir(ep, lower="none")
    zeros = next(node for node in model.graph if node.op_type == "aten.zeros.default")
    assert zeros.attributes["dtype"].value == int(ir.DataType.DOUBLE)
    assert zeros.attributes["device"].value == "cpu"
    cat = next(node for node in model.graph if node.op_type == "aten.cat.default")
    assert len(cat.inputs) == 2
    assert cat.inputs[1] is zeros.outputs[0]
    assert model.graph.outputs[0].shape == ir.Shape([4, 3])
    assert model.graph.outputs[0].dtype == ir.DataType.DOUBLE
    onnx.checker.check_model(ir.to_proto(model), check_custom_domain=False)


def test_shared_dynamic_dimensions_warn_and_keep_one_consistent_name():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = Model().eval()
    with pytest.warns(UserWarning, match="shares the same shape constraints"):
        program = torch_onnx.export(
            model,
            (torch.randn(2, 3), torch.randn(2, 3)),
            dynamic_shapes=(["left_batch", None], ["right_batch", None]),
            verbose=False,
        )
    try:
        assert all(
            value.shape[0].value == "left_batch" for value in program.model.graph.inputs
        )
        x, y = torch.randn(5, 3), torch.randn(5, 3)
        torch.testing.assert_close(program(x, y)[0], model(x, y))
    finally:
        program.release()


@pytest.mark.parametrize(
    "axes", [{0: "batch", 1: None}, ["batch", None], ("batch", None)]
)
def test_dynamic_shapes_back_conversion_keeps_only_dynamic_axes(axes):
    result = _dynamic_shapes.from_dynamic_shapes_to_dynamic_axes(
        (axes, None), ["x", "y"], RuntimeError("original capture failure")
    )
    assert result == {"x": [0]}


def test_dynamic_shapes_back_conversion_preserves_original_failure_as_cause():
    original = RuntimeError("original capture failure")
    with pytest.raises(ValueError, match="number of input names") as error:
        _dynamic_shapes.from_dynamic_shapes_to_dynamic_axes(
            ({0: "batch"}, {0: "batch"}), ["x"], original
        )
    assert error.value.__cause__ is original
