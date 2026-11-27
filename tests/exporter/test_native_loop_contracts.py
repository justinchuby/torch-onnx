"""Native Loop construction supports scalar carried state and zero iterations."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from torch_onnx import _ir_builder, _ir_passes
from torch_onnx._torchlib.ops import hop


def _value(name, dtype):
    return ir.Value(name=name, shape=ir.Shape([]), type=ir.TensorType(dtype))


def _function(name):
    counter = _value("counter", ir.DataType.INT64)
    total = _value("total", ir.DataType.FLOAT)
    limit = _value("limit", ir.DataType.INT64)
    graph = ir.Graph(
        [counter, total, limit], [], nodes=[], name=name, opset_imports={"": 28}
    )
    op = _ir_builder.TranslationContext(_ir_builder.GraphBuilder(graph, opset=28))
    outputs = (
        [op.Less(counter, limit)]
        if name == "condition"
        else [op.Add(counter, 1), op.Add(total, 0.25)]
    )
    graph.outputs.extend(outputs)
    return ir.Function("test.scalar_loop", name, graph=graph, attributes=[])


@pytest.mark.parametrize("limit", [0, 1, 4])
def test_native_loop_scalar_carried_state_matches_python(limit):
    condition, body = _function("condition"), _function("body")
    maximum = _value("maximum", ir.DataType.INT64)
    graph = ir.Graph(
        [maximum], [], nodes=[], name="scalar_loop", opset_imports={"": 28}
    )
    builder = _ir_builder.GraphBuilder(graph, opset=28)
    outputs = hop.higher_order_while_loop(
        _ir_builder.TranslationContext(builder), condition, body, [0, 1.0], [maximum]
    )
    for output, dtype in zip(
        outputs, [ir.DataType.INT64, ir.DataType.FLOAT], strict=True
    ):
        output.shape = ir.Shape([])
        output.type = ir.TensorType(dtype)
    graph.outputs.extend(outputs)
    model = ir.Model(graph, ir_version=10, functions=[condition, body])
    _ir_passes.add_opset_imports(model)
    ir.passes.common.NameFixPass()(model)
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    actual = ReferenceEvaluator(proto).run(
        None, {"maximum": np.array(limit, dtype=np.int64)}
    )
    np.testing.assert_equal(actual[0], np.array(limit, dtype=np.int64))
    np.testing.assert_equal(actual[1], np.array(1.0 + 0.25 * limit, dtype=np.float32))
