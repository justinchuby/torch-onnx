# Owner(s): ["module: onnx"]
"""Check inferred facts against execution of small composed native graphs."""

from __future__ import annotations

from unittest import mock

import numpy as np
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification

from torch_onnx import GraphBuilder, TranslationContext


class TestInferenceSoundness(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_expression_symbols_do_not_alias_anonymous_input_dimensions(self):
        graph = ir.Graph([], [], nodes=[], name="expression_symbols")
        op = TranslationContext(GraphBuilder(graph))
        x = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["2*_d0"]),
        )
        y = ir.Value(
            name="y",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([None]),
        )
        graph.inputs.extend([x, y])
        result = op.Reshape(x, op.Concat(op.Shape(y), [-1], axis=0))
        graph.outputs.append(result)
        evaluator = ReferenceEvaluator(ir.to_proto(ir.Model(graph, ir_version=10)))

        # The author's _d0 in x has no relation to y's unknown dimension.
        for length, expected in ((6, (2, 3)), (10, (2, 5))):
            with self.subTest(length=length):
                (actual,) = evaluator.run(
                    None,
                    {
                        "x": np.zeros(length, dtype=np.float32),
                        "y": np.zeros(2, dtype=np.float32),
                    },
                )
                self.assertEqual(actual.shape, expected)
                if result.shape is not None:
                    for inferred, observed in zip(
                        result.shape, actual.shape, strict=True
                    ):
                        if isinstance(inferred, ir.SymbolicDim):
                            try:
                                concrete = int(inferred.value)
                            except (TypeError, ValueError):
                                continue
                        else:
                            concrete = inferred
                        self.assertEqual(concrete, observed)

    @common_utils.parametrize(
        "value,intermediate_dtype",
        [
            (2, ir.DataType.BOOL),
            (16777217, ir.DataType.FLOAT),
            (257, ir.DataType.UINT8),
        ],
    )
    def test_cast_roundtrip_constants_match_runtime(self, value, intermediate_dtype):
        graph = ir.Graph([], [], nodes=[], name="cast_roundtrip")
        op = TranslationContext(GraphBuilder(graph))
        original = op.constant([value], dtype=ir.DataType.INT64)
        converted = op.Cast(original, to=intermediate_dtype)
        result = op.Cast(converted, to=ir.DataType.INT64)
        graph.outputs.append(result)
        (actual,) = ReferenceEvaluator(ir.to_proto(ir.Model(graph, ir_version=10))).run(
            None, {}
        )

        # Inference may decline folding, but any claimed constant must be exact.
        if result.const_value is not None:
            np.testing.assert_array_equal(result.const_value.numpy(), actual)

    @common_utils.parametrize("optional", [False, True])
    def test_failed_inference_restores_nested_type_dtype_aliases(self, optional):
        graph = ir.Graph([], [], nodes=[], name="nested_type_rollback")
        builder = GraphBuilder(graph)
        leaf = ir.TensorType(ir.DataType.FLOAT)
        type_ = ir.OptionalType(leaf) if optional else ir.SequenceType(leaf)
        value = ir.Value(name="input", type=type_)
        alias = ir.Value(name="alias", type=type_)

        def fail_after_dtype_change(context, node):
            node.inputs[0].dtype = ir.DataType.DOUBLE
            raise ValueError("failed after mutating a nested type")

        with mock.patch(
            "torch_onnx._ir_builder.onnx_shape_inference.registry.get",
            return_value=fail_after_dtype_change,
        ):
            builder.node("Identity", [value], output_types=[type_])

        self.assertIs(value.type, type_)
        self.assertEqual(value.dtype, ir.DataType.FLOAT)
        self.assertEqual(alias.dtype, ir.DataType.FLOAT)
        self.assertEqual(leaf.dtype, ir.DataType.FLOAT)

    def test_loop_scan_length_is_not_unconditionally_max_trip_count(self):
        iteration = ir.Value(
            name="iteration",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([]),
        )
        condition = ir.Value(
            name="condition",
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape([]),
        )
        body = ir.Graph([iteration, condition], [], nodes=[], name="stop_after_one")
        body_op = TranslationContext(GraphBuilder(body))
        stop = body_op.constant(False, dtype=ir.DataType.BOOL)
        scan_item = body_op.constant([7], dtype=ir.DataType.INT64)
        body.outputs.extend([stop, scan_item])

        graph = ir.Graph([], [], nodes=[], name="early_terminating_loop")
        builder = GraphBuilder(graph)
        op = TranslationContext(builder)
        (result,) = builder.node(
            "Loop",
            [
                op.constant(3, dtype=ir.DataType.INT64),
                op.constant(True, dtype=ir.DataType.BOOL),
            ],
            attributes={"body": body},
            output_types=[ir.TensorType(ir.DataType.INT64)],
        )
        graph.outputs.append(result)
        (actual,) = ReferenceEvaluator(ir.to_proto(ir.Model(graph, ir_version=10))).run(
            None, {}
        )
        self.assertEqual(actual.shape, (1, 1))
        if result.shape is not None:
            for inferred, observed in zip(result.shape, actual.shape, strict=True):
                if isinstance(inferred, int):
                    self.assertEqual(inferred, observed)


common_utils.instantiate_parametrized_tests(TestInferenceSoundness)
