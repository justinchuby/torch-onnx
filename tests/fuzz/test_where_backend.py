"""A narrow backend observation kept separate from exporter correctness tests."""

from __future__ import annotations

import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx.reference import ReferenceEvaluator


class BackendSignedZeroMismatch(AssertionError):
    """Only the confirmed ORT Where signed-zero discrepancy is expected here."""


@pytest.mark.integration
@pytest.mark.xfail(
    condition=sys.platform == "darwin"
    and platform.machine() == "arm64"
    and ort.__version__ == "1.32.0",
    reason="ORT 1.32.0 macOS arm64 Where loses -0; eager and ONNX reference preserve it",
    raises=BackendSignedZeroMismatch,
    strict=True,
)
def test_where_selects_signed_zero_bits_like_reference_and_eager():
    data = np.array([-0.0, 0.0], dtype=np.float32)
    condition = np.array([True, False])
    expected = torch.where(
        torch.from_numpy(condition), torch.from_numpy(data), torch.from_numpy(data)
    ).numpy()
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Where", ["cond", "x", "x"], ["y"])],
            "where_signed_zero",
            [
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [2]),
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2]),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 26)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"cond": condition, "x": data}
    reference = ReferenceEvaluator(model).run(None, feeds)[0]
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(np.signbit(reference), np.signbit(expected))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    actual = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    if not np.array_equal(np.signbit(actual), np.signbit(expected)):
        raise BackendSignedZeroMismatch(
            f"expected signbits {np.signbit(expected)}, got {np.signbit(actual)}"
        )
