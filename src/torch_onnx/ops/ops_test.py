"""Test the ONNX custom operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx
from torch.testing._internal import common_device_type, common_utils

import torch_onnx.ops
from torch_onnx.ops._testing import onnx_opinfo

if TYPE_CHECKING:
    from torch.testing._internal.opinfo import core as opinfo_core


def assert_no_onnx_ops(graph: torch.fx.Graph) -> None:
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target._namespace == "onnx":
            raise AssertionError(f"Found an ONNX op in the graph: {node}")


class OpTest(common_utils.TestCase):
    @common_device_type.ops(
        onnx_opinfo.op_db,
        allowed_dtypes=[torch.float32],
    )
    def test_op_(self, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo):
        class Model(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return op.op(*args, **kwargs)

        for i, sample in enumerate(
            op.sample_inputs(
                device,
                dtype,
                requires_grad=False,
            )
        ):
            inputs = (sample.input, *sample.args)
            # Provide the repr to subtest because tensors are not serializable in parallel test runs
            with self.subTest(
                subtest="exported_program",
                sample_num=i,
                inputs=repr(
                    [
                        f"Tensor<{inp.shape}, dtype={inp.dtype}>"
                        if isinstance(inp, torch.Tensor)
                        else inp
                        for inp in inputs
                    ]
                ),
                kwargs=repr(sample.kwargs),
            ):
                ep = torch.export.export(
                    Model(), inputs, kwargs=sample.kwargs, strict=False
                )
                torch.testing.assert_close(
                    ep.module()(*inputs, **sample.kwargs),
                    op.op(*inputs, **sample.kwargs),
                    equal_nan=True,
                )

            with self.subTest(
                subtest="decomp_to_aten",
                sample_num=i,
                inputs=repr(
                    [
                        f"Tensor<{inp.shape}, dtype={inp.dtype}>"
                        if isinstance(inp, torch.Tensor)
                        else inp
                        for inp in inputs
                    ]
                ),
                kwargs=repr(sample.kwargs),
            ):
                decomped = ep.run_decompositions(
                    torch_onnx.ops.onnx_aten_decomp_table()
                )
                assert_no_onnx_ops(decomped.graph)
                torch.testing.assert_close(
                    decomped.module()(*inputs, **sample.kwargs),
                    op.op(*inputs, **sample.kwargs),
                    equal_nan=True,
                )

    def test_op_can_be_exported_to_onnx(self):
        pass

    def test_op_results_matches_onnx_runtime(self):
        pass


common_device_type.instantiate_device_type_tests(OpTest, globals(), only_for=["cpu"])

if __name__ == "__main__":
    common_utils.run_tests()
