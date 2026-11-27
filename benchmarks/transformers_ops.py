"""Compare native tensor lowerings with their former strategies on one CPU thread.

Run: python benchmarks/transformers_ops.py > transformers-ops.json
"""

from __future__ import annotations

import json
import platform
import statistics
import time

import onnx_ir as ir
import onnxruntime as ort
import torch

import torch_onnx


def old_rms(op, input, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    if weight is None:
        weight = op.ConstantOfShape(
            op.Shape(input), value=ir.tensor([1.0], dtype=input.dtype)
        )
    return op.RMSNormalization(input, weight, axis=-len(normalized_shape), epsilon=eps)


def old_index(op, self, indices, values, accumulate=False):
    original = indices[0]
    mask = original
    rank = len(self.shape)
    mask_rank = len(original.shape)
    for _ in range(rank - mask_rank):
        mask = op.Unsqueeze(mask, [-1])
    mask = op.Expand(mask, op.Shape(self))
    coordinates = op.Transpose(op.NonZero(mask), perm=[1, 0])
    count = op.ReduceSum(
        op.Cast(op.Reshape(original, [-1]), to=ir.DataType.INT64), keepdims=1
    )
    selection = op.Concat(count, op.Shape(self, start=mask_rank), axis=0)
    updates = op.Reshape(op.Expand(values, selection), [-1])
    return op.ScatterND(self, coordinates, updates)


def old_scatter(op, self, mask, source):
    self = op.Expand(self, op.Shape(mask))
    mask = op.Expand(mask, op.Shape(self))
    coordinates = op.Transpose(op.NonZero(mask), perm=[1, 0])
    count = op.Gather(op.Shape(coordinates), [0], axis=0)
    updates = op.Slice(op.Reshape(source, [-1]), [0], count, [0])
    return op.ScatterND(self, coordinates, updates)


class Index(torch.nn.Module):
    def forward(self, x, mask, value):
        return torch.index_put(x, (mask,), value)


class Scatter(torch.nn.Module):
    def forward(self, x, mask, source):
        return x.masked_scatter(mask, source)


def benchmark(name, model, args, target, baseline):
    sessions, nodes = {}, {}
    for strategy in ("previous", "native"):
        if strategy == "previous":
            torch_onnx.register_custom_translation(target, baseline)
        try:
            program = torch_onnx.export(
                model.eval(), args, opset_version=23, verbose=False
            )
        finally:
            if strategy == "previous":
                torch_onnx.unregister_custom_translation(target)
        nodes[strategy] = len(program.model.graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        sessions[strategy] = ort.InferenceSession(
            program.model_proto.SerializeToString(),
            options,
            providers=["CPUExecutionProvider"],
        )
        feed = {
            value.name: tensor.numpy()
            for value, tensor in zip(program.model.graph.inputs, args, strict=True)
        }
        program.release()
        with torch.no_grad():
            expected = model(*args)
        actual = torch.from_numpy(sessions[strategy].run(None, feed)[0])
        torch.testing.assert_close(actual, expected)
        for _ in range(10):
            sessions[strategy].run(None, feed)
    samples = {strategy: [] for strategy in sessions}
    for iteration in range(200):
        order = list(sessions) if iteration % 2 else list(reversed(sessions))
        for strategy in order:
            start = time.perf_counter()
            sessions[strategy].run(None, feed)
            samples[strategy].append(time.perf_counter() - start)
    medians = {
        strategy: statistics.median(values) for strategy, values in samples.items()
    }
    return {
        "name": name,
        "seconds": medians,
        "nodes": nodes,
        "speedup": medians["previous"] / medians["native"],
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    x, mask = torch.randn(64, 512), torch.rand(64, 512) > 0.5
    row_mask = torch.arange(64) % 2 == 0
    row_values = torch.randn(32, 512)
    results = [
        benchmark(
            "nonaffine_rms",
            torch.nn.RMSNorm(32, elementwise_affine=False),
            (torch.randn(8, 128, 32),),
            torch.ops.aten.rms_norm.default,
            old_rms,
        ),
        benchmark(
            "scalar_bool_index_put",
            Index(),
            (x, mask, torch.tensor(0.25)),
            torch.ops.aten.index_put.default,
            old_index,
        ),
        benchmark(
            "row_bool_index_put",
            Index(),
            (x, row_mask, row_values),
            torch.ops.aten.index_put.default,
            old_index,
        ),
        benchmark(
            "masked_scatter",
            Scatter(),
            (x, mask, torch.randn(64 * 512)),
            torch.ops.aten.masked_scatter.default,
            old_scatter,
        ),
    ]
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
                "threads": 1,
                "warmups": 10,
                "iterations": 200,
                "results": results,
            },
            indent=2,
        )
    )
