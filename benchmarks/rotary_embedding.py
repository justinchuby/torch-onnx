"""Compare broadcast-aware RoPE fusion with a valid unfused graph on CPU.

Run: python benchmarks/rotary_embedding.py > rotary.json
"""

from __future__ import annotations

import contextlib
import json
import platform
import statistics
import time
from unittest import mock

import numpy as np
import onnxruntime as ort
import torch

import torch_onnx


class Rotary(torch.nn.Module):
    def forward(self, x, frequencies):
        repeated = torch.cat((frequencies, frequencies), -1)
        cos, sin = repeated.cos().unsqueeze(1), repeated.sin().unsqueeze(1)
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def benchmark(shape):
    _batch, _heads, length, depth = shape
    torch.manual_seed(0)
    model = Rotary().eval()
    x = torch.randn(shape)
    frequencies = torch.randn(1, length, depth // 2)
    feed = {"x": x.numpy(), "frequencies": frequencies.numpy()}
    sessions, nodes = {}, {}
    for name in ("unfused", "fused"):
        context = (
            mock.patch("torch_onnx._onnx_fusions.fuse", return_value=None)
            if name == "unfused"
            else contextlib.nullcontext()
        )
        with context:
            program = torch_onnx.export(
                model, (x, frequencies), opset_version=23, verbose=False
            )
        nodes[name] = [node.op_type for node in program.model.graph]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        sessions[name] = ort.InferenceSession(
            program.model_proto.SerializeToString(),
            options,
            providers=["CPUExecutionProvider"],
        )
        program.release()
        actual = sessions[name].run(None, feed)[0]
        np.testing.assert_allclose(
            actual, model(x, frequencies).numpy(), rtol=1e-5, atol=1e-6
        )
        for _ in range(10):
            sessions[name].run(None, feed)
    samples = {name: [] for name in sessions}
    for repeat in range(100):
        order = list(sessions) if repeat % 2 else list(reversed(sessions))
        for name in order:
            start = time.perf_counter()
            sessions[name].run(None, feed)
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "shape": shape,
        "seconds": medians,
        "speedup": medians["unfused"] / medians["fused"],
        "nodes": nodes,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
                "threads": 1,
                "warmups": 10,
                "iterations": 100,
                "results": [
                    benchmark(shape)
                    for shape in (
                        (2, 2, 4, 8),
                        (2, 8, 128, 64),
                        (4, 16, 512, 64),
                    )
                ],
            },
            indent=2,
        )
    )
