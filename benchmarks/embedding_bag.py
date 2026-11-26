"""Compare a saved old embedding lowering against current tensor scans on one host."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch

from torch_onnx._torchlib.ops import embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Direct translations avoid capture/optimization hiding the implementation.
    spec = importlib.util.spec_from_file_location(
        "embedding_fixture",
        Path(__file__).resolve().parents[1]
        / "tests/exporter/test_embedding_bag_tensor_scan.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    old = types.ModuleType("old_embedding")
    source = args.old_source.read_text().replace(
        "from torch_onnx._torchlib._torchlib_registry import onnx_impl",
        "def onnx_impl(*args, **kwargs):\n    return lambda function: function",
    )
    exec(compile(source, str(args.old_source), "exec"), old.__dict__)
    generator = torch.Generator().manual_seed(0)
    results = []
    for bags, length, width in [(4, 4, 16), (256, 16, 64), (1024, 32, 128)]:
        weight = torch.randn(2048, width, generator=generator)
        indices = torch.randint(0, 2048, (bags * length,), generator=generator)
        offsets = torch.arange(0, bags * length, length)
        feed = {
            "weight": weight.numpy(),
            "indices": indices.numpy(),
            "offsets": offsets.numpy(),
        }
        for mode in (0, 1, 2):
            sessions = {}
            nodes = {}
            for name, function in [
                ("old", old.aten_embedding_bag_padding_idx),
                ("scan", embedding.aten_embedding_bag_padding_idx),
            ]:
                model = fixture.build_model(function, mode=mode)
                nodes[name] = len(fixture.node_types(model.graph))
                sessions[name] = fixture.session(model)
            expected = torch.ops.aten.embedding_bag.padding_idx(
                weight, indices, offsets, False, mode, False, None, False, None
            )
            for runtime in sessions.values():
                for actual, reference in zip(
                    runtime.run(None, feed), expected, strict=True
                ):
                    np.testing.assert_allclose(
                        actual, reference.numpy(), rtol=2e-5, atol=2e-6
                    )
                for _ in range(5):
                    runtime.run(None, feed)
            samples = {name: [] for name in sessions}
            for repeat in range(15):
                # Alternate execution order to reduce drift bias.
                for name in list(sessions) if repeat % 2 else list(reversed(sessions)):
                    start = time.perf_counter()
                    sessions[name].run(None, feed)
                    samples[name].append(time.perf_counter() - start)
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            results.append(
                {
                    "bags": bags,
                    "length": length,
                    "width": width,
                    "mode": mode,
                    "seconds": medians,
                    "nodes_including_subgraphs": nodes,
                    "ratio_scan_over_old": medians["scan"] / medians["old"],
                }
            )
            print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
