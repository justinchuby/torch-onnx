# mypy: allow-untyped-defs
from __future__ import annotations

__all__ = [
    "VerificationInfo",
    "SearchResult",
    "verify_onnx_program",
    "minimize_inaccurate_subgraph",
]

import copy
import dataclasses
import logging
import math
import operator
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import torch
from torch._functorch import fx_minifier
from torch.utils import _pytree

from torch_onnx import _core, _onnx_program

if TYPE_CHECKING:
    import torch.fx


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VerificationInfo:
    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime


@dataclasses.dataclass
class SearchResult:
    graph_module: torch.fx.GraphModule
    inputs: Sequence[Any]

    @property
    def graph(self) -> torch.fx.Graph:
        return self.graph_module.graph


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.numel() == 0 or actual.numel() == 0:
        return math.inf, math.inf, torch.tensor(math.inf), torch.tensor(math.inf)
    if expected.dtype == torch.bool:
        expected = expected.to(torch.float32)
        actual = actual.to(torch.float32)
    abs_diff = torch.abs(expected - actual)
    eps = 1e-7
    normalizer = torch.abs(expected) + eps
    rel_diff = abs_diff / normalizer

    max_absolute_difference = abs_diff.max().item()
    max_relative_difference = rel_diff.max().item()

    return max_absolute_difference, max_relative_difference, abs_diff, rel_diff


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> list[VerificationInfo]:
    exported_program = onnx_program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNX program does not contain an exported_program. "
            "Please provide an exported_program to verify the ONNX program."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    results = []
    for torch_output, onnx_output, output_val in zip(
        torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
    ):
        name = output_val.name
        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            torch_output, onnx_output
        )
        abs_diff = abs_diff.flatten()
        rel_diff = rel_diff.flatten()
        bins = torch.tensor([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10])
        abs_diff_hist = torch.histogram(abs_diff, bins=bins)
        rel_diff_hist = torch.histogram(rel_diff, bins=bins)
        results.append(
            VerificationInfo(
                name=str(name),
                max_abs_diff=max_abs_diff,
                max_rel_diff=max_rel_diff,
                abs_diff_hist=abs_diff_hist,
                rel_diff_hist=rel_diff_hist,
                expected_dtype=torch_output.dtype,
                actual_dtype=onnx_output.dtype,
            )
        )
    return results


def _exported_program_to_fx_graph_module_and_inputs(
    exported_program: torch.export.ExportedProgram,
) -> tuple[torch.fx.GraphModule, Sequence[torch.Tensor]]:
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm = exported_program.graph_module
    fx_inputs = exported_program._graph_module_flat_inputs(
        *exported_program.example_inputs
    )
    return fx_gm, fx_inputs


def _normalize_getitem_nodes(fx_gm: torch.fx.GraphModule):
    """This function turns all operator getitem nodes in ExportedProgram FX graph to

    new nodes composed of "computation + getitem". The normalization duplicates
    some computations in the graph but would make the graph more friendly for
    partitioning in FX minifier.
    """
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0

    fx_gm = copy.deepcopy(fx_gm)
    graph = fx_gm.graph
    for n in graph.nodes:
        if n.target != operator.getitem:
            continue

        src_n, key = n.args
        assert n.op == "call_function"
        with graph.inserting_after(n):
            new_n = graph.call_function(
                lambda src_target, key, args, kwargs: operator.getitem(
                    src_target(*args, **kwargs), key
                ),
                (src_n.target, key, src_n.args, src_n.kwargs),
            )
            n.replace_all_uses_with(new_n)

    graph.eliminate_dead_code()
    fx_gm.graph = graph
    return fx_gm


def _erase_unused_inputs(fx_gm: torch.fx.GraphModule, inputs: Sequence[torch.Tensor]):
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm = copy.deepcopy(fx_gm)
    args = fx_gm.graph.process_inputs(*inputs)
    args_iter = iter(args)

    graph = fx_gm.graph
    new_inputs = []
    for n in graph.nodes:
        if n.op == "placeholder":
            if n.target.startswith("*"):
                new_inputs += list(args_iter)
            elif len(n.users) > 0:
                new_inputs.append(next(args_iter))
            else:
                graph.erase_node(n)
                next(args_iter)
    new_inputs = tuple(new_inputs)
    fx_gm.graph = graph
    return fx_gm, new_inputs


def _lift_dead_ops_to_outputs(fx_gm: torch.fx.GraphModule):
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm = copy.deepcopy(fx_gm)

    new_outputs = []
    graph = fx_gm.graph
    nodes = list(graph.nodes)
    assert nodes[-1].op == "output" and sum(n.op == "output" for n in nodes) == 1
    for node in nodes:
        if node.op not in ("placeholder", "output") and len(node.users) == 0:
            new_outputs.append(node)  # noqa: PERF401

    output_node = nodes[-1]
    # FX output node returns the first arg as is.
    # ref: https://github.com/pytorch/pytorch/blob/1a578df57cc0f417f671634e564c62ef5d9a97e2/torch/fx/interpreter.py#L337
    new_outputs, _ = _pytree.tree_flatten([new_outputs, output_node.args[0]])
    output_node.update_arg(0, tuple(new_outputs))

    fx_gm.graph = graph
    return fx_gm


def _normalize_minified_fx_gm(
    fx_gm: torch.fx.GraphModule, inputs: Sequence[torch.Tensor]
):
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm, inputs = _erase_unused_inputs(fx_gm, inputs)
    fx_gm = _lift_dead_ops_to_outputs(fx_gm)
    return fx_gm, inputs


def _erase_trivial_outputs(fx_gm: torch.fx.GraphModule):
    """Remove output nodes directly connected to an input node."""
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm = copy.deepcopy(fx_gm)

    graph = fx_gm.graph
    nodes = list(graph.nodes)
    assert nodes[-1].op == "output" and sum(n.op == "output" for n in nodes) == 1
    output_node = nodes[-1]

    outputs, _ = _pytree.tree_flatten(output_node.args[0])
    new_outputs = [output for output in outputs if output.op != "placeholder"]
    output_node.update_arg(0, tuple(new_outputs))

    fx_gm.recompile()
    return fx_gm


def _erase_sub_gm_from_gm(
    fx_gm: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    sub_gm: torch.fx.GraphModule,
    sub_inputs: Sequence[torch.Tensor],
):
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0
    fx_gm = copy.deepcopy(fx_gm)
    fx_inputs = list(inputs)

    class EraseNodeInterpreter(torch.fx.Interpreter):
        def run_node(self, node):
            nonlocal fx_gm, fx_inputs
            res = super().run_node(node)
            if node.op not in ("placeholder", "output"):
                to_erase = next(m for m in fx_gm.graph.nodes if m.name == node.name)
                # Raise the output (tensor) of the erased node to be an input of
                # the new model graph. Some raised inputs may become unused later
                # when all the users are within the erased subgraph, those inputs
                # will be removed by the followed `_erase_unused_inputs` pass.
                with fx_gm.graph.inserting_before(to_erase):
                    new_input = fx_gm.graph.placeholder(node.name + "__value")
                to_erase.replace_all_uses_with(new_input)

                fx_gm.graph.erase_node(to_erase)
                fx_inputs.append(res)
            return res

    interpreter = EraseNodeInterpreter(sub_gm)
    interpreter.run(*sub_inputs)

    fx_gm.graph.lint()
    fx_gm.recompile()

    # Ops prior to the erased subgraph may be dangling. Lift them as outputs.
    fx_gm = _lift_dead_ops_to_outputs(fx_gm)
    fx_gm = _erase_trivial_outputs(fx_gm)
    fx_gm, fx_inputs = _erase_unused_inputs(fx_gm, fx_inputs)

    fx_gm.graph.lint()
    fx_gm.recompile()
    return fx_gm, fx_inputs


def _dump_state(fx_g: torch.fx.GraphModule, inps: Sequence[Any]):
    # TODO(justinchuby): Offload the actual inputs
    inps_text = _pytree.tree_map(lambda i: (i.shape, i.dtype, i.device.type), inps)
    print(
        f"""
# Working Repro with {len(fx_g.graph.nodes)} nodes
inps = {inps_text}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]
{fx_g.code}
"""
    )


def minimize_inaccurate_subgraph(
    exported_program: torch.export.ExportedProgram,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Iterator[SearchResult]:
    """Find the subgraph with error and minimize it."""
    # Adapted from https://github.com/google-ai-edge/ai-edge-torch/blob/a54d10d4fcf53339d32b00dda71918e810064e22/ai_edge_torch/debug/culprit.py
    # Original code Copyright 2024 The AI Edge Torch Authors.
    # Apache License, Version 2.0

    def _export_and_verify(
        torch_module: torch.fx.GraphModule,
        inputs: Any,
    ) -> bool:
        try:
            exported_program = torch.export.export(torch_module, tuple(inputs))
            onnx_model = _core.exported_program_to_ir(exported_program)
        except Exception:
            logger.exception("Failed to export the program. Stop minimizing.")
            # Treat this as a success because we don't want to minimize any more
            return False
        onnx_program = _onnx_program.ONNXProgram(onnx_model, exported_program)
        verification_info = verify_onnx_program(onnx_program)
        for info in verification_info:
            if info.max_abs_diff > atol or info.max_rel_diff > rtol:
                logger.warning("Found inaccuracy: %s", info)
                return True
        return False

    # Get the subgraph with error
    fx_gm, fx_inputs = _exported_program_to_fx_graph_module_and_inputs(exported_program)
    found_inaccuracies_num = 0
    while True:
        try:
            fx_gm = _normalize_getitem_nodes(fx_gm)
            raw_min_fx_gm, raw_min_inputs = fx_minifier.minifier(
                fx_gm,
                fx_inputs,
                _export_and_verify,
                dump_state=_dump_state,
            )
            min_fx_gm, min_inputs = _normalize_minified_fx_gm(
                raw_min_fx_gm, raw_min_inputs
            )
            found_inaccuracies_num += 1
            yield SearchResult(min_fx_gm, min_inputs)
            fx_gm, fx_inputs = _erase_sub_gm_from_gm(
                fx_gm, fx_inputs, raw_min_fx_gm, raw_min_inputs
            )
        except RuntimeError as e:  # noqa: PERF203
            if (
                str(e) == "Input graph did not fail the tester"
                and found_inaccuracies_num > 0
            ):
                break
            raise
