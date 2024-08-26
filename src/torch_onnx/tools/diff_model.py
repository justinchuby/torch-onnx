"""Compute the numerical difference between two ONNX models."""

import gc
import logging
import os
import typing
from typing import TYPE_CHECKING, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxscript import ir
from torch_onnx import _verification

if TYPE_CHECKING:
    import torch.fx


logger = logging.getLogger(__name__)


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


def _add_outputs(model: ir.Model, value_names: Sequence[str]) -> None:
    """Add outputs to the model."""
    value_mapping = _create_value_mapping(model.graph)

    # Check that the values can be made outputs
    for value_name in value_names:
        if value_name not in value_mapping:
            raise ValueError(f"Value '{value_name}' is not in the graph")
        value = value_mapping[value_name]
        if value.shape is None:
            raise ValueError(
                f"Value '{value_name}' has unknown shape and cannot be an output"
            )
        if value.type is None:
            raise ValueError(
                f"Value '{value_name}' has unknown type and cannot be an output"
            )

    for value_name in value_names:
        model.graph.outputs.append(value_mapping[value_name])


def _infer_shapes(model: ir.Model) -> ir.Model:
    """Infer the shapes of the values in the model."""
    proto = ir.serde.serialize_model(model)
    inferred_model = onnx.shape_inference.infer_shapes(proto, data_prop=True)
    return ir.serde.deserialize_model(inferred_model)


def _process_onnx_model(
    model_path: str | os.PathLike,
    values: Sequence[str],
    keep_original_outputs: bool = True,
) -> tuple[str, Sequence[str]]:
    """Process the ONNX model and create a temporary model file.

    Returns:
        The path to the temporary model file and the names of all outputs.
    """
    model = ir.load(model_path)
    model = _infer_shapes(model)
    if not keep_original_outputs:
        model.graph.outputs.clear()
    _add_outputs(model, values)

    output_names = [output.name for output in model.graph.outputs]
    assert all(output_names), f"Output names must not be None or empty: {output_names}"

    model_dir = os.path.dirname(model_path)
    model_file = os.path.basename(model_path)
    temp_model_path = os.path.join(model_dir, f"temp_{model_file}")
    ir.save(model, temp_model_path)

    output_names = typing.cast(Sequence[str], output_names)
    return temp_model_path, output_names


def _process_exported_program(
    ep: torch.export.ExportedProgram,
    values: Sequence[str],
    keep_original_outputs: bool = True,
) -> None:
    """Add outputs to the exported program."""
    graph: torch.fx.Graph = ep.graph
    nodes = list(graph.nodes)
    new_outputs = []
    node_mapping = dict((node.name, node) for node in nodes)
    for value in values:
        if value not in node_mapping:
            raise ValueError(f"Value '{value}' is not in the graph")
        new_outputs.append(node_mapping[value])

    if not keep_original_outputs:
        for node in graph.nodes:
            if node.op != "output":
                continue
            graph.erase_node(node)

    last_node = list(graph.nodes)[-1]
    with graph.inserting_after(last_node):
        for node in new_outputs:
            graph.output((node,))

    ep.graph_module.recompile()


def _ort_session_initializer(model: str | bytes) -> ort.InferenceSession:
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(
        model, providers=providers, sess_options=session_options
    )


def _run_session(
    session: ort.InferenceSession, inputs: Sequence[np.ndarray]
) -> Sequence[np.ndarray]:
    # We don't expect non-tensor as inputs
    input_names = [input.name for input in session.get_inputs()]
    ort_input = dict(zip(input_names, inputs))
    run_options = ort.RunOptions()
    run_options.log_severity_level = 3  # 3: Error
    return session.run(None, ort_input, run_options=run_options)


def _compare_outputs(
    outputs_1: Sequence[torch.Tensor],
    outputs_2: Sequence[torch.Tensor],
    value_names_1: Sequence[str],
    value_names_2: Sequence[str],
) -> tuple[list[_verification.VerificationInfo], list[_verification.VerificationInfo]]:
    # The other is the expected
    results_1 = []
    results_2 = []
    for output_1, output_2, value_name_1, value_name_2 in zip(
        outputs_1, outputs_2, value_names_1, value_names_2
    ):
        try:
            # First, treat output_2 as the expected output
            max_abs_diff_1, max_rel_diff_1, abs_diff_1, rel_diff_1 = (
                _verification._compare_tensors(output_2, output_1)
            )
            abs_diff_1 = abs_diff_1.flatten()
            rel_diff_1 = rel_diff_1.flatten()
            bins = torch.tensor([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10])
            abs_diff_hist_1 = torch.histogram(abs_diff_1, bins=bins)
            rel_diff_hist_2 = torch.histogram(rel_diff_1, bins=bins)
            # TODO: Check which is the expected val when computing the diff
            results_1.append(
                _verification.VerificationInfo(
                    name=value_name_1,
                    max_abs_diff=max_abs_diff_1,
                    max_rel_diff=max_rel_diff_1,
                    abs_diff_hist=abs_diff_hist_1,
                    rel_diff_hist=rel_diff_hist_2,
                    expected_dtype=output_2.dtype,
                    actual_dtype=output_1.dtype,
                )
            )

            # Second, treat output_1 as the expected output
            max_abs_diff_2, max_rel_diff_2, abs_diff_2, rel_diff_2 = (
                _verification._compare_tensors(output_1, output_2)
            )
            abs_diff_2 = abs_diff_2.flatten()
            rel_diff_2 = rel_diff_2.flatten()
            abs_diff_hist_2 = torch.histogram(abs_diff_2, bins=bins)
            rel_diff_hist_2 = torch.histogram(rel_diff_2, bins=bins)
            results_2.append(
                _verification.VerificationInfo(
                    name=value_name_2,
                    max_abs_diff=max_abs_diff_2,
                    max_rel_diff=max_rel_diff_2,
                    abs_diff_hist=abs_diff_hist_2,
                    rel_diff_hist=rel_diff_hist_2,
                    expected_dtype=output_1.dtype,
                    actual_dtype=output_2.dtype,
                )
            )
        except Exception:  # noqa: PERF203
            logger.exception(
                "Error comparing outputs '%s' and '%s'", value_name_1, value_name_2
            )

    return results_1, results_2


def diff_exported_program(
    model_path: str | os.PathLike,
    exported_program: torch.export.ExportedProgram,
    value_names: Sequence[str] | Sequence[tuple[str, str]],
    inputs: Sequence[torch.Tensor],
    keep_original_outputs: bool = True,
) -> list[_verification.VerificationInfo]:
    """Compare the outputs of an ONNX model and an exported program.

    Args:
        model_path: The path to the ONNX model.
        exported_program: The exported program.
        value_names: The names of the values to compare. If provided as a list, then
            the same names in the ONNX model and the exported program will be compared.
            If provided as a list of tuples, then the first element of each tuple will
            be the name of the value in the ONNX model and the second element will be
            the name of the value in the exported program.
        inputs: The inputs to the models.
        keep_original_outputs: Whether to keep the original outputs of the models.

    Returns:
        A list of verification information for each value.
    """
    np_inputs = [
        input.numpy(force=True) if isinstance(input, torch.Tensor) else input
        for input in inputs
    ]

    if value_names and isinstance(value_names[0], tuple):
        onnx_names = [pair[0] for pair in value_names]
        torch_names = [pair[1] for pair in value_names]
    else:
        onnx_names = typing.cast(Sequence[str], value_names)
        torch_names = typing.cast(Sequence[str], value_names)

    temp_model_path, model_output_names = _process_onnx_model(
        model_path, onnx_names, keep_original_outputs
    )
    _process_exported_program(exported_program, torch_names, keep_original_outputs)
    # Run two models with the same inputs and compare the outputs
    ort_session = _ort_session_initializer(temp_model_path)
    outputs_onnx = _run_session(ort_session, np_inputs)
    del ort_session
    gc.collect()
    outputs_onnx = [torch.tensor(output) for output in outputs_onnx]

    # TODO: Handle kwargs
    outputs_torch = exported_program.module()(*inputs)

    # Compare the outputs
    results, _ = _compare_outputs(
        outputs_onnx, outputs_torch, model_output_names, model_output_names
    )
    return results


def diff(
    model_1_path: str | os.PathLike,
    model_2_path: str | os.PathLike,
    value_pairs: Sequence[tuple[str, str]],
    inputs: Sequence[np.ndarray | torch.Tensor],
    keep_original_outputs: bool = True,
) -> tuple[list[_verification.VerificationInfo], list[_verification.VerificationInfo]]:
    inputs = [
        input.numpy(force=True) if isinstance(input, torch.Tensor) else input
        for input in inputs
    ]

    temp_model_1_path, model_1_output_names = _process_onnx_model(
        model_1_path, [pair[0] for pair in value_pairs], keep_original_outputs
    )
    temp_model_2_path, model_2_output_names = _process_onnx_model(
        model_2_path, [pair[1] for pair in value_pairs], keep_original_outputs
    )

    # Run two models with the same inputs and compare the outputs
    ort_session_1 = _ort_session_initializer(temp_model_1_path)
    outputs_1 = _run_session(ort_session_1, inputs)
    del ort_session_1
    gc.collect()
    ort_session_2 = _ort_session_initializer(temp_model_2_path)
    outputs_2 = _run_session(ort_session_2, inputs)
    del ort_session_2
    gc.collect()

    outputs_1 = [torch.tensor(output) for output in outputs_1]
    outputs_2 = [torch.tensor(output) for output in outputs_2]

    # Compare the outputs
    return _compare_outputs(
        outputs_1, outputs_2, model_1_output_names, model_2_output_names
    )
