"""Compute the numerical difference between two ONNX models."""

import gc
import os
from typing import Sequence
import numpy as np
from onnxscript import ir
import torch
from torch_onnx import _onnx_program, _verification
import onnx
import onnxruntime as ort


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
    for value_name in value_mapping:
        if value_name not in value_names:
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


def _process_onnx_model(model_path: str | os.PathLike, values: Sequence[str], keep_original_outputs: bool= True) -> str:
    """Process the ONNX model and create a temporary model file."""
    model = ir.load(model_path)
    model = _infer_shapes(model)
    if not keep_original_outputs:
        model.graph.outputs.clear()
    _add_outputs(model, values)
    model_dir = os.path.dirname(model_path)
    model_file = os.path.basename(model_path)
    temp_model_path = os.path.join(model_dir, f"temp_{model_file}")
    ir.save(model, temp_model_path)
    return temp_model_path


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

def _run_session(session, inputs) -> Sequence[torch.Tensor]:
    # We don't expect non-tensor as inputs
    ort_input = dict(zip(session.get_inputs(), inputs))
    run_options = ort.RunOptions()
    run_options.log_severity_level = 3  # 3: Error
    return session.run(None, ort_input, run_options=run_options)


def _compare_outputs(outputs_1: Sequence[torch.Tensor], outputs_2: Sequence[torch.Tensor],
                     value_names_1: Sequence[str], value_names_2: Sequence[str]) -> tuple[list[_verification.VerificationInfo], list[_verification.VerificationInfo]]:
    # The other is the expected
    results_1 = []
    results_2 = []
    for output_1, output_2, value_name_1, value_name_2 in zip(
        outputs_1, outputs_2, value_names_1, value_names_2
    ):
        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _verification._compare_tensors(
            output_1, output_2
        )
        abs_diff = abs_diff.flatten()
        rel_diff = rel_diff.flatten()
        bins = torch.tensor([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10])
        abs_diff_hist = torch.histogram(abs_diff, bins=bins)
        rel_diff_hist = torch.histogram(rel_diff, bins=bins)
        # TODO: Check which is the expected val when computing the diff
        results_1.append(
            _verification.VerificationInfo(
                name=value_name_1,
                max_abs_diff=max_abs_diff,
                max_rel_diff=max_rel_diff,
                abs_diff_hist=abs_diff_hist,
                rel_diff_hist=rel_diff_hist,
                expected_dtype=output_2.dtype,
                actual_dtype=output_1.dtype,
            )
        )
    return results_1, results_2

def diff_exported_program(
    exported_program: torch.export.ExportedProgram,
    model_path: str | os.PathLike,
    values: Sequence[str],
    inputs: Sequence[np.ndarray],
) -> tuple[list[_verification.VerificationInfo], _verification.VerificationInfo]:
    pass


def diff(
    model_1_path: str | os.PathLike,
    model_2_path: str | os.PathLike,
    value_pairs: Sequence[tuple[str, str]],
    inputs: Sequence[np.ndarray],
    keep_original_outputs: bool = True,
) -> tuple[list[_verification.VerificationInfo], _verification.VerificationInfo]:
    temp_model_1_path = _process_onnx_model(model_1_path, [pair[0] for pair in value_pairs], keep_original_outputs)
    temp_model_2_path = _process_onnx_model(model_2_path, [pair[1] for pair in value_pairs], keep_original_outputs)

    # Run two models with the same inputs and compare the outputs
    ort_session_1 = _ort_session_initializer(temp_model_1_path)
    outputs_1 = _run_session(ort_session_1, inputs)
    del ort_session_1
    gc.collect()
    ort_session_2 = _ort_session_initializer(temp_model_2_path)
    outputs_2 = _run_session(ort_session_2, inputs)
    del ort_session_2
    gc.collect()

    # Compare the outputs
    verification_info = _verification.(outputs_1, outputs_2)

