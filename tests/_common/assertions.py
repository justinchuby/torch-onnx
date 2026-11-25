"""Standardized export, serialization, and output assertions for recovered tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import onnx
import onnx_ir as ir
import torch
from torch.utils import _pytree

import torch_onnx

_MISSING = object()


def _normalize_args(args: Any) -> tuple[Any, ...]:
    return args if isinstance(args, tuple) else (args,)


def _copy_value(value: Any) -> Any:
    memo: dict[int, object] = {}

    def _copy(item: object) -> object:
        if isinstance(item, torch.Tensor) and not item.is_leaf:
            if id(item) not in memo:
                detached = item.detach().requires_grad_(item.requires_grad)
                memo[id(item)] = copy.deepcopy(detached, memo)
            return memo[id(item)]
        return copy.deepcopy(item, memo)

    return _pytree.tree_map(_copy, value)


def _adapt_expected_outputs(expected: Any) -> tuple[torch.Tensor, ...]:
    flat_outputs, _ = _pytree.tree_flatten(expected)
    adapted: list[torch.Tensor] = []
    for output in flat_outputs:
        if output is None:
            continue
        if not isinstance(output, torch.Tensor):
            adapted.append(torch.tensor(output))
        elif torch.is_complex(output):
            adapted.append(torch.view_as_real(output))
        else:
            adapted.append(output)
    return tuple(adapted)


def reload_onnx_program(
    program: torch_onnx.ONNXProgram,
    *,
    external_data: bool = False,
) -> torch_onnx.ONNXProgram:
    """Save an ONNXProgram to disk and reload it into a fresh ONNXProgram."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recovered.onnx"
        program.save(path, external_data=external_data)
        loaded_model = (
            ir.serde.deserialize_model(onnx.load(path))
            if external_data
            else ir.load(path)
        )
        return torch_onnx.ONNXProgram(loaded_model, program.exported_program)


def assert_program_outputs(
    program: torch_onnx.ONNXProgram,
    args: Any = (),
    kwargs: dict[str, Any] | None = None,
    *,
    expected: Any = _MISSING,
    reference: Callable[..., Any] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Assert ONNXProgram outputs match an eager reference callable or expected value."""
    if (expected is _MISSING) == (reference is None):
        raise ValueError("Specify exactly one of 'expected' or 'reference'.")
    call_args = _normalize_args(args)
    call_kwargs = {} if kwargs is None else kwargs
    if reference is not None:
        ref_args, ref_kwargs = _copy_value((call_args, call_kwargs))
        expected = reference(*ref_args, **ref_kwargs)
    expected_tensors = _adapt_expected_outputs(expected)
    actual_tensors = tuple(program(*call_args, **call_kwargs))
    torch.testing.assert_close(
        actual_tensors,
        expected_tensors,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )
    return actual_tensors


def assert_program_roundtrip(
    program: torch_onnx.ONNXProgram,
    *,
    examples: Iterable[Any] | None = None,
    reference: Callable[..., Any] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    strategy: str | None = None,
    external_data: bool = False,
) -> torch_onnx.ONNXProgram:
    """Validate ONNX schema, save/reload roundtrip, and numerical outputs."""
    onnx.checker.check_model(program.model_proto, full_check=True)
    reloaded = reload_onnx_program(program, external_data=external_data)
    if reference is None:
        torch_onnx.testing.assert_onnx_program(
            program, strategy=strategy, rtol=rtol, atol=atol
        )
        torch_onnx.testing.assert_onnx_program(
            reloaded, strategy=None, rtol=rtol, atol=atol
        )
        if examples is not None:
            for example in examples:
                example_args = _normalize_args(example)
                for candidate in (program, reloaded):
                    torch_onnx.testing.assert_onnx_program(
                        candidate,
                        args=example_args,
                        kwargs={},
                        strategy=None,
                        rtol=rtol,
                        atol=atol,
                    )
    else:
        if examples is None:
            raise ValueError("Provide 'examples' when asserting against 'reference'.")
        example_list = list(examples)
        for candidate in (program, reloaded):
            for example in example_list:
                assert_program_outputs(
                    candidate,
                    example,
                    reference=reference,
                    rtol=rtol,
                    atol=atol,
                )
    return reloaded


def export_and_assert(
    model: torch.nn.Module,
    args: Any = (),
    kwargs: dict[str, Any] | None = None,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    additional_test_inputs: Iterable[Any] | None = None,
    check_serialization: bool = False,
    external_data: bool = False,
    strategy: str | None = "TorchExportNonStrictStrategy",
    **export_kwargs: Any,
) -> torch_onnx.ONNXProgram:
    """Export with torch_onnx, check ONNX validity, and assert numerical parity."""
    call_args = _normalize_args(args)
    export_options: dict[str, Any] = {"verbose": False, **export_kwargs}
    if kwargs is not None:
        export_options["kwargs"] = kwargs
    program = torch_onnx.export(model, call_args, **export_options)
    if check_serialization or external_data:
        assert_program_roundtrip(
            program,
            examples=additional_test_inputs,
            rtol=rtol,
            atol=atol,
            strategy=strategy,
            external_data=external_data,
        )
    else:
        onnx.checker.check_model(program.model_proto, full_check=True)
        torch_onnx.testing.assert_onnx_program(
            program, strategy=strategy, rtol=rtol, atol=atol
        )
        if additional_test_inputs is not None:
            for example in additional_test_inputs:
                torch_onnx.testing.assert_onnx_program(
                    program,
                    args=_normalize_args(example),
                    kwargs={},
                    strategy=None,
                    rtol=rtol,
                    atol=atol,
                )
    return program
