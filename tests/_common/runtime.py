# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Shared harness extracted without changing historical regression semantics."""

from __future__ import annotations

import torch
from torch.testing._internal import common_utils

import torch_onnx
from _common.assertions import (
    assert_program_outputs,
    assert_program_roundtrip,
    reload_onnx_program,
)


class RecoveredRuntimeHarness(common_utils.TestCase):
    def reload_program(
        self,
        program: torch_onnx.ONNXProgram,
        *,
        external_data: bool = False,
    ) -> torch_onnx.ONNXProgram:
        return reload_onnx_program(program, external_data=external_data)

    def assert_program_outputs(
        self,
        program: torch_onnx.ONNXProgram,
        args=(),
        kwargs=None,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        **expectation,
    ) -> tuple[torch.Tensor, ...]:
        return assert_program_outputs(
            program,
            args,
            kwargs,
            rtol=rtol,
            atol=atol,
            **expectation,
        )

    def assert_program_roundtrip(
        self,
        program: torch_onnx.ONNXProgram,
        *,
        examples=None,
        reference=None,
        rtol: float | None = None,
        atol: float | None = None,
        strategy: str | None = None,
        external_data: bool = False,
    ) -> torch_onnx.ONNXProgram:
        return assert_program_roundtrip(
            program,
            examples=examples,
            reference=reference,
            rtol=rtol,
            atol=atol,
            strategy=strategy,
            external_data=external_data,
        )

    @torch.no_grad()
    def run_test(
        self,
        model,
        input_args=None,
        input_kwargs=None,
        *,
        rtol=1e-3,
        atol=1e-7,
        opset_version: int | None = None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        dynamic_shapes=None,
        remained_onnx_input_idx=None,
        training: bool | None = False,
    ):
        args = input_args if isinstance(input_args, tuple) else (input_args,)
        # A trailing dictionary is legacy kwargs syntax unless followed by {}.
        export_args = (
            (*args, {})
            if args and isinstance(args[-1], dict) and input_kwargs is None
            else args
        )
        if training is not None:
            model.train(training)
        target_opset = (
            getattr(self, "opset_version", None)
            if opset_version is None
            else opset_version
        )
        program = torch_onnx.export(
            model,
            export_args,
            optimize=True,
            verbose=False,
            kwargs={} if input_kwargs is None else input_kwargs,
            opset_version=target_opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
        )
        if output_names is not None:
            self.assertEqual(
                [value.name for value in program.model.graph.outputs][
                    : len(output_names)
                ],
                list(output_names),
            )
        if remained_onnx_input_idx is not None:
            # Modern export preserves the call signature, even for constant outputs.
            self.assertEqual(
                sum(bool(value.uses()) for value in program.model.graph.inputs),
                len(remained_onnx_input_idx),
            )
        self.assert_program_roundtrip(
            program,
            examples=additional_test_inputs,
            rtol=rtol,
            atol=atol,
            strategy=None,
        )
        return program
