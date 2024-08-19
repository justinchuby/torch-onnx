# Owner(s): ["module: onnx"]
# mypy: allow-untyped-defs

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import logging
import os
import unittest
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import onnxruntime
import pytest
import pytorch_test_common
import torch
import torch_onnx
import torch_onnx.errors
from torch.testing._internal import common_utils
from torch.testing._internal.opinfo import core as opinfo_core
from torch.types import Number

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable, torch.export.ExportedProgram]
_InputArgsType = Optional[
    Union[torch.Tensor, int, float, bool, Sequence[Any], Mapping[str, Any]]
]
_OutputsType = Sequence[_NumericType]


def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def assert_dynamic_shapes(onnx_program: torch.onnx.ONNXProgram, dynamic_shapes: bool):
    """Assert whether the exported model has dynamic shapes or not.

    Args:
        onnx_program (torch.onnx.ONNXProgram): The output of torch.onnx.dynamo_export.
        dynamic_shapes (bool): Whether the exported model has dynamic shapes or not.
            When True, raises if graph inputs don't have at least one dynamic dimension
            When False, raises if graph inputs have at least one dynamic dimension.

    Raises:
        AssertionError: If the exported model has dynamic shapes and dynamic_shapes is False and vice-versa.
    """

    if dynamic_shapes is None:
        return

    model_proto = onnx_program.model_proto
    # Process graph inputs
    dynamic_inputs = []
    for inp in model_proto.graph.input:
        dynamic_inputs += [
            dim
            for dim in inp.type.tensor_type.shape.dim
            if dim.dim_value == 0 and dim.dim_param != ""
        ]
    assert dynamic_shapes == (
        len(dynamic_inputs) > 0
    ), "Dynamic shape check failed for graph inputs"


def parameterize_class_name(cls: type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(common_utils.TestCase):
    opset_version = _constants.ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False
    check_shape = True
    check_dtype = True

    def setUp(self):
        super().setUp()
        onnxruntime.set_seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self.is_script_test_enabled = True

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            try:
                return run_model_test(
                    self,
                    m,
                    input_args=input_args,
                    input_kwargs=input_kwargs,
                    rtol=rtol,
                    atol=atol,
                    do_constant_folding=do_constant_folding,
                    dynamic_axes=dynamic_axes,
                    additional_test_inputs=additional_test_inputs,
                    input_names=input_names,
                    output_names=output_names,
                    fixed_batch_size=fixed_batch_size,
                    training=training,
                    remained_onnx_input_idx=remained_onnx_input_idx,
                    flatten=flatten,
                    ignore_none=ignore_none,
                    verbose=verbose,
                )
            except torch_onnx.errors.TorchExportError:
                self.skipTest("torch.export errors are skipped")

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script_test_enabled and self.is_script:
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        if not is_model_script and not self.is_script:
            _run_test(model, tracing_remained_onnx_input_idx)


    def run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
        self,
        model: _ModelType,
        input_args: tuple[_InputArgsType],
        *,
        input_kwargs: Mapping[str, _InputArgsType] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        """Compare the results of PyTorch model with exported ONNX model

        Args:
            model: PyTorch model
            input_args: torch input arguments
            input_kwargs: torch input kwargs
            rtol: relative tolerance.
            atol: absolute tolerance.
        """

        # TODO: Here
        return

# The min onnx opset version to test for
FX_MIN_ONNX_OPSET_VERSION = 18
# The max onnx opset version to test for
FX_MAX_ONNX_OPSET_VERSION = 18
FX_TESTED_OPSETS = range(FX_MIN_ONNX_OPSET_VERSION, FX_MAX_ONNX_OPSET_VERSION + 1)

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.uint8,
)

QINT_TYPES = (
    torch.qint8,
    torch.quint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    # torch.float64,  ORT doesn't support
)

COMPLEX_TYPES = (
    # torch.complex32,  NOTE: torch.complex32 is experimental in torch
    torch.complex64,
    # torch.complex128,  ORT doesn't support
)

TESTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    *INT_TYPES,
    # Floating types
    *FLOAT_TYPES,
    # Complex types
    *COMPLEX_TYPES,
)


@dataclasses.dataclass
class DecorateMeta:
    """Information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py

    Attributes:
        op_name: The name of the operator.
        variant_name: The name of the OpInfo variant.
        decorator: The decorator to apply to the test case.
        opsets: The opsets to apply the decorator to.
        dtypes: The dtypes to apply the decorator to.
        reason: The reason for skipping.
        test_behavior: The behavior of the test case. [skip or xfail]
        matcher: The matcher to apply to the test case.
        enabled_if: Whether to enable test behavior. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Collection[int | Callable[[int], bool]] | None
    dtypes: Collection[torch.dtype] | None
    reason: str
    test_behavior: str
    matcher: Callable[[Any], bool] | None = None
    enabled_if: bool = True
    model_type: pytorch_test_common.TorchModelType | None = None

    def contains_opset(self, opset: int) -> bool:
        if self.opsets is None:
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], bool] | None = None,
    enabled_if: bool = True,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    """Expects a OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable xfail. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        opsets=opsets,
        dtypes=dtypes,
        enabled_if=enabled_if,
        matcher=matcher,
        reason=reason,
        test_behavior="xfail",
        model_type=model_type,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], Any] | None = None,
    enabled_if: bool = True,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable skip. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_behavior="skip",
        model_type=model_type,
    )


def skip_slow(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], Any] | None = None,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    """Skips a test case in OpInfo that is too slow.

    It needs further investigation to understand why it is slow.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=common_utils.slowTest,
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=not common_utils.TEST_WITH_SLOW,
        test_behavior="skip",
        model_type=model_type,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list.

    Args:
        all_opinfos: All OpInfos.
        test_class_name: The name of the test class.
        base_test_name: The name of the test method.
        opset: The opset to decorate for.
        skip_or_xfails: DecorateMeta's.
    """
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        if not decorate_meta.contains_opset(opset):
            # Skip does not apply to this opset
            continue
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        assert (
            opinfo is not None
        ), f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        assert decorate_meta.model_type is None, (
            f"Tested op: {decorate_meta.op_name} in wrong position! "
            "If model_type needs to be specified, it should be "
            "put under SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE."
        )
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def opsets_before(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is before the specified."""

    def compare(other_opset: int):
        return other_opset < opset

    return compare


def opsets_after(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is after the specified."""

    def compare(other_opset: int):
        return other_opset > opset

    return compare


def reason_onnx_script_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    """Formats the reason: ONNX script doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX script"


def reason_onnx_runtime_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    """Formats the reason: ONNX Runtime doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX Runtime"


def reason_onnx_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    """Formats the reason: ONNX doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the ONNX Spec"


def reason_dynamo_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    """Formats the reason: Dynamo doesn't support the given dtypes."""
    return (
        f"{operator} on {dtypes or 'certain dtypes'} not supported by the Dynamo Spec"
    )


def reason_jit_tracer_error(info: str) -> str:
    """Formats the reason: JIT tracer errors."""
    return f"JIT tracer error on {info}"


def reason_flaky() -> str:
    """Formats the reason: test is flaky."""
    return "flaky test"


@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: str | None = None, reason: str = ""
):
    """This context manager is used to handle the different behaviors of xfail and skip.

    Args:
        test_behavior (optional[str]): From DecorateMeta name, can be 'skip', 'xfail', or None.
        reason (optional[str]): The reason for the failure or skip.

    Raises:
        e: Any exception raised by the test case if it's not an expected failure.
    """

    # We need to skip as soon as possible, as SegFault might also be a case.
    if test_behavior == "skip":
        pytest.skip(reason=(reason))

    try:
        yield
    # We could use `except (AssertionError, RuntimeError, ...) as e:`, but it needs
    # to go over all test cases to find the right exception type.
    except Exception as e:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise e
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
