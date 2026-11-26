r"""Direct ONNX IR construction for plain-callable translations.

Only Python constants are converted implicitly. Tensor casts, symbolic shape
construction, sequences, and control flow remain explicit ONNX operations.
This module does not depend on ONNX Script evaluation or annotations.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import warnings
from typing import TYPE_CHECKING, Any

import onnx
import onnx_ir as ir
import onnx_shape_inference

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


logger = logging.getLogger(__name__)


_CURRENT: contextvars.ContextVar[GraphBuilder | None] = contextvars.ContextVar(
    "onnx_ir_graph_builder", default=None
)


def validate_opset_version(opset: int) -> None:
    """Validate that the requested ONNX opset version is a positive integer."""
    if type(opset) is not int or opset < 1:
        raise ValueError(
            f"ONNX opset must be a positive integer, got {opset!r}. "
            "Pass an integer >= 1."
        )


def current_builder() -> GraphBuilder:
    r"""Return the builder installed by ``GraphBuilder.activate()``."""
    builder = _CURRENT.get()
    if builder is None:
        raise RuntimeError(
            "No ONNX IR builder is active. Run the translation inside "
            "'with builder.activate():' or pass the builder explicitly."
        )
    return builder


class TranslationContext:
    r"""ONNX operator call helpers bound to an explicitly supplied builder.

    Named calls default to returning a single :class:`onnx_ir.Value` unless the
    operator is known to have multiple/optional/variadic outputs or ``outputs=`` /
    ``output_types=`` is passed. Output dtypes and symbolic shapes are inferred
    incrementally using ``onnx_shape_inference`` when an inference rule is
    available. Final ATen return values also receive captured ``ExportedProgram``
    metadata in ``_core.py``. When a default-domain operator
    schema is missing from the installed ``onnx`` package, named calls warn once
    per operator per builder and construct a single output node unless
    ``outputs=`` or ``output_types=`` specifies otherwise. Use ``builder`` for
    raw tuple-returning calls, custom domains, functions, and explicit subgraph
    construction.

    No activation or ambient builder lookup is needed. Exporter injection is
    separate from this construction API; helpers receive the context explicitly.

    Examples::

        op = TranslationContext(GraphBuilder(graph))
        result = op.Add(x, op.Mul(y, 2))
        values, indices = op.TopK(result, [1])
        parts = op.Split(result, outputs=2, axis=0)
    """

    def __init__(self, builder: GraphBuilder) -> None:
        self.builder = builder

    def constant(self, value: Any, *, dtype: ir.DataType) -> ir.Value:
        """Create a typed constant using the builder's graph-local cache."""
        return self.builder.constant(value, dtype=dtype)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            schema: onnx.defs.OpSchema | None = onnx.defs.get_schema(
                name, self.builder.opset, ""
            )
        except onnx.defs.SchemaError:
            schema = None

        def call(
            *inputs: Any,
            outputs: int | None = None,
            output_types: Sequence[ir.TypeProtocol | None] | None = None,
            **attributes: Any,
        ) -> ir.Value | tuple[ir.Value, ...]:
            if outputs is not None and (type(outputs) is not int or outputs < 0):
                raise ValueError(
                    f"{name} outputs must be a nonnegative integer, got {outputs!r}. "
                    "Specify the number of ONNX output values."
                )
            explicit_outputs = outputs is not None or output_types is not None
            effective_outputs = (
                1 if schema is None and not explicit_outputs else outputs
            )
            values = self.builder.call(
                name,
                *inputs,
                _num_outputs=effective_outputs,
                _output_types=output_types,
                **attributes,
            )
            if schema is not None:
                if schema.min_output == schema.max_output == 1:
                    return values[0]
                return values
            if not explicit_outputs or (
                outputs is None and output_types is not None and len(values) == 1
            ):
                return values[0]
            return values

        return call


class GraphBuilder:
    r"""Append native nodes to one graph without changing existing exports.

    Always returns tuples of plain ir.Value objects. A builder has graph-local
    constants and scoped activation; no symbolic Python operators are installed.
    Model assembly and optimization belong to the caller.

    Example::

        graph = ir.Graph([], [], nodes=[], name="translation")
        builder = GraphBuilder(graph)
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
        with builder.activate():
            (result,) = current_builder().call("Add", x, 1)
        graph.inputs.append(x)
        graph.outputs.append(result)
    """

    def __init__(self, graph: ir.Graph, *, opset: int = 18) -> None:
        validate_opset_version(opset)
        existing = graph.opset_imports.get("")
        if existing is not None and existing != opset:
            raise ValueError(
                f"Graph imports ONNX opset {existing}, not {opset}. "
                "Construct the builder with the graph's opset."
            )
        self.graph = graph
        self.opset = opset
        self.functions: dict[ir.OperatorIdentifier, ir.Function] = {}
        self._constants: dict[tuple[ir.DataType, tuple[int, ...], bytes], ir.Value] = {}
        self._warned_missing_schemas: set[str] = set()
        onnx_shape_inference.registry.collect()
        self._shape_context: onnx_shape_inference.ShapeInferenceContext | None = None

    @contextlib.contextmanager
    def activate(self) -> Iterator[GraphBuilder]:
        r"""Install this builder, restoring the previous scope after exceptions."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def constant(self, value: Any, *, dtype: ir.DataType) -> ir.Value:
        r"""Create a typed tensor constant; empty tensors require an explicit dtype."""
        try:
            tensor = ir.tensor(value, dtype=dtype)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot construct an ONNX constant with dtype {dtype}. "
                "Supply rectangular Python data representable by that dtype. "
                "Build symbolic shape lists explicitly with Reshape/Concat."
            ) from error
        array = tensor.numpy()
        # Object/string arrays contain pointers rather than stable value bytes.
        cacheable = array.dtype.kind not in ("O", "U", "S")
        key = (dtype, tuple(array.shape), array.tobytes())
        if cacheable and key in self._constants:
            return self._constants[key]
        result = self.node(
            "Constant",
            attributes={"value": tensor},
            output_types=[ir.TensorType(dtype)],
        )[0]
        result.shape = ir.Shape(array.shape)
        result.const_value = tensor
        if cacheable:
            self._constants[key] = result
        return result

    def node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None] = (),
        *,
        attributes: Mapping[str, Any] | None = None,
        output_types: Sequence[ir.TypeProtocol | None],
        domain: str = "",
        version: int | None = None,
        overload: str = "",
    ) -> tuple[ir.Value, ...]:
        r"""Append a raw node with explicit types, including custom/control-flow nodes.

        Subgraphs are explicit ir.Graph attributes, constructed with their own
        builder. Captures, graph outputs, function dependencies, and names avoiding
        shadowing of ancestor values are the caller's responsibility.
        """
        if version is None:
            if domain:
                raise ValueError(
                    f"Missing opset version for {domain}::{op_type}. Pass version=."
                )
            version = self.opset
        existing = self.graph.opset_imports.get(domain)
        if existing is not None and existing != version:
            raise ValueError(
                f"Graph imports {domain!r} version {existing}, but {op_type} "
                f"requests {version}. Use one opset version per domain."
            )
        if any(
            value is not None and not isinstance(value, ir.Value) for value in inputs
        ):
            raise TypeError(
                f"{domain}::{op_type} raw inputs must be ir.Value or None. "
                "Use constant(..., dtype=...) or call() for Python constants."
            )
        trimmed = list(inputs)
        while trimmed and trimmed[-1] is None:
            trimmed.pop()
        outputs = [ir.Value(type=type_) for type_ in output_types]
        node = ir.Node(
            domain,
            op_type,
            inputs=trimmed,
            attributes=ir.convenience.convert_attributes(
                {
                    key: value
                    for key, value in (attributes or {}).items()
                    if value is not None
                }
            ),
            outputs=outputs,
            version=version,
            overload=overload,
        )
        self.graph.append(node)
        self.graph.opset_imports[domain] = version
        self._infer_node(node, version)
        return tuple(outputs)

    def _infer_node(self, node: ir.Node, version: int) -> None:
        """Enrich intermediate values without requiring every op to have a rule."""
        infer = onnx_shape_inference.registry.get(
            node.domain, node.op_type, version=version
        )
        if infer is None:
            return
        context = self._shape_context
        if context is None:
            context = onnx_shape_inference.ShapeInferenceContext(
                self.graph.opset_imports
            )
            self._shape_context = context
            values = (*self.graph.inputs, *node.inputs)
        else:
            values = node.inputs
        context.opset_imports = self.graph.opset_imports
        for value in values:
            if value is not None and value.shape is not None:
                context.reserve_symbol_names(
                    str(dim)
                    for dim in value.shape
                    if isinstance(dim, ir.SymbolicDim) and dim.value is not None
                )
        try:
            for value in node.inputs:
                if value is not None:
                    context.name_anonymous_dims(value)
            infer(context, node)
            # onnx-shape-inference 0.4.3 flattens nonconstant Unique inputs
            # even with axis=. Correct this before downstream Shape rules run.
            if (
                node.domain == ""
                and node.op_type == "Unique"
                and "axis" in node.attributes
            ):
                data = node.inputs[0]
                output = node.outputs[0]
                if data is not None and data.const_value is None:
                    if data.shape is None:
                        output.shape = None
                    elif output.shape is not None and output.shape.rank() == 1:
                        dims = list(data.shape)
                        dims[node.attributes["axis"].as_int()] = output.shape[0]
                        output.shape = ir.Shape(dims)
        except Exception:
            # Partial graphs, unresolved function attributes and custom operators
            # can lack the information needed for inference. Capture metadata and
            # final model validation remain authoritative for exporter correctness.
            logger.debug(
                "Shape inference failed for %s::%s",
                node.domain,
                node.op_type,
                exc_info=True,
            )

    def call(
        self,
        op_type: str,
        *inputs: Any,
        domain: str = "",
        version: int | None = None,
        _num_outputs: int | None = None,
        _output_types: Sequence[ir.TypeProtocol | None] | None = None,
        **attributes: Any,
    ) -> tuple[ir.Value, ...]:
        r"""Call an ONNX schema, converting constants but never casting tensors.

        Inputs are positional, attributes are keyword-only, variadics are expanded
        positionally, and optional holes use None. Ambiguous output arity requires
        _num_outputs or _output_types. Output types may be supplied explicitly.
        Symbolic inference enriches output types, shapes, and shape-tensor data
        when an inference rule is available.
        """
        num_outputs = _num_outputs
        output_types = _output_types
        if version is None:
            version = self.graph.opset_imports.get(domain)
        if version is None:
            if domain:
                raise ValueError(
                    f"Missing opset version for {domain}::{op_type}. Pass version=."
                )
            version = self.opset
        existing = self.graph.opset_imports.get(domain)
        if existing is not None and existing != version:
            raise ValueError(
                f"{domain}::{op_type} requests version {version}, but graph imports "
                f"{existing}. Use the graph's opset version."
            )
        try:
            schema: onnx.defs.OpSchema | None = onnx.defs.get_schema(
                op_type, version, domain
            )
        except onnx.defs.SchemaError as error:
            if domain:
                raise ValueError(
                    f"No ONNX schema for {domain}::{op_type} at version {version}. "
                    "Correct the operator/opset, install an ONNX build with that "
                    "schema, or use node() with explicit output types."
                ) from error
            schema = None
        if schema is None:
            return self._call_without_schema(
                op_type,
                inputs,
                version=version,
                num_outputs=num_outputs,
                output_types=output_types,
                attributes=attributes,
            )
        if (
            not domain
            and op_type == "ScatterElements"
            and attributes.get("reduction") == "mean"
            and schema.since_version < 29
        ):
            if version < 29:
                raise ValueError(
                    f"ScatterElements reduction='mean' requires opset_version >= 29, "
                    f"got {version}. Select opset_version >= 29 or use explicit "
                    "sum/count scatter reduction."
                )
            return self._call_without_schema(
                op_type,
                inputs,
                version=version,
                num_outputs=num_outputs if num_outputs is not None else 1,
                output_types=output_types,
                attributes=attributes,
                stale_since_version=schema.since_version,
            )
        try:
            return self._call_with_schema(
                schema,
                op_type,
                inputs,
                domain=domain,
                version=version,
                num_outputs=num_outputs,
                output_types=output_types,
                attributes=attributes,
            )
        except (TypeError, ValueError):
            if (
                not domain
                and version > onnx.defs.onnx_opset_version()
                and self._can_fallback_stale_schema(
                    schema, inputs, num_outputs, output_types
                )
            ):
                fallback_outputs = (
                    num_outputs
                    if num_outputs is not None
                    else (
                        schema.min_output
                        if schema.min_output == schema.max_output
                        else None
                    )
                )
                return self._call_without_schema(
                    op_type,
                    inputs,
                    version=version,
                    num_outputs=fallback_outputs,
                    output_types=output_types,
                    attributes=attributes,
                    stale_since_version=schema.since_version,
                )
            raise

    @staticmethod
    def _can_fallback_stale_schema(
        schema: onnx.defs.OpSchema,
        inputs: tuple[Any, ...],
        num_outputs: int | None,
        output_types: Sequence[ir.TypeProtocol | None] | None,
    ) -> bool:
        if num_outputs is not None and (
            type(num_outputs) is not int or num_outputs < 0
        ):
            return False
        if (
            output_types is not None
            and num_outputs is not None
            and num_outputs != len(output_types)
        ):
            return False
        signature = ir.schemas.OpSignature.from_op_schema(schema)
        parameters = [
            p for p in signature.params if isinstance(p, ir.schemas.Parameter)
        ]
        for index, value in enumerate(inputs):
            if value is None and index < len(parameters) and parameters[index].required:
                return False
        return True

    @staticmethod
    def _attribute_output_dtype(
        op_type: str,
        domain: str,
        attributes: Mapping[str, Any],
    ) -> ir.DataType | None:
        if domain:
            return None
        if op_type in ("Cast", "BitCast") and attributes.get("to") is not None:
            to_attr = attributes["to"]
            return ir.DataType(
                to_attr.as_int() if isinstance(to_attr, ir.Attr) else to_attr
            )
        if op_type == "ConstantOfShape":
            value = attributes.get("value")
            if value is None:
                return ir.DataType.FLOAT
            if isinstance(value, ir.Attr):
                return value.as_tensor().dtype
            return value.dtype
        if op_type == "Constant":
            value = attributes.get("value")
            if value is not None:
                return (
                    value.as_tensor().dtype
                    if isinstance(value, ir.Attr)
                    else value.dtype
                )
            for suffix, dtype in (
                ("int", ir.DataType.INT64),
                ("float", ir.DataType.FLOAT),
                ("string", ir.DataType.STRING),
            ):
                if (
                    attributes.get(f"value_{suffix}") is not None
                    or attributes.get(f"value_{suffix}s") is not None
                ):
                    return dtype
        return None

    def _warn_missing_standard_schema(
        self,
        op_type: str,
        version: int,
        *,
        stale_since_version: int | None = None,
    ) -> None:
        if op_type in self._warned_missing_schemas:
            return
        self._warned_missing_schemas.add(op_type)
        available = onnx.defs.onnx_opset_version()
        if stale_since_version is not None:
            detail = (
                f"Standard ONNX operator {op_type!r} at opset {version} exceeds "
                f"installed ONNX {onnx.__version__} schemas (default-domain schemas "
                f"available through opset {available}, resolved {op_type} "
                f"since_version={stale_since_version})."
            )
        else:
            detail = (
                f"Standard ONNX operator {op_type!r} at opset {version} was not "
                f"found in installed ONNX {onnx.__version__} schemas (default-domain "
                f"schemas available through opset {available})."
            )
        warnings.warn(
            f"{detail} Constructing the node without target-opset schema validation. "
            "Upgrade the 'onnx' package if you need onnx.checker or "
            "reference-evaluator validation for this operator.",
            UserWarning,
            stacklevel=3,
        )

    def _call_without_schema(
        self,
        op_type: str,
        inputs: tuple[Any, ...],
        *,
        version: int,
        num_outputs: int | None,
        output_types: Sequence[ir.TypeProtocol | None] | None,
        attributes: Mapping[str, Any],
        stale_since_version: int | None = None,
    ) -> tuple[ir.Value, ...]:
        if num_outputs is not None and (
            type(num_outputs) is not int or num_outputs < 0
        ):
            raise ValueError(
                f"{op_type} _num_outputs must be a nonnegative integer, "
                f"got {num_outputs!r}."
            )
        attribute_dtype = self._attribute_output_dtype(op_type, "", attributes)
        default_type = (
            ir.TensorType(attribute_dtype) if attribute_dtype is not None else None
        )
        if output_types is not None:
            if num_outputs is not None and num_outputs != len(output_types):
                raise ValueError(
                    "_num_outputs disagrees with _output_types. Supply matching lengths."
                )
            resolved_output_types: Sequence[ir.TypeProtocol | None] = [
                explicit if explicit is not None else default_type
                for explicit in output_types
            ]
            if default_type is not None:
                for index, explicit in enumerate(output_types):
                    if explicit is not None and explicit != default_type:
                        raise TypeError(
                            f"{op_type} output {index} type {explicit} conflicts with "
                            f"attribute dtype {default_type}. Correct _output_types."
                        )
        elif num_outputs is not None:
            resolved_output_types = [default_type] * num_outputs
        else:
            available = onnx.defs.onnx_opset_version()
            raise ValueError(
                f"Cannot determine output arity for standard ONNX operator {op_type!r} "
                f"at opset {version} because installed ONNX {onnx.__version__} only "
                f"provides default-domain schemas through opset {available}. Pass "
                "_num_outputs= / _output_types= on GraphBuilder.call (or use "
                "TranslationContext, which defaults to 1 output), or check the "
                "operator spelling."
            )
        bound_dtype = next(
            (
                value.dtype
                for value in inputs
                if isinstance(value, ir.Value) and value.dtype is not None
            ),
            None,
        )
        prepared: list[ir.Value | tuple[Any, ir.DataType] | None] = []
        for index, value in enumerate(inputs):
            if value is None or isinstance(value, ir.Value):
                prepared.append(value)
                continue
            dtype = bound_dtype or self._constant_dtype(
                value, op_type, f"input[{index}]"
            )
            prepared.append((value, dtype))
        self._warn_missing_standard_schema(
            op_type,
            version,
            stale_since_version=stale_since_version,
        )
        resolved = [
            self.constant(value[0], dtype=value[1])
            if isinstance(value, tuple)
            else value
            for value in prepared
        ]
        return self.node(
            op_type,
            resolved,
            domain="",
            version=version,
            attributes=attributes,
            output_types=resolved_output_types,
        )

    def _call_with_schema(
        self,
        schema: onnx.defs.OpSchema,
        op_type: str,
        inputs: tuple[Any, ...],
        *,
        domain: str,
        version: int,
        num_outputs: int | None,
        output_types: Sequence[ir.TypeProtocol | None] | None,
        attributes: Mapping[str, Any],
    ) -> tuple[ir.Value, ...]:
        if not schema.min_input <= len(inputs) <= schema.max_input:
            raise ValueError(
                f"{op_type} expects {schema.min_input}..{schema.max_input} inputs, "
                f"got {len(inputs)}. Supply optional holes as None and expand variadics."
            )
        signature = ir.schemas.OpSignature.from_op_schema(schema)
        parameters = [
            p for p in signature.params if isinstance(p, ir.schemas.Parameter)
        ]
        bindings: dict[ir.schemas.TypeConstraintParam, ir.TypeProtocol] = {}
        expanded = []
        for index, value in enumerate(inputs):
            param = parameters[min(index, len(parameters) - 1)]
            expanded.append((param, value))
            if value is None:
                if param.required:
                    raise ValueError(
                        f"{op_type} input {param.name!r} is required. Supply a typed value."
                    )
                continue
            if not isinstance(value, ir.Value):
                continue
            if value.type is None:
                continue
            constraint = param.type_constraint
            if value.type not in constraint.allowed_types:
                raise TypeError(
                    f"{op_type} input {param.name!r} has type {value.type}, expected "
                    f"{constraint.allowed_types}. Insert an explicit Cast."
                )
            if param.homogeneous:
                previous = bindings.get(constraint)
                if previous is not None and previous != value.type:
                    raise TypeError(
                        f"{op_type} input {param.name!r} has type {value.type}, "
                        f"but {constraint.name} is bound to {previous}. "
                        "Insert an explicit Cast; tensor promotion is not automatic."
                    )
                bindings[constraint] = value.type

        tensor_bindings = set(bindings)
        prepared: list[ir.Value | tuple[Any, ir.DataType] | None] = []
        for param, value in expanded:
            if value is None or isinstance(value, ir.Value):
                prepared.append(value)
                continue
            constraint = param.type_constraint
            type_ = (
                bindings.get(constraint)
                if param.homogeneous and constraint in tensor_bindings
                else None
            )
            if type_ is None and len(constraint.allowed_types) == 1:
                type_ = next(iter(constraint.allowed_types))
            if type_ is None:
                type_ = ir.TensorType(self._constant_dtype(value, op_type, param.name))
            if (
                not isinstance(type_, ir.TensorType)
                or type_ not in constraint.allowed_types
            ):
                raise TypeError(
                    f"{op_type} input {param.name!r} cannot convert a Python constant "
                    f"to {type_}. Build sequences and optionals explicitly with ONNX nodes."
                )
            if param.homogeneous:
                previous = bindings.get(constraint)
                if previous is not None and previous != type_:
                    raise TypeError(
                        f"{op_type} constant input {param.name!r} inferred {type_}, "
                        f"but {constraint.name} is bound to {previous}. "
                        "Supply explicitly typed constants for mixed Python dtypes."
                    )
                bindings[constraint] = type_
            prepared.append((value, type_.dtype))

        if num_outputs is not None and (
            type(num_outputs) is not int or num_outputs < 0
        ):
            raise ValueError(
                f"{op_type} _num_outputs must be a nonnegative integer, "
                f"got {num_outputs!r}."
            )
        if output_types is not None:
            if num_outputs is not None and num_outputs != len(output_types):
                raise ValueError(
                    "_num_outputs disagrees with _output_types. Supply matching lengths."
                )
            num_outputs = len(output_types)
        if num_outputs is None:
            if schema.min_output != schema.max_output:
                raise ValueError(
                    f"{op_type} has optional/variadic outputs. "
                    "Pass _num_outputs= or _output_types= explicitly."
                )
            num_outputs = schema.min_output
        if not schema.min_output <= num_outputs <= schema.max_output:
            raise ValueError(
                f"{op_type} requires {schema.min_output}..{schema.max_output} outputs, "
                f"got {num_outputs}. Correct num_outputs or output_types."
            )
        for name in attributes:
            if name not in schema.attributes:
                raise ValueError(
                    f"{op_type} has no attribute {name!r}. Check its ONNX schema."
                )
        for name, attribute in schema.attributes.items():
            if attribute.required and attributes.get(name) is None:
                raise ValueError(
                    f"{op_type} requires attribute {name!r}. Supply its value."
                )
        typed_attributes = {
            name: ir.convenience.convert_attribute(
                name, value, ir.AttributeType(schema.attributes[name].type)
            )
            for name, value in attributes.items()
            if value is not None
        }
        # Seed explicit output types and dtype-defining attributes before the
        # node is passed to symbolic type/shape inference.
        attribute_dtype = self._attribute_output_dtype(
            op_type, domain, typed_attributes
        )
        default_type = (
            ir.TensorType(attribute_dtype) if attribute_dtype is not None else None
        )
        resolved_outputs: list[ir.TypeProtocol | None] = []
        for index in range(num_outputs):
            type_ = default_type
            if output_types is not None and output_types[index] is not None:
                explicit = output_types[index]
                if type_ is not None and explicit != type_:
                    raise TypeError(
                        f"{op_type} output {index} type {explicit} conflicts with "
                        f"attribute dtype {type_}. Correct _output_types."
                    )
                type_ = explicit
            resolved_outputs.append(type_)
        resolved = [
            self.constant(value[0], dtype=value[1])
            if isinstance(value, tuple)
            else value
            for value in prepared
        ]
        return self.node(
            op_type,
            resolved,
            domain=domain,
            version=version,
            attributes=typed_attributes,
            output_types=resolved_outputs,
        )

    @staticmethod
    def _constant_dtype(value: Any, op_type: str, name: str) -> ir.DataType:
        if isinstance(value, bool):
            return ir.DataType.BOOL
        if isinstance(value, int):
            return ir.DataType.INT64
        if isinstance(value, float):
            return ir.DataType.FLOAT
        if isinstance(value, str):
            return ir.DataType.STRING
        if isinstance(value, list | tuple) and value:
            dtypes = {
                GraphBuilder._constant_dtype(item, op_type, name) for item in value
            }
            if len(dtypes) == 1:
                return dtypes.pop()
        raise TypeError(
            f"{op_type} input {name!r} has no unambiguous Python constant dtype. "
            "Use constant(value, dtype=...) for empty/mixed data. Construct symbolic "
            "shape lists explicitly with Reshape/Concat."
        )

    def call_function(
        self,
        function: ir.Function,
        *inputs: ir.Value,
        version: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> tuple[ir.Value, ...]:
        r"""Call an explicit function; expose definitions for caller model assembly."""
        if len(inputs) != len(function.inputs):
            raise ValueError(
                f"Function {function.name} expects {len(function.inputs)} inputs, "
                f"got {len(inputs)}. Supply all function inputs."
            )
        for value, formal in zip(inputs, function.inputs, strict=False):
            if (
                formal.type is not None
                and value.type is not None
                and value.type != formal.type
            ):
                raise TypeError(
                    f"Function {function.name} input {formal.name} expects {formal.type}, "
                    f"got {value.type}. Set types or insert an explicit Cast."
                )
        supplied = attributes or {}
        for name in supplied:
            if name not in function.attributes:
                raise ValueError(
                    f"Function {function.name} has no attribute {name!r}. "
                    "Supply only attributes declared by the function."
                )
        typed_attributes = {}
        for name, formal in function.attributes.items():
            if supplied.get(name) is not None:
                typed_attributes[name] = ir.convenience.convert_attribute(
                    name, supplied[name], formal.type
                )
            elif formal.value is None:
                raise ValueError(
                    f"Function {function.name} requires attribute {name!r}. "
                    "Supply it in attributes=."
                )
        identifier = function.identifier()
        previous = self.functions.get(identifier)
        if previous is not None and previous is not function:
            raise ValueError(
                f"Function {identifier} is already defined. Reuse it or choose another name."
            )
        outputs = self.node(
            function.name,
            inputs,
            domain=function.domain,
            version=version,
            overload=function.overload,
            attributes=typed_attributes,
            output_types=[value.type for value in function.outputs],
        )
        self.functions[identifier] = function
        return outputs
