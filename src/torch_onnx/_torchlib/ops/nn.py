"""torch.ops.aten operators under the `core` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
import torch

from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import _promotion, common

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_onnx._ir_builder import TranslationContext


aten = torch.ops.aten


def _validate_attention_heads(
    op: TranslationContext,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    *,
    enable_gqa: bool,
) -> None:
    shapes = [common.require_shape(op, tensor) for tensor in (query, key, value)]
    if any(len(shape) < 3 for shape in shapes):
        raise ValueError(
            "SDPA head validation requires query, key and value with at least 3 dimensions."
        )
    q, k, v = (shape[-3] for shape in shapes)
    if not (isinstance(q, int) and isinstance(k, int) and isinstance(v, int)):
        # Dynamic head counts are checked by the ONNX runtime, not Python comparisons.
        return
    if enable_gqa:
        if not (k > 0 and q > k == v and q % k == 0):
            raise ValueError(
                "SDPA (GQA or MQA) requires positive kv heads, q_num_heads > kv_num_heads and divisible head counts."
            )
    elif not q == k == v:
        raise ValueError(
            "SDPA (MHA) requires matching query, key and value head counts."
        )


@onnx_impl(aten.gelu.default, opset_introduced=20)
def aten_gelu_opset20(
    op: TranslationContext,
    self: ir.Value,
    approximate: str = "none",
) -> ir.Value:
    """gelu(Tensor self, *, str approximate="none") -> Tensor"""
    return op.Gelu(self, approximate=approximate)


@onnx_impl(aten.group_norm.default, opset_introduced=21)
def aten_group_norm(
    op: TranslationContext,
    input: ir.Value,
    num_groups: int,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> ir.Value:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    c = op.Shape(input, start=1, end=2)
    if weight is None:
        weight = op.ConstantOfShape(c, value=ir.tensor([1.0], dtype=input.dtype))
    if bias is None:
        bias = op.ConstantOfShape(c, value=ir.tensor([0.0], dtype=input.dtype))
    result = op.GroupNormalization(
        input, weight, bias, epsilon=eps, num_groups=num_groups
    )
    result.type = input.type
    result.shape = input.shape
    return result


@onnx_impl(aten.rms_norm.default, opset_introduced=23)
def aten_rms_norm(
    op: TranslationContext,
    input: ir.Value,
    normalized_shape: Sequence[int],
    weight: ir.Value | None = None,
    eps: float | None = None,
) -> ir.Value:
    """rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor"""

    # Default eps value if not provided
    if eps is None:
        eps = torch.finfo(_promotion.source_dtype(op, input)).eps

    # Calculate axis: the first normalization dimension
    # For normalized_shape with D dimensions, normalize over last D dimensions
    # Since ONNX RMSNormalization supports negative axis values, we use -len(normalized_shape)
    # which correctly maps to the first axis of the normalized dimensions
    normalized_dims = len(normalized_shape)
    axis = -normalized_dims

    # Create weight tensor if not provided
    if weight is None:
        weight = op.ConstantOfShape(
            op.Shape(input, start=axis),
            value=ir.tensor([1], dtype=input.dtype),
        )

    return op.RMSNormalization(input, weight, axis=axis, epsilon=eps)


@onnx_impl(aten.scaled_dot_product_attention.default, opset_introduced=23)
def aten_scaled_dot_product_attention_23(
    op: TranslationContext,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> ir.Value:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor

    Reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        2. https://onnx.ai/onnx/operators/onnx__Attention.html

    Attempts to convert SDPA to Attention onnx op and fallbacks to an onnx graph equivalent to the following PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = (
            torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            if is_causal
            else attn_mask
        )
        attn_mask = (
            attn_mask.masked_fill(not attn_mask, -float("inf"))
            if attn_mask.dtype == torch.bool
            else attn_mask
        )
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    if is_causal and attn_mask is not None:
        raise AssertionError("is_causal and attn_mask cannot be set at the same time")
    if any(
        len(common.require_shape(op, tensor)) != 4 for tensor in (query, key, value)
    ):
        # ONNX Attention has a different 3D layout (packed multi-head features).
        # Use PyTorch's rank-polymorphic math semantics for other input ranks.
        # Import locally because neural also uses the shared attention helpers.
        from torch_onnx._torchlib.ops import neural

        return neural.aten_scaled_dot_product_attention(
            op, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    _validate_attention_heads(op, query, key, value, enable_gqa=enable_gqa)

    # Attention onnx op can only handle non-training scenarios where dropout is disabled.
    if dropout_p == 0:
        # NOTE: num_heads attributes (q_num_heads/kv_num_heads) should not be specified for 4D.
        # They are not populated with 4D inputs because this information directly comes from input shapes:
        # `q_num_heads=query.shape[1]` and `kv_num_heads=key.shape[1]`.
        # This dimension is usually static but it could not be dynamic if also given as an attribute.
        # num_heads attributes are needed for 3D attention inputs:
        # (shape: [B, S, N*H]), 4D shape is ([B, N, S, H]).

        Y, _, _, _ = op.Attention(
            query,
            key,
            value,
            attn_mask,
            outputs=4,
            scale=scale,
            is_causal=is_causal,
        )
        return Y

    scale_value = op.cast_like(
        _attention_scale(op, query) if scale is None else scale, query
    )

    if is_causal:
        attn_mask = _causal_attention_mask(op, query, key)

    if enable_gqa:
        key, value = _attention_repeat_kv_for_group_query(op, query, key, value)

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            op, query, key, value, scale_value, dropout_p
        )

    return _aten_scaled_dot_product_attention_float_mask_onnx(
        op, query, key, value, attn_mask, scale_value, dropout_p
    )


def _attention_repeat_kv_for_group_query(
    op: TranslationContext, query: ir.Value, key: ir.Value, value: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """Expand key and value for group query attention.

    repeat_interleave is applied on key and value to match the number of heads in query.

    Args:
        query: Tensor of shape [..., q_num_heads, q_S, E]
        key: Tensor of shape [..., k_num_heads, kv_S, E]
        value: Tensor of shape [..., v_num_heads, kv_S, Ev]

    Returns:
        Tuple of (expanded_key, expanded_value) where:
            - expanded_key: Tensor of shape [..., q_num_heads, kv_S, E]
            - expanded_value: Tensor of shape [..., q_num_heads, kv_S, Ev]
    """

    _validate_attention_heads(op, query, key, value, enable_gqa=True)

    q_num_heads = op.Shape(query, start=-3, end=-2)

    def repeat(tensor: ir.Value) -> ir.Value:
        # Retain each tensor's batch prefix so MatMul can broadcast it later.
        prefix = op.Shape(tensor, end=-3)
        heads = op.Shape(tensor, start=-3, end=-2)
        tail = op.Shape(tensor, start=-2)
        repeats = op.Div(q_num_heads, heads)
        expanded = op.Expand(
            op.Unsqueeze(tensor, [-3]),
            op.Concat(prefix, heads, repeats, tail, axis=0),
        )
        return op.Reshape(
            expanded, op.Concat(prefix, q_num_heads, tail, axis=0), allowzero=1
        )

    return repeat(key), repeat(value)


def _attention_scale(op: TranslationContext, query: ir.Value) -> ir.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    q_shape = op.Shape(query)
    q_last_dim = op.Gather(q_shape, op.Constant(value_ints=[-1]))
    embedding_size = op.cast_like(q_last_dim, query)
    cast_one = op.cast_like(1.0, query)
    scale = op.Div(cast_one, op.Sqrt(embedding_size))
    return scale


def _causal_attention_mask(
    op: TranslationContext, query: ir.Value, key: ir.Value
) -> ir.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float("inf"))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    q_shape = op.Shape(query)
    k_shape = op.Shape(key)

    target_length = op.Slice(
        q_shape,
        op.Constant(value_ints=[-2]),
        op.Constant(value_ints=[-1]),
    )
    source_length = op.Slice(
        k_shape,
        op.Constant(value_ints=[-2]),
        op.Constant(value_ints=[-1]),
    )
    # attn_mask = torch.ones(L, S) := {
    size = op.Concat(target_length, source_length, axis=0)
    attn_mask = op.Expand(op.Constant(value_float=1.0), size)
    # }
    attn_mask = op.Trilu(attn_mask, upper=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    attn_mask = op.Where(
        op.Equal(attn_mask, op.Constant(value_float=0.0)),
        op.Constant(value_float=-float("inf")),
        op.Constant(value_float=0.0),
    )
    attn_mask = op.cast_like(attn_mask, query)
    return attn_mask


def _aten_scaled_dot_product_attention_no_mask_onnx(
    op: TranslationContext,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    scale: ir.Value,
    dropout_p: float,
) -> ir.Value:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]),
        key_second_last_dim,
        key_last_dim,
        axis=0,
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(
        key_transposed,
        op.cast_like(op.Sqrt(scale), key_transposed),
    )
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p, outputs=2)
    return op.MatMul(attn_weight, value)


def _aten_scaled_dot_product_attention_float_mask_onnx(
    op: TranslationContext,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value,
    scale: ir.Value,
    dropout_p: float,
) -> ir.Value:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]),
        key_second_last_dim,
        key_last_dim,
        axis=0,
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.Add(
            op.MatMul(query_scaled, key_transposed_scaled),
            attn_mask,
        ),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p, outputs=2)
    return op.MatMul(attn_weight, value)
