# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR embedding translations."""

from __future__ import annotations

import onnx_ir as ir
import torch

from torch_onnx._ir_builder import TranslationContext
from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import common as common_ops

aten = torch.ops.aten


@onnx_impl(aten.embedding.default)
def aten_embedding(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> ir.Value:
    # embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor

    return op.Gather(weight, indices)


@onnx_impl(aten.embedding_renorm.default)
def aten_embedding_renorm(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    max_norm: float,
    norm_type: float = 2.0,
) -> ir.Value:
    """embedding_renorm(Tensor weight, Tensor indices, float max_norm, float norm_type) -> Tensor"""

    unique_indices, _, _, _ = op.Unique(indices, outputs=4)
    partial_weight = op.Gather(weight, unique_indices)
    # partial_weight_norm = sum(|w|^p)^(1/p)
    if norm_type == 1.0:
        # This is not necessary, but op.ReduceL1 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL1(partial_weight, [1], keepdims=True)
    elif norm_type == 2.0:
        # This is not necessary, but op.ReduceL2 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL2(partial_weight, [1], keepdims=True)
    else:
        # Abs -> Pow -> ReduceSum -> Pow -> Pow
        partial_weight_abs = op.Abs(partial_weight)
        partial_weight_pow = op.Pow(
            partial_weight_abs,
            op.constant(norm_type, dtype=ir.DataType.FLOAT),
        )
        partial_weight_norm = op.ReduceSum(partial_weight_pow, [1], keepdims=True)
        pow_value = op.cast_like(1.0 / norm_type, weight)
        partial_weight_norm = op.Pow(partial_weight_norm, pow_value)

    max_norm_value = op.cast_like(max_norm, weight)
    # This is to avoid weight is zero
    err = op.cast_like(1e-7, weight)
    partial_weight_norm_ = op.Add(partial_weight_norm, err)
    scales = op.Div(max_norm_value, partial_weight_norm_)
    partial_weight_renorm = op.Mul(partial_weight, scales)
    # Set values to renormed values where weight_norm > max_norm, but keep the original values where weight_norm <= max_norm
    partial_weight_renorm = op.Where(
        op.Greater(partial_weight_norm, max_norm_value),
        partial_weight_renorm,
        partial_weight,
    )
    value = op.ScatterND(
        weight,
        op.Unsqueeze(unique_indices, [1]),
        partial_weight_renorm,
    )
    return value


@onnx_impl(aten.embedding_bag.default)
def aten_embedding_bag(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    offsets: ir.Value,
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: ir.Value | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)"""
    return aten_embedding_bag_padding_idx(
        op,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@onnx_impl(
    (
        aten.embedding_bag.padding_idx,
        aten._embedding_bag.default,
        aten._embedding_bag_forward_only.default,
    )
)
def aten_embedding_bag_padding_idx(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    offsets: ir.Value,
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: ir.Value | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)

    We add default values for the attributes to accommodate _embedding_bag as well:
    _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1)
    """
    del scale_grad_by_freq, sparse
    builder = op.builder
    if mode not in (0, 1, 2):
        raise ValueError(
            f"embedding_bag mode must be 0 (sum), 1 (mean), or 2 (max), got {mode}."
        )
    if per_sample_weights is not None and mode != 0:
        raise ValueError("embedding_bag per_sample_weights requires mode=0 (sum).")
    # assert(rank(indices) in [1,2])
    # assert(rank(offsets) == 1)
    # assert(op.Size(per_sample_weights) == op.Size(indices))
    neg_1 = op.constant([-1], dtype=ir.DataType.INT64)
    # Assume indices is shape(5,2), indices_1d is shape(10,)
    indices_1d = op.Reshape(indices, neg_1)
    indices_size = op.Shape(indices_1d)
    if per_sample_weights is None:
        # Set per_sample_weights to 1.0, because cannot check 'None' in ONNX-Script
        # Size of persample_weights is the same as indices, and should be 1d tensor
        per_sample_weights = op.Expand(1, indices_size)
        # Dtype of per_sample_weights is the same as weight
        per_sample_weights = op.cast_like(per_sample_weights, weight)
    else:
        per_sample_weights = op.Reshape(per_sample_weights, neg_1)
    # Get weight out according to indices_1d,
    # Get weight out according to indices,
    # e.g. indices=[3,1,4,5,3] means get weight[[3,1,4,5,3]]
    new_weight = op.Gather(weight, indices_1d)
    # This happens after first step of Gather. Because Shape(indices)==Shape(per_sample_weights)
    new_weight = op.Mul(new_weight, op.Unsqueeze(per_sample_weights, [1]))
    weight_dim_1 = op.Shape(weight, start=1, end=2)
    normalized_indices = indices_1d
    # When padding_idx is None, use the standard embedding_bag implementation
    if padding_idx is not None:
        num_embeddings = op.Shape(
            weight, start=0, end=1
        )  # Get number of rows in weight
        num_embeddings_scalar = op.Squeeze(num_embeddings)
        if padding_idx < 0:
            padding_idx = op.Add(padding_idx, num_embeddings_scalar)
        normalized_indices = op.Where(
            op.Less(indices_1d, 0),
            op.Add(indices_1d, num_embeddings_scalar),
            indices_1d,
        )

    # Assume indices is shape(5,2) reshape to (10,), offsets=[0,2,3], include_last_offset = False
    # [0,2,3] -> [0:2], [2:3], [3:10]
    num_bag = op.Size(offsets)  # 3 bags, means 10 is the last index
    if include_last_offset:
        num_bag = op.Sub(num_bag, 1)  # 2 bags, means 3 is the last index
        if padding_idx is not None:
            # Change(by ScatterElement setting) the last element to 'end'
            # [0,2,3] -> [0,2,end]
            offsets = op.ScatterElements(offsets, [-1], indices_size)
    else:
        # Change [0,2,3] -> [0,2,3,end], means [0:2],[2:3],[3:end]
        offsets = op.Concat(offsets, indices_size, axis=0)  # Replace end with number
    # Loop scan outputs have fixed per-bag shapes, so no tensor sequences or
    # repeated append/concatenate allocations are needed.
    new_weight = op.Cast(new_weight, to=ir.DataType.FLOAT)
    # Used for iterator
    i = ir.Value(
        name="bag_iteration", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
    )
    condition = ir.Value(
        name="bag_condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
    )
    body = ir.Graph([i, condition], [], nodes=[], name="embedding_bag_body")
    local = builder.subgraph(body)
    local_op = TranslationContext(local)
    # Process each bag
    start_pos = local_op.Gather(offsets, i)
    end_pos = local_op.Gather(offsets, local_op.Add(i, 1))
    last_bag = local_op.Equal(i, local_op.Sub(num_bag, 1))
    range_end = end_pos
    if mode == 2 and padding_idx is None:
        # max
        # The last bag
        range_end = local_op.Where(last_bag, local_op.Size(indices_1d), end_pos)
    positions = local_op.Range(start_pos, range_end, 1)
    # empty tensor
    curr_indices = local_op.Gather(normalized_indices, positions)
    if padding_idx is not None:
        # Call the existing function for handling padding_idx
        # Something like the 'append' operation: keep only positions not equal to padding_idx.
        keep = local_op.Not(local_op.Equal(curr_indices, padding_idx))
        positions = local_op.Compress(positions, keep, axis=0)
        curr_indices = local_op.Compress(curr_indices, keep, axis=0)
    row_weight = local_op.Gather(new_weight, positions)
    count = local_op.Size(positions)
    empty = local_op.Equal(count, 0)
    if mode == 2 and padding_idx is None:
        empty = local_op.Equal(start_pos, end_pos)
    expand_shape = local_op.Concat([1], weight_dim_1, axis=0)
    then_branch = ir.Graph([], [], nodes=[], name="embedding_bag_empty")
    sub = builder.subgraph(then_branch)
    sub_op = TranslationContext(sub)
    # Empty input get zero value output, not empty output
    # row_result should be 0, need to generate (1,N) shape tensor with 0 values
    zero_row = sub_op.Expand(sub.constant([0.0], dtype=ir.DataType.FLOAT), expand_shape)
    zero_max = sub_op.Expand(sub.constant([0], dtype=ir.DataType.INT64), expand_shape)
    zero_row.type = ir.TensorType(ir.DataType.FLOAT)
    zero_max.type = ir.TensorType(ir.DataType.INT64)
    common_ops.finish_subgraph(op, then_branch, [zero_row, zero_max])
    else_branch = ir.Graph([], [], nodes=[], name="embedding_bag_nonempty")
    sub = builder.subgraph(else_branch)
    sub_op = TranslationContext(sub)
    if mode == 0:  # sum
        row_result = sub_op.ReduceSum(row_weight, [0])
    elif mode == 1:  # mean
        if padding_idx is None:
            # The last bag
            row_result = sub_op.ReduceSum(row_weight, [0])
            # When include_last_offset=False, offsets=[0,2,3] -> [0,2,3,10], denominator=10-3=7
            # When include_last_offset=True, offsets=[0,2,3], denominator=10-2=8
            denominator = sub_op.Sub(
                sub_op.Gather(sub_op.Shape(indices), 0),
                start_pos,
            )
            denominator = sub_op.Where(last_bag, denominator, count)
            denominator = sub_op.cast_like(sub_op.Max(denominator, 1), row_weight)
            row_result = sub_op.Div(row_result, denominator)
        else:
            row_result = sub_op.ReduceMean(row_weight, [0])
    else:  # mode == 2, max
        # max
        row_result = sub_op.ReduceMax(row_weight, [0])
    if mode == 2:
        argmax = sub_op.ArgMax(row_weight, axis=0, keepdims=0)
        max_indices = sub_op.Unsqueeze(sub_op.Gather(curr_indices, argmax), [0])
    else:
        max_indices = sub_op.Expand(
            sub.constant([0], dtype=ir.DataType.INT64), expand_shape
        )
    row_result.type = ir.TensorType(ir.DataType.FLOAT)
    max_indices.type = ir.TensorType(ir.DataType.INT64)
    common_ops.finish_subgraph(op, else_branch, [row_result, max_indices])
    row_result, max_indices = local.node(
        "If",
        [empty],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[
            ir.TensorType(ir.DataType.FLOAT),
            ir.TensorType(ir.DataType.INT64),
        ],
    )
    row_result = local_op.Squeeze(row_result, [0])
    max_indices = local_op.Squeeze(max_indices, [0])
    scan_count = count
    if mode == 1 and padding_idx is None:
        # ATen reports the remaining indices as the last mean bag size even
        # when an explicit end offset truncates its numerator.
        scan_count = local_op.Where(
            last_bag, local_op.Sub(local_op.Size(indices_1d), start_pos), count
        )
    common_ops.finish_subgraph(
        op, body, [condition, row_result, scan_count, max_indices]
    )
    result, bag_size, max_indices = builder.node(
        "Loop",
        [num_bag, op.constant(True, dtype=ir.DataType.BOOL)],
        attributes={"body": body},
        output_types=[
            ir.TensorType(ir.DataType.FLOAT),
            ir.TensorType(ir.DataType.INT64),
            ir.TensorType(ir.DataType.INT64),
        ],
    )
    # An unexecuted scan may have an unspecified trailing shape at runtime.
    # Explicit target shapes also cover zero bags and zero embedding width.
    result_shape = op.Concat(op.Reshape(num_bag, [1]), weight_dim_1, axis=0)
    result = op.Reshape(result, result_shape, allowzero=1)
    result = op.cast_like(result, weight)
    bag_size = op.Reshape(bag_size, op.Reshape(num_bag, [1]), allowzero=1)
    max_indices = op.Reshape(max_indices, result_shape, allowzero=1)
    # Preserve the auxiliary outputs for consumers that use them.
    if mode == 0 and padding_idx is None:
        offset2bag = op.constant([], dtype=ir.DataType.INT64)
    else:
        indices_length = op.Gather(op.Shape(indices), 0)
        starts = op.Slice(offsets, [0], op.Reshape(num_bag, [1]))
        # Count bag starts at each index, including duplicate starts for empty
        # bags. Prefix sums assign each position to its latest bag in O(N+B)
        # space/work instead of broadcasting an N-by-B membership matrix.
        markers = op.Expand(
            op.constant(0, dtype=ir.DataType.INT64),
            op.Reshape(op.Add(indices_length, 1), [1]),
        )
        starts_count = op.Expand(
            op.constant(1, dtype=ir.DataType.INT64), op.Shape(starts)
        )
        markers = op.ScatterElements(markers, starts, starts_count, reduction="add")
        offset2bag = op.Sub(
            op.Slice(op.CumSum(markers, 0), [0], op.Reshape(indices_length, [1])),
            1,
        )
    if mode == 0:  # sum
        bag_size = op.Expand(
            op.constant(0, dtype=ir.DataType.INT64),
            op.Shape(offsets),
        )
        if not include_last_offset:
            bag_size = op.Slice(bag_size, [0], [-1])
        max_indices = op.Identity(bag_size)
    elif mode == 1:  # mean
        max_indices = op.Identity(bag_size)
    # shape = (bag_size.dim[0], weight.dim[1])
    return result, offset2bag, bag_size, max_indices
