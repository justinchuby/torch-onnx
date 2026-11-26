# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR vision translations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import onnx_ir as ir

from torch_onnx._torchlib._torchlib_registry import onnx_impl
from torch_onnx._torchlib.ops import common as common_ops

if TYPE_CHECKING:
    from torch_onnx._ir_builder import TranslationContext


_INT64_MAX = 9223372036854775807


@onnx_impl("torchvision::nms", opset_introduced=18)
def torchvision_nms(
    op: TranslationContext, boxes: ir.Value, scores: ir.Value, iou_threshold: float
) -> ir.Value:
    """nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor"""
    # boxes: [num_batches, spatial_dimension, 4]
    boxes = op.Unsqueeze(boxes, [0])
    # scores: [num_batches, num_classes, spatial_dimension]
    scores = op.Unsqueeze(scores, [0, 1])
    # nms_out: [num_selected_indices, 3] where each column is [batch_index, class_index, box_index]
    nms_out = op.NonMaxSuppression(boxes, scores, _INT64_MAX, iou_threshold)
    return op.Reshape(op.Slice(nms_out, [2], [3], [1]), [-1])


def _process_batch_indices_for_roi_align(op: TranslationContext, rois):
    # Extract batch indices from the first column (index 0) of rois
    indices = op.Slice(rois, [0], [1], [1])
    indices = op.Squeeze(indices, [1])
    return op.Cast(indices, to=ir.DataType.INT64)


def _process_rois_for_roi_align(op: TranslationContext, rois):
    # Extract roi coordinates from columns 1, 2, 3, 4 (x1, y1, x2, y2)
    return op.Slice(rois, [1], [5], [1])


def _process_sampling_ratio_for_roi_align(op: TranslationContext, sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn(
            "ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. "
            "The model will be exported with a sampling_ratio of 0.",
            stacklevel=2,
        )
        sampling_ratio = 0
    return sampling_ratio


@onnx_impl("torchvision::roi_align", opset_introduced=18)
def torchvision_roi_align(
    op: TranslationContext,
    input,
    rois,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    """torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor"""
    batch_indices = _process_batch_indices_for_roi_align(op, rois)
    rois_coords = _process_rois_for_roi_align(op, rois)
    coordinate_transformation_mode = "half_pixel" if aligned else "output_half_pixel"
    sampling_ratio = _process_sampling_ratio_for_roi_align(op, sampling_ratio)

    return op.RoiAlign(
        input,
        rois_coords,
        batch_indices,
        coordinate_transformation_mode=coordinate_transformation_mode,
        spatial_scale=spatial_scale,
        output_height=pooled_height,
        output_width=pooled_width,
        sampling_ratio=sampling_ratio,
    )


@onnx_impl("torchvision::ps_roi_align", opset_introduced=18)
def torchvision_ps_roi_align(
    op: TranslationContext,
    input: ir.Value,
    rois: ir.Value,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int,
):
    channels = input.shape[1] if input.shape is not None else None
    bins = pooled_height * pooled_width
    if not isinstance(channels, int) or channels % bins:
        raise NotImplementedError(
            "torchvision ps_roi_align requires a static channel count divisible by "
            "pooled_height * pooled_width. Export with a static channel dimension."
        )
    aligned = torchvision_roi_align(
        op,
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        True,
    )
    indices = [channel * bins + channel % bins for channel in range(channels)]
    selected = op.Gather(op.Flatten(aligned, axis=1), indices, axis=1)
    shape = [-1, channels // bins, pooled_height, pooled_width]
    result = op.Reshape(selected, shape)
    mapping = op.Reshape(
        op.Constant(value=ir.tensor(list(range(channels)), dtype=ir.DataType.INT32)),
        [1, channels // bins, pooled_height, pooled_width],
    )
    return result, op.Expand(mapping, op.Shape(result))


@onnx_impl("torchvision::roi_pool", opset_introduced=18)
def torchvision_roi_pool(
    op: TranslationContext,
    input: ir.Value,
    boxes: ir.Value,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
) -> tuple[ir.Value, ir.Value]:
    """torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width) -> (Tensor, Tensor)"""
    # MaxRoiPool expects boxes in format [batch_index, x1, y1, x2, y2]
    pooled = op.MaxRoiPool(
        input,
        boxes,
        pooled_shape=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
    )
    input_dtype = common_ops.require_dtype(op, input)
    boxes_dtype = common_ops.require_dtype(op, boxes)
    half = op.constant(0.5, dtype=boxes_dtype)
    one_float = op.constant(1.0, dtype=boxes_dtype)
    shape = op.Shape(input)
    height, width = op.Gather(shape, 2), op.Gather(shape, 3)
    scaled = op.Mul(
        op.Slice(boxes, [1], [5], [1]),
        op.constant(spatial_scale, dtype=boxes_dtype),
    )
    # torchvision rounds half away from zero, unlike ONNX Round.
    coordinates = op.Cast(
        op.Mul(op.Sign(scaled), op.Floor(op.Add(op.Abs(scaled), half))),
        to=ir.DataType.INT64,
    )

    def bins(
        op: TranslationContext,
        start_index: int,
        end_index: int,
        count: int,
        limit: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        start = op.Gather(coordinates, start_index, axis=1)
        end = op.Gather(coordinates, end_index, axis=1)
        extent = op.Max(op.Add(op.Sub(end, start), 1), 1)
        count_float = op.constant(float(count), dtype=boxes_dtype)
        size = op.Div(op.CastLike(extent, boxes), count_float)
        positions = op.CastLike(op.Range(0, count, 1), boxes)
        size = op.Unsqueeze(size, [1])
        start = op.Unsqueeze(start, [1])
        first = op.Add(
            start, op.Cast(op.Floor(op.Mul(size, positions)), to=ir.DataType.INT64)
        )
        last = op.Add(
            start,
            op.Cast(
                op.Ceil(op.Mul(size, op.Add(positions, one_float))),
                to=ir.DataType.INT64,
            ),
        )
        first, last = op.Clip(first, 0, limit), op.Clip(last, 0, limit)
        largest = op.ReduceMax(
            op.Concat(op.Reshape(op.Sub(last, first), [-1]), [1], axis=0),
            keepdims=0,
        )
        return first, last, op.Range(0, largest, 1)

    top, bottom, rows = bins(op, 1, 3, pooled_height, height)
    left, right, columns = bins(op, 0, 2, pooled_width, width)
    rows = op.Add(op.Unsqueeze(top, [2, 3, 4]), op.Unsqueeze(rows, [0, 1, 2, 4]))
    columns = op.Add(op.Unsqueeze(left, [1, 3, 4]), op.Unsqueeze(columns, [0, 1, 2, 3]))
    valid = op.And(
        op.Less(rows, op.Unsqueeze(bottom, [2, 3, 4])),
        op.Less(columns, op.Unsqueeze(right, [1, 3, 4])),
    )
    indices = op.Add(op.Mul(rows, width), columns)
    candidate_shape = op.Shape(indices)
    roi_count = op.Gather(candidate_shape, [0])
    candidate_count = op.ReduceProd(op.Slice(candidate_shape, [1], [5]), keepdims=1)
    bin_count = op.ReduceProd(op.Slice(candidate_shape, [3], [5]), keepdims=1)
    flat_indices = op.Reshape(
        op.Min(indices, op.Sub(op.Mul(height, width), 1)),
        op.Concat(roi_count, [1], candidate_count, axis=0),
    )
    batch = _process_batch_indices_for_roi_align(op, boxes)
    features = op.Reshape(
        op.Gather(input, batch, axis=0),
        op.Concat(
            roi_count,
            op.Gather(shape, [1]),
            op.Reshape(op.Mul(height, width), [1]),
            axis=0,
        ),
    )
    flat_indices = op.Expand(
        flat_indices,
        op.Concat(roi_count, op.Gather(shape, [1]), candidate_count, axis=0),
    )
    candidates = op.Reshape(
        op.GatherElements(features, flat_indices, axis=2),
        op.Concat(op.Shape(pooled), bin_count, axis=0),
    )
    indices = op.Reshape(
        indices,
        op.Concat(roi_count, [1, pooled_height, pooled_width], bin_count, axis=0),
    )
    valid = op.Reshape(valid, op.Shape(indices))
    min_value = op.constant(input_dtype.min, dtype=input_dtype)
    matches = op.And(valid, op.Equal(candidates, op.Unsqueeze(pooled, [-1])))
    matches = op.And(matches, op.Greater(candidates, min_value))
    # The first maximum in row-major order wins; empty bins have index -1.
    int64_max = op.constant(_INT64_MAX, dtype=ir.DataType.INT64)
    minus_one = op.constant(-1, dtype=ir.DataType.INT64)
    argmax = op.ReduceMin(op.Where(matches, indices, int64_max), [-1], keepdims=0)
    argmax = op.Where(op.Equal(argmax, int64_max), minus_one, argmax)
    return pooled, op.Cast(argmax, to=ir.DataType.INT32)


@onnx_impl("torchvision::deform_conv2d", opset_introduced=19)
def torchvision_deform_conv2d(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None,
    bias: ir.Value | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
    use_mask: bool,
):
    """torchvision::deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor? mask, Tensor? bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> Tensor"""

    return op.DeformConv(
        input,
        weight,
        offset,
        bias,
        mask if use_mask else None,
        dilations=(dilation_h, dilation_w),
        strides=(stride_h, stride_w),
        pads=(pad_h, pad_w, pad_h, pad_w),
        group=groups,
        offset_group=offset_groups,
    )
