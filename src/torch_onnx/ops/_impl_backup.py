from __future__ import annotations

from typing import Sequence

import torch
import torch.fx


def Abs_13(X: torch.Tensor) -> torch.Tensor:
    return torch.abs(X)


def Acos_7(input: torch.Tensor) -> torch.Tensor:
    return torch.acos(input)


def Acosh_9(input: torch.Tensor) -> torch.Tensor:
    return torch.acosh(input)


def Add_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.add(A, B)


def AffineGrid_20(
    theta: torch.Tensor, size: torch.Tensor, *, align_corners: int = 0
) -> torch.Tensor:
    return torch.nn.functional.affine_grid(
        theta, size.numpy(force=True).tolist(), align_corners=bool(align_corners)
    )


def And_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.logical_and(A, B)


def ArgMax_13(
    data: torch.Tensor, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> torch.Tensor:
    result = torch.argmax(data, dim=axis, keepdim=bool(keepdims))
    if select_last_index:
        result = torch.flip(result, dims=[axis])
    return result


def ArgMin_13(
    data: torch.Tensor, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> torch.Tensor:
    result = torch.argmin(data, dim=axis, keepdim=bool(keepdims))
    if select_last_index:
        result = torch.flip(result, dims=[axis])
    return result


def Asin_7(input: torch.Tensor) -> torch.Tensor:
    return torch.asin(input)


def Asinh_9(input: torch.Tensor) -> torch.Tensor:
    return torch.asinh(input)


def Atan_7(input: torch.Tensor) -> torch.Tensor:
    return torch.atan(input)


def Atanh_9(input: torch.Tensor) -> torch.Tensor:
    return torch.atanh(input)


def AveragePool_19(
    X: torch.Tensor,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    count_include_pad: int = 0,
    dilations: list[int] | None = None,
    kernel_shape: list[int],
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def BatchNormalization_15(
    X: torch.Tensor,
    scale: torch.Tensor,
    B: torch.Tensor,
    input_mean: torch.Tensor,
    input_var: torch.Tensor,
    *,
    epsilon: float = 9.999999747378752e-06,
    momentum: float = 0.8999999761581421,
    training_mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.nn.functional.batch_norm(
            X, input_mean, input_var, scale, B, bool(training_mode), momentum, epsilon
        ),
        input_mean,
        input_var,
    )


def Bernoulli_15(
    input: torch.Tensor, *, dtype: int | None = None, seed: float | None = None
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.bernoulli(input).to(dtype)


def BitShift_11(X: torch.Tensor, Y: torch.Tensor, *, direction: str) -> torch.Tensor:
    if direction == "LEFT":
        return torch.bitwise_left_shift(X, Y)
    elif direction == "RIGHT":
        return torch.bitwise_right_shift(X, Y)
    else:
        raise ValueError("direction must be 'LEFT' or 'RIGHT'")


def BitwiseAnd_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(A, B)


def BitwiseNot_18(X: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_not(X)


def BitwiseOr_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_or(A, B)


def BitwiseXor_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_xor(A, B)


def BlackmanWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    return torch.blackman_window(size.item(), periodic=bool(periodic)).to(
        output_datatype
    )


def Cast_21(input: torch.Tensor, *, saturate: int = 1, to: int) -> torch.Tensor:
    return input.to(to)


def CastLike_21(
    input: torch.Tensor, target_type: torch.Tensor, *, saturate: int = 1
) -> torch.Tensor:
    return input.to(target_type.dtype)


def Ceil_13(X: torch.Tensor) -> torch.Tensor:
    return torch.ceil(X)


def Celu_12(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.celu(X, alpha)


def CenterCropPad_18(
    input_data: torch.Tensor, shape: torch.Tensor, *, axes: list[int] | None = None
) -> torch.Tensor:
    return torch.nn.functional.pad(input_data, shape.tolist(), mode="constant", value=0)


def Clip_13(
    input: torch.Tensor,
    min: torch.Tensor | None = None,
    max: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.clamp(input, min=min, max=max)


def Col2Im_18(
    input: torch.Tensor,
    image_shape: torch.Tensor,
    block_shape: torch.Tensor,
    *,
    dilations: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def Compress_11(
    input: torch.Tensor, condition: torch.Tensor, *, axis: int | None = None
) -> torch.Tensor:
    return torch.masked_select(input, condition)


def Concat_13(*inputs: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.cat(inputs, dim=axis)


def Constant_21(
    *,
    value: torch.Tensor | None = None,
    value_float: float | None = None,
    value_floats: list[float] | None = None,
    value_int: int | None = None,
    value_ints: list[int] | None = None,
    value_string: str | None = None,
    value_strings: list[str] | None = None,
) -> torch.Tensor:
    if value is not None:
        return value
    elif value_float is not None:
        return torch.tensor(value_float)
    elif value_floats is not None:
        return torch.tensor(value_floats)
    elif value_int is not None:
        return torch.tensor(value_int)
    elif value_ints is not None:
        return torch.tensor(value_ints)
    elif value_string is not None:
        return torch.tensor(value_string)
    elif value_strings is not None:
        return torch.tensor(value_strings)
    else:
        raise ValueError("No value provided for Constant")


def ConstantOfShape_21(
    input: torch.Tensor, *, value: torch.Tensor | None = None
) -> torch.Tensor:
    if value is None:
        value = torch.tensor(0)
    return torch.full(input.tolist(), value.item())


def Conv_11(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: list[int] | None = None,
    group: int = 1,
    kernel_shape: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    return torch.nn.functional.conv2d(X, W, B, strides, pads, dilations, group)


def ConvInteger_10(
    x: torch.Tensor,
    w: torch.Tensor,
    x_zero_point: torch.Tensor | None = None,
    w_zero_point: torch.Tensor | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: list[int] | None = None,
    group: int = 1,
    kernel_shape: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def ConvTranspose_11(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: list[int] | None = None,
    group: int = 1,
    kernel_shape: list[int] | None = None,
    output_padding: list[int] | None = None,
    output_shape: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    return torch.nn.functional.conv_transpose2d(
        X, W, B, strides, pads, output_padding, group, dilations
    )


def Cos_7(input: torch.Tensor) -> torch.Tensor:
    return torch.cos(input)


def Cosh_9(input: torch.Tensor) -> torch.Tensor:
    return torch.cosh(input)


def CumSum_14(
    x: torch.Tensor, axis: torch.Tensor, *, exclusive: int = 0, reverse: int = 0
) -> torch.Tensor:
    result = torch.cumsum(x, dim=axis.item())
    if exclusive:
        result = torch.cat(
            [torch.zeros_like(result[..., :1]), result[..., :-1]], dim=axis.item()
        )
    if reverse:
        result = torch.flip(result, dims=[axis.item()])
    return result


def DFT_20(
    input: torch.Tensor,
    dft_length: torch.Tensor | None = None,
    axis: torch.Tensor | None = None,
    *,
    inverse: int = 0,
    onesided: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


def DeformConv_19(
    X: torch.Tensor,
    W: torch.Tensor,
    offset: torch.Tensor,
    B: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    dilations: list[int] | None = None,
    group: int = 1,
    kernel_shape: list[int] | None = None,
    offset_group: int = 1,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def DepthToSpace_13(
    input: torch.Tensor, *, blocksize: int, mode: str = "DCR"
) -> torch.Tensor:
    return torch.nn.functional.pixel_shuffle(input, blocksize)


def DequantizeLinear_21(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: torch.Tensor | None = None,
    *,
    axis: int = 1,
    block_size: int = 0,
) -> torch.Tensor:
    return torch.dequantize(x)


def Det_11(X: torch.Tensor) -> torch.Tensor:
    return torch.det(X)


def Div_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.div(A, B)


def Dropout_13(
    data: torch.Tensor,
    ratio: torch.Tensor | None = None,
    training_mode: torch.Tensor | None = None,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.nn.functional.dropout(
        data, p=ratio.item() if ratio is not None else 0.5, training=bool(training_mode)
    ), data


def DynamicQuantizeLinear_11(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def Einsum_12(*Inputs: torch.Tensor, equation: str) -> torch.Tensor:
    return torch.einsum(equation, Inputs)


def Elu_6(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.elu(X, alpha)


def Equal_19(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.eq(A, B)


def Erf_13(input: torch.Tensor) -> torch.Tensor:
    return torch.erf(input)


def Exp_13(input: torch.Tensor) -> torch.Tensor:
    return torch.exp(input)


def Expand_13(input: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    return input.expand(shape.tolist())


def EyeLike_9(
    input: torch.Tensor, *, dtype: int | None = None, k: int = 0
) -> torch.Tensor:
    return torch.eye(input.size(0), input.size(1), dtype=dtype, device=input.device)


def Flatten_21(input: torch.Tensor, *, axis: int = 1) -> torch.Tensor:
    return torch.flatten(input, start_dim=axis)


def Floor_13(X: torch.Tensor) -> torch.Tensor:
    return torch.floor(X)


def GRU_14(
    X: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    B: torch.Tensor | None = None,
    sequence_lens: torch.Tensor | None = None,
    initial_h: torch.Tensor | None = None,
    *,
    activation_alpha: list[float] | None = None,
    activation_beta: list[float] | None = None,
    activations: list[str] | None = None,
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    layout: int = 0,
    linear_before_reset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def Gather_13(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    return torch.gather(data, axis, indices)


def GatherElements_13(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    return torch.gather(data, axis, indices)


def GatherND_13(
    data: torch.Tensor, indices: torch.Tensor, *, batch_dims: int = 0
) -> torch.Tensor:
    return torch.gather(data, batch_dims, indices)


def Gelu_20(X: torch.Tensor, *, approximate: str = "none") -> torch.Tensor:
    return torch.nn.functional.gelu(X, approximate=approximate)


def Gemm_13(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> torch.Tensor:
    if transA:
        A = A.t()
    if transB:
        B = B.t()
    Y = alpha * torch.matmul(A, B)
    if C is not None:
        Y += beta * C
    return Y


def GlobalAveragePool_1(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_avg_pool2d(X, (1, 1))


def GlobalLpPool_2(X: torch.Tensor, *, p: int = 2) -> torch.Tensor:
    return torch.nn.functional.lp_pool2d(X, norm_type=p, kernel_size=X.size()[2:])


def GlobalMaxPool_1(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_max_pool2d(X, (1, 1))


def Greater_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.gt(A, B)


def GreaterOrEqual_16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.ge(A, B)


def GridSample_20(
    X: torch.Tensor,
    grid: torch.Tensor,
    *,
    align_corners: int = 0,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return torch.nn.functional.grid_sample(
        X, grid, mode=mode, padding_mode=padding_mode, align_corners=bool(align_corners)
    )


def GroupNormalization_21(
    X: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    epsilon: float = 9.999999747378752e-06,
    num_groups: int,
    stash_type: int = 1,
) -> torch.Tensor:
    return torch.nn.functional.group_norm(X, num_groups, scale, bias, eps=epsilon)


def HammingWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    return torch.hamming_window(size.item(), periodic=bool(periodic)).to(
        output_datatype
    )


def HannWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    return torch.hann_window(size.item(), periodic=bool(periodic)).to(output_datatype)


def HardSigmoid_6(
    X: torch.Tensor, *, alpha: float = 0.20000000298023224, beta: float = 0.5
) -> torch.Tensor:
    return torch.nn.functional.hardsigmoid(X, alpha, beta)


def HardSwish_14(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.hardswish(X)


def Hardmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    return torch.nn.functional.hardmax(input, dim=axis)


def Identity_21(input: torch.Tensor) -> torch.Tensor:
    return input


def If_21(
    cond: torch.Tensor,
    *,
    else_branch: torch.fx.GraphModule,
    then_branch: torch.fx.GraphModule,
) -> list[torch.Tensor]:
    if cond.item():
        return then_branch()
    else:
        return else_branch()


def ImageDecoder_20(
    encoded_stream: torch.Tensor, *, pixel_format: str = "RGB"
) -> torch.Tensor:
    raise NotImplementedError


def InstanceNormalization_6(
    input: torch.Tensor,
    scale: torch.Tensor,
    B: torch.Tensor,
    *,
    epsilon: float = 9.999999747378752e-06,
) -> torch.Tensor:
    return torch.nn.functional.instance_norm(input, scale, B, eps=epsilon)


def IsInf_20(
    X: torch.Tensor, *, detect_negative: int = 1, detect_positive: int = 1
) -> torch.Tensor:
    return torch.isinf(X) & ((X > 0) if detect_positive else (X < 0))


def IsNaN_20(X: torch.Tensor) -> torch.Tensor:
    return torch.isnan(X)


def LRN_13(
    X: torch.Tensor,
    *,
    alpha: float = 9.999999747378752e-05,
    beta: float = 0.75,
    bias: float = 1.0,
    size: int,
) -> torch.Tensor:
    return torch.nn.functional.local_response_norm(
        X, size, alpha=alpha, beta=beta, k=bias
    )


def LSTM_14(
    X: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    B: torch.Tensor | None = None,
    sequence_lens: torch.Tensor | None = None,
    initial_h: torch.Tensor | None = None,
    initial_c: torch.Tensor | None = None,
    P: torch.Tensor | None = None,
    *,
    activation_alpha: list[float] | None = None,
    activation_beta: list[float] | None = None,
    activations: list[str] | None = None,
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    input_forget: int = 0,
    layout: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def LayerNormalization_17(
    X: torch.Tensor,
    Scale: torch.Tensor,
    B: torch.Tensor | None = None,
    *,
    axis: int = -1,
    epsilon: float = 9.999999747378752e-06,
    stash_type: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.nn.functional.layer_norm(X, X.size()[axis:], Scale, B, eps=epsilon),
        Scale,
        B,
    )


def LeakyRelu_16(
    X: torch.Tensor, *, alpha: float = 0.009999999776482582
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(X, negative_slope=alpha)


def Less_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.lt(A, B)


def LessOrEqual_16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.le(A, B)


def Log_13(input: torch.Tensor) -> torch.Tensor:
    return torch.log(input)


def LogSoftmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    return torch.nn.functional.log_softmax(input, dim=axis)


def Loop_21(
    M: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    *v_initial: torch.Tensor,
    body: torch.fx.GraphModule,
) -> list[torch.Tensor]:
    raise NotImplementedError


def LpNormalization_1(
    input: torch.Tensor, *, axis: int = -1, p: int = 2
) -> torch.Tensor:
    return torch.nn.functional.normalize(input, p=p, dim=axis)


def LpPool_18(
    X: torch.Tensor,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: list[int] | None = None,
    kernel_shape: list[int],
    p: int = 2,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def MatMul_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


def MatMulInteger_10(
    A: torch.Tensor,
    B: torch.Tensor,
    a_zero_point: torch.Tensor | None = None,
    b_zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def Max_13(*data_0: torch.Tensor) -> torch.Tensor:
    return torch.max(*data_0)


def MaxPool_12(
    X: torch.Tensor,
    *,
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: list[int] | None = None,
    kernel_shape: list[int],
    pads: list[int] | None = None,
    storage_order: int = 0,
    strides: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.nn.functional.max_pool2d(
        X,
        kernel_shape,
        stride=strides,
        padding=pads,
        dilation=dilations,
        ceil_mode=bool(ceil_mode),
    ), X


def MaxRoiPool_1(
    X: torch.Tensor,
    rois: torch.Tensor,
    *,
    pooled_shape: list[int],
    spatial_scale: float = 1.0,
) -> torch.Tensor:
    raise NotImplementedError


def MaxUnpool_11(
    X: torch.Tensor,
    I: torch.Tensor,
    output_shape: torch.Tensor | None = None,
    *,
    kernel_shape: list[int],
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def Mean_13(*data_0: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.stack(data_0), dim=0)


def MeanVarianceNormalization_13(
    X: torch.Tensor, *, axes: Sequence[int] = (0, 2, 3)
) -> torch.Tensor:
    # FIXME
    mean = torch.mean(X, dim=axes, keepdim=True)
    std = torch.std(X, dim=axes, keepdim=True)
    return (X - mean) / std


def MelWeightMatrix_17(
    num_mel_bins: torch.Tensor,
    dft_length: torch.Tensor,
    sample_rate: torch.Tensor,
    lower_edge_hertz: torch.Tensor,
    upper_edge_hertz: torch.Tensor,
    *,
    output_datatype: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


def Min_13(*data_0: torch.Tensor) -> torch.Tensor:
    # FIXME
    return torch.min(*data_0)


def Mish_18(X: torch.Tensor) -> torch.Tensor:
    return X * torch.tanh(torch.nn.functional.softplus(X))


def Mod_13(A: torch.Tensor, B: torch.Tensor, *, fmod: int = 0) -> torch.Tensor:
    return torch.fmod(A, B) if fmod else torch.remainder(A, B)


def Mul_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mul(A, B)


def Multinomial_7(
    input: torch.Tensor,
    *,
    dtype: int = 6,
    sample_size: int = 1,
    seed: float | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.multinomial(input, sample_size).to(dtype)


def Neg_13(X: torch.Tensor) -> torch.Tensor:
    return torch.neg(X)


def NegativeLogLikelihoodLoss_13(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return torch.nn.functional.nll_loss(
        input, target, weight, ignore_index=ignore_index, reduction=reduction
    )


def NonMaxSuppression_11(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: torch.Tensor | None = None,
    iou_threshold: torch.Tensor | None = None,
    score_threshold: torch.Tensor | None = None,
    *,
    center_point_box: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


def NonZero_13(X: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(X, as_tuple=False)


def Not_1(X: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(X)


def OneHot_11(
    indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor, *, axis: int = -1
) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, num_classes=depth.item()).to(
        values.dtype
    )


def Optional_15(
    input: torch.Tensor | None = None, *, type: torch.dtype | None = None
) -> torch.Tensor:
    return input if input is not None else torch.tensor([], dtype=type)


def Or_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(A, B)


def PRelu_16(X: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.prelu(X, slope)


def Pad_21(
    data: torch.Tensor,
    pads: torch.Tensor,
    constant_value: torch.Tensor | None = None,
    axes: torch.Tensor | None = None,
    *,
    mode: str = "constant",
) -> torch.Tensor:
    pad = pads.tolist()
    if constant_value is None:
        constant_value = torch.tensor(0)
    return torch.nn.functional.pad(data, pad, mode=mode, value=constant_value.item())


def Pow_15(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.pow(X, Y)


def QLinearConv_10(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero_point: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero_point: torch.Tensor,
    B: torch.Tensor | None = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: list[int] | None = None,
    group: int = 1,
    kernel_shape: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def QLinearMatMul_21(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    a_zero_point: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    b_zero_point: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero_point: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError


def QuantizeLinear_21(
    x: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero_point: torch.Tensor | None = None,
    *,
    axis: int = 1,
    block_size: int = 0,
    output_dtype: int = 0,
    saturate: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


def RNN_14(
    X: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    B: torch.Tensor | None = None,
    sequence_lens: torch.Tensor | None = None,
    initial_h: torch.Tensor | None = None,
    *,
    activation_alpha: list[float] | None = None,
    activation_beta: list[float] | None = None,
    activations: list[str] = ("Tanh", "Tanh"),
    clip: float | None = None,
    direction: str = "forward",
    hidden_size: int | None = None,
    layout: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def RandomNormal_1(
    *,
    dtype: int = 1,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
    shape: list[int],
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.normal(mean, scale, size=shape).to(dtype)


def RandomNormalLike_1(
    input: torch.Tensor,
    *,
    dtype: int | None = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.normal(mean, scale, size=input.size()).to(
        dtype if dtype is not None else input.dtype
    )


def RandomUniform_1(
    *,
    dtype: int = 1,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
    shape: list[int],
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.empty(shape).uniform_(low, high).to(dtype)


def RandomUniformLike_1(
    input: torch.Tensor,
    *,
    dtype: int | None = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    return (
        torch.empty_like(input)
        .uniform_(low, high)
        .to(dtype if dtype is not None else input.dtype)
    )


def Range_11(
    start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    return torch.arange(start.item(), limit.item(), delta.item())


def Reciprocal_13(X: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(X)


def ReduceL1_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.sum(
        torch.abs(data),
        dim=axes.tolist() if axes is not None else None,
        keepdim=bool(keepdims),
    )


def ReduceL2_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.sqrt(
        torch.sum(
            torch.square(data),
            dim=axes.tolist() if axes is not None else None,
            keepdim=bool(keepdims),
        )
    )


def ReduceLogSum_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.log(
        torch.sum(
            data,
            dim=axes.tolist() if axes is not None else None,
            keepdim=bool(keepdims),
        )
    )


def ReduceLogSumExp_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.logsumexp(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceMax_20(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.max(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceMean_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.mean(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceMin_20(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.min(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceProd_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.prod(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceSum_13(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.sum(
        data, dim=axes.tolist() if axes is not None else None, keepdim=bool(keepdims)
    )


def ReduceSumSquare_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    return torch.sum(
        torch.square(data),
        dim=axes.tolist() if axes is not None else None,
        keepdim=bool(keepdims),
    )


def RegexFullMatch_20(X: torch.Tensor, *, pattern: str | None = None) -> torch.Tensor:
    raise NotImplementedError


def Relu_14(X: torch.Tensor) -> torch.Tensor:
    return torch.relu(X)


def Reshape_21(
    data: torch.Tensor, shape: torch.Tensor, *, allowzero: int = 0
) -> torch.Tensor:
    return torch.reshape(data, shape.tolist())


def Resize_19(
    X: torch.Tensor,
    roi: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    sizes: torch.Tensor | None = None,
    *,
    antialias: int = 0,
    axes: list[int] | None = None,
    coordinate_transformation_mode: str = "half_pixel",
    cubic_coeff_a: float = -0.75,
    exclude_outside: int = 0,
    extrapolation_value: float = 0.0,
    keep_aspect_ratio_policy: str = "stretch",
    mode: str = "nearest",
    nearest_mode: str = "round_prefer_floor",
) -> torch.Tensor:
    raise NotImplementedError


def RoiAlign_16(
    X: torch.Tensor,
    rois: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    coordinate_transformation_mode: str = "half_pixel",
    mode: str = "avg",
    output_height: int = 1,
    output_width: int = 1,
    sampling_ratio: int = 0,
    spatial_scale: float = 1.0,
) -> torch.Tensor:
    raise NotImplementedError


def Round_11(X: torch.Tensor) -> torch.Tensor:
    return torch.round(X)


def STFT_17(
    signal: torch.Tensor,
    frame_step: torch.Tensor,
    window: torch.Tensor | None = None,
    frame_length: torch.Tensor | None = None,
    *,
    onesided: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


def Scan_21(
    *initial_state_and_scan_inputs: torch.Tensor,
    body: torch.fx.GraphModule,
    num_scan_inputs: int,
    scan_input_axes: list[int] | None = None,
    scan_input_directions: list[int] | None = None,
    scan_output_axes: list[int] | None = None,
    scan_output_directions: list[int] | None = None,
) -> list[torch.Tensor]:
    raise NotImplementedError


def Scatter_11(
    data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    return data.scatter_(axis, indices, updates)


def ScatterElements_18(
    data: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    *,
    axis: int = 0,
    reduction: str = "none",
) -> torch.Tensor:
    if reduction == "none":
        return data.scatter(axis, indices, updates)
    elif reduction == "add":
        return data.scatter_add(axis, indices, updates)
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def ScatterND_18(
    data: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    if reduction == "none":
        return data.scatter_nd(indices, updates)
    elif reduction == "add":
        return data.scatter_nd_add(indices, updates)
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def Selu_6(
    X: torch.Tensor,
    *,
    alpha: float = 1.6732631921768188,
    gamma: float = 1.0507010221481323,
) -> torch.Tensor:
    return gamma * (
        torch.nn.functional.selu(X)
        if alpha == 1.6732631921768188
        else torch.nn.functional.selu(X, alpha)
    )


def Shape_21(
    data: torch.Tensor, *, end: int | None = None, start: int = 0
) -> torch.Tensor:
    shape = data.shape[start:end]
    return torch.tensor(shape, dtype=torch.int64)


def Shrink_9(
    input: torch.Tensor, *, bias: float = 0.0, lambd: float = 0.5
) -> torch.Tensor:
    return torch.where(
        input < -lambd,
        input + bias,
        torch.where(input > lambd, input - bias, torch.zeros_like(input)),
    )


def Sigmoid_13(X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(X)


def Sign_13(input: torch.Tensor) -> torch.Tensor:
    return torch.sign(input)


def Sin_7(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(input)


def Sinh_9(input: torch.Tensor) -> torch.Tensor:
    return torch.sinh(input)


def Size_21(data: torch.Tensor) -> torch.Tensor:
    return torch.tensor(data.numel(), dtype=torch.int64)


def Slice_13(
    data: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    axes: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
) -> torch.Tensor:
    slices = [slice(None)] * data.dim()
    for i, axis in enumerate(axes.tolist() if axes is not None else range(data.dim())):
        slices[axis] = slice(
            starts[i].item(),
            ends[i].item(),
            steps[i].item() if steps is not None else None,
        )
    return data[tuple(slices)]


def Softmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    return torch.nn.functional.softmax(input, dim=axis)


def SoftmaxCrossEntropyLoss_13(
    scores: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = torch.nn.functional.cross_entropy(
        scores, labels, weight=weights, ignore_index=ignore_index, reduction=reduction
    )
    return loss, torch.argmax(scores, dim=1)


def Softplus_1(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(X)


def Softsign_1(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softsign(input)


def SpaceToDepth_13(input: torch.Tensor, *, blocksize: int) -> torch.Tensor:
    return torch.nn.functional.pixel_unshuffle(input, blocksize)


def Split_18(
    input: torch.Tensor,
    split: torch.Tensor | None = None,
    *,
    axis: int = 0,
    num_outputs: int | None = None,
) -> list[torch.Tensor]:
    if split is not None:
        split_sizes = split.tolist()
    else:
        split_sizes = [input.size(axis) // num_outputs] * num_outputs
    return torch.split(input, split_sizes, dim=axis)


def SplitToSequence_11(
    input: torch.Tensor,
    split: torch.Tensor | None = None,
    *,
    axis: int = 0,
    keepdims: int = 1,
) -> torch.Tensor:
    splits = Split_18(input, split, axis=axis)
    if keepdims:
        splits = [s.unsqueeze(axis) for s in splits]
    return torch.stack(splits, dim=axis)


def Sqrt_13(X: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(X)


def Squeeze_21(data: torch.Tensor, axes: torch.Tensor | None = None) -> torch.Tensor:
    if axes is not None:
        return torch.squeeze(data, dim=tuple(axes.tolist()))
    return torch.squeeze(data)


def StringConcat_20(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def StringNormalizer_10(
    X: torch.Tensor,
    *,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str | None = None,
    stopwords: list[str] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def StringSplit_20(
    X: torch.Tensor, *, delimiter: str | None = None, maxsplit: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def Sub_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.sub(A, B)


def Sum_13(*data_0: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.stack(data_0), dim=0)


def Tan_7(input: torch.Tensor) -> torch.Tensor:
    return torch.tan(input)


def Tanh_13(input: torch.Tensor) -> torch.Tensor:
    return torch.tanh(input)


def TfIdfVectorizer_9(
    X: torch.Tensor,
    *,
    max_gram_length: int,
    max_skip_count: int,
    min_gram_length: int,
    mode: str,
    ngram_counts: list[int],
    ngram_indexes: list[int],
    pool_int64s: list[int] | None = None,
    pool_strings: list[str] | None = None,
    weights: list[float] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def ThresholdedRelu_10(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.threshold(X, alpha, 0)


def Tile_13(input: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
    return input.repeat(repeats.tolist())


def TopK_11(
    X: torch.Tensor,
    K: torch.Tensor,
    *,
    axis: int = -1,
    largest: int = 1,
    sorted: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(X, K.item(), dim=axis, largest=bool(largest), sorted=bool(sorted))


def Transpose_21(data: torch.Tensor, *, perm: list[int] | None = None) -> torch.Tensor:
    return data.permute(perm)


def Trilu_14(
    input: torch.Tensor, k: torch.Tensor | None = None, *, upper: int = 1
) -> torch.Tensor:
    return (
        torch.tril(input, diagonal=k.item() if k is not None else 0)
        if not upper
        else torch.triu(input, diagonal=k.item() if k is not None else 0)
    )


def Unique_11(
    X: torch.Tensor, *, axis: int | None = None, sorted: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def Unsqueeze_21(data: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    for axis in sorted(axes.tolist()):
        data = torch.unsqueeze(data, axis)
    return data


def Upsample_10(
    X: torch.Tensor, scales: torch.Tensor, *, mode: str = "nearest"
) -> torch.Tensor:
    size = [int(dim * scale) for dim, scale in zip(X.shape[2:], scales.tolist()[2:])]
    return torch.nn.functional.interpolate(X, size=size, mode=mode)


def Where_16(condition: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, X, Y)


def Xor_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.logical_xor(A, B)
