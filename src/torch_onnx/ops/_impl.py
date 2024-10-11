from __future__ import annotations

import torch
import torch.fx
import functools

_ONNX_DECOMP_TABLE = {}


def _register_op(func):
    func_name = func.__name__
    torch_op = torch.library.custom_op(f"onnx::{func_name}", mutates_args=[])(func)
    _ONNX_DECOMP_TABLE[torch_op] = func
    return torch_op


@_register_op
def Abs_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Acos_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Acosh_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Add_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def AffineGrid_20(
    theta: torch.Tensor, size: torch.Tensor, *, align_corners: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def And_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ArgMax_13(
    data: torch.Tensor, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ArgMin_13(
    data: torch.Tensor, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Asin_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Asinh_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Atan_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Atanh_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def AveragePool_22(
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


@_register_op
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
    raise NotImplementedError


@_register_op
def Bernoulli_22(
    input: torch.Tensor, *, dtype: int | None = None, seed: float | None = None
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BitShift_11(X: torch.Tensor, Y: torch.Tensor, *, direction: str) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BitwiseAnd_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BitwiseNot_18(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BitwiseOr_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BitwiseXor_18(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def BlackmanWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Cast_21(input: torch.Tensor, *, saturate: int = 1, to: int) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def CastLike_21(
    input: torch.Tensor, target_type: torch.Tensor, *, saturate: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Ceil_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Celu_12(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def CenterCropPad_18(
    input_data: torch.Tensor, shape: torch.Tensor, *, axes: list[int] | None = None
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Clip_13(
    input: torch.Tensor,
    min: torch.Tensor | None = None,
    max: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def Compress_11(
    input: torch.Tensor, condition: torch.Tensor, *, axis: int | None = None
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Concat_13(*inputs: torch.Tensor, axis: int) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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
    raise NotImplementedError


@_register_op
def ConstantOfShape_21(
    input: torch.Tensor, *, value: torch.Tensor | None = None
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Conv_22(
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
    raise NotImplementedError


@_register_op
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


@_register_op
def ConvTranspose_22(
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
    raise NotImplementedError


@_register_op
def Cos_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Cosh_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def CumSum_14(
    x: torch.Tensor, axis: torch.Tensor, *, exclusive: int = 0, reverse: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def DFT_20(
    input: torch.Tensor,
    dft_length: torch.Tensor | None = None,
    axis: torch.Tensor | None = None,
    *,
    inverse: int = 0,
    onesided: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def DeformConv_22(
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


@_register_op
def DepthToSpace_13(
    input: torch.Tensor, *, blocksize: int, mode: str = "DCR"
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def DequantizeLinear_21(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero_point: torch.Tensor | None = None,
    *,
    axis: int = 1,
    block_size: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Det_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Div_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Dropout_22(
    data: torch.Tensor,
    ratio: torch.Tensor | None = None,
    training_mode: torch.Tensor | None = None,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def DynamicQuantizeLinear_11(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def Einsum_12(*Inputs: torch.Tensor, equation: str) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Elu_22(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Equal_19(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Erf_13(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Exp_13(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Expand_13(input: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def EyeLike_22(
    input: torch.Tensor, *, dtype: int | None = None, k: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Flatten_21(input: torch.Tensor, *, axis: int = 1) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Floor_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GRU_22(
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


@_register_op
def Gather_13(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GatherElements_13(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GatherND_13(
    data: torch.Tensor, indices: torch.Tensor, *, batch_dims: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Gelu_20(X: torch.Tensor, *, approximate: str = "none") -> torch.Tensor:
    raise NotImplementedError


@_register_op
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
    raise NotImplementedError


@_register_op
def GlobalAveragePool_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GlobalLpPool_22(X: torch.Tensor, *, p: int = 2) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GlobalMaxPool_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Greater_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GreaterOrEqual_16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GridSample_22(
    X: torch.Tensor,
    grid: torch.Tensor,
    *,
    align_corners: int = 0,
    mode: str = "linear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def GroupNormalization_21(
    X: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    epsilon: float = 9.999999747378752e-06,
    num_groups: int,
    stash_type: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def HammingWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def HannWindow_17(
    size: torch.Tensor, *, output_datatype: int = 1, periodic: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def HardSigmoid_22(
    X: torch.Tensor, *, alpha: float = 0.20000000298023224, beta: float = 0.5
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def HardSwish_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Hardmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Identity_21(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def If_21(
    cond: torch.Tensor,
    *,
    else_branch: torch.fx.GraphModule,
    then_branch: torch.fx.GraphModule,
) -> list[torch.Tensor]:
    raise NotImplementedError


@_register_op
def ImageDecoder_20(
    encoded_stream: torch.Tensor, *, pixel_format: str = "RGB"
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def InstanceNormalization_22(
    input: torch.Tensor,
    scale: torch.Tensor,
    B: torch.Tensor,
    *,
    epsilon: float = 9.999999747378752e-06,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def IsInf_20(
    X: torch.Tensor, *, detect_negative: int = 1, detect_positive: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def IsNaN_20(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def LRN_13(
    X: torch.Tensor,
    *,
    alpha: float = 9.999999747378752e-05,
    beta: float = 0.75,
    bias: float = 1.0,
    size: int,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def LSTM_22(
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


@_register_op
def LayerNormalization_17(
    X: torch.Tensor,
    Scale: torch.Tensor,
    B: torch.Tensor | None = None,
    *,
    axis: int = -1,
    epsilon: float = 9.999999747378752e-06,
    stash_type: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def LeakyRelu_16(
    X: torch.Tensor, *, alpha: float = 0.009999999776482582
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Less_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def LessOrEqual_16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Log_13(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def LogSoftmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Loop_21(
    M: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    *v_initial: torch.Tensor,
    body: torch.fx.GraphModule,
) -> list[torch.Tensor]:
    raise NotImplementedError


@_register_op
def LpNormalization_22(
    input: torch.Tensor, *, axis: int = -1, p: int = 2
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def LpPool_22(
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


@_register_op
def MatMul_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def MatMulInteger_10(
    A: torch.Tensor,
    B: torch.Tensor,
    a_zero_point: torch.Tensor | None = None,
    b_zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Max_13(*data_0: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def MaxPool_22(
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
    raise NotImplementedError


@_register_op
def MaxRoiPool_22(
    X: torch.Tensor,
    rois: torch.Tensor,
    *,
    pooled_shape: list[int],
    spatial_scale: float = 1.0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def MaxUnpool_22(
    X: torch.Tensor,
    I: torch.Tensor,
    output_shape: torch.Tensor | None = None,
    *,
    kernel_shape: list[int],
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Mean_13(*data_0: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def MeanVarianceNormalization_13(
    X: torch.Tensor, *, axes: list[int] = (0, 2, 3)
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def Min_13(*data_0: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Mish_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Mod_13(A: torch.Tensor, B: torch.Tensor, *, fmod: int = 0) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Mul_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Multinomial_22(
    input: torch.Tensor,
    *,
    dtype: int = 6,
    sample_size: int = 1,
    seed: float | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Neg_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def NegativeLogLikelihoodLoss_22(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def NonZero_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Not_1(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def OneHot_11(
    indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor, *, axis: int = -1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Optional_15(
    input: torch.Tensor | None = None, *, type: torch.dtype | None = None
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Or_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def PRelu_16(X: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Pad_21(
    data: torch.Tensor,
    pads: torch.Tensor,
    constant_value: torch.Tensor | None = None,
    axes: torch.Tensor | None = None,
    *,
    mode: str = "constant",
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Pow_15(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
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


@_register_op
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


@_register_op
def RNN_22(
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


@_register_op
def RandomNormal_22(
    *,
    dtype: int = 1,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
    shape: list[int],
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def RandomNormalLike_22(
    input: torch.Tensor,
    *,
    dtype: int | None = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def RandomUniform_22(
    *,
    dtype: int = 1,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
    shape: list[int],
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def RandomUniformLike_22(
    input: torch.Tensor,
    *,
    dtype: int | None = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Range_11(
    start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Reciprocal_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceL1_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceL2_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceLogSum_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceLogSumExp_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceMax_20(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceMean_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceMin_20(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceProd_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceSum_13(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ReduceSumSquare_18(
    data: torch.Tensor,
    axes: torch.Tensor | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def RegexFullMatch_20(X: torch.Tensor, *, pattern: str | None = None) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Relu_14(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Reshape_21(
    data: torch.Tensor, shape: torch.Tensor, *, allowzero: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def RoiAlign_22(
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


@_register_op
def Round_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def STFT_17(
    signal: torch.Tensor,
    frame_step: torch.Tensor,
    window: torch.Tensor | None = None,
    frame_length: torch.Tensor | None = None,
    *,
    onesided: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def Scatter_11(
    data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ScatterElements_18(
    data: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    *,
    axis: int = 0,
    reduction: str = "none",
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def ScatterND_18(
    data: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Selu_22(
    X: torch.Tensor,
    *,
    alpha: float = 1.6732631921768188,
    gamma: float = 1.0507010221481323,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Shape_21(
    data: torch.Tensor, *, end: int | None = None, start: int = 0
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Shrink_9(
    input: torch.Tensor, *, bias: float = 0.0, lambd: float = 0.5
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sigmoid_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sign_13(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sin_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sinh_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Size_21(data: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Slice_13(
    data: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    axes: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Softmax_13(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def SoftmaxCrossEntropyLoss_13(
    scores: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    ignore_index: int | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def Softplus_22(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Softsign_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def SpaceToDepth_13(input: torch.Tensor, *, blocksize: int) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Split_18(
    input: torch.Tensor,
    split: torch.Tensor | None = None,
    *,
    axis: int = 0,
    num_outputs: int | None = None,
) -> list[torch.Tensor]:
    raise NotImplementedError


@_register_op
def SplitToSequence_11(
    input: torch.Tensor,
    split: torch.Tensor | None = None,
    *,
    axis: int = 0,
    keepdims: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sqrt_13(X: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Squeeze_21(data: torch.Tensor, axes: torch.Tensor | None = None) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def StringConcat_20(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def StringNormalizer_10(
    X: torch.Tensor,
    *,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str | None = None,
    stopwords: list[str] | None = None,
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def StringSplit_20(
    X: torch.Tensor, *, delimiter: str | None = None, maxsplit: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def Sub_14(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Sum_13(*data_0: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Tan_22(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Tanh_13(input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
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


@_register_op
def ThresholdedRelu_22(X: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Tile_13(input: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def TopK_11(
    X: torch.Tensor,
    K: torch.Tensor,
    *,
    axis: int = -1,
    largest: int = 1,
    sorted: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def Transpose_21(data: torch.Tensor, *, perm: list[int] | None = None) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Trilu_14(
    input: torch.Tensor, k: torch.Tensor | None = None, *, upper: int = 1
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Unique_11(
    X: torch.Tensor, *, axis: int | None = None, sorted: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@_register_op
def Unsqueeze_21(data: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Upsample_10(
    X: torch.Tensor, scales: torch.Tensor, *, mode: str = "nearest"
) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Where_16(condition: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


@_register_op
def Xor_7(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
