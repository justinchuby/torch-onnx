from __future__ import annotations

import onnx
import torch
from onnxscript import ir
from torch.testing._internal import (
    common_dtype,
)
from torch.testing._internal.opinfo import core as opinfo_core

import torch_onnx
import torch_onnx.ops
from torch_onnx import _schemas

_ONNX_DTYPE_TO_TORCH: dict[ir.DataType, torch.dtype] = {
    ir.DataType.BFLOAT16: torch.bfloat16,
    ir.DataType.BOOL: torch.bool,
    ir.DataType.FLOAT16: torch.float16,
    ir.DataType.FLOAT: torch.float32,
    ir.DataType.DOUBLE: torch.float64,
    ir.DataType.FLOAT8E4M3FN: torch.float8_e4m3fn,
    ir.DataType.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,
    ir.DataType.FLOAT8E5M2: torch.float8_e5m2,
    ir.DataType.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,
    ir.DataType.INT16: torch.int16,
    ir.DataType.INT32: torch.int32,
    ir.DataType.INT64: torch.int64,
    ir.DataType.INT8: torch.int8,
    ir.DataType.UINT8: torch.uint8,
    ir.DataType.UINT16: torch.uint16,
    ir.DataType.UINT32: torch.uint32,
    ir.DataType.UINT64: torch.uint64,
}


def _get_supported_dtypes(op: str, opset_version: int) -> common_dtype._dispatch_dtypes:
    op_schema = onnx.defs.get_schema(op, opset_version)
    op_signature = _schemas.OpSignature.from_opschema(op_schema)
    if not op_signature.params or isinstance(
        op_signature.params[0], _schemas.AttributeParameter
    ):
        return common_dtype.all_types_and_half()
    supported_dtypes = [
        _ONNX_DTYPE_TO_TORCH[type_.dtype]
        for type_ in op_signature.params[0].type_constraint.allowed_types
        if type_.dtype in _ONNX_DTYPE_TO_TORCH
    ]

    return common_dtype._dispatch_dtypes(supported_dtypes)


class OnnxUnaryUfuncInfo(opinfo_core.UnaryUfuncInfo):
    def __init__(self, name: str, **kwargs) -> None:
        opinfo_name = f"ops.onnx.{name}"
        op = getattr(torch_onnx.ops, name)
        op_name, opset_version = name.rsplit("_", 1)
        dtypes = _get_supported_dtypes(op_name, int(opset_version))
        super().__init__(opinfo_name, op=op, dtypes=dtypes, **kwargs)


class OnnxBinaryUfuncInfo(opinfo_core.BinaryUfuncInfo):
    def __init__(self, name: str, **kwargs) -> None:
        opinfo_name = f"ops.onnx.{name}"
        op = getattr(torch_onnx.ops, name)
        op_name, opset_version = name.rsplit("_", 1)
        dtypes = _get_supported_dtypes(op_name, int(opset_version))
        super().__init__(opinfo_name, op=op, dtypes=dtypes, **kwargs)


class OnnxReductionOpInfo(opinfo_core.ReductionOpInfo):
    # TODO: Implement
    pass


class PlaceHolder(opinfo_core.OpInfo):
    pass


op_db = [
    OnnxUnaryUfuncInfo("Abs_13"),
    OnnxUnaryUfuncInfo("Acos_22"),
    OnnxUnaryUfuncInfo("Acosh_22"),
    OnnxBinaryUfuncInfo("Add_14"),
    # PlaceHolder("AffineGrid_20"),
    OnnxBinaryUfuncInfo("And_7"),
    # PlaceHolder("ArgMax_13"),
    # PlaceHolder("ArgMin_13"),
    # OnnxUnaryUfuncInfo("Asin_22"),
    # OnnxUnaryUfuncInfo("Asinh_22"),
    # OnnxUnaryUfuncInfo("Atan_22"),
    # OnnxUnaryUfuncInfo("Atanh_22"),
    # PlaceHolder("AveragePool_22"),
    # PlaceHolder("BatchNormalization_15"),
    # PlaceHolder("Bernoulli_22"),
    # PlaceHolder("BitShift_11"),
    # OnnxBinaryUfuncInfo("BitwiseAnd_18"),
    # OnnxBinaryUfuncInfo("BitwiseNot_18"),
    # OnnxBinaryUfuncInfo("BitwiseOr_18"),
    # OnnxBinaryUfuncInfo("BitwiseXor_18"),
    # PlaceHolder("BlackmanWindow_17"),
    # PlaceHolder("Cast_21"),
    # PlaceHolder("CastLike_21"),
    # OnnxUnaryUfuncInfo("Ceil_13"),
    # PlaceHolder("Celu_12"),
    # PlaceHolder("CenterCropPad_18"),
    # PlaceHolder("Clip_13"),
    # PlaceHolder("Col2Im_18"),
    # PlaceHolder("Compress_11"),
    # PlaceHolder("Concat_13"),
    # PlaceHolder("Constant_21"),
    # PlaceHolder("ConstantOfShape_21"),
    # PlaceHolder("Conv_22"),
    # PlaceHolder("ConvInteger_10"),
    # PlaceHolder("ConvTranspose_22"),
    # OnnxUnaryUfuncInfo("Cos_22"),
    # OnnxUnaryUfuncInfo("Cosh_22"),
    # PlaceHolder("CumSum_14"),
    # PlaceHolder("DFT_20"),
    # PlaceHolder("DeformConv_22"),
    # PlaceHolder("DepthToSpace_13"),
    # PlaceHolder("DequantizeLinear_21"),
    # PlaceHolder("Det_22"),
    # OnnxBinaryUfuncInfo("Div_14"),
    # PlaceHolder("Dropout_22"),
    # PlaceHolder("DynamicQuantizeLinear_11"),
    # PlaceHolder("Einsum_12"),
    # PlaceHolder("Elu_22"),
    # OnnxBinaryUfuncInfo("Equal_19"),
    # PlaceHolder("Erf_13"),
    # OnnxUnaryUfuncInfo("Exp_13"),
    # PlaceHolder("Expand_13"),
    # PlaceHolder("EyeLike_22"),
    # PlaceHolder("Flatten_21"),
    # OnnxUnaryUfuncInfo("Floor_13"),
    # PlaceHolder("GRU_22"),
    # PlaceHolder("Gather_13"),
    # PlaceHolder("GatherElements_11"),
    # PlaceHolder("GatherND_13"),
    # PlaceHolder("Gelu_20"),
    # PlaceHolder("Gemm_13"),
    # PlaceHolder("GlobalAveragePool_22"),
    # PlaceHolder("GlobalLpPool_22"),
    # PlaceHolder("GlobalMaxPool_22"),
    # OnnxBinaryUfuncInfo("Greater_13"),
    # OnnxBinaryUfuncInfo("GreaterOrEqual_16"),
    # PlaceHolder("GridSample_22"),
    # PlaceHolder("GroupNormalization_21"),
    # PlaceHolder("HammingWindow_17"),
    # PlaceHolder("HannWindow_17"),
    # PlaceHolder("HardSigmoid_22"),
    # PlaceHolder("HardSwish_22"),
    # PlaceHolder("Hardmax_13"),
    # PlaceHolder("Identity_21"),
    # PlaceHolder("ImageDecoder_20"),
    # PlaceHolder("InstanceNormalization_22"),
    # PlaceHolder("IsInf_20"),
    # PlaceHolder("IsNaN_20"),
    # PlaceHolder("LRN_13"),
    # PlaceHolder("LSTM_22"),
    # PlaceHolder("LayerNormalization_17"),
    # PlaceHolder("LeakyRelu_16"),
    # OnnxBinaryUfuncInfo("Less_13"),
    # OnnxBinaryUfuncInfo("LessOrEqual_16"),
    # OnnxUnaryUfuncInfo("Log_13"),
    # PlaceHolder("LogSoftmax_13"),
    # PlaceHolder("Loop_21"),
    # PlaceHolder("LpNormalization_22"),
    # PlaceHolder("LpPool_22"),
    # OnnxUnaryUfuncInfo("MatMul_13"),
    # PlaceHolder("MatMulInteger_10"),
    # OnnxUnaryUfuncInfo("Max_13"),
    # PlaceHolder("MaxPool_22"),
    # PlaceHolder("MaxRoiPool_22"),
    # PlaceHolder("MaxUnpool_22"),
    # PlaceHolder("Mean_13"),
    # PlaceHolder("MeanVarianceNormalization_13"),
    # PlaceHolder("MelWeightMatrix_17"),
    # PlaceHolder("Min_13"),
    # PlaceHolder("Mish_22"),
    # OnnxBinaryUfuncInfo("Mod_13"),
    # OnnxBinaryUfuncInfo("Mul_14"),
    # PlaceHolder("Multinomial_22"),
    # OnnxUnaryUfuncInfo("Neg_13"),
    # PlaceHolder("NegativeLogLikelihoodLoss_22"),
    # PlaceHolder("NonMaxSuppression_11"),
    # OnnxUnaryUfuncInfo("NonZero_13"),
    # OnnxUnaryUfuncInfo("Not_1"),
    # PlaceHolder("OneHot_11"),
    # PlaceHolder("Optional_15"),
    # OnnxBinaryUfuncInfo("Or_7"),
    # PlaceHolder("PRelu_16"),
    # PlaceHolder("Pad_21"),
    # OnnxBinaryUfuncInfo("Pow_15"),
    # PlaceHolder("QLinearConv_10"),
    # PlaceHolder("QLinearMatMul_21"),
    # PlaceHolder("QuantizeLinear_21"),
    # PlaceHolder("RNN_22"),
    # PlaceHolder("RandomNormal_22"),
    # PlaceHolder("RandomNormalLike_22"),
    # PlaceHolder("RandomUniform_22"),
    # PlaceHolder("RandomUniformLike_22"),
    # PlaceHolder("Range_11"),
    # OnnxUnaryUfuncInfo("Reciprocal_13"),
    # PlaceHolder("ReduceL1_18"),
    # PlaceHolder("ReduceL2_18"),
    # PlaceHolder("ReduceLogSum_18"),
    # PlaceHolder("ReduceLogSumExp_18"),
    # PlaceHolder("ReduceMax_20"),
    # PlaceHolder("ReduceMean_18"),
    # PlaceHolder("ReduceMin_20"),
    # PlaceHolder("ReduceProd_18"),
    # PlaceHolder("ReduceSum_13"),
    # PlaceHolder("ReduceSumSquare_18"),
    # PlaceHolder("RegexFullMatch_20"),
    # PlaceHolder("Relu_14"),
    # PlaceHolder("Reshape_21"),
    # PlaceHolder("Resize_19"),
    # PlaceHolder("RoiAlign_22"),
    # PlaceHolder("Round_22"),
    # PlaceHolder("STFT_17"),
    # PlaceHolder("Scan_21"),
    # PlaceHolder("Scatter_11"),
    # PlaceHolder("ScatterElements_18"),
    # PlaceHolder("ScatterND_18"),
    # PlaceHolder("Selu_22"),
    # PlaceHolder("Shape_21"),
    # PlaceHolder("Shrink_9"),
    # PlaceHolder("Sigmoid_13"),
    # PlaceHolder("Sign_13"),
    # PlaceHolder("Sin_22"),
    # PlaceHolder("Sinh_22"),
    # PlaceHolder("Size_21"),
    # PlaceHolder("Slice_13"),
    # PlaceHolder("Softmax_13"),
    # PlaceHolder("SoftmaxCrossEntropyLoss_13"),
    # OnnxUnaryUfuncInfo("Softplus_22"),
    # OnnxUnaryUfuncInfo("Softsign_22"),
    # PlaceHolder("SpaceToDepth_13"),
    # PlaceHolder("Split_18"),
    # PlaceHolder("SplitToSequence_11"),
    # OnnxUnaryUfuncInfo("Sqrt_13"),
    # PlaceHolder("Squeeze_21"),
    # PlaceHolder("StringConcat_20"),
    # PlaceHolder("StringNormalizer_10"),
    # PlaceHolder("StringSplit_20"),
    # OnnxBinaryUfuncInfo("Sub_14"),
    # PlaceHolder("Sum_13"),
    # OnnxUnaryUfuncInfo("Tan_22"),
    # OnnxUnaryUfuncInfo("Tanh_13"),
    # PlaceHolder("TfIdfVectorizer_9"),
    # PlaceHolder("ThresholdedRelu_22"),
    # PlaceHolder("Tile_13"),
    # PlaceHolder("TopK_11"),
    # PlaceHolder("Transpose_21"),
    # PlaceHolder("Trilu_14"),
    # PlaceHolder("Unique_11"),
    # PlaceHolder("Unsqueeze_21"),
    # PlaceHolder("Upsample_10"),
    # PlaceHolder("Where_16"),
    # OnnxBinaryUfuncInfo("Xor_7"),
]
