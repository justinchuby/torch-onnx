import torch


def Adagrad_1(R: torch.Tensor, T: torch.Tensor, inputs: torch.Tensor, decay_factor: float, epsilon: float, norm_coefficient: float,) -> torch.Tensor: ...
r"""
Compute one iteration of ADAGRAD, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
some parameters:

 - The initial learning-rate "R".
 - The update count "T". That is, the number of training iterations conducted.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A learning-rate decay factor "decay_factor".
 - A small constant "epsilon" to avoid dividing-by-zero.

At each ADAGRAD iteration, the optimized tensors are moved along a direction
computed based on their estimated gradient and accumulated squared gradient. Assume
that only a single tensor "X" is updated by this operator. We need the value of "X",
its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
parameters are given as attributes because they are usually constants. Also, the
corresponding output tensors are the new value of "X" (called "X_new"), and then
the new accumulated squared gradient (called "H_new"). Those outputs are computed
from the given inputs following the pseudo code below.

Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
numpy-style broadcasting support. The pseudo code to compute those outputs is:

  // Compute a scalar learning-rate factor. At the first update of X, T is generally
  // 0 (0-based update index) or 1 (1-based update index).
  r = R / (1 + T * decay_factor);

  // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
  G_regularized = norm_coefficient * X + G;

  // Compute new accumulated squared gradient.
  H_new = H + G_regularized * G_regularized;

  // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
  // computes element-wise square-root.
  H_adaptive = Sqrt(H_new) + epsilon

  // Compute the new value of "X".
  X_new = X - r * G_regularized / H_adaptive;

If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
be concatenated too) and then just reuse the entire pseudo code.

Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
In that reference paper, this operator is a special case of the Figure 1's composite mirror
descent update.
"""


def TreeEnsemble_5(X: torch.Tensor, aggregate_function: int, leaf_targetids: ints, leaf_weights: tensor, membership_values: tensor, n_targets: int, nodes_falseleafs: ints, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: tensor, nodes_missing_value_tracks_true: ints, nodes_modes: tensor, nodes_splits: tensor, nodes_trueleafs: ints, nodes_truenodeids: ints, post_transform: int, tree_roots: ints,) -> torch.Tensor: ...
r"""
Tree Ensemble operator.  Returns the regressed values for each input in a batch.
Inputs have dimensions `[N, F]` where `N` is the input batch size and `F` is the number of input features.
Outputs have dimensions `[N, num_targets]` where `N` is the batch size and `num_targets` is the number of targets, which is a configurable attribute.

The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.

All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.

This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.
"""


def TreeEnsembleClassifier_1(X: torch.Tensor, base_values: floats, class_ids: ints, class_nodeids: ints, class_treeids: ints, class_weights: floats, classlabels_int64s: ints, classlabels_strings: strings, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, post_transform: string,) -> torch.Tensor: ...
r"""
Tree Ensemble classifier.  Returns the top class for each of N inputs.

The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.

Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.

One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
"""


def TreeEnsembleClassifier_3(X: torch.Tensor, base_values: floats, base_values_as_tensor: tensor, class_ids: ints, class_nodeids: ints, class_treeids: ints, class_weights: floats, class_weights_as_tensor: tensor, classlabels_int64s: ints, classlabels_strings: strings, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_hitrates_as_tensor: tensor, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, nodes_values_as_tensor: tensor, post_transform: string,) -> torch.Tensor: ...
r"""
Tree Ensemble classifier. Returns the top class for each of N inputs.

The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.

Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.

One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
"""


def TreeEnsembleClassifier_5(X: torch.Tensor, base_values: floats, base_values_as_tensor: tensor, class_ids: ints, class_nodeids: ints, class_treeids: ints, class_weights: floats, class_weights_as_tensor: tensor, classlabels_int64s: ints, classlabels_strings: strings, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_hitrates_as_tensor: tensor, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, nodes_values_as_tensor: tensor, post_transform: string,) -> torch.Tensor: ...
r"""
This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
To encode class labels, use a LabelEncoder operator.
Tree Ensemble classifier. Returns the top class for each of N inputs.

The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.

Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.

One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
"""


def Scaler_1(X: torch.Tensor, offset: floats, scale: floats,) -> torch.Tensor: ...
r"""
Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
"""


def SVMRegressor_1(X: torch.Tensor, coefficients: floats, kernel_params: floats, kernel_type: string, n_supports: int, one_class: int, post_transform: string, rho: floats, support_vectors: floats,) -> torch.Tensor: ...
r"""
Support Vector Machine regression prediction and one-class SVM anomaly detection.
"""


def OneHotEncoder_1(X: torch.Tensor, cats_int64s: ints, cats_strings: strings, zeros: int,) -> torch.Tensor: ...
r"""
Replace each input element with an array of ones and zeros, where a single
one is placed at the index of the category that was passed in. The total category count
will determine the size of the extra dimension of the output array Y.

For example, if we pass a tensor with a single value of 4, and a category count of 8,
the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.

This operator assumes every input feature is from the same set of categories.

If the input is a tensor of float, int32, or double, the data will be cast
to integers and the cats_int64s category list will be used for the lookups.
"""


def LinearRegressor_1(X: torch.Tensor, coefficients: floats, intercepts: floats, post_transform: string, targets: int,) -> torch.Tensor: ...
r"""
Generalized linear regression evaluation.

If targets is set to 1 (default) then univariate regression is performed.

If targets is set to M then M sets of coefficients must be passed in as a sequence
and M results will be output for each input n in N.

The coefficients array is of length n, and the coefficients for each target are contiguous.
Intercepts are optional but if provided must match the number of targets.
"""


def LabelEncoder_1(X: torch.Tensor, classes_strings: strings, default_int64: int, default_string: string,) -> torch.Tensor: ...
r"""
Converts strings to integers and vice versa.

If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.

Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.

When converting from integers to strings, the string is fetched from the
'classes_strings' list, by simple indexing.

When converting from strings to integers, the string is looked up in the list
and the index at which it is found is used as the converted value.
"""


def LabelEncoder_2(X: torch.Tensor, default_float: float, default_int64: int, default_string: string, keys_floats: floats, keys_int64s: ints, keys_strings: strings, values_floats: floats, values_int64s: ints, values_strings: strings,) -> torch.Tensor: ...
r"""
Maps each element in the input tensor to another value.

The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value.

Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].

Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.

For key look-up, bit-wise comparison is used so even a float NaN can be
mapped to a value in 'values_*' attribute.
"""


def LabelEncoder_4(X: torch.Tensor, default_float: float, default_int64: int, default_string: string, default_tensor: tensor, keys_floats: floats, keys_int64s: ints, keys_strings: strings, keys_tensor: tensor, values_floats: floats, values_int64s: ints, values_strings: strings, values_tensor: tensor,) -> torch.Tensor: ...
r"""
Maps each element in the input tensor to another value.

The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
must have the same length. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value. The type of the 'default_*'
attribute must match the 'values_*' attribute chosen.

Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].

Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.

Float keys with value 'NaN' match any input 'NaN' value regardless of bit
value. If a key is repeated, the last key takes precedence.
"""


def Imputer_1(X: torch.Tensor, imputed_value_floats: floats, imputed_value_int64s: ints, replaced_value_float: float, replaced_value_int64: int,) -> torch.Tensor: ...
r"""
Replaces inputs that equal one value with another, leaving all other elements alone.

This operator is typically used to replace missing values in situations where they have a canonical
representation, such as -1, 0, NaN, or some extreme value.

One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
which one depends on whether floats or integers are being processed.

The imputed_value attribute length can be 1 element, or it can have one element per input feature.
In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
"""


def Binarizer_1(X: torch.Tensor, threshold: float,) -> torch.Tensor: ...
r"""
Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
"""


def Gelu_20(X: torch.Tensor, approximate: string,) -> torch.Tensor: ...
r"""
Gelu takes one input data (Tensor<T>) and produces one
output data (Tensor<T>) where the gaussian error linear units function,
$y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
If the attribute "approximate" is set to "tanh", the function estimation,
$y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
to the tensor elementwise.
"""


def AffineGrid_20(theta: torch.Tensor, size: torch.Tensor, align_corners: int,) -> torch.Tensor: ...
r"""
Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
```
[r00, r01, r02, t0]   [x]   [x']
[r10, r11, r12, t1] * [y] = [y']
[r20, r21, r22, t2]   [z]   [z']
[0,   0,   0,   1 ]   [1]   [1 ]
```
where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
The output `grid` contains positions in the output space.

When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
```
v            v            v            v
|-------------------|------------------|
-1                  0                  1
```
When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
```
    v        v         v         v
|------------------|-------------------|
-1                 0                   1
```
"""


def Greater_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def Greater_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Greater_9(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Greater_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def DeformConv_19(X: torch.Tensor, W: torch.Tensor, offset: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, dilations: ints, group: int, kernel_shape: ints, offset_group: int, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
"""


def BitwiseXor_18(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulting from performing the bitwise `xor` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def BitwiseOr_18(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulting from performing the bitwise `or` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def BitwiseAnd_18(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulting from performing the bitwise `and` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def STFT_17(signal: torch.Tensor, frame_step: torch.Tensor, window: torch.Tensor, frame_length: torch.Tensor, onesided: int,) -> torch.Tensor: ...
r"""
Computes the Short-time Fourier Transform of the signal.
"""


def Mish_18(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```
"""


def MelWeightMatrix_17(num_mel_bins: torch.Tensor, dft_length: torch.Tensor, sample_rate: torch.Tensor, lower_edge_hertz: torch.Tensor, upper_edge_hertz: torch.Tensor, output_datatype: int,) -> torch.Tensor: ...
r"""
Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
"""


def CastLike_15(input: torch.Tensor, target_type: torch.Tensor, ) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
"""


def CastLike_19(input: torch.Tensor, target_type: torch.Tensor, saturate: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
"""


def CastLike_21(input: torch.Tensor, target_type: torch.Tensor, saturate: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
"""


def OptionalGetElement_15(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.
"""


def OptionalGetElement_18(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
"""


def Optional_15(input: torch.Tensor, type: type_proto,) -> torch.Tensor: ...
r"""
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
"""


def HardSwish_14(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
"""


def LessOrEqual_12(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def LessOrEqual_16(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Celu_12(X: torch.Tensor, alpha: float,) -> torch.Tensor: ...
r"""
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
"""


def ConcatFromSequence_11(input_sequence: torch.Tensor, axis: int, new_axis: int,) -> torch.Tensor: ...
r"""
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
"""


def DictVectorizer_1(X: torch.Tensor, int64_vocabulary: ints, string_vocabulary: strings,) -> torch.Tensor: ...
r"""
Uses an index mapping to convert a dictionary to an array.

Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
the key type. The index into the vocabulary array at which the key is found is then
used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.

The key type of the input map must correspond to the element type of the defined vocabulary attribute.
Therefore, the output array will be equal in length to the index mapping vector parameter.
All keys in the input dictionary must be present in the index mapping vector.
For each item in the input dictionary, insert its value in the output array.
Any keys not present in the input dictionary, will be zero in the output array.

For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
"""


def CastMap_1(X: torch.Tensor, cast_to: string, map_form: string, max_map: int,) -> torch.Tensor: ...
r"""
Converts a map to a tensor.
The map key must be an int64 and the values will be ordered
in ascending order based on this key.
The operator supports dense packing or sparse packing.
If using sparse packing, the key cannot exceed the max_map-1 value.
"""


def Or_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def Or_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def SequenceMap_17(input_sequence: torch.Tensor, additional_inputs: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.
"""


def SequenceAt_11(input_sequence: torch.Tensor, position: torch.Tensor, ) -> torch.Tensor: ...
r"""
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
"""


def SequenceInsert_11(input_sequence: torch.Tensor, tensor: torch.Tensor, position: torch.Tensor, ) -> torch.Tensor: ...
r"""
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
"""


def GatherND_11(data: torch.Tensor, indices: torch.Tensor, ) -> torch.Tensor: ...
r"""
Given `data` tensor of rank `r` >= 1, and `indices` tensor of rank `q` >= 1, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r` (inclusive)

3) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r` => error condition

2) If `indices_shape[-1] == r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `r`. Let us think of each such `r` ranked tensor as `indices_slice`.
   Each *scalar value* corresponding to `data[indices_slice]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r`. Let us think of each such tensors as `indices_slice`.
   Each *tensor slice* corresponding to `data[indices_slice , :]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, and 4 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]
"""


def GatherND_12(data: torch.Tensor, indices: torch.Tensor, batch_dims: int,) -> torch.Tensor: ...
r"""
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  batch_dims = 0

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  batch_dims = 0

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

`Example 5`

  batch_dims = 1

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[1],[0]]             # indices_shape = [2, 1]

  output  = [[2,3],[4,5]]             # output_shape = [2, 2]
"""


def GatherND_13(data: torch.Tensor, indices: torch.Tensor, batch_dims: int,) -> torch.Tensor: ...
r"""
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```
"""


def HannWindow_17(size: torch.Tensor, output_datatype: int, periodic: int,) -> torch.Tensor: ...
r"""
Generates a Hann window as described in the paper https://ieeexplore.ieee.org/document/1455106.
"""


def ScatterND_11(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, ) -> torch.Tensor: ...
r"""
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
That is, two or more `updates` for the same index-location is not supported.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] = updates[idx]

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

This operator is the inverse of GatherND.

Example 1:
```
  data    = [1, 2, 3, 4, 5, 6, 7, 8]
  indices = [[4], [3], [1], [7]]
  updates = [9, 10, 11, 12]
  output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
  data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  indices = [[0], [2]]
  updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
  output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
"""


def ScatterND_13(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, ) -> torch.Tensor: ...
r"""
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
That is, two or more `updates` for the same index-location is not supported.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] = updates[idx]

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

This operator is the inverse of GatherND.

Example 1:
```
  data    = [1, 2, 3, 4, 5, 6, 7, 8]
  indices = [[4], [3], [1], [7]]
  updates = [9, 10, 11, 12]
  output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
  data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  indices = [[0], [2]]
  updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
  output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
"""


def ScatterND_16(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, reduction: string,) -> torch.Tensor: ...
r"""
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:
    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] = updates[idx]
The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. This ensures that the output value does not depend on the iteration order.
When `reduction` is set to "add", `output` is calculated as follows:
    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] += updates[idx]
When `reduction` is set to "mul", `output` is calculated as follows:
    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] *= updates[idx]
This operator is the inverse of GatherND.
Example 1:
```
  data    = [1, 2, 3, 4, 5, 6, 7, 8]
  indices = [[4], [3], [1], [7]]
  updates = [9, 10, 11, 12]
  output  = [1, 11, 3, 10, 9, 6, 7, 12]
```
Example 2:
```
  data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  indices = [[0], [2]]
  updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
  output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
"""


def ScatterND_18(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, reduction: string,) -> torch.Tensor: ...
r"""
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
`indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = updates[idx]
```

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. This ensures that the output value does not depend on the iteration order.
When `reduction` is set to some reduction function `f`, `output` is calculated as follows:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = f(output[indices[idx]], updates[idx])
```

where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherND.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data    = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [[4], [3], [1], [7]]
updates = [9, 10, 11, 12]
output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
indices = [[0], [2]]
updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
"""


def Col2Im_18(input: torch.Tensor, image_shape: torch.Tensor, block_shape: torch.Tensor, dilations: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The operator rearranges column blocks back into a multidimensional image

Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
but it only supports *batched* multi-dimensional image tensors.
Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

NOTE:
  Although specifying image_shape looks redundant because it could be calculated from
  convolution formulas, it is required as input for more advanced scenarios as explained
  at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)
"""


def Det_11(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
"""


def ScatterElements_11(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
"""


def ScatterElements_13(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
"""


def ScatterElements_16(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int, reduction: string,) -> torch.Tensor: ...
r"""
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.
For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.
`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
corresponding to the [i][j] entry is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```
When `reduction` is set to "add", the update corresponding to the [i][j] entry is performed as below:
```
  output[indices[i][j]][j] += updates[i][j] if axis = 0,
  output[i][indices[i][j]] += updates[i][j] if axis = 1,
```
When `reduction` is set to "mul", the update corresponding to the [i][j] entry is performed as below:
```
  output[indices[i][j]][j] *= updates[i][j] if axis = 0,
  output[i][indices[i][j]] *= updates[i][j] if axis = 1,
```
This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.
Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
"""


def ScatterElements_18(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int, reduction: string,) -> torch.Tensor: ...
r"""
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = updates[i][j] if axis = 0,
output[i][indices[i][j]] = updates[i][j] if axis = 1,
```
When `reduction` is set to some reduction function `f`, the update corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
```
where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
indices = [
    [1, 0, 2],
    [0, 2, 1],
]
updates = [
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2],
]
output = [
    [2.0, 1.1, 0.0]
    [1.0, 0.0, 2.2]
    [0.0, 2.1, 1.2]
]
```
Example 2:
```
data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
indices = [[1, 3]]
updates = [[1.1, 2.1]]
axis = 1
output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
"""


def ZipMap_1(X: torch.Tensor, classlabels_int64s: ints, classlabels_strings: strings,) -> torch.Tensor: ...
r"""
Creates a map from the input and the attributes.

The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).

The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.
"""


def BitShift_11(X: torch.Tensor, Y: torch.Tensor, direction: string,) -> torch.Tensor: ...
r"""
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def GatherElements_11(data: torch.Tensor, indices: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
  out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
  out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
  out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
  data = [
      [1, 2],
      [3, 4],
  ]
  indices = [
      [0, 0],
      [1, 0],
  ]
  axis = 1
  output = [
      [
        [1, 1],
        [4, 3],
      ],
  ]
```
Example 2:
```
  data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
  ]
  indices = [
      [1, 2, 0],
      [2, 0, 0],
  ]
  axis = 0
  output = [
      [
        [4, 8, 3],
        [7, 2, 3],
      ],
  ]
```
"""


def GatherElements_13(data: torch.Tensor, indices: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```
"""


def SplitToSequence_11(input: torch.Tensor, split: torch.Tensor, axis: int, keepdims: int,) -> torch.Tensor: ...
r"""
Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.
"""


def DynamicQuantizeLinear_11(x: torch.Tensor, ) -> torch.Tensor: ...
r"""
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
"""


def Softmax_1(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
"""


def Softmax_11(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the softmax values of the corresponding input.
"""


def Softmax_13(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The "axis" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.
"""


def RegexFullMatch_20(X: torch.Tensor, pattern: string,) -> torch.Tensor: ...
r"""
RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.
"""


def Round_11(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
"""


def CumSum_11(x: torch.Tensor, axis: torch.Tensor, exclusive: int, reverse: int,) -> torch.Tensor: ...
r"""
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
"""


def CumSum_14(x: torch.Tensor, axis: torch.Tensor, exclusive: int, reverse: int,) -> torch.Tensor: ...
r"""
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
"""


def Momentum_1(R: torch.Tensor, T: torch.Tensor, inputs: torch.Tensor, alpha: float, beta: float, mode: string, norm_coefficient: float,) -> torch.Tensor: ...
r"""
Compute one iteration of stochastic gradient update with momentum.
This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. As you can imagine, SG with momentum requires
several parameters:

 - The learning-rate "R".
 - The update count "T". That is, the number of conducted training iterations. It should
   be zero in the first training iteration.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
 - The scaling coefficient of current gradient "beta".
 - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
   be used.

For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
momentum (called "V_new").

This operator supports two different momentum algorithms. Set the attribute "mode" to
"nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
"standard" to use standard momentum. Computation details are described subsequently.

Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.

Pseudo code for SG with standard momentum:

  // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
  // values of all elements in X.
  G_regularized = norm_coefficient * X + G

  // In the first training iteration, beta should always be 1.
  beta_adjusted = T > 0 ? beta : 1

  // Compute the current momentum based on previous momentum and the current gradient.
  V_new = alpha * V + beta_adjusted * G_regularized

  // Update X.
  X_new = X - R * V_new

Pseudo code for SG with Nesterov's momentum:

  // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
  // values of all elements in X.
  G_regularized = norm_coefficient * X + G;

  // In the first training iteration, beta should always be 1.
  beta_adjusted = T > 0 ? beta : 1

  // Compute the current momentum based on previous momentum and the current gradient.
  V_new = alpha * V + beta_adjusted * G_regularized;

  // Compute final update direction and then update X.
  X_new = X - R * (G_regularized + alpha * V_new)

If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
be concatenated too) and then our pseudo code becomes applicable.
"""


def Tan_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the tangent of the given input tensor, element-wise.
"""


def Trilu_14(input: torch.Tensor, k: torch.Tensor, upper: int,) -> torch.Tensor: ...
r"""
Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute "upper" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the "upper" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
"""


def Selu_1(X: torch.Tensor, alpha: float, consumed_inputs: ints, gamma: float,) -> torch.Tensor: ...
r"""
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
"""


def Selu_6(X: torch.Tensor, alpha: float, gamma: float,) -> torch.Tensor: ...
r"""
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
"""


def RoiAlign_10(X: torch.Tensor, rois: torch.Tensor, batch_indices: torch.Tensor, mode: string, output_height: int, output_width: int, sampling_ratio: int, spatial_scale: float,) -> torch.Tensor: ...
r"""
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
"""


def RoiAlign_16(X: torch.Tensor, rois: torch.Tensor, batch_indices: torch.Tensor, coordinate_transformation_mode: string, mode: string, output_height: int, output_width: int, sampling_ratio: int, spatial_scale: float,) -> torch.Tensor: ...
r"""
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
"""


def Reciprocal_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
"""


def Reciprocal_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
"""


def Reciprocal_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
"""


def ReverseSequence_10(input: torch.Tensor, sequence_lens: torch.Tensor, batch_axis: int, time_axis: int,) -> torch.Tensor: ...
r"""
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
"""


def NonMaxSuppression_10(boxes: torch.Tensor, scores: torch.Tensor, max_output_boxes_per_class: torch.Tensor, iou_threshold: torch.Tensor, score_threshold: torch.Tensor, center_point_box: int,) -> torch.Tensor: ...
r"""
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
"""


def NonMaxSuppression_11(boxes: torch.Tensor, scores: torch.Tensor, max_output_boxes_per_class: torch.Tensor, iou_threshold: torch.Tensor, score_threshold: torch.Tensor, center_point_box: int,) -> torch.Tensor: ...
r"""
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
"""


def IsInf_10(X: torch.Tensor, detect_negative: int, detect_positive: int,) -> torch.Tensor: ...
r"""
Map infinity to true and other values to false.
"""


def IsInf_20(X: torch.Tensor, detect_negative: int, detect_positive: int,) -> torch.Tensor: ...
r"""
Map infinity to true and other values to false.
"""


def Equal_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def Equal_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Equal_11(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Equal_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Equal_19(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def LpPool_1(X: torch.Tensor, auto_pad: string, kernel_shape: ints, p: float, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
LpPool consumes an input tensor X and applies Lp pooling across the
the tensor according to kernel sizes, stride sizes, and pad lengths.
Lp pooling consisting of computing the Lp norm on all values of a subset
of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing.
"""


def LpPool_2(X: torch.Tensor, auto_pad: string, kernel_shape: ints, p: int, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
LpPool consumes an input tensor X and applies Lp pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
Lp pooling consisting of computing the Lp norm on all values of a subset
of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing.
"""


def LpPool_11(X: torch.Tensor, auto_pad: string, kernel_shape: ints, p: int, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
LpPool consumes an input tensor X and applies Lp pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
Lp pooling consisting of computing the Lp norm on all values of a subset
of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing.
"""


def LpPool_18(X: torch.Tensor, auto_pad: string, ceil_mode: int, dilations: ints, kernel_shape: ints, p: int, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
LpPool consumes an input tensor X and applies Lp pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
Lp pooling consisting of computing the Lp norm on all values of a subset
of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
```
"""


def QuantizeLinear_10(x: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, ) -> torch.Tensor: ...
r"""
The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
"""


def QuantizeLinear_13(x: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
"""


def QuantizeLinear_19(x: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, axis: int, saturate: int,) -> torch.Tensor: ...
r"""
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.
"""


def QuantizeLinear_21(x: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, axis: int, block_size: int, output_dtype: int, saturate: int,) -> torch.Tensor: ...
r"""
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
"""


def QLinearConv_10(x: torch.Tensor, x_scale: torch.Tensor, x_zero_point: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, w_zero_point: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, B: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
"""


def Elu_1(X: torch.Tensor, alpha: float, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
"""


def Elu_6(X: torch.Tensor, alpha: float,) -> torch.Tensor: ...
r"""
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
"""


def ConvInteger_10(x: torch.Tensor, w: torch.Tensor, x_zero_point: torch.Tensor, w_zero_point: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
"""


def TreeEnsembleRegressor_1(X: torch.Tensor, aggregate_function: string, base_values: floats, n_targets: int, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, post_transform: string, target_ids: ints, target_nodeids: ints, target_treeids: ints, target_weights: floats,) -> torch.Tensor: ...
r"""
Tree Ensemble regressor.  Returns the regressed values for each input in N.

All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.

All fields prefixed with target_ are tuples of votes at the leaves.

A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.

All trees must have their node ids start at 0 and increment by 1.

Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
"""


def TreeEnsembleRegressor_3(X: torch.Tensor, aggregate_function: string, base_values: floats, base_values_as_tensor: tensor, n_targets: int, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_hitrates_as_tensor: tensor, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, nodes_values_as_tensor: tensor, post_transform: string, target_ids: ints, target_nodeids: ints, target_treeids: ints, target_weights: floats, target_weights_as_tensor: tensor,) -> torch.Tensor: ...
r"""
Tree Ensemble regressor.  Returns the regressed values for each input in N.

All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.

All fields prefixed with target_ are tuples of votes at the leaves.

A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.

All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
All trees must have their node ids start at 0 and increment by 1.

Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
"""


def TreeEnsembleRegressor_5(X: torch.Tensor, aggregate_function: string, base_values: floats, base_values_as_tensor: tensor, n_targets: int, nodes_falsenodeids: ints, nodes_featureids: ints, nodes_hitrates: floats, nodes_hitrates_as_tensor: tensor, nodes_missing_value_tracks_true: ints, nodes_modes: strings, nodes_nodeids: ints, nodes_treeids: ints, nodes_truenodeids: ints, nodes_values: floats, nodes_values_as_tensor: tensor, post_transform: string, target_ids: ints, target_nodeids: ints, target_treeids: ints, target_weights: floats, target_weights_as_tensor: tensor,) -> torch.Tensor: ...
r"""
This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
functionality.

Tree Ensemble regressor.  Returns the regressed values for each input in N.

All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.

All fields prefixed with target_ are tuples of votes at the leaves.

A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.

All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
All trees must have their node ids start at 0 and increment by 1.

Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
"""


def QLinearMatMul_10(a: torch.Tensor, a_scale: torch.Tensor, a_zero_point: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor, b_zero_point: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
"""


def QLinearMatMul_21(a: torch.Tensor, a_scale: torch.Tensor, a_zero_point: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor, b_zero_point: torch.Tensor, y_scale: torch.Tensor, y_zero_point: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
"""


def MatMulInteger_10(A: torch.Tensor, B: torch.Tensor, a_zero_point: torch.Tensor, b_zero_point: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
"""


def StringNormalizer_10(X: torch.Tensor, case_change_action: string, is_case_sensitive: int, locale: string, stopwords: strings,) -> torch.Tensor: ...
r"""
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
"""


def MeanVarianceNormalization_9(X: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
A MeanVarianceNormalization Function: Perform mean variance normalization
on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
"""


def MeanVarianceNormalization_13(X: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
A MeanVarianceNormalization Function: Perform mean variance normalization
on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
"""


def LogSoftmax_1(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the logsoftmax (log of softmax) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the logsoftmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
"""


def LogSoftmax_11(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the logsoftmax (log of softmax) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the logsoftmax values of the corresponding input.
"""


def LogSoftmax_13(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The "axis" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
"""


def TfIdfVectorizer_9(X: torch.Tensor, max_gram_length: int, max_skip_count: int, min_gram_length: int, mode: string, ngram_counts: ints, ngram_indexes: ints, pool_int64s: ints, pool_strings: strings, weights: floats,) -> torch.Tensor: ...
r"""
This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

The output vector (denoted by Y) stores the count of each n-gram;
Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.
"""


def Range_11(start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor, ) -> torch.Tensor: ...
r"""
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
"""


def Gather_1(data: torch.Tensor, indices: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
Example 2:
```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [[1.0, 1.9]],
      [[2.3, 3.9]],
      [[4.5, 5.9]],
  ]
```
"""


def Gather_11(data: torch.Tensor, indices: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

axis = 0 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
axis = 1 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [[1.0, 1.9]],
      [[2.3, 3.9]],
      [[4.5, 5.9]],
  ]
```
"""


def Gather_13(data: torch.Tensor, indices: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

If `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```
"""


def NonZero_9(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the indices of the elements that are non-zero
(in row-major order - by dimension).
NonZero behaves similar to numpy.nonzero:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
"""


def NonZero_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the indices of the elements that are non-zero
(in row-major order - by dimension).
NonZero behaves similar to numpy.nonzero:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
"""


def Mean_1(data_0: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Mean_6(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Mean_8(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Mean_13(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def BlackmanWindow_17(size: torch.Tensor, output_datatype: int, periodic: int,) -> torch.Tensor: ...
r"""
Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.
"""


def Sub_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Performs element-wise binary subtraction (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Sub_6(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary subtraction (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Sub_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Sub_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Sub_14(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
"""


def Sign_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
"""


def Sign_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
"""


def IsNaN_9(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns which elements of the input are NaN.
"""


def IsNaN_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns which elements of the input are NaN.
"""


def IsNaN_20(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns which elements of the input are NaN.
"""


def SequenceErase_11(input_sequence: torch.Tensor, position: torch.Tensor, ) -> torch.Tensor: ...
r"""
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
"""


def Max_1(data_0: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Max_6(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Max_8(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Max_12(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Max_13(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Shrink_9(input: torch.Tensor, bias: float, lambd: float,) -> torch.Tensor: ...
r"""
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
"""


def Sinh_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic sine of the given input tensor element-wise.
"""


def Mod_10(A: torch.Tensor, B: torch.Tensor, fmod: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Mod_13(A: torch.Tensor, B: torch.Tensor, fmod: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary modulus (with Numpy-style broadcasting support).
The sign of the remainder is the same as that of the Divisor.

Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
(in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
This attribute is set to 0 by default causing the behavior to be like integer mod.
Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

If the input type is floating point, then `fmod` attribute must be set to 1.

In case of dividend being zero, the results will be platform dependent.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Scatter_9(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.
Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
"""


def Scatter_11(data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
"""


def OneHot_9(indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [0, depth) will result in one-hot representation with all 'off_value' values in the
output tensor.
"""


def OneHot_11(indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
output tensor.

when axis = 0:
output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

when axis = -1:
output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
"""


def StringConcat_20(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)
"""


def ReduceProd_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the product of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceProd_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the product of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceProd_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the product of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceProd_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the product of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def MaxUnpool_9(X: torch.Tensor, I: torch.Tensor, output_shape: torch.Tensor, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
"""


def MaxUnpool_11(X: torch.Tensor, I: torch.Tensor, output_shape: torch.Tensor, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
"""


def EyeLike_9(input: torch.Tensor, dtype: int, k: int,) -> torch.Tensor: ...
r"""
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
"""


def ConstantOfShape_9(input: torch.Tensor, value: tensor,) -> torch.Tensor: ...
r"""
Generate a tensor with given value and shape.
"""


def ConstantOfShape_20(input: torch.Tensor, value: tensor,) -> torch.Tensor: ...
r"""
Generate a tensor with given value and shape.
"""


def ConstantOfShape_21(input: torch.Tensor, value: tensor,) -> torch.Tensor: ...
r"""
Generate a tensor with given value and shape.
"""


def Compress_9(input: torch.Tensor, condition: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
In case axis is not provided, input is flattened before elements are selected.
Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
"""


def Compress_11(input: torch.Tensor, condition: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
In case axis is not provided, input is flattened before elements are selected.
Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
"""


def GlobalMaxPool_1(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
GlobalMaxPool consumes an input tensor X and applies max pooling across
the values in the same channel. This is equivalent to MaxPool with kernel size
equal to the spatial dimension of input tensor.
"""


def Scan_8(sequence_lens: torch.Tensor, initial_state_and_scan_inputs: torch.Tensor, body: graph, directions: ints, num_scan_inputs: int,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops). All these tensors are required to
have the same shape in each iteration of the loop (a restriction imposed to enable efficient
memory allocation). Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs).

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The operation supports batching, and the batch-axis is required to be 0.
When multiple scan_input tensors are used, they must all have the same batch-size,
and they must all have the same maximum-sequence-length (the dimensionality of the
sequence axis or scan axis). The sequence axis or scan axis is required to be 1.

The operation has an optional sequence_lens input (of shape [BATCH_SIZE]) to
allow variable length sequences of length <= the maximum-sequence-length. If this
input is not specified, all sequences are assumed to be of length equal to
maximum-sequence-length. For variable length input sequences, the scan_outputs
will consist of a sequence of same length as the input, padded to the
maximum-sequence-length.

The optional attribute directions can be used to scan a sequence in the reverse direction.
If this attribute is omitted, all sequences are scanned in the forward direction.
A bidirectional scan be performed by specifying the same tensor input twice in the
scan_inputs, once with a forward direction, and once with a backward direction.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body
    > (sequence_lengths, init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // T.shape[0] denotes the batch-size of T
    // The batch-size of scan_1, ..., scan_m are all required to be equal
    batch_size = scan_1.shape[0];

    // scan_i.shape[1] denotes the (max) sequence-length of scan_i
    // scan_i.shape[1] is required to be equal to scan_j.shape[1] for all i,j.
    max_sequence_length = scan_1.shape[1];

    for (int batch = 0; batch < batch_size; ++batch) {
        // initialize state-variables
        st_1 = init_1; ... st_n = init_n;
        // initialize scan-output variables: [] denotes an empty tensor
        scan_out_1 = []; ...; scan_out_k = [];
        // identify number of iterations:
        N = (sequence_lengths specified) ? sequence_lengths[batch] : max_sequence_length;

        // execute loop
        for (int t = 0; t < N; ++t) {
            // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
            // of rank one less than T obtained by indexing T at position t along axis k.
            si_1 = (scan_1<axis=0>[batch])<axis=1>[t];
            ... ;
            si_m = (scan_m<axis=0>[batch])<axis=1>[t];
            // execute loop-body
            st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
            // accumulate the scan-output elements
            scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
        }
        // accumulate the outputs for this batch:
        bst_1[batch] = st_1; ..., bst_n[batch] = st_n;
        // Note scan-outputs will have size max_sequence_length, but only first N values will be meaningful.
        // The remaining values have an undefined value.
        b_scan_out_1[batch] = scan_out_1; ...; b_scan_out_k[batch] = scan_out_k;
    }
    return bst_1, ..., bst_n, b_scan_out_1, ..., b_scan_out_k;



*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1]("", %H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def Scan_9(initial_state_and_scan_inputs: torch.Tensor, body: graph, num_scan_inputs: int, scan_input_axes: ints, scan_input_directions: ints, scan_output_axes: ints, scan_output_directions: ints,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def Scan_11(initial_state_and_scan_inputs: torch.Tensor, body: graph, num_scan_inputs: int, scan_input_axes: ints, scan_input_directions: ints, scan_output_axes: ints, scan_output_directions: ints,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def Scan_16(initial_state_and_scan_inputs: torch.Tensor, body: graph, num_scan_inputs: int, scan_input_axes: ints, scan_input_directions: ints, scan_output_axes: ints, scan_output_directions: ints,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def Scan_19(initial_state_and_scan_inputs: torch.Tensor, body: graph, num_scan_inputs: int, scan_input_axes: ints, scan_input_directions: ints, scan_output_axes: ints, scan_output_directions: ints,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def Scan_21(initial_state_and_scan_inputs: torch.Tensor, body: graph, num_scan_inputs: int, scan_input_axes: ints, scan_input_directions: ints, scan_output_axes: ints, scan_output_directions: ints,) -> torch.Tensor: ...
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }
"""


def DequantizeLinear_10(x: torch.Tensor, x_scale: torch.Tensor, x_zero_point: torch.Tensor, ) -> torch.Tensor: ...
r"""
The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
"""


def DequantizeLinear_13(x: torch.Tensor, x_scale: torch.Tensor, x_zero_point: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
"""


def DequantizeLinear_19(x: torch.Tensor, x_scale: torch.Tensor, x_zero_point: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
"""


def DequantizeLinear_21(x: torch.Tensor, x_scale: torch.Tensor, x_zero_point: torch.Tensor, axis: int, block_size: int,) -> torch.Tensor: ...
r"""
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 types quantization, but the dequantization formula remains the same
for consistency, and `x_scale` still determines the output type.
"""


def ThresholdedRelu_10(X: torch.Tensor, alpha: float,) -> torch.Tensor: ...
r"""
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
"""


def ReduceL1_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L1 norm of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceL1_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L1 norm of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceL1_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceL1_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Expand_8(input: torch.Tensor, shape: torch.Tensor, ) -> torch.Tensor: ...
r"""
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
"""


def Expand_13(input: torch.Tensor, shape: torch.Tensor, ) -> torch.Tensor: ...
r"""
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
"""


def Multinomial_7(input: torch.Tensor, dtype: int, sample_size: int, seed: float,) -> torch.Tensor: ...
r"""
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
"""


def Asin_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
"""


def Xor_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def Xor_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Einsum_12(Inputs: torch.Tensor, equation: string,) -> torch.Tensor: ...
r"""
An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term2] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
"""


def ReduceSum_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceSum_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceSum_13(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Floor_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
"""


def Floor_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
"""


def Floor_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
"""


def ReduceSumSquare_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the sum square of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceSumSquare_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the sum square of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceSumSquare_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the sum square of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceSumSquare_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the sum square of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Upsample_1(X: torch.Tensor, height_scale: float, mode: string, width_scale: float,) -> torch.Tensor: ...
r"""
Upsample the input tensor.
The width and height of the output tensor are:
  output_width = floor(input_width * width_scale),
  output_height = floor(input_height * height_scale).
Example:
  Given `data` tensor, width_scale, height_scale, mode,
  Upsample the input 4-D tensor in nearest mode:
  data = [[[
      [1, 2],
      [3, 4]
  ]]]
  width_scale = 2
  height_scale = 2
  mode = "nearest"
  output = [[[
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]
  ]]]
"""


def Upsample_7(X: torch.Tensor, mode: string, scales: floats,) -> torch.Tensor: ...
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
"""


def Upsample_9(X: torch.Tensor, scales: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
"""


def Upsample_10(X: torch.Tensor, scales: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
"""


def And_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def And_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Tile_1(input: torch.Tensor, tiles: torch.Tensor, axis: torch.Tensor, ) -> torch.Tensor: ...
r"""
Repeat the elements of a tensor along an axis.
"""


def Tile_6(input: torch.Tensor, repeats: torch.Tensor, ) -> torch.Tensor: ...
r"""
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
"""


def Tile_13(input: torch.Tensor, repeats: torch.Tensor, ) -> torch.Tensor: ...
r"""
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
"""


def MaxRoiPool_1(X: torch.Tensor, rois: torch.Tensor, pooled_shape: ints, spatial_scale: float,) -> torch.Tensor: ...
r"""
ROI max pool consumes an input tensor X and region of interests (RoIs) to
apply max pooling across each RoI, to produce output 4-D tensor of shape
(num_rois, channels, pooled_shape[0], pooled_shape[1]).
"""


def RandomUniform_1( dtype: int, high: float, low: float, seed: float, shape: ints,) -> torch.Tensor: ...
r"""
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
"""


def Squeeze_1(data: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""


def Squeeze_11(data: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""


def Squeeze_13(data: torch.Tensor, axes: torch.Tensor, ) -> torch.Tensor: ...
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""


def Squeeze_21(data: torch.Tensor, axes: torch.Tensor, ) -> torch.Tensor: ...
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""


def Acosh_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic arccosine of the given input tensor element-wise.
"""


def ReduceLogSum_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceLogSum_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceLogSum_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceLogSum_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the log sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Split_1(input: torch.Tensor, split: torch.Tensor, axis: int, split: ints,) -> torch.Tensor: ...
r"""
Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
"""


def Split_2(input: torch.Tensor, axis: int, split: ints,) -> torch.Tensor: ...
r"""
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
"""


def Split_11(input: torch.Tensor, axis: int, split: ints,) -> torch.Tensor: ...
r"""
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
"""


def Split_13(input: torch.Tensor, split: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
"""


def Split_18(input: torch.Tensor, split: torch.Tensor, axis: int, num_outputs: int,) -> torch.Tensor: ...
r"""
Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.
"""


def Where_9(condition: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Where_16(condition: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Sqrt_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
"""


def Sqrt_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
"""


def Sqrt_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
"""


def Softsign_1(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
"""


def Softplus_1(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
"""


def ReduceMax_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMax_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMax_12(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMax_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceMax_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceMax_20(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


If the input data type is Boolean, the comparison should consider `False < True`.

The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ImageDecoder_20(encoded_stream: torch.Tensor, pixel_format: string,) -> torch.Tensor: ...
r"""
Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
stream, invalid format, it will return an empty matrix).
The following image formats are supported:
* BMP
* JPEG (note: Lossless JPEG support is optional)
* JPEG2000
* TIFF
* PNG
* WebP
* Portable image format (PBM, PGM, PPM, PXM, PNM)
Decoded images follow a channel-last layout: (Height, Width, Channels).
**JPEG chroma upsampling method:**
When upsampling the chroma components by a factor of 2, the pixels are linearly interpolated so that the
centers of the output pixels are 1/4 and 3/4 of the way between input pixel centers.
When rounding, 0.5 is rounded down and up at alternative pixels locations to prevent bias towards
larger values (ordered dither pattern).
Considering adjacent input pixels A, B, and C, B is upsampled to pixels B0 and B1 so that
```
B0 = round_half_down((1/4) * A + (3/4) * B)
B1 = round_half_up((3/4) * B + (1/4) * C)
```
This method,  is the default chroma upsampling method in the well-established libjpeg-turbo library,
also referred as "smooth" or "fancy" upsampling.
"""


def Cos_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the cosine of the given input tensor, element-wise.
"""


def HardSigmoid_1(X: torch.Tensor, alpha: float, beta: float, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
"""


def HardSigmoid_6(X: torch.Tensor, alpha: float, beta: float,) -> torch.Tensor: ...
r"""
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
"""


def SpaceToDepth_1(input: torch.Tensor, blocksize: int,) -> torch.Tensor: ...
r"""
SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
"""


def SpaceToDepth_13(input: torch.Tensor, blocksize: int,) -> torch.Tensor: ...
r"""
SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
"""


def FeatureVectorizer_1(X: torch.Tensor, inputdimensions: ints,) -> torch.Tensor: ...
r"""
Concatenates input tensors into one continuous output.

All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
Inputs are copied to the output maintaining the order of the input arguments.

All inputs must be integers or floats, while the output will be all floating point values.
"""


def GreaterOrEqual_12(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def GreaterOrEqual_16(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Acos_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
"""


def Erf_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Computes the error function of the given input tensor element-wise.
"""


def Erf_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Computes the error function of the given input tensor element-wise.
"""


def Size_1(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
"""


def Size_13(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
"""


def Size_19(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
"""


def Size_21(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
"""


def AveragePool_1(X: torch.Tensor, auto_pad: string, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
AveragePool consumes an input tensor X and applies average pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
average pooling consisting of computing the average on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
The output of each pooling window is divided by the number of elements exclude pad.
"""


def AveragePool_7(X: torch.Tensor, auto_pad: string, count_include_pad: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
AveragePool consumes an input tensor X and applies average pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
average pooling consisting of computing the average on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
"""


def AveragePool_10(X: torch.Tensor, auto_pad: string, ceil_mode: int, count_include_pad: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
AveragePool consumes an input tensor X and applies average pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
average pooling consisting of computing the average on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled

```
* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
"""


def AveragePool_11(X: torch.Tensor, auto_pad: string, ceil_mode: int, count_include_pad: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
or when ceil_mode is disabled:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
 ```

 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
"""


def AveragePool_19(X: torch.Tensor, auto_pad: string, ceil_mode: int, count_include_pad: int, dilations: ints, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
AveragePool consumes an input tensor X and applies average pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
average pooling consisting of computing the average on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape is calculated differently
depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
```
VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```
The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
"""


def Tanh_1(input: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Calculates the hyperbolic tangent of the given input tensor element-wise.
"""


def Tanh_6(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic tangent of the given input tensor element-wise.
"""


def Tanh_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic tangent of the given input tensor element-wise.
"""


def Transpose_1(data: torch.Tensor, perm: ints,) -> torch.Tensor: ...
r"""
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
"""


def Transpose_13(data: torch.Tensor, perm: ints,) -> torch.Tensor: ...
r"""
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
"""


def Transpose_21(data: torch.Tensor, perm: ints,) -> torch.Tensor: ...
r"""
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
"""


def Shape_1(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
"""


def Shape_13(data: torch.Tensor, ) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
"""


def Shape_15(data: torch.Tensor, end: int, start: int,) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r-1], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
"""


def Shape_19(data: torch.Tensor, end: int, start: int,) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r-1], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
"""


def Shape_21(data: torch.Tensor, end: int, start: int,) -> torch.Tensor: ...
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r-1], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
"""


def TopK_1(X: torch.Tensor, axis: int, k: int,) -> torch.Tensor: ...
r"""
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:
  -Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
"""


def TopK_10(X: torch.Tensor, K: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:
  -Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
   contains the indices of the top k elements (original indices from the input
   tensor).

Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
"""


def TopK_11(X: torch.Tensor, K: torch.Tensor, axis: int, largest: int, sorted: int,) -> torch.Tensor: ...
r"""
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
"""


def Sum_1(data_0: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Sum_6(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Sum_8(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Sum_13(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Atanh_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic arctangent of the given input tensor element-wise.
"""


def Relu_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
"""


def Relu_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
"""


def Relu_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
"""


def Relu_14(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
"""


def NegativeLogLikelihoodLoss_12(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, ignore_index: int, reduction: string,) -> torch.Tensor: ...
r"""
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
When an optional "weight" is provided, the sample loss is calculated as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
loss is zero for the case when target-value equals ignore_index.

    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
    mean(loss), if "weight" is not provided,
or if weight is provided,
    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).
See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
Example 1:
    // negative log likelihood loss, "none" reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1]
    // print(loss)
    // [[-3. -2.]
    //  [-0. -2.]]
Example 2:
    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
    loss = np.sum(loss)
    // print(loss)
    // -1.1
Example 3:
    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
            weight_total = weight_total + weight[c]
    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57
"""


def NegativeLogLikelihoodLoss_13(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, ignore_index: int, reduction: string,) -> torch.Tensor: ...
r"""
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```

When an optional "weight" is provided, the sample loss is calculated as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
```

loss is zero for the case when target-value equals ignore_index.

```
loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
```

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

```
mean(loss), if "weight" is not provided,
```

or if weight is provided,

```
sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
```

If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

```
// negative log likelihood loss, "none" reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
          [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]

loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1]

// print(loss)
// [[-3. -2.]
//  [-0. -2.]]
```

Example 2:

```
// weighted negative log likelihood loss, sum reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]

loss = np.sum(loss)
// print(loss)
// -1.1
```

Example 3:

```
// weighted negative log likelihood loss, mean reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
weight_total = 0
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]
        weight_total = weight_total + weight[c]

loss = np.sum(loss) / weight_total
// print(loss)
// -1.57
```
"""


def SequenceLength_11(input_sequence: torch.Tensor, ) -> torch.Tensor: ...
r"""
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
"""


def ReduceMin_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMin_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMin_12(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMin_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceMin_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceMin_20(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


If the input data type is Boolean, the comparison should consider `False < True`.

The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Mul_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Performs element-wise binary multiplication (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Mul_6(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary multiplication (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Mul_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Mul_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Mul_14(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
"""


def RandomUniformLike_1(input: torch.Tensor, dtype: int, high: float, low: float, seed: float,) -> torch.Tensor: ...
r"""
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
"""


def ConvTranspose_1(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, output_padding: ints, output_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
"""


def ConvTranspose_11(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, output_padding: ints, output_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
"""


def Sin_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the sine of the given input tensor, element-wise.
"""


def Gradient_1(Inputs: torch.Tensor, xs: strings, y: string, zs: strings,) -> torch.Tensor: ...
r"""
Gradient operator computes the partial derivatives of a specific tensor w.r.t.
some other tensors. This operator is widely used in gradient-based training
algorithms. To illustrate its use, let's consider a computation graph,

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

, where W and Z are trainable tensors. Note that operators' attributes are
omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
Y with respect to W (Z). The user can compute gradient by inserting Gradient
operator to form another graph shown below.

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
|      |   |   "xs" followed by "zs")
|      v   v
'---> Gradient(xs=["W", "Z"], zs=["X"], y="Y")
       |   |
       |   '-----------------------------------> dY/dW (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

By definition, the tensor "y" is a function of independent variables in "xs"
and "zs". Since we only compute the gradient of "y" w.r.t. the differentiable
variables in "xs", this Gradient only outputs dY/dW and dY/dZ. Note that "H"
cannot appear in "xs" and "zs". The reason is that "H" can be determined by
tensors "W" and "X" and therefore "H" is not an independent variable.

All outputs are optional. If needed, for example, user can assign an empty
string to the 1st output name of that Gradient to skip the generation of dY/dW.
Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done via

```
W --> Conv --> H --> Gemm --> Y
       ^       |      ^
       |       |      |
       X       |      Z
       .-------'      |
       |   .----------'
       |   | (H/Z is the 1st/2nd input of Gradient as shown in "xs")
       v   v
      Gradient(xs=["H", "Z"], y="Y")
       |   |
       |   '-----------------------------------> dY/dH (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

It is possible to represent high-order differentiation using Gradient operators.
For example, given the following linear model:

```
W --> Gemm --> Y --> Loss --> O
       ^              ^
       |              |
       X              L
```

To compute the 2nd order derivative of O with respect to W (denoted by
d^2O/dW^2), one can do

```
W --> Gemm --> Y --> Loss --> O
|      ^              ^
|      |              |
|      X .------------L
|      | |            |
|      | |            v
+------+-+> Gradient(xs=["X", "W"], zs=["L"], y="O") ---> dO/dX (1st output of Gradient)
|      | |    |
|      | |    '---> dO/dW (2nd output of Gradient)
|      v v
'---> Gradient(xs=["X", "W"], zs=["L"], y="dO/dW") ---> d(dO/dW)dX (1st output of
       |                                                  Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

The tensors named in attributes "xs", "zs", and "y" define the differentiated
computation graph, and the inputs to Gradient node define the values at
which the gradient is computed. We can feed different tensors to the identified
graph. For example, one can compute the gradient of Y with respect to H at
a specific value of H, H_1, by providing that value as an input to the Gradient
node.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

          Z_1 (2nd input of Gradient)
           |
           v
H_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dH when H = H_1 and Y = Y_1.
           |
           '------------------------------> dY/dZ (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in "xs" and "zs", the
computation can be optimized. More specifically, intermediate variables in
forward pass can be reused if the gradient is computed via reverse-mode
auto-differentiation.
"""


def SVMClassifier_1(X: torch.Tensor, classlabels_ints: ints, classlabels_strings: strings, coefficients: floats, kernel_params: floats, kernel_type: string, post_transform: string, prob_a: floats, prob_b: floats, rho: floats, support_vectors: floats, vectors_per_class: ints,) -> torch.Tensor: ...
r"""
Support Vector Machine classifier
"""


def Sigmoid_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
"""


def Sigmoid_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
"""


def Sigmoid_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
"""


def RandomNormalLike_1(input: torch.Tensor, dtype: int, mean: float, scale: float, seed: float,) -> torch.Tensor: ...
r"""
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
"""


def Asinh_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic arcsine of the given input tensor element-wise.
"""


def RNN_1(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, output_sequence: int,) -> torch.Tensor: ...
r"""
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*Ri + Wbi + Rbi)
"""


def RNN_7(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int,) -> torch.Tensor: ...
r"""
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def RNN_14(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, layout: int,) -> torch.Tensor: ...
r"""
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `t` - time step (t-1 means previous time step)
* `Wi` - W parameter weight matrix for input gate
* `Ri` - R recurrence weight matrix for input gate
* `Wbi` - W parameter bias vector for input gate
* `Rbi` - R parameter bias vector for input gate
* `WBi` - W parameter weight matrix for backward input gate
* `RBi` - R recurrence weight matrix for backward input gate
* `WBbi` - WR bias vectors for backward input gate
* `RBbi` - RR bias vectors for backward input gate
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

* Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Pad_1(data: torch.Tensor, mode: string, paddings: ints, value: float,) -> torch.Tensor: ...
r"""
Given `data` tensor, paddings, mode, and value.
Example:
  Insert 0 paddings to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  paddings = [0, 0, 2, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
"""


def Pad_2(data: torch.Tensor, mode: string, pads: ints, value: float,) -> torch.Tensor: ...
r"""
Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
"""


def Pad_11(data: torch.Tensor, pads: torch.Tensor, constant_value: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]
"""


def Pad_13(data: torch.Tensor, pads: torch.Tensor, constant_value: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]
"""


def Pad_18(data: torch.Tensor, pads: torch.Tensor, constant_value: torch.Tensor, axes: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```
"""


def Pad_19(data: torch.Tensor, pads: torch.Tensor, constant_value: torch.Tensor, axes: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array

4) `wrap` - wrap-around padding as if the data tensor forms a torus


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Example 4 (`wrap` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [2, 1, 1, 1]

mode = 'wrap'

output = [
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
]
```
"""


def Pad_21(data: torch.Tensor, pads: torch.Tensor, constant_value: torch.Tensor, axes: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array

4) `wrap` - wrap-around padding as if the data tensor forms a torus


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Example 4 (`wrap` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [2, 1, 1, 1]

mode = 'wrap'

output = [
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
]
```
"""


def Conv_1(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The convolution operator consumes an input tensor and a filter, and
computes the output.
"""


def Conv_11(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, auto_pad: string, dilations: ints, group: int, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
The convolution operator consumes an input tensor and a filter, and
computes the output.
"""


def Slice_1(data: torch.Tensor, axes: ints, ends: ints, starts: ints,) -> torch.Tensor: ...
r"""
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
"""


def Slice_10(data: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, axes: torch.Tensor, steps: torch.Tensor, ) -> torch.Tensor: ...
r"""
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If a negative value is passed for step, it represents slicing backward.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
"""


def Slice_11(data: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, axes: torch.Tensor, steps: torch.Tensor, ) -> torch.Tensor: ...
r"""
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represents number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
when slicing forward and 'INT_MIN' when slicing backward.
If a negative value is passed for step, it represents slicing backward.
However step value cannot be 0.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
"""


def Slice_13(data: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, axes: torch.Tensor, steps: torch.Tensor, ) -> torch.Tensor: ...
r"""
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
`dims` are the dimensions of `input` and `steps[i] = 1`.

All negative elements of `axes` are made non-negative by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
`starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
and `[0, dims[axes[i]]-1]` for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `steps[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
axes = [0, 1]
starts = [1, 0]
ends = [2, 3]
steps = [1, 2]
result = [
    [5, 7],
]
```

Example 2:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
```
"""


def InstanceNormalization_1(input: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, consumed_inputs: ints, epsilon: float,) -> torch.Tensor: ...
r"""
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.
"""


def InstanceNormalization_6(input: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, epsilon: float,) -> torch.Tensor: ...
r"""
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.
"""


def ReduceLogSumExp_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum exponent of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceLogSumExp_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum exponent of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceLogSumExp_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceLogSumExp_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Neg_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
"""


def Neg_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
"""


def Neg_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
"""


def BitwiseNot_18(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the bitwise not of the input tensor element-wise.
"""


def Reshape_1(data: torch.Tensor, consumed_inputs: ints, shape: ints,) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
"""


def Reshape_5(data: torch.Tensor, shape: torch.Tensor, ) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
"""


def Reshape_13(data: torch.Tensor, shape: torch.Tensor, ) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
"""


def Reshape_14(data: torch.Tensor, shape: torch.Tensor, allowzero: int,) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
"""


def Reshape_19(data: torch.Tensor, shape: torch.Tensor, allowzero: int,) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
"""


def Reshape_21(data: torch.Tensor, shape: torch.Tensor, allowzero: int,) -> torch.Tensor: ...
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
"""


def LayerNormalization_17(X: torch.Tensor, Scale: torch.Tensor, B: torch.Tensor, axis: int, epsilon: float, stash_type: int,) -> torch.Tensor: ...
r"""
This is layer normalization defined in ONNX as function.
The overall computation can be split into two stages.
The first stage is standardization, which makes the
normalized elements have zero mean and unit variances.
The computation required by standardization can be
described by the following equations.
```
Mean = ReduceMean<axes=normalized_axes>(X)
D = Sub(X, Mean)
DD = Mul(D, D)
Var = ReduceMean<axes=normalized_axes>(DD)
VarEps = Add(Var, epsilon)
StdDev = Sqrt(VarEps)
InvStdDev = Reciprocal(StdDev)
Normalized = Mul(D, InvStdDev)
```
where `normalized_axes` is `[axis, ..., rank of X - 1]`.
The variables `Var` and `StdDev` stand for variance and
standard deviation, respectively. The second output is
`Mean` and the last one is `InvStdDev`.
Depending on `stash_type` attribute, the actual computation
must happen in different floating-point precision.
For example, if `stash_type` is 1, this operator casts
all input variables to 32-bit float, perform the computation, and
finally cast `Normalized` back to the original type of `X`.
The second stage then scales and shifts the outcome of the
first stage using
```
NormalizedScaled = Mul(Normalized, Scale)
Y = Add(NormalizedScaled, B)
```
The second stage doesn't depends on `stash_type`.
All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
The same variable (i.e., input, output, and attribute) uses
the same name in the equations above and this operator's definition.
Let `d[i]` indicate the i-th dimension of `X`.
If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
`Y` and `X` have the same shape. This operator supports unidirectional broadcasting
(tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
for more details please check [the doc](Broadcasting.md).
"""


def ReduceL2_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L2 norm of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceL2_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L2 norm of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceL2_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceL2_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ArrayFeatureExtractor_1(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Select elements of the input tensor based on the indices passed.

The indices are applied to the last axes of the tensor.
"""


def Min_1(data_0: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Min_6(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
"""


def Min_8(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Min_12(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Min_13(data_0: torch.Tensor, ) -> torch.Tensor: ...
r"""
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Flatten_1(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""


def Flatten_9(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""


def Flatten_11(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""


def Flatten_13(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""


def Flatten_21(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""


def RandomNormal_1( dtype: int, mean: float, scale: float, seed: float, shape: ints,) -> torch.Tensor: ...
r"""
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
"""


def CenterCropPad_18(input_data: torch.Tensor, shape: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
Center crop or pad an input to given dimensions.

The crop/pad dimensions can be specified for a subset of the `axes`. Non-specified dimensions will not be
cropped or padded.

If the input dimensions are bigger than the crop shape, a centered cropping window is extracted from the input.
If the input dimensions are smaller than the crop shape, the input is padded on each side equally,
so that the input is centered in the output.
"""


def HammingWindow_17(size: torch.Tensor, output_datatype: int, periodic: int,) -> torch.Tensor: ...
r"""
Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.
"""


def OptionalHasElement_15(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
"""


def OptionalHasElement_18(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
"""


def Bernoulli_15(input: torch.Tensor, dtype: int, seed: float,) -> torch.Tensor: ...
r"""
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
"""


def Loop_1(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]
      %keepgoing[BOOL, scalar]
      %b[INT32, scalar]
    ) {
      %my_local = Add(%a, %b)
      %b_out = Sub(%a, %b)
      %keepgoing_out = Greater(%my_local, %b_out)
      %user_defined_vals = Add(%b, %b)
      return %keepgoing_out, %b_out, %user_defined_vals
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      for (int i=0; i < max_trip_count && keepgoing; ++i) {
        /* User-defined code (loop body) */
        int my_local = a + b; // Reading values in the enclosing scope is fine
        b = a - b; // writes fine if we specify b as a loop-carried dependency
        keepgoing = my_local > b; // keepgoing is a loop-carried dependency
        user_defined_vals[i] = b + b;
        /* End user-defined code */
      }
      // my_local = 123; // Can't do this. my_local was defined in the body

      // These below values are live-out from the loop and therefore accessible
      b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable a here) are in scope and can
   be referenced in the inputs of the loop.
2) Any variables which you wish to make available in the enclosing scope (i.e.
   the variables b and keepgoing) must be declared as either loop-carried
   dependencies (both at the op inputs and output and at the body net input and
   output) or scan_outputs.
3) Values created in the body cannot be accessed in the enclosing scope.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).
"""


def Loop_11(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).
"""


def Loop_13(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
"""


def Loop_16(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var).

* input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

* input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
"""


def Loop_19(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var).

* input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

* input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
"""


def Loop_21(M: torch.Tensor, cond: torch.Tensor, v_initial: torch.Tensor, body: graph,) -> torch.Tensor: ...
r"""
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var).

* input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

* input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
"""


def Log_1(input: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Calculates the natural log of the given input tensor, element-wise.
"""


def Log_6(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the natural log of the given input tensor, element-wise.
"""


def Log_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the natural log of the given input tensor, element-wise.
"""


def LeakyRelu_1(X: torch.Tensor, alpha: float, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
"""


def LeakyRelu_6(X: torch.Tensor, alpha: float,) -> torch.Tensor: ...
r"""
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
"""


def LeakyRelu_16(X: torch.Tensor, alpha: float,) -> torch.Tensor: ...
r"""
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
"""


def BatchNormalization_1(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, consumed_inputs: ints, epsilon: float, is_test: int, momentum: float, spatial: int,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
"""


def BatchNormalization_6(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, epsilon: float, is_test: int, momentum: float, spatial: int,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
"""


def BatchNormalization_7(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, epsilon: float, momentum: float, spatial: int,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def BatchNormalization_9(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, epsilon: float, momentum: float,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def BatchNormalization_14(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, input_mean: torch.Tensor, input_var: torch.Tensor, epsilon: float, momentum: float, training_mode: int,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

Output case #1: Y, running_mean, running_var (training_mode=True)
Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

where:

current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)

Notice that ReduceVar refers to the population variance, and it equals to
sum(sqrd(x_i - x_avg)) / N
where N is the population size (this formula does not use sample size N - 1).

```

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def BatchNormalization_15(X: torch.Tensor, scale: torch.Tensor, B: torch.Tensor, input_mean: torch.Tensor, input_var: torch.Tensor, epsilon: float, momentum: float, training_mode: int,) -> torch.Tensor: ...
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Cosh_9(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the hyperbolic cosine of the given input tensor element-wise.
"""


def Cast_1(input: torch.Tensor, to: string,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
"""


def Cast_6(input: torch.Tensor, to: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
"""


def Cast_9(input: torch.Tensor, to: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
"""


def Cast_13(input: torch.Tensor, to: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules:

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.
"""


def Cast_19(input: torch.Tensor, saturate: int, to: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules
if the destination type is not a float 8 type.

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.

Float 8 type were introduced to speed up the training of
deep models. By default the conversion of a float *x* obeys
to the following rules. `[x]` means the value rounded to
the target mantissa width.

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | +/- FLT_MAX | NaN | FLT_MAX | NaN |
| [x] > FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX |
| [x] < -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else | RNE | RNE | RNE | RNE |

The behavior changes if the parameter 'saturate' is set to False.
The rules then become:

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | NaN | NaN | +/- Inf | NaN |
| [x] > FLT_MAX | NaN | NaN | Inf | NaN |
| [x] < -FLT_MAX | NaN | NaN | -Inf | NaN |
| else | RNE | RNE | RNE | RNE |
"""


def Cast_21(input: torch.Tensor, saturate: int, to: int,) -> torch.Tensor: ...
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules
if the destination type is not a float 8 type.

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.

Float 8 type were introduced to speed up the training of
deep models. By default the conversion of a float *x* obeys
to the following rules. `[x]` means the value rounded to
the target mantissa width.

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | +/- FLT_MAX | NaN | FLT_MAX | NaN |
| [x] > FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX |
| [x] < -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else | RNE | RNE | RNE | RNE |

The behavior changes if the parameter 'saturate' is set to False.
The rules then become:

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | NaN | NaN | +/- Inf | NaN |
| [x] > FLT_MAX | NaN | NaN | Inf | NaN |
| [x] < -FLT_MAX | NaN | NaN | -Inf | NaN |
| else | RNE | RNE | RNE | RNE |
"""


def Not_1(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the negation of the input tensor element-wise.
"""


def LSTM_1(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, initial_c: torch.Tensor, P: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, input_forget: int, output_sequence: int,) -> torch.Tensor: ...
r"""
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
"""


def LSTM_7(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, initial_c: torch.Tensor, P: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, input_forget: int,) -> torch.Tensor: ...
r"""
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def LSTM_14(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, initial_c: torch.Tensor, P: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, input_forget: int, layout: int,) -> torch.Tensor: ...
r"""
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Unsqueeze_1(data: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
"""


def Unsqueeze_11(data: torch.Tensor, axes: ints,) -> torch.Tensor: ...
r"""
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
"""


def Unsqueeze_13(data: torch.Tensor, axes: torch.Tensor, ) -> torch.Tensor: ...
r"""
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example, given an input tensor (`data`) of shape [3, 4, 5], then
Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
"""


def Unsqueeze_21(data: torch.Tensor, axes: torch.Tensor, ) -> torch.Tensor: ...
r"""
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example, given an input tensor (`data`) of shape [3, 4, 5], then
Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
"""


def ArgMax_1(data: torch.Tensor, axis: int, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.
"""


def ArgMax_11(data: torch.Tensor, axis: int, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
The input tensor must not be empty.
The type of the output tensor is integer.
"""


def ArgMax_12(data: torch.Tensor, axis: int, keepdims: int, select_last_index: int,) -> torch.Tensor: ...
r"""
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""


def ArgMax_13(data: torch.Tensor, axis: int, keepdims: int, select_last_index: int,) -> torch.Tensor: ...
r"""
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""


def LRN_1(X: torch.Tensor, alpha: float, beta: float, bias: float, size: int,) -> torch.Tensor: ...
r"""
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
"""


def LRN_13(X: torch.Tensor, alpha: float, beta: float, bias: float, size: int,) -> torch.Tensor: ...
r"""
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
of shape `(N x C x D1 x D2, ..., Dk)`, its region is
`{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

`square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

`Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
"""


def SequenceEmpty_11( dtype: int,) -> torch.Tensor: ...
r"""
Construct an empty tensor sequence, with given data type.
"""


def Adam_1(R: torch.Tensor, T: torch.Tensor, inputs: torch.Tensor, alpha: float, beta: float, epsilon: float, norm_coefficient: float, norm_coefficient_post: float,) -> torch.Tensor: ...
r"""
Compute one iteration of Adam, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. First of all, Adam requires
some parameters:

 - The learning-rate "R".
 - The update count "T". That is, the number of training iterations conducted.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A small constant "epsilon" to avoid dividing-by-zero.
 - Two coefficients, "alpha" and "beta".

At each Adam iteration, the optimized tensors are moved along a direction
computed based on their exponentially-averaged historical gradient and
exponentially-averaged historical squared gradient. Assume that only a tensor
"X" is being optimized. The rest of required information is

 - the value of "X",
 - "X"'s gradient (denoted by "G"),
 - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
 - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

Some of those parameters are passed into this operator as input tensors and others
are stored as this operator's attributes. Specifically, this operator's input tensor
list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
the second input, and so on. Other parameters are given as attributes because they
are constants. Moreover, the corresponding output tensors are

 - the new value of "X" (called "X_new"),
 - the new exponentially-averaged historical gradient (denoted by "V_new"), and
 - the new exponentially-averaged historical squared gradient (denoted by "H_new").

Those outputs are computed following the pseudo code below.

Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
numpy-style broadcasting support. The pseudo code to compute those outputs is:

  // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
  G_regularized = norm_coefficient * X + G

  // Update exponentially-averaged historical gradient.
  V_new = alpha * V + (1 - alpha) * G_regularized

  // Update exponentially-averaged historical squared gradient.
  H_new = beta * H + (1 - beta) * G_regularized * G_regularized

  // Compute the element-wise square-root of H_new. V_new will be element-wisely
  // divided by H_sqrt for a better update direction.
  H_sqrt = Sqrt(H_new) + epsilon

  // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
  R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

  // Compute new value of "X".
  X_new = X - R_adjusted * V_new / H_sqrt

  // Post-update regularization.
  X_final = (1 - norm_coefficient_post) * X_new

If there are multiple inputs to be optimized, the pseudo code will be applied
independently to each of them.
"""


def Exp_1(input: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Calculates the exponential of the given input tensor, element-wise.
"""


def Exp_6(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the exponential of the given input tensor, element-wise.
"""


def Exp_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the exponential of the given input tensor, element-wise.
"""


def SoftmaxCrossEntropyLoss_12(scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, ignore_index: int, reduction: string,) -> torch.Tensor: ...
r"""
Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
"""


def SoftmaxCrossEntropyLoss_13(scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, ignore_index: int, reduction: string,) -> torch.Tensor: ...
r"""
Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

* shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.
* shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
```
or
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
```

loss is zero for the case when label-value equals ignore_index.
```
l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
```

where:
```
p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
```

Finally, L is optionally reduced:

* If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
* If reduction = 'sum', the output is scalar: Sum(L).
* If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
  where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
"""


def Concat_1(inputs: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Concatenate a list of tensors into a single tensor
"""


def Concat_4(inputs: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Concatenate a list of tensors into a single tensor
"""


def Concat_11(inputs: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
"""


def Concat_13(inputs: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
"""


def GRU_1(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, output_sequence: int,) -> torch.Tensor: ...
r"""
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
"""


def GRU_3(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, linear_before_reset: int, output_sequence: int,) -> torch.Tensor: ...
r"""
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
"""


def GRU_7(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, linear_before_reset: int,) -> torch.Tensor: ...
r"""
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def GRU_14(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor, sequence_lens: torch.Tensor, initial_h: torch.Tensor, activation_alpha: floats, activation_beta: floats, activations: strings, clip: float, direction: string, hidden_size: int, layout: int, linear_before_reset: int,) -> torch.Tensor: ...
r"""
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def If_1(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def If_11(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def If_13(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def If_16(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def If_19(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def If_21(cond: torch.Tensor, else_branch: graph, then_branch: graph,) -> torch.Tensor: ...
r"""
If conditional
"""


def CategoryMapper_1(X: torch.Tensor, cats_int64s: ints, cats_strings: strings, default_int64: int, default_string: string,) -> torch.Tensor: ...
r"""
Converts strings to integers and vice versa.

Two sequences of equal length are used to map between integers and strings,
with strings and integers at the same index detailing the mapping.

Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.

If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.
"""


def GridSample_16(X: torch.Tensor, grid: torch.Tensor, align_corners: int, mode: string, padding_mode: string,) -> torch.Tensor: ...
r"""
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For input `X` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out).

The tensor `X` contains values at centers of square pixels in a H by W 2-dimensional image.
The tensor `grid` describes normalized positions where the output `Y` is to be computed
using a specified interpolation method (the mode) and a padding mode (for grid positions falling outside the 2-dimensional image).

Elements in `grid[N, H_out, W_out]` are size-2 vectors specifying positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[N, C, H_out, W_out]`.

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).
"""


def GridSample_20(X: torch.Tensor, grid: torch.Tensor, align_corners: int, mode: string, padding_mode: string,) -> torch.Tensor: ...
r"""
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
"""


def Clip_1(input: torch.Tensor, consumed_inputs: ints, max: float, min: float,) -> torch.Tensor: ...
r"""
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
"""


def Clip_6(input: torch.Tensor, max: float, min: float,) -> torch.Tensor: ...
r"""
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
"""


def Clip_11(input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, ) -> torch.Tensor: ...
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""


def Clip_12(input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, ) -> torch.Tensor: ...
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""


def Clip_13(input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, ) -> torch.Tensor: ...
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""


def Abs_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
"""


def Abs_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
"""


def Abs_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.
"""


def Identity_1(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def Identity_13(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def Identity_14(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def Identity_16(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def Identity_19(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def Identity_21(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Identity operator
"""


def PRelu_1(X: torch.Tensor, slope: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
"""


def PRelu_6(X: torch.Tensor, slope: torch.Tensor, ) -> torch.Tensor: ...
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
"""


def PRelu_7(X: torch.Tensor, slope: torch.Tensor, ) -> torch.Tensor: ...
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
"""


def PRelu_9(X: torch.Tensor, slope: torch.Tensor, ) -> torch.Tensor: ...
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
"""


def PRelu_16(X: torch.Tensor, slope: torch.Tensor, ) -> torch.Tensor: ...
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
"""


def MatMul_1(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
"""


def MatMul_9(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
"""


def MatMul_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
"""


def DFT_17(input: torch.Tensor, dft_length: torch.Tensor, axis: int, inverse: int, onesided: int,) -> torch.Tensor: ...
r"""
Computes the discrete Fourier transform of input.
"""


def DFT_20(input: torch.Tensor, dft_length: torch.Tensor, axis: torch.Tensor, inverse: int, onesided: int,) -> torch.Tensor: ...
r"""
Computes the discrete Fourier Transform (DFT) of the input.

Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
DFT is computed and `M` denotes the conceptual "all other dimensions,"
the DFT `y[m, k]` of shape `[M, N]` is defined as

$$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$

and the inverse transform is defined as

$$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$

where $j$ is the imaginary unit.

The actual shape of the output is specified in the "output" section.

Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html
"""


def Resize_10(X: torch.Tensor, scales: torch.Tensor, mode: string,) -> torch.Tensor: ...
r"""
Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
"""


def Resize_11(X: torch.Tensor, roi: torch.Tensor, scales: torch.Tensor, sizes: torch.Tensor, coordinate_transformation_mode: string, cubic_coeff_a: float, exclude_outside: int, extrapolation_value: float, mode: string, nearest_mode: string,) -> torch.Tensor: ...
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
"""


def Resize_13(X: torch.Tensor, roi: torch.Tensor, scales: torch.Tensor, sizes: torch.Tensor, coordinate_transformation_mode: string, cubic_coeff_a: float, exclude_outside: int, extrapolation_value: float, mode: string, nearest_mode: string,) -> torch.Tensor: ...
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
"""


def Resize_18(X: torch.Tensor, roi: torch.Tensor, scales: torch.Tensor, sizes: torch.Tensor, antialias: int, axes: ints, coordinate_transformation_mode: string, cubic_coeff_a: float, exclude_outside: int, extrapolation_value: float, keep_aspect_ratio_policy: string, mode: string, nearest_mode: string,) -> torch.Tensor: ...
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is: <br/>
  `output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)` <br/>
if input \"sizes\" is not specified.
"""


def Resize_19(X: torch.Tensor, roi: torch.Tensor, scales: torch.Tensor, sizes: torch.Tensor, antialias: int, axes: ints, coordinate_transformation_mode: string, cubic_coeff_a: float, exclude_outside: int, extrapolation_value: float, keep_aspect_ratio_policy: string, mode: string, nearest_mode: string,) -> torch.Tensor: ...
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \"sizes\" is not specified.
"""


def Atan_7(input: torch.Tensor, ) -> torch.Tensor: ...
r"""
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
"""


def GlobalLpPool_1(X: torch.Tensor, p: float,) -> torch.Tensor: ...
r"""
GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
the values in the same channel. This is equivalent to LpPool with kernel size
equal to the spatial dimension of input tensor.
"""


def GlobalLpPool_2(X: torch.Tensor, p: int,) -> torch.Tensor: ...
r"""
GlobalLpPool consumes an input tensor X and applies lp pool pooling across
the values in the same channel. This is equivalent to LpPool with kernel size
equal to the spatial dimension of input tensor.
"""


def SequenceConstruct_11(inputs: torch.Tensor, ) -> torch.Tensor: ...
r"""
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
"""


def GlobalAveragePool_1(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
GlobalAveragePool consumes an input tensor X and applies average pooling across
the values in the same channel. This is equivalent to AveragePool with kernel size
equal to the spatial dimension of input tensor.
"""


def Unique_11(X: torch.Tensor, axis: int, sorted: int,) -> torch.Tensor: ...
r"""
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
```
input_X = [2, 1, 1, 3, 4, 3]
attribute_sorted = 0
attribute_axis = None
output_Y = [2, 1, 3, 4]
output_indices = [0, 1, 3, 4]
output_inverse_indices = [0, 1, 1, 2, 3, 2]
output_counts = [1, 2, 2, 1]
```

Example 2:
```
input_X = [[1, 3], [2, 3]]
attribute_sorted = 1
attribute_axis = None
output_Y = [1, 2, 3]
output_indices = [0, 2, 1]
output_inverse_indices = [0, 2, 1, 2]
output_counts = [1, 1, 2]
```

Example 3:
```
input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
attribute_sorted = 1
attribute_axis = 0
output_Y = [[1, 0, 0], [2, 3, 4]]
output_indices = [0, 2]
output_inverse_indices = [0, 0, 1]
output_counts = [2, 1]
```

Example 4:
```
input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
            [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
attribute_sorted = 1
attribute_axis = 1
```

intermediate data are presented below for better understanding:
there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
```
A: [[1, 1], [1, 1]],
   [[0, 1], [0, 1]],
   [[2, 1], [2, 1]],
   [[0, 1], [0, 1]].
```

there are 3 unique subtensors:
```
[[1, 1], [1, 1]],
[[0, 1], [0, 1]],
[[2, 1], [2, 1]].
```

sorted unique subtensors:
```
B: [[0, 1], [0, 1]],
   [[1, 1], [1, 1]],
   [[2, 1], [2, 1]].
```

output_Y is constructed from B:
```
[[[0. 1.], [1. 1.], [2. 1.]],
 [[0. 1.], [1. 1.], [2. 1.]]]
```

output_indices is to map from B to A:
```
[1, 0, 2]
```

output_inverse_indices is to map from A to B:
```
[1, 0, 2, 0]
```

output_counts:
```
[2, 1, 1]
```
"""


def ArgMin_1(data: torch.Tensor, axis: int, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.
"""


def ArgMin_11(data: torch.Tensor, axis: int, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
The input tensor must not be empty.
The type of the output tensor is integer.
"""


def ArgMin_12(data: torch.Tensor, axis: int, keepdims: int, select_last_index: int,) -> torch.Tensor: ...
r"""
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""


def ArgMin_13(data: torch.Tensor, axis: int, keepdims: int, select_last_index: int,) -> torch.Tensor: ...
r"""
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""


def Add_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Performs element-wise binary addition (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Add_6(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary addition (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Add_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Add_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Add_14(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
"""


def Constant_1( value: tensor,) -> torch.Tensor: ...
r"""
A constant tensor.
"""


def Constant_9( value: tensor,) -> torch.Tensor: ...
r"""
A constant tensor.
"""


def Constant_11( sparse_value: sparse_tensor, value: tensor,) -> torch.Tensor: ...
r"""
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
"""


def Constant_12( sparse_value: sparse_tensor, value: tensor, value_float: float, value_floats: floats, value_int: int, value_ints: ints, value_string: string, value_strings: strings,) -> torch.Tensor: ...
r"""
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
"""


def Constant_13( sparse_value: sparse_tensor, value: tensor, value_float: float, value_floats: floats, value_int: int, value_ints: ints, value_string: string, value_strings: strings,) -> torch.Tensor: ...
r"""
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
"""


def Constant_19( sparse_value: sparse_tensor, value: tensor, value_float: float, value_floats: floats, value_int: int, value_ints: ints, value_string: string, value_strings: strings,) -> torch.Tensor: ...
r"""
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
"""


def Constant_21( sparse_value: sparse_tensor, value: tensor, value_float: float, value_floats: floats, value_int: int, value_ints: ints, value_string: string, value_strings: strings,) -> torch.Tensor: ...
r"""
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
"""


def Pow_1(X: torch.Tensor, Y: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Pow_7(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Pow_12(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Pow_13(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Pow_15(X: torch.Tensor, Y: torch.Tensor, ) -> torch.Tensor: ...
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def MaxPool_1(X: torch.Tensor, auto_pad: string, kernel_shape: ints, pads: ints, strides: ints,) -> torch.Tensor: ...
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
The output of each pooling window is maximum number of elements exclude pad.
"""


def MaxPool_8(X: torch.Tensor, auto_pad: string, kernel_shape: ints, pads: ints, storage_order: int, strides: ints,) -> torch.Tensor: ...
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
The output of each pooling window is maximum number of elements exclude pad.
"""


def MaxPool_10(X: torch.Tensor, auto_pad: string, ceil_mode: int, dilations: ints, kernel_shape: ints, pads: ints, storage_order: int, strides: ints,) -> torch.Tensor: ...
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled

```
* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```
The output of each pooling window is maximum number of elements exclude pad.
"""


def MaxPool_11(X: torch.Tensor, auto_pad: string, ceil_mode: int, dilations: ints, kernel_shape: ints, pads: ints, storage_order: int, strides: ints,) -> torch.Tensor: ...
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled

```
* pad_shape[i] is sum of pads along axis i
```

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```
The output of each pooling window is maximum number of elements exclude pad.
"""


def MaxPool_12(X: torch.Tensor, auto_pad: string, ceil_mode: int, dilations: ints, kernel_shape: ints, pads: ints, storage_order: int, strides: ints,) -> torch.Tensor: ...
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape is calculated differently
depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```
or
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
```
if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
```
VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
```
VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
```
And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
```
The output of each pooling window is maximum number of elements exclude pad.
"""


def Div_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Performs element-wise binary division (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Div_6(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Performs element-wise binary division (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
"""


def Div_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Div_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Div_14(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
"""


def ReduceMean_1(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the mean of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMean_11(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the mean of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
"""


def ReduceMean_13(data: torch.Tensor, axes: ints, keepdims: int,) -> torch.Tensor: ...
r"""
Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def ReduceMean_18(data: torch.Tensor, axes: torch.Tensor, keepdims: int, noop_with_empty_axes: int,) -> torch.Tensor: ...
r"""
Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
"""


def Normalizer_1(X: torch.Tensor, norm: string,) -> torch.Tensor: ...
r"""
Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':



    Max: Y = X / max(X)

    L1:  Y = X / sum(X)

    L2:  Y = sqrt(X^2 / sum(X^2)}

    In all modes, if the divisor is zero, Y == X.


    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.
"""


def Less_1(A: torch.Tensor, B: torch.Tensor, axis: int, broadcast: int,) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
"""


def Less_7(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Less_9(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Less_13(A: torch.Tensor, B: torch.Tensor, ) -> torch.Tensor: ...
r"""
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
"""


def Dropout_1(data: torch.Tensor, consumed_inputs: ints, is_test: int, ratio: float,) -> torch.Tensor: ...
r"""
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
"""


def Dropout_6(data: torch.Tensor, is_test: int, ratio: float,) -> torch.Tensor: ...
r"""
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
"""


def Dropout_7(data: torch.Tensor, ratio: float,) -> torch.Tensor: ...
r"""
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Dropout_10(data: torch.Tensor, ratio: float,) -> torch.Tensor: ...
r"""
Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Dropout_12(data: torch.Tensor, ratio: torch.Tensor, training_mode: torch.Tensor, seed: int,) -> torch.Tensor: ...
r"""
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Dropout_13(data: torch.Tensor, ratio: torch.Tensor, training_mode: torch.Tensor, seed: int,) -> torch.Tensor: ...
r"""
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def DepthToSpace_1(input: torch.Tensor, blocksize: int,) -> torch.Tensor: ...
r"""
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
"""


def DepthToSpace_11(input: torch.Tensor, blocksize: int, mode: string,) -> torch.Tensor: ...
r"""
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])

tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])

y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])


In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])

tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])

y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
"""


def DepthToSpace_13(input: torch.Tensor, blocksize: int, mode: string,) -> torch.Tensor: ...
r"""
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```
"""


def Ceil_1(X: torch.Tensor, consumed_inputs: ints,) -> torch.Tensor: ...
r"""
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
"""


def Ceil_6(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
"""


def Ceil_13(X: torch.Tensor, ) -> torch.Tensor: ...
r"""
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
"""


def Hardmax_1(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the hardmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
"""


def Hardmax_11(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the hardmax values of the corresponding input.
"""


def Hardmax_13(input: torch.Tensor, axis: int,) -> torch.Tensor: ...
r"""
The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The "axis" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
"""


def LpNormalization_1(input: torch.Tensor, axis: int, p: int,) -> torch.Tensor: ...
r"""
Given a matrix, apply Lp-normalization along the provided axis.
"""


def StringSplit_20(X: torch.Tensor, delimiter: string, maxsplit: int,) -> torch.Tensor: ...
r"""
StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.

The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the `delimiter` substring. This tensor has one additional rank compared to the input tensor in order to store the substrings for each input element (where the input tensor is not empty). Note that, in order to ensure the same number of elements are present in the final dimension, this tensor will pad empty strings as illustrated in the examples below. Consecutive delimiters are not grouped together and are deemed to delimit empty strings, except if the `delimiter` is unspecified or is the empty string (""). In the case where the `delimiter` is unspecified or the empty string, consecutive whitespace characters are regarded as a single separator and leading or trailing whitespace is removed in the output.

The second output tensor represents the number of substrings generated. `maxsplit` can be used to limit the number of splits performed - after the `maxsplit`th split if the string is not fully split, the trailing suffix of input string after the final split point is also added. For elements where fewer splits are possible than specified in `maxsplit`, it has no effect.
"""


def GroupNormalization_18(X: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, epsilon: float, num_groups: int,) -> torch.Tensor: ...
r"""
A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
"""


def GroupNormalization_21(X: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, epsilon: float, num_groups: int, stash_type: int,) -> torch.Tensor: ...
r"""
A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

The overall computation has two stages: the first stage normalizes the elements to
have zero mean and unit variance for each instance in each group, and the second
stage scales and shifts the results of the first stage. The floating-point precision
used in the first stage is determined by the `stash_type` attribute. For example,
if `stash_type` is 1, the operator casts all input variables to 32-bit float,
performs the computation, and finally casts the normalized results back to the
original type of `X`. The second stage does not depend on `stash_type`.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
"""


def LinearClassifier_1(X: torch.Tensor, classlabels_ints: ints, classlabels_strings: strings, coefficients: floats, intercepts: floats, multi_class: int, post_transform: string,) -> torch.Tensor: ...
r"""
Linear classifier
"""


def Gemm_1(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, broadcast: int, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
"""


def Gemm_6(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, broadcast: int, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
"""


def Gemm_7(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
"""


def Gemm_9(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
"""


def Gemm_11(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""


def Gemm_13(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float, transA: int, transB: int,) -> torch.Tensor: ...
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""
