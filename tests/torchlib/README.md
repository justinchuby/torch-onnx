# Native operator coverage

The OpInfo suite compares native ONNX translations with PyTorch's operator
samples. It requires the development dependencies and PyTorch's internal test
harness. Expanded OpInfo and model integration suites are opt-in:

```bash
python -m pytest
python -m pytest tests/torchlib/test_ops.py -m opinfo -k add
python -m pytest tests/exporter/test_small_models_e2e.py -m integration
```

`ops_test_data.py` indexes supported functions, source-supported dtypes,
numerical tolerances and sample-specific skips. `extra_opinfo.py` supplies
additional samples. Unsupported placeholder implementations do not add support.

Implementations receive an explicit first `TranslationContext` argument and
plain `onnx_ir.Value` inputs. Use `op.Add(...)` for schema-named operators and
`op.builder` for explicit variable arity, subgraphs and captures. Helpers receive
the context explicitly. Tensor casts, shape construction and control flow are
explicit; the builder converts Python constants but never promotes tensors.
Registration selects the target, real/complex branch and highest eligible opset,
not annotation/dtype overloads.

| Family | Native modules | Focused regression suite |
| --- | --- | --- |
| Scalar/tensor arithmetic, views, axes, indexing, scattering, masking, reductions | core, special | test_native_leaf_ops, test_native_structural_ops |
| Factories, windows, random sampling, products, Tensor[] | factories, linalg, sequences | test_native_factory_sequence_ops |
| GRU/LSTM, embedding bags, padding, interpolation, quantization | recurrent, embedding, residual | test_native_recurrent_residual_ops |
| Convolution, normalization, pooling, attention | normalization, neural, nn | test_native_neural_ops |
| FFT, prims, special, optional vision, decomposed quantization | fft, prims, special, vision, quantized_decomposed | test_native_final_ops |
| Registry, global custom translations, context, promotion | registry, common, _promotion | test_native_registration, test_custom_translation, test_ir_builder, test_native_type_promotion |

Focused suites prohibit legacy graph construction and exercise actual FX
exports with optimization both off and on. They use ONNX Runtime or ONNX's
reference evaluator as appropriate. Negative padding and negative-axis gathers
use ONNX Runtime because ONNX 1.23.2's reference kernels do not support those
cases correctly. Random tests check shape, dtype and range, not identical streams.
Boolean SplitToSequence uses the reference evaluator for older ORT kernels.

Recurrent hidden size must be known, although batch/sequence sizes may be
dynamic. Empty allocations are represented by zeros. Device/generator/layout
limitations and ONNX Hamming coefficient tolerances remain those of the source.
Training normalization is not forced into inference; saved inference statistics
retain the source CUDA convention rather than CPU's empty outputs. Attention
auxiliary outputs remain zero placeholders, not usable log-sum-exp statistics,
sequence offsets, debug masks or RNG state. Grouped-query fallback needs static
head counts to validate them at translation time; symbolic counts are checked
by the runtime. Vision targets require the optional namespace to be loaded;
registration does not import torchvision. DeformConv requires opset19, while
DFT's axis input starts at opset20.

Set `CATCH_ORT_SEGFAULT=1` for isolated runtime evaluation or
`CREATE_REPRODUCTION_REPORT=1` to write a failure reproduction report. Keep
source numerical coverage and skips when adding `TorchLibOpInfo` mappings.
