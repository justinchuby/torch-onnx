"""Exercise upstream Transformers' real exporter and reversible patches, offline."""

from __future__ import annotations

import copy

import onnx
import onnx_ir as ir
import pytest
import torch
from transformers import (
    BertConfig,
    BertModel,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    ViTConfig,
    ViTModel,
)
from transformers.exporters import (
    ExportedModel,
    OnnxConfig,
    OnnxExporter,
    base,
    exporter_onnx,
)
from transformers.exporters.utils import get_leaf_tensors

import torch_onnx
from torch_onnx import _core, _exporter_compat

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def patched_exporter(monkeypatch):
    torch.manual_seed(0)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    programs = []
    original = _core.export

    def export(*args, **kwargs):
        program = original(*args, **kwargs)
        assert type(program) is torch_onnx.ONNXProgram
        programs.append(program)
        return program

    monkeypatch.setattr(_core, "export", export)
    torch_onnx.patch_torch(verbose=False)
    try:
        yield programs
    finally:
        for program in programs:
            program.release()
        torch_onnx.unpatch_torch()
        torch.set_num_threads(threads)


def make_model(family, *, use_cache=False):
    common = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
        max_position_embeddings=64,
        use_cache=use_cache,
    )
    if family == "bert":
        return BertModel(BertConfig(**common)).eval()
    if family == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**common)).eval()
    if family == "moe":
        return Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **common,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=16,
            )
        ).eval()
    if family == "hybrid":
        common["num_hidden_layers"] = 2
        return Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(
                **common,
                linear_conv_kernel_dim=3,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
            )
        ).eval()
    if family == "vit":
        return ViTModel(
            ViTConfig(
                image_size=8,
                patch_size=4,
                num_channels=3,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        ).eval()
    raise ValueError(family)


def inputs(family, batch, length):
    if family == "vit":
        return {
            "pixel_values": torch.arange(
                batch * 3 * length * length, dtype=torch.float32
            ).reshape(batch, 3, length, length)
            / 100,
            "interpolate_pos_encoding": True,
        }
    return {
        "input_ids": torch.arange(batch * length).reshape(batch, length) % 31,
        "attention_mask": torch.ones(batch, length, dtype=torch.int64),
    }


def assert_outputs(actual, expected):
    assert type(actual) is type(expected)
    actual_tensors, expected_tensors = (
        get_leaf_tensors(actual),
        get_leaf_tensors(expected),
    )
    assert actual_tensors.keys() == expected_tensors.keys()
    assert expected_tensors
    for name, expected_tensor in expected_tensors.items():
        torch.testing.assert_close(
            actual_tensors[name], expected_tensor, rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("family", ["bert", "qwen3", "moe", "hybrid", "vit"])
@pytest.mark.parametrize("dynamic,optimize", [(False, False), (True, True)])
def test_transformers_forward_export_and_reload(
    family,
    dynamic,
    optimize,
    tmp_path,
    patched_exporter,
    monkeypatch,
):
    from torch.onnx._internal.exporter import _core as torch_core

    model = make_model(family)
    # Capture the interpolation branch rather than the training-resolution shortcut.
    sample = inputs(family, 2, 12 if family == "vit" else 4)
    upstream_prepare = torch_core._prepare_exported_program_for_export
    fixes = exporter_onnx.apply_fx_node_fixes
    calls = []

    def apply_fixes(*args, **kwargs):
        calls.append(args[0])
        return fixes(*args, **kwargs)

    monkeypatch.setattr(exporter_onnx, "apply_fx_node_fixes", apply_fixes)
    exported = OnnxExporter().export(
        model,
        copy.deepcopy(sample),
        config=OnnxConfig(dynamic=dynamic, opset_version=23, optimize=optimize),
    )
    assert len(patched_exporter) == 1
    assert exported.program is patched_exporter[0].model
    assert (
        calls.count("onnx") >= 2
    )  # Transformers owns pre/post-decomposition FX repairs.
    assert torch_core._prepare_exported_program_for_export is upstream_prepare
    onnx.checker.check_model(ir.to_proto(exported.program), full_check=True)
    assert (
        _exporter_compat.constant_fold_policy()(
            type("Resize", (), {"op_type": "Resize", "domain": ""})()
        )
        is None
    )
    if family == "vit":
        assert [value.name for value in exported.program.graph.inputs] == [
            "pixel_values"
        ]
    cases = [sample]
    if dynamic:
        cases.append(inputs(family, 3, 16 if family == "vit" else 6))
    exported.save_pretrained(tmp_path)
    loaded = ExportedModel.from_pretrained(tmp_path)
    for data in cases:
        with torch.no_grad():
            expected = model(**copy.deepcopy(data))
        assert_outputs(exported(**copy.deepcopy(data)), expected)
        assert_outputs(loaded(**copy.deepcopy(data)), expected)


@pytest.mark.parametrize("multi_token_decode", [False, True])
def test_transformers_generation_components_and_reload(
    multi_token_decode,
    tmp_path,
    monkeypatch,
    patched_exporter,
):
    model = make_model("qwen3", use_cache=True)
    captured = {}
    decompose = base.decompose_for_generation

    def capture(*args, **kwargs):
        components = decompose(*args, **kwargs)
        captured.update(
            {
                name: (submodel, copy.deepcopy(data))
                for name, (submodel, data) in components.items()
            }
        )
        return components

    monkeypatch.setattr(base, "decompose_for_generation", capture)
    exported = OnnxExporter().export_for_generation(
        model,
        inputs("qwen3", 2, 4),
        config=OnnxConfig(dynamic=True, opset_version=23, optimize=True),
        multi_token_decode=multi_token_decode,
    )
    assert set(exported.components) == {"prefill", "decode"}
    assert len(patched_exporter) == 2
    for component in exported.components.values():
        onnx.checker.check_model(ir.to_proto(component.program), full_check=True)
    exported.save_pretrained(tmp_path)
    loaded = ExportedModel.from_pretrained(tmp_path)
    for name, (submodel, data) in captured.items():
        with torch.no_grad():
            expected = submodel(**copy.deepcopy(data))
        assert_outputs(exported.runners[name](**copy.deepcopy(data)), expected)
        assert_outputs(loaded.runners[name](**copy.deepcopy(data)), expected)


def test_transformers_optional_ort_fusions(tmp_path):
    model = make_model("qwen3")
    sample = inputs("qwen3", 2, 4)
    exported = OnnxExporter().export(
        model,
        copy.deepcopy(sample),
        config=OnnxConfig(
            dynamic=True, opset_version=23, optimize=True, ort_fusions="cpu"
        ),
    )
    onnx.checker.check_model(ir.to_proto(exported.program), full_check=True)
    exported.save_pretrained(tmp_path)
    loaded = ExportedModel.from_pretrained(tmp_path)
    for data in (sample, inputs("qwen3", 3, 6)):
        with torch.no_grad():
            expected = model(**copy.deepcopy(data))
        assert_outputs(exported(**copy.deepcopy(data)), expected)
        assert_outputs(loaded(**copy.deepcopy(data)), expected)


@pytest.mark.parametrize("family", ["bert", "qwen3", "vit"])
def test_transformers_default_config(family):
    model = make_model(family)
    sample = inputs(family, 2, 12 if family == "vit" else 4)
    exported = OnnxExporter().export(model, copy.deepcopy(sample), config=OnnxConfig())
    assert exported.program.opset_imports[""] == 28
    onnx.checker.check_model(ir.to_proto(exported.program), full_check=True)
    if family == "vit":
        assert any(node.op_type == "Resize" for node in exported.program.graph)
    with torch.no_grad():
        expected = model(**copy.deepcopy(sample))
    if family == "qwen3":
        # ORT 1.32 cannot expand Swish at opset 25-28. Verify the default
        # graph numerically with ONNX reference; explicit-opset-23 cases above
        # continue exercising the real Transformers ORT runner and reload.
        from onnx.reference import ReferenceEvaluator

        proto = ir.to_proto(exported.program)
        feed = {value.name: sample[value.name].numpy() for value in proto.graph.input}
        (actual,) = ReferenceEvaluator(proto).run(None, feed)
        torch.testing.assert_close(
            torch.from_numpy(actual), expected.logits, rtol=1e-4, atol=1e-5
        )
    else:
        assert_outputs(exported(**copy.deepcopy(sample)), expected)


@pytest.mark.parametrize("family", ["bert", "qwen3", "moe", "hybrid", "vit"])
def test_native_torchlib_without_transformers_operator_patches(
    family, monkeypatch, tmp_path
):
    """Native operator lowerings work while Transformers retains its FX fixes."""
    import contextlib

    monkeypatch.setattr(
        exporter_onnx, "apply_patches", lambda _backend: contextlib.nullcontext()
    )
    monkeypatch.setattr(exporter_onnx, "apply_onnx_ir_fixes", lambda _program: None)
    monkeypatch.setattr(exporter_onnx, "_ONNX_TRANSLATION_TABLE", {})
    model = make_model(family)
    sample = inputs(family, 2, 12 if family == "vit" else 4)
    exported = OnnxExporter().export(
        model,
        copy.deepcopy(sample),
        config=OnnxConfig(dynamic=True, opset_version=23, optimize=True),
    )
    onnx.checker.check_model(ir.to_proto(exported.program), full_check=True)
    exported.save_pretrained(tmp_path)
    loaded = ExportedModel.from_pretrained(tmp_path)
    for data in (sample, inputs(family, 3, 16 if family == "vit" else 6)):
        with torch.no_grad():
            expected = model(**copy.deepcopy(data))
        assert_outputs(exported(**copy.deepcopy(data)), expected)
        assert_outputs(loaded(**copy.deepcopy(data)), expected)
