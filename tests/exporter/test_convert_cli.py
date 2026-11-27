"""Exercise the installed CLI implementation, including real pt2-to-ONNX conversion."""

import runpy
import sys

import onnx
import onnx_ir as ir
import pytest
import torch

import torch_onnx
from torch_onnx.bin import convert


@pytest.mark.parametrize("flag", ["--report", "--external_data"])
@pytest.mark.parametrize(
    "value,expected", [("true", True), ("FALSE", False), ("1", True), ("0", False)]
)
def test_boolean_option_values_are_not_python_string_truthiness(
    monkeypatch, flag, value, expected
):
    monkeypatch.setattr(sys, "argv", ["torch-onnx", "in.pt2", "out.onnx", flag, value])
    assert getattr(convert._parse_args(), flag.removeprefix("--")) is expected


@pytest.mark.parametrize("flag", ["--report", "--external_data"])
def test_invalid_boolean_option_is_actionable(monkeypatch, capsys, flag):
    monkeypatch.setattr(
        sys, "argv", ["torch-onnx", "in.pt2", "out.onnx", flag, "maybe"]
    )
    with pytest.raises(SystemExit) as error:
        convert._parse_args()
    assert error.value.code == 2
    assert "true or false" in capsys.readouterr().err


def test_help_does_not_attempt_to_load_a_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["torch-onnx", "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(convert.__file__, run_name="__main__")
    assert error.value.code == 0
    assert "pt2" in capsys.readouterr().out


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("report", [False, True])
def test_cli_converts_a_real_exported_program_and_preserves_external_data_choice(
    tmp_path, monkeypatch, external, report
):
    model = torch.nn.Linear(64, 64).eval()
    x = torch.randn(1, 64)
    ep = torch.export.export(model, (x,))
    source, destination = tmp_path / "model.pt2", tmp_path / "model.onnx"
    torch.export.save(ep, source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch-onnx",
            str(source),
            str(destination),
            "--external_data",
            str(external),
            "--report",
            str(report),
        ],
    )
    convert.main()
    onnx.checker.check_model(str(destination), full_check=True)
    proto = onnx.load(destination, load_external_data=False)
    assert (
        any(
            value.data_location == onnx.TensorProto.EXTERNAL
            for value in proto.graph.initializer
        )
        is external
    )
    assert bool(list(tmp_path.glob("*_success.md"))) is report
    loaded = torch_onnx.ONNXProgram(ir.load(destination), ep)
    try:
        torch.testing.assert_close(loaded(x)[0], model(x), rtol=1e-4, atol=1e-5)
    finally:
        loaded.release()


def test_cli_preserves_missing_input_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["torch-onnx", str(tmp_path / "missing.pt2"), str(tmp_path / "model.onnx")],
    )
    with pytest.raises(FileNotFoundError):
        convert.main()
