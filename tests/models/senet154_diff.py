import torch
import torch_onnx
import torch_onnx.tools
import torch_onnx.tools.diff_model
from monai.networks.nets import SENet154


def main():
    model = SENet154(spatial_dims=3, in_channels=2, num_classes=2).eval()
    data = (torch.randn(2, 2, 64, 64, 64),)
    # torch_onnx.export(model, data, verify=True, report=True)
    ep = torch.export.export(model, data)
    onnx_program = torch_onnx.export(ep, data)
    onnx_program.save("senet154.onnx", external_data=True)
    assert onnx_program.exported_program is not None
    results = torch_onnx.tools.diff_model.diff_exported_program(
        "senet154.onnx",
        onnx_program.exported_program,
        [
            "relu_126",
            "relu_150",
            "sigmoid_37",
            "relu_154",
            "sigmoid_38",
            "getitem_368",
            "mul_38",
            "add_38",
            "relu_158",
            # "relu_162",
            # "relu_174",
            # "relu_186",
            # "relu_190",
        ],
        data,
        keep_original_outputs=False,
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
