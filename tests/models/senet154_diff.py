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
        ["relu_2", "getitem_9"],
        data,
        keep_original_outputs=True,
    )
    print(results)


if __name__ == "__main__":
    main()
