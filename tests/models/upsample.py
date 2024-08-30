import torch
from monai.networks.nets import SegResNet

import torch_onnx

torch_onnx.patch_torch(
    report=True, profile=True, verify=True, dump_exported_program=False, fallback=False
)


def main():
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).eval()
    data = torch.randn(1, 4, 224, 224, 128)

    # Check that the upsample op is not decomposed
    torch.onnx.export(model, (data,), "upsample.onnx")


if __name__ == "__main__":
    main()
