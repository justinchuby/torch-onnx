import torch
import torch_onnx
from monai.networks.nets import SegResNet

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

    program = torch.onnx.export(model, (data,))
    print(program)


if __name__ == "__main__":
    main()
