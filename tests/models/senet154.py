import torch
from monai.networks.nets import SENet154

import torch_onnx


def main():
    model = SENet154(spatial_dims=3, in_channels=2, num_classes=2).eval()
    data = (torch.randn(2, 2, 64, 64, 64),)
    # torch_onnx.export(model, data, verify=True, report=True)
    ep = torch.export.export(model, data)
    results = list(
        torch_onnx.verification.minimize_inaccurate_subgraph(ep, rtol=10, atol=1)
    )
    for i, result in enumerate(results):
        print(f"------Result {i}------")
        print(result)


if __name__ == "__main__":
    main()
