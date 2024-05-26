import torch
import torchvision


def test():
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    sample_input = (torch.randn(4, 3, 224, 224), )
    exported = torch.export.export(resnet18, sample_input)


if __name__ == '__main__':
    test()
