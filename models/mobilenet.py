import torch
from torch import nn, Tensor
from typing import List, Optional, Callable

from models.common import _make_divisible, Conv2dNormActivation, DepthWiseSeparableConv2d

import torchvision.models._utils as _utils

__all__ = ["mobilenet_v1_025"]


class MobileNetV1(nn.Module):
    def __init__(self, width_mult: float = 1.0, num_classes: int = 1000):
        super().__init__()

        filters = [32, 64, 128, 256, 512, 1024]
        filters = [_make_divisible(filter * width_mult) for filter in filters]

        self.stage1: List[nn.Module] = nn.Sequential(
            Conv2dNormActivation(in_channels=3,  out_channels=filters[0], kernel_size=3, stride=2),
            DepthWiseSeparableConv2d(filters[0],  out_channels=filters[1], stride=1),
            DepthWiseSeparableConv2d(filters[1], out_channels=filters[2], stride=2),
            DepthWiseSeparableConv2d(filters[2], out_channels=filters[2], stride=1),
            DepthWiseSeparableConv2d(filters[2], out_channels=filters[3], stride=2),
            DepthWiseSeparableConv2d(filters[3], out_channels=filters[3], stride=1),  # (5) P / 8 -> 640 / 8 = 80
        )
        self.stage2: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(filters[3], out_channels=filters[4], stride=2),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),  # (11) P / 16 -> 640 / 16 = 40
        )
        self.stage3: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[5], stride=2),
            DepthWiseSeparableConv2d(filters[5], out_channels=filters[5], stride=1),  # (13) P / 32 -> 640 / 32 = 20
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[5], num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def mobilenet_v1_025(pretrained: bool = True, num_classes: int = 1000):
    model = MobileNetV1(width_mult=0.25, num_classes=num_classes)
    state_dict = None
    if pretrained:
        try:
            state_dict = torch.load("weights/mobilenetv1_025.pretrained", weights_only=True)
        except:
            print("Could not find pre-trained backbone model!")
            
        if state_dict:
            model.load_state_dict(state_dict)
            print("Pre-trained MobileNetV1_0.25 weights successfully loaded!")
    return model


if __name__ == "__main__":
    model = mobilenet_v1_025()

    x = torch.randn(1, 3, 640, 640)
    t = _utils.IntermediateLayerGetter(model, {'stage1': 1, 'stage2': 2, 'stage3': 3})

    a, b, c = list(t(x).values())

    print(a.size())
    print(b.size())
    print(c.size())
