import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils


from typing import List, Tuple

from models.common import Conv2dNormActivation, DepthWiseSeparableConv2d, DepthwiseConv2d


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = None, num_classes: int = 2, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.class_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=64, out_channels=anchors[1] * 2, kernel_size=3),
            DepthwiseConv2d(in_channels=128, out_channels=anchors[0] * 2, kernel_size=3),
            DepthwiseConv2d(in_channels=256, out_channels=anchors[0] * 2, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=anchors[1] * 2, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.class_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 2) for out in outputs], dim=1)
        return outputs


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = None, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.bbox_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=64, out_channels=anchors[1] * 4, kernel_size=3),
            DepthwiseConv2d(in_channels=128, out_channels=anchors[0] * 4, kernel_size=3),
            DepthwiseConv2d(in_channels=256, out_channels=anchors[0] * 4, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=anchors[1] * 4, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.bbox_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 4) for out in outputs], dim=1)
        return outputs


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = None, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=64, out_channels=anchors[1] * 10, kernel_size=3),
            DepthwiseConv2d(in_channels=128, out_channels=anchors[0] * 10, kernel_size=3),
            DepthwiseConv2d(in_channels=256, out_channels=anchors[0] * 10, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=anchors[1] * 10, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.landmark_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 10) for out in outputs], dim=1)
        return outputs


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1: List[nn.Module] = nn.Sequential(
            Conv2dNormActivation(in_channels=3, out_channels=16, stride=2),
            DepthWiseSeparableConv2d(in_channels=16, out_channels=32, stride=1),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=32, stride=2),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=32, stride=1),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=64, stride=2),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, stride=1),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, stride=1),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, stride=1)
        )

        self.stage2: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(in_channels=64, out_channels=128, stride=2),
            DepthWiseSeparableConv2d(in_channels=128, out_channels=128, stride=1),
            DepthWiseSeparableConv2d(in_channels=128, out_channels=128, stride=1)
        )

        self.stage3: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(in_channels=128, out_channels=256, stride=2),
            DepthWiseSeparableConv2d(in_channels=256, out_channels=256, stride=1)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            DepthwiseConv2d(64, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        return x


class Slim(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.anchors = [2, 3]
        backbone = FeatureExtractor()
        self.fx = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        self.class_head = ClassHead(anchors=self.anchors)
        self.bbox_head = BboxHead(anchors=self.anchors)
        self.landmark_head = LandmarkHead(anchors=self.anchors)

    def forward(self, x):
        outputs = self.fx(x)
        features = list(outputs.values())

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions)
        return output
