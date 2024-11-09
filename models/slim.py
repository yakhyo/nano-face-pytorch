"""
Author: Yakhyokhuja Valikhujaev
Date: 2024-11-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from models.common import Conv2dNormActivation, DepthWiseSeparableConv2d, DepthwiseConv2d


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 64, num_classes: int = 2, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.class_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=in_channels, out_channels=anchors[1] * 2, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 2, out_channels=anchors[0] * 2, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 4, out_channels=anchors[0] * 2, kernel_size=3),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=anchors[1] * 2, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.class_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 2) for out in outputs], dim=1)
        return outputs


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 64, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.bbox_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=in_channels, out_channels=anchors[1] * 4, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 2, out_channels=anchors[0] * 4, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 4, out_channels=anchors[0] * 4, kernel_size=3),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=anchors[1] * 4, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.bbox_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 4) for out in outputs], dim=1)
        return outputs


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 64, anchors: List[int] = [2, 3]) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList([
            DepthwiseConv2d(in_channels=in_channels, out_channels=anchors[1] * 10, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 2, out_channels=anchors[0] * 10, kernel_size=3),
            DepthwiseConv2d(in_channels=in_channels * 4, out_channels=anchors[0] * 10, kernel_size=3),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=anchors[1] * 10, kernel_size=3, padding=1)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for feature, layer in zip(x, self.landmark_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        outputs = torch.cat([out.view(out.shape[0], -1, 10) for out in outputs], dim=1)
        return outputs


class SlimFace(nn.Module):
    def __init__(self, cfg: dict = None) -> None:
        super().__init__()
        self.num_classes = 2
        self.anchors = [2, 3]
        self.stage1: List[nn.Module] = nn.Sequential(
            Conv2dNormActivation(in_channels=3, out_channels=16, stride=2),
            DepthWiseSeparableConv2d(in_channels=16, out_channels=32),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=32, stride=2),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=32),
            DepthWiseSeparableConv2d(in_channels=32, out_channels=64, stride=2),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64)
        )
        self.stage2: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(in_channels=64, out_channels=128, stride=2),
            DepthWiseSeparableConv2d(in_channels=128, out_channels=128),
            DepthWiseSeparableConv2d(in_channels=128, out_channels=128)
        )
        self.stage3: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(in_channels=128, out_channels=256, stride=2),
            DepthWiseSeparableConv2d(in_channels=256, out_channels=256)
        )
        self.stage4 = nn.Sequential(
            Conv2dNormActivation(in_channels=256, out_channels=64, kernel_size=1, norm_layer=None),
            DepthwiseConv2d(64, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )
        # Detection heads
        self.class_head = ClassHead(in_channels=64, num_classes=self.num_classes, anchors=self.anchors)
        self.bbox_head = BboxHead(in_channels=64, anchors=self.anchors)
        self.landmark_head = LandmarkHead(in_channels=64, anchors=self.anchors)

    def forward(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)

        x = self.stage2(x)
        features.append(x)

        x = self.stage3(x)
        features.append(x)

        x = self.stage4(x)
        features.append(x)

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions)
        return output
