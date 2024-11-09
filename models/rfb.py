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


class BasicRFB(nn.Module):
    """
    Basic Receptive Field Block (RFB) for feature extraction.
    """

    def __init__(self, in_channels: int, out_channels: int, scale: float = 0.1, map_reduce: int = 8) -> None:
        super().__init__()
        self.scale = scale
        inter_channels = in_channels // map_reduce

        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, inter_channels, kernel_size=1, activation_layer=None),
            Conv2dNormActivation(inter_channels, 2 * inter_channels, kernel_size=3),
            Conv2dNormActivation(
                2 * inter_channels,
                2 * inter_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                activation_layer=None
            )
        )
        self.branch2 = nn.Sequential(
            Conv2dNormActivation(in_channels, inter_channels, kernel_size=1,  activation_layer=None),
            Conv2dNormActivation(inter_channels, 2 * inter_channels, kernel_size=3),
            Conv2dNormActivation(
                2 * inter_channels,
                2 * inter_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                activation_layer=None
            )
        )
        self.branch3 = nn.Sequential(
            Conv2dNormActivation(in_channels, inter_channels, kernel_size=1, activation_layer=None),
            Conv2dNormActivation(inter_channels, 3 * inter_channels // 2, kernel_size=3),
            Conv2dNormActivation(3 * inter_channels // 2, 2 * inter_channels, kernel_size=3),
            Conv2dNormActivation(
                2 * inter_channels,
                2 * inter_channels,
                kernel_size=3,
                padding=5,
                dilation=5,
                activation_layer=None
            )
        )

        self.conv_linear = Conv2dNormActivation(6 * inter_channels, out_channels, kernel_size=1, activation_layer=None)
        self.shortcut = Conv2dNormActivation(in_channels, out_channels, kernel_size=1, activation_layer=None)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch1(x)
        b3 = self.branch1(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_linear(out) * self.scale + self.shortcut(x)
        out = self.relu(out)
        return out


class RFB(nn.Module):
    """
    RFB-based face detection model.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        self.num_classes = 2
        self.anchors = [2, 3]
        self.stage1 = nn.Sequential(
            Conv2dNormActivation(3, 16, stride=2),
            DepthWiseSeparableConv2d(16, 32),
            DepthWiseSeparableConv2d(32, 32, stride=2),
            DepthWiseSeparableConv2d(32, 32),
            DepthWiseSeparableConv2d(32, 64, stride=2),
            DepthWiseSeparableConv2d(64, 64),
            DepthWiseSeparableConv2d(64, 64),
            BasicRFB(64, 64, scale=1.0)
        )

        self.stage2 = nn.Sequential(
            DepthWiseSeparableConv2d(64, 128, stride=2),
            DepthWiseSeparableConv2d(128, 128),
            DepthWiseSeparableConv2d(128, 128)
        )

        self.stage3 = nn.Sequential(
            DepthWiseSeparableConv2d(128, 256, stride=2),
            DepthWiseSeparableConv2d(256, 256)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            DepthwiseConv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Detection heads
        self.class_head = ClassHead(in_channels=64, num_classes=self.num_classes, anchors=self.anchors)
        self.bbox_head = BboxHead(in_channels=64, anchors=self.anchors)
        self.landmark_head = LandmarkHead(in_channels=64, anchors=self.anchors)

    def forward(self, x: torch.Tensor) -> tuple:
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
            return bbox_regressions, classifications, landmark_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions
