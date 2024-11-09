from collections import OrderedDict
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class DepthWiseSeparableConv2d(nn.Sequential):
    """DepthWise Separable Convolutional with Depthwise and Pointwise layers followed by nn.BatchNorm2d and nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None
    ) -> None:

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels
            ),  # Depthwise
            Conv2dNormActivation(in_channels, out_channels, kernel_size=1)  # Pointwise
        ]

        super().__init__(*layers)


class DepthwiseConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: Optional = None) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels
            ),
            nn.ReLU(inplace=True),  # Activation
            nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Pointwise
        ]
        super().__init__(*layers)


class SSH(nn.Module):
    """
    SSH (Single Stage Headless) Module for feature extraction.
    Combines 3x3, 5x5, and 7x7 convolutions with batch normalization and optional ReLU activations.
    """

    def __init__(self, in_channel: int, out_channels: int) -> None:
        """
        Initializes the SSH module.

        Args:
            in_channel (int): Number of input channels.
            out_channels (int): Number of output channels, must be divisible by 4.
        """
        super().__init__()

        assert out_channels % 4 == 0, "Output channel must be divisible by 4."

        # 3x3 Convolution branch
        self.conv3X3 = Conv2dNormActivation(in_channel, out_channels // 2, kernel_size=3, activation_layer=None)

        # 5x5 Convolution branch
        self.conv5X5_1 = Conv2dNormActivation(in_channel, out_channels // 4, kernel_size=3)
        self.conv5X5_2 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None
        )

        # 7x7 Convolution branch
        self.conv7X7_2 = Conv2dNormActivation(out_channels // 4, out_channels // 4, kernel_size=3)
        self.conv7x7_3 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SSH module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map after applying SSH operations and ReLU activation.
        """
        conv3X3 = self.conv3X3(x)
        conv5X5 = self.conv5X5_2(self.conv5X5_1(x))
        conv7X7 = self.conv7x7_3(self.conv7X7_2(self.conv5X5_1(x)))

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        return F.relu(out, inplace=True)


class FPN(nn.Module):
    """
    FPN (Feature Pyramid Network) for multi-scale feature map extraction and merging.
    Uses 1x1 convolutions for output layers and 3x3 convolutions for merging layers.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        """
        Initializes the FPN module.

        Args:
            in_channels_list (list of int): List of input channel sizes for each pyramid level.
            out_channels (int): Number of output channels for the feature pyramid.
        """
        super().__init__()
        # Define 1x1 convolution output layers
        self.output1 = Conv2dNormActivation(in_channels_list[0], out_channels, kernel_size=1)
        self.output2 = Conv2dNormActivation(in_channels_list[1], out_channels, kernel_size=1)
        self.output3 = Conv2dNormActivation(in_channels_list[2], out_channels, kernel_size=1)

        # Define merge layers using 3x3 convolutions
        self.merge1 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3)
        self.merge2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3)

    def forward(self, inputs) -> List[Tensor]:
        """
        Forward pass of the FPN module.

        Args:
            inputs (dict or list): Input feature maps from different levels of the pyramid.

        Returns:
            list: List of merged output feature maps at different scales.
        """
        inputs = list(inputs.values())

        # Apply output layers to each feature map
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        # Merge outputs with upsampling and addition
        upsample3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = self.merge2(output2 + upsample3)

        upsample2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = self.merge1(output1 + upsample2)

        # Return merged feature maps
        return [output1, output2, output3]
