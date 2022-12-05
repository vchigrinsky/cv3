"""Simple convoluational network
"""

import math

from torch import Tensor

import torch
from torch import nn


class ConvNet(nn.Module):
    """Convolutional network
    """

    def __init__(
        self, out_channels: int = 0, in_channels: int = 16,  
        weights: str = None,
    ):
        """Creates base convolutional model

        Args:
            out_channels: output channels
            in_channels: input channels
            weights: path to weights for initialization 
                or None for random ResNet-style initialization
        """

        super().__init__()

        self.in_channels = in_channels

        channels = in_channels
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels * 2)
        self.relu1 = nn.ReLU()

        channels *= 2
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        channels *= 2
        if out_channels == 0:
            out_channels = channels
        self.conv3 = nn.Conv2d(channels, out_channels, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.init_weights(weights)

        self.out_channels = out_channels

    def init_weights(self, path: str = None):
        """Initialize weights

        Arguments:
            path: path to .pth file with weights to initialize model with
        """

        if path is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = (m.kernel_size[0] * m.kernel_size[1]) \
                        * m.out_channels
                    nn.init.normal_(
                        m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out)
                    )

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        else:
            self.load_state_dict(torch.load(path))

    def forward(self, x: Tensor) -> Tensor:
        """Pass tensor through net

        Args:
            x: input tensor
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x
