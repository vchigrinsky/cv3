"""CNN stem layer
"""

import math

from torch import Tensor

from torch import nn


class Stem(nn.Module):
    """Stem module: Conv -> BN -> ReLU
    """

    def __init__(
        self, out_channels: int = 16, in_channels: int = 3, 
        kernel_size: int = 5, pool_size: int = 2
    ):
        """Creates stem module

        Args:
            out_channels: output stem channels
            in_channels: input channels, 3 or 1 (rgb or gray)
            kernel_size: conv2d kernel size
            pool_size: maxpool2d kernel size
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)

        self.init_weights()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def init_weights(self):
        """ResNet-style weights initialization
        """

        fan_out = self.conv.kernel_size[0] * self.conv.kernel_size[1] \
            * self.conv.out_channels
        nn.init.normal_(
            self.conv.weight, mean=0.0, std=math.sqrt(2.0 / fan_out)
        )

        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Pass tensor through stem

        Args:
            x: input tensor
        """

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
