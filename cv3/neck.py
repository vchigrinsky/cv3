"""Model neck modules
"""

from torch import Tensor

from torch import nn


class Neck(nn.Module):
    """CNN neck (average pooling, flattening, linear and batch norm layers)
    """

    def __init__(self, out_channels: int, in_channels: int):
        """Creates neck module

        Args:
            out_channels: output channels
            in_channels: input channels
        """

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

        self.init_weights()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def init_weights(self):
        """ResNet-style weights initialization
        """

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: Tensor) -> Tensor: 
        """Pass tensor through net neck

        Args:
            x: input tensor
        """

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)

        return x
