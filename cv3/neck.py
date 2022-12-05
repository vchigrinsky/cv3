"""Model neck modules
"""

from torch import Tensor

import torch
from torch import nn


class Neck(nn.Module):
    """CNN neck (average pooling, flattening, linear and batch norm layers)
    """

    def __init__(
        self, out_channels: int, in_channels: int, weights: str = None
    ):
        """Creates neck module

        Args:
            out_channels: output channels
            in_channels: input channels
            weights: path to weights for initialization 
                or None for random ResNet-style initialization
        """

        super().__init__()

        self.in_channels = in_channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

        self.init_weights(weights)

        self.out_channels = out_channels

    def init_weights(self, path: str = None):
        """Initialize weights

        Arguments:
            path: path to .pth file with weights to initialize model with
        """

        if path is None:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

        else:
            self.load_state_dict(torch.load(path))

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
