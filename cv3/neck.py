"""Model neck modules
"""

from torch import Tensor

from torch import nn


class Neck(nn.Module):
    """CNN neck (average pooling, flattening, linear and batch norm layers)
    """

    def __init__(
        self, in_channels: int, descriptor_size: int, init_weights: bool = True
    ):
        """Creates neck module

        Args:
            in_channels: input channels
            descriptor_size: output channels
            init_weights: resnet-style weights initialization flag
        """

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, descriptor_size, bias=False)
        self.bn = nn.BatchNorm1d(descriptor_size)

        if init_weights:
            self.init_weights()

        self.descriptor_size = descriptor_size

    def init_weights(self):
        """Performs ResNet-style weight initialization
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
