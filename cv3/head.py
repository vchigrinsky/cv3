"""Classification neck and head modules
"""

from torch import Tensor

from torch import nn


class Neck(nn.Module):
    """CNN neck (average pooling, flattening, linear and batch norm layers)
    """

    def __init__(
        self, channels: int, descriptor_size: int, 
        linear_bias: bool = True,
        init_weights: bool = False
    ):
        """Creates neck module

        Args:
            channels: input channels
            descriptor_size: output channels
            linear_bias: bias in linear layer flag
            init_weights: resnet-style weights initialization flag
        """

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(channels, descriptor_size, bias=linear_bias)
        self.bn = nn.BatchNorm1d(descriptor_size)

        if init_weights:
            self.init_weights()

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


class Head(nn.Module):
    """CNN head (linear layer)
    """

    def __init__(
        self, descriptor_size: int, classes: int, 
        init_weights: bool = False
    ):
        """Creates model head

        Args:
            descriptor_size: input channels
            classes: output channels
            init_weights: resnet-style weights initialization flag
        """

        super().__init__()

        self.linear = nn.Linear(descriptor_size, classes, bias=True)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Performs ResNet-style weight initialization
        """

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Pass tensor through net head

        Args:
            x: input tensor
        """

        x = self.linear(x)

        return x
