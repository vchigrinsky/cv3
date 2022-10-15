"""Simple convoluational network
"""

import math

from torch import Tensor

from torch import nn


class ConvNet(nn.Module):
    """Convolutional network
    """

    def __init__(
        self, channels: int = 3, 
        conv_bias: bool = True,
        init_weights: bool = False,
        descriptor_size: int = None  # deprecated
    ):
        """Creates base convolutional model

        Args:
            channels: input channels, 3 or 1
            conv_bias: bias in convolutional layer flag
            init_weights: resnet-style weights initialization flag
            descriptor_size: deprecated
        """

        super().__init__()

        self.conv1 = nn.Conv2d(channels, 16, 7, bias=conv_bias)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 5, bias=conv_bias)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, bias=conv_bias)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, bias=conv_bias)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Performs ResNet-style weight initialization
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(
                    m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out)
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        return x
