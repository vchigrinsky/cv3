"""PyTorch model wrappers
"""

from torch import Tensor

from torch import nn


class ConvNet(nn.Module):
    """Base convolutional network
    """

    def __init__(self):
        """Creates base convolutional model
        """

        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 128)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, x: Tensor):
        """Pass tensor through net

        Args:
            x: input tensor
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

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

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)

        return x


class Head(nn.Module):
    """CNN head (linear layer)
    """

    def __init__(self, descriptor_size: int, n_classes: int):
        """Creates model head

        Args:
            descriptor_size: input channels
            n_classes: output channels
        """

        super().__init__()

        self.linear = nn.Linear(descriptor_size, n_classes)

    def forward(self, x: Tensor):
        """Pass tensor through net head

        Args:
            x: input tensor
        """

        x = self.linear(x)

        return x
