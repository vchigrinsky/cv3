"""Classification head modules
"""

from torch import Tensor

from torch import nn


class Head(nn.Module):
    """CNN head (linear layer)
    """

    def __init__(self, classes: int, descriptor_size: int):
        """Creates model head

        Args:
            classes: output channels
            descriptor_size: input channels
        """

        super().__init__()

        self.linear = nn.Linear(descriptor_size, classes, bias=True)

        self.init_weights()

    def init_weights(self, path: str = None):
        """ResNet-style weights initialization
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
