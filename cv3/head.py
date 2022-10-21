"""Classification head modules
"""

from torch import Tensor

from torch import nn


class Head(nn.Module):
    """CNN head (linear layer)
    """

    def __init__(
        self, descriptor_size: int, classes: int, 
        init_weights: bool = True
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
