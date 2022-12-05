"""Classification head modules
"""

from torch import Tensor

import torch
from torch import nn


class Head(nn.Module):
    """CNN head (linear layer)
    """

    def __init__(
        self, classes: int, descriptor_size: int, weights: str = None
    ):
        """Creates model head

        Args:
            classes: output channels
            descriptor_size: input channels
            weights: path to weights for initialization 
                or None for random ResNet-style initialization
        """

        super().__init__()

        self.linear = nn.Linear(descriptor_size, classes, bias=True)

        self.init_weights(weights)

    def init_weights(self, path: str = None):
        """Initialize weights

        Arguments:
            path: path to .pth file with weights to initialize model with
        """

        if path is None:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.linear.bias)

        else:
            self.load_state_dict(torch.load(path))

    def forward(self, x: Tensor) -> Tensor:
        """Pass tensor through net head

        Args:
            x: input tensor
        """

        x = self.linear(x)

        return x
