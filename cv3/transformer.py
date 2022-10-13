"""PyTorch tensor transformer
"""

from torch import Tensor

import torch
import torchvision.transforms.functional as F


class Transformer:
    """PyTorch tensor transformer
    """

    def __init__(
        self, height: int, width: int, 
        mean: (float, float, float), std: (float, float, float)
    ):
        """Creates empty transformer
        """

        self.resize = {'size': [height, width]}
        self.normalize = {'mean': mean, 'std': std}

        self.transforms = list()

    def __call__(self, tensor: Tensor):
        """Pass input tensor through transforms

        Args:
            tensor: input tensor
        """

        tensor = F.resize(tensor, **self.resize)

        for transform in self.transforms:
            pass

        tensor = tensor.to(torch.float32).div(255.)
        tensor = F.normalize(tensor, **self.normalize)

        return tensor
