"""PyTorch tensor transformer
"""

from torch import Tensor

import torch
import torchvision.transforms


class Transformer:
    """PyTorch tensor transformer
    """

    def __init__(
        self, 
        resize: (int, int) or int = None,
        crop: (int, int) or int = None,
        mean: (float, float, float) or float = None, 
        std: (float, float, float) or float = None
    ):
        """Creates empty transformer

        Args:
            resize: [height, width], if a single value 
                is passed then the smallest side will be fitted to this
            crop: center crop size, [height, width], if a single value 
                is passed then it is height = width
            mean: normalization mean
            std: normalization std
        """

        if resize is not None:
            self.resize = torchvision.transforms.Resize(resize)
        else:
            self.resize = None

        if crop is not None:
            self.crop = torchvision.transforms.CenterCrop(crop)
        else:
            self.crop = None

        if mean is not None and std is not None:
            self.normalize = torchvision.transforms.Normalize(mean, std)
        else:
            self.normalize = None

        self.transforms = list()

    def add_random_crop(self, size: (int, int) or int, **kwargs):
        """Creates torchvision.transforms.RandomCrop and adds it to transforms

        Args:
            size: [height, width] or height = width if a single value is passed
            **kwargs: torchvision.transforms.RandomCrop args
        """

        self.transforms.append(
            torchvision.transforms.RandomCrop(size, **kwargs)
        )

    def __call__(self, tensor: Tensor) -> Tensor:
        """Pass input tensor through transforms

        Args:
            tensor: input tensor

        Returns:
            tensor after transforms
        """

        if self.resize is not None:
            tensor = self.resize(tensor)

        if self.crop is not None:
            tensor = self.crop(tensor)

        for transform in self.transforms:
            tensor = transform(tensor)

        tensor = tensor.to(torch.float32).div(255.)
        
        if self.normalize is not None:
            tensor = self.normalize(tensor)

        return tensor
