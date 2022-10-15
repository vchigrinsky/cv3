"""PyTorch tensor transformer
"""

from torch import Tensor

import torch
import torchvision.transforms


class Transformer:
    """PyTorch tensor transformer
    """

    def __init__(
        self, height: int, width: int, 
        mean: (float, float, float) or float, 
        std: (float, float, float) or float,
        crop_height: int = None, crop_width: int = None,
        mode: str = 'train'
    ):
        """Creates empty transformer

        Args:
            height: resize height
            width: resize width
            mean: normalization mean
            std: normalization std
            crop_height: crop height
            crop_width: crop width
            mode: train or test
        """

        assert mode in {'train', 'test'}

        if crop_height is None:
            crop_height = height
        if crop_width is None:
            crop_width = width

        self.resize = torchvision.transforms.Resize(size=[height, width])

        if mode == 'train':
            self.crop = torchvision.transforms.RandomCrop(
                size=[crop_height, crop_width]
            )
        elif mode == 'test':
            self.crop = torchvision.transforms.CenterCrop(
                size=[crop_height, crop_width]
            )

        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

        self.transforms = list()

    def __call__(self, tensor: Tensor) -> Tensor:
        """Pass input tensor through transforms

        Args:
            tensor: input tensor

        Returns:
            tensor after transforms
        """

        tensor = self.resize(tensor)
        tensor = self.crop(tensor)

        for transform in self.transforms:
            pass

        tensor = tensor.to(torch.float32).div(255.)
        tensor = self.normalize(tensor)

        return tensor
