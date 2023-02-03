"""PyTorch tensor transform module
"""

import torch
from torch import Tensor

import torchvision.transforms.functional as F
from torchvision.transforms import \
    RandomCrop


class TransformModule:
    """PyTorch tensor transformer
    """

    def __init__(self, *transforms):
        """Creates empty transform

        Args:
            transforms: sequence of transforms with arguments
        """

        self.uint8_transforms = list()
        self.float_transforms = list()

        for transform in transforms:
            name = transform.pop('transform')

            if name in ('random_crop', 'center_crop'):
                size = transform.pop('size')
                transform['output_size'] \
                    = size if isinstance(size, tuple) else (size, size)
            
            if name in ('normalize'):
                self.float_transforms.append((name, transform))
            else:
                self.uint8_transforms.append((name, transform))

    def __call__(self, tensor: Tensor) -> Tensor:
        """Pass input tensor through transforms

        Args:
            tensor: input uint8 tensor

        Returns:
            float tensor after transforms 
        """

        for transform, arguments in self.uint8_transforms:
            if transform == 'resize':
                tensor = F.resize(tensor, **arguments)
            elif transform == 'center_crop':
                tensor = F.center_crop(tensor, **arguments)
            elif transform == 'random_crop':
                i, j, h, w = RandomCrop.get_params(tensor, **arguments)
                tensor = F.crop(tensor, i, j, h, w)
            else:
                raise NotImplementedError(transform)

        tensor = tensor.to(torch.float32).div(255.)

        for transform, arguments in self.float_transforms:
            if transform == 'normalize':
                tensor = F.normalize(tensor, **arguments)
            else:
                raise NotImplementedError(transform)

        return tensor
