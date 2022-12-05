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

    def __call__(self, tensor: Tensor) -> (Tensor, tuple):
        """Pass input tensor through transforms

        Args:
            tensor: input uint8 tensor

        Returns:
            float tensor after transforms 
                and tuple of uint8 and float performed transforms
        """

        uint8_transforms = list()
        for transform, arguments in self.uint8_transforms:
            if transform == 'resize':
                tensor = F.resize(tensor, **arguments)
                uint8_transforms.append(('resize', arguments))
            elif transform == 'center_crop':
                tensor = F.center_crop(tensor, **arguments)
                uint8_transforms.append(('center_crop', arguments))
            elif transform == 'random_crop':
                i, j, h, w = RandomCrop.get_params(tensor, **arguments)
                tensor = F.crop(tensor, i, j, h, w)
                uint8_transforms.append(
                    ('crop', {'i': i, 'j': j, 'h': h, 'w': w})
                )
            else:
                raise NotImplementedError(transform)

        tensor = tensor.to(torch.float32).div(255.)

        float_transforms = list()
        for transform, arguments in self.float_transforms:
            if transform == 'normalize':
                tensor = F.normalize(tensor, **arguments)
                float_transforms.append(('normalize', arguments))
            else:
                raise NotImplementedError(transform)

        return tensor, (uint8_transforms, float_transforms)
