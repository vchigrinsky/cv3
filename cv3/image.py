"""Image class a wrapper over torch.Tensor, numpy.ndarray
"""

from torch import Tensor
from numpy import ndarray as Array
from PIL.Image import Image as PILImage

import torch
import numpy as np
import PIL


class Image:
    """Image class: a wrapper over torch.Tensor, numpy.ndarray 
    and PIL.Image.Image
    """

    def __init__(self, source: Tensor or Array or PILImage):
        """Creates Image object

        Args:
            source: Tensor (C x H x W; C = 1 or 3; uint8), 
                Array (H x W x C or H x W; uint8) 
                or PILImage ("L" or "RGB" mode)
        """

        if isinstance(source, Tensor):
            assert source.ndim == 3, 'tensor must have 3 dimensions'
            assert source.size(0) in {1, 3}, 'tensor must have 1 or 3 channels'
            assert source.dtype == torch.uint8, 'tensor must be uint8'

            self.tensor = source

        elif isinstance(source, Array):
            assert source.ndim in {2, 3}, 'array must have 2 or 3 dimensions'
            if source.ndim == 3:
                assert source.shape[2] == 3, \
                    'array must have 3 channels if it has 3 dimensions'
            assert source.dtype == np.uint8, 'array must be uint8'

            if source.ndim == 3:
                self.tensor = torch.as_tensor(source).permute(2, 0, 1)
            elif source.ndim == 1:
                self.tensor = torch.as_tensor(source).unsqueeze(0)
            else:
                raise NotImplementedError

        elif isinstance(source, PILImage):
            assert source.mode in {'L', 'RGB'}, \
                'PIL image must be in L or RGB mode'

            if source.mode == 'RGB':
                self.tensor = torch.as_tensor(
                    np.asarray(source)
                ).permute(2, 0, 1)
            elif source.mode == 'L':
                self.tensor = torch.as_tensor(
                    np.asarray(source)
                ).unsqueeze(0)
            else:
                raise NotImplementedError

        self.channels = self.tensor.size(0)

    @property
    def array(self) -> Array:
        """Image represented as numpy ndarray
        """

        if self.channels == 3:
            return self.tensor.permute(1, 2, 0).numpy()
        elif self.channels == 1:
            return self.tensor.squeeze(0).numpy()
        else:
            raise NotImplementedError

    def _repr_png_(self):
        """Visualization in Jupyter notebook
        """

        if self.tensor.size(0) == 3:
            return PIL.Image.fromarray(self.array)._repr_png_()
