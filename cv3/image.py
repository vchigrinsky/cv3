"""Image class: a wrapper over torch.Tensor, numpy.ndarray and PIL.Image.Image
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
            source: Tensor (CxHxW; C=3; uint8), Array (HxWxC; C=3; uint8) 
                or PILImage (mode='RGB')
        """

        if isinstance(source, Tensor):
            assert source.ndim == 3, 'tensor must have 3 dimensions'
            assert source.size(0) == 3, 'tensor must have 3 channels'
            assert source.dtype == torch.uint8, 'tensor must be uint8'

            self.tensor = source

        elif isinstance(source, Array):
            assert source.ndim == 3, 'array must have 3 dimensions'
            assert source.shape[2] == 3, 'array must have 3 channels'
            assert source.dtype == np.uint8, 'array must be uint8'

            self.tensor = torch.as_tensor(source).permute(2, 0, 1)

        elif isinstance(source, PILImage):
            assert source.mode == 'RGB', 'PIL image must be in RGB mode'

            self.tensor = torch.as_tensor(np.asarray(source)).permute(2, 0, 1)

    @property
    def array(self) -> Array:
        """Image represented as numpy ndarray
        """

        return self.tensor.permute(1, 2, 0).numpy()

    def _repr_png_(self):
        """Visualization in Jupyter notebook
        """

        return PIL.Image.fromarray(self.array)._repr_png_()
