"""Input-output utilities
"""

import os.path as osp
import torchvision
import pandas as pd

from .image import Image
from .table import Table


def read_image(path: str) -> Image:
    """Reads image from disk

    Args:
        path: a path to an png file on disk
    """

    assert osp.exists(path), f'{path} is missing'
    assert path.endswith('.png'), 'image on disk must be PNG'

    return Image(torchvision.io.read_image(path))


def read_table(path: str) -> Table:
    """Reads table from disk

    Args:
        path: a path to a csv file on disk
    """

    assert osp.exists(path), f'{path} is missing'
    assert path.endswith('.csv'), 'table on disk must be CSV'

    return Table(pd.read_csv(path))
