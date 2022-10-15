"""Input-output utilities
"""

from .image import Image
from pandas import DataFrame

import os.path as osp

import pandas as pd

import torch
import torchvision


class Table:
    """Table class: a wrapper over pandas.DataFrame
    """

    def __init__(self, source: DataFrame):
        """Creates table

        Args:
            source: pandas DataFrame
        """

        self.dataframe = source

    def __len__(self) -> int:
        """Table length
        """

        return len(self.dataframe)

    @property
    def columns(self) -> tuple:

        return tuple(self.dataframe.columns)

    @classmethod
    def read(cls, path: str):
        """Reads table from disk

        Args:
            path: a path to a csv file on disk
        """

        assert osp.exists(path), f'{path} is missing'
        assert path.endswith('.csv'), 'table on disk must be CSV'

        return cls(pd.read_csv(path))


class ImageTable(Table):
    """Table of images, must have "id" column
    """

    def __init__(self, source: DataFrame, prefix: str, mode: str = 'rgb'):
        """Creates table of images

        Args:
            source: table source, must have "id" column
            prefix: common prefix for images to read
            mode: in which mode read images, "rgb" or "gray"
        """

        assert mode in {'rgb', 'gray'}

        Table.__init__(self, source)

        assert 'id' in self.columns, 'image table should have "id" column'
        self.ids = list(self.dataframe.id)

        self.prefix = prefix

        if mode == 'rgb':
            self.channels = 3
        elif mode == 'gray':
            self.channels = 1
        else:
            raise NotImplementedError

    def __getitem__(self, index: int) -> Image:
        """Gets image

        Args:
            index: index in table
        """

        path = osp.join(self.prefix, self.ids[index])

        assert osp.exists(path), f'{path} is missing'
        assert path.endswith('.png'), 'image on disk must be PNG'

        if self.channels == 3:
            mode = torchvision.io.image.ImageReadMode.RGB
        elif self.channels == 1:
            mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            raise NotImplementedError

        return Image(torchvision.io.read_image(path, mode))

    @classmethod
    def read(cls, path: str, prefix: str, mode: str = 'rgb'):
        """Reads image table from disk

        Args:
            path: a path to a csv file on disk
            prefix: common prefix for images to read
        """

        assert osp.exists(path), f'{path} is missing'
        assert path.endswith('.csv'), 'table on disk must be CSV'

        return cls(pd.read_csv(path), prefix)


class LabeledImageTable(ImageTable):
    """Table of images with labels, must have "id" and "label" columns
    """

    def __init__(self, source: DataFrame, prefix: str, mode: str = 'rgb'):
        """Creates table of images with labels

        Args:
            source: table source, must have "id" and "label" column
            prefix: common prefix for images to read
            mode: in which mode read images, "rgb" or "gray"
        """

        ImageTable.__init__(self, source, prefix, mode)

        assert 'label' in self.columns, \
            'labeled image table should have "label" column'
        self.labels = list(self.dataframe.label)
        self.map_labels_to_range()
        self.classes = len(self.labels.unique())

        self.prefix = prefix

    def map_labels_to_range(self):
        """Maps custom labels to int range
        """

        mapping = dict(zip(
            sorted(set(self.labels)), range(len(set(self.labels)))
        ))
        self.labels = torch.as_tensor(
            [mapping[x] for x in self.labels], dtype=torch.int32
        )

    def __getitem__(self, index: int) -> (Image, int):
        """Gets image and label

        Args:
            index: index in table
        """

        return ImageTable.__getitem__(self, index), self.labels[index].item()
