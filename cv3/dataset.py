"""PyTorch dataset wrapper on table
"""

from .image import Image
from .io import ImageTable, LabeledImageTable
from .transformer import Transformer

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper on table
    """

    def __init__(
        self, path: str, prefix: str, 
        transformer: Transformer, 
        has_labels: bool = False
    ):
        """Creates PyTorch dataset on table

        Args:
            path: path to a table on disk
            prefix: common prefix of images
            has_labels: boolean flag
        """

        super().__init__()

        self.has_labels = has_labels
        self.table = LabeledImageTable.read(path, prefix) if self.has_labels \
            else ImageTable.read(path, prefix)

        self.transformer = transformer

    def __len__(self) -> int:
        """Dataset length
        """

        return len(self.table)

    def __getitem__(self, index: int) -> int:
        """Gets tensor and optionally label from dataset
        """

        if self.has_labels:
            image, label = self.table[index]
            return self.transformer(image.tensor), label

        else:
            image = self.table[index]
            return self.transformer(image.tensor)
