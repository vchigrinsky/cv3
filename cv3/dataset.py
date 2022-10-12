"""PyTorch dataset wrapper on table
"""

from .image import Image
from .io import ImageTable, LabeledImageTable
from .transformer import Transformer

from torch.utils.data import Dataset as TensorDataset


class Dataset(TensorDataset):
    """PyTorch dataset wrapper on table
    """

    def __init__(
        self, table: ImageTable or LabeledImageTable, 
        transformer: Transformer
    ):
        """Creates PyTorch dataset on table

        Args:
            table: image table (may be labeled or not)
            transformer: preprocess module
        """

        super().__init__()

        self.table = table
        self.transformer = transformer

    @classmethod
    def read(
        cls, path: str, prefix: str, 
        transformer: Transformer, 
        labeled: bool = False
    ):
        """Reads table and creates dataset

        Args:
            path: path to a table on disk
            prefix: common prefix of images
            transformer: preprocess module
            labeled: boolean flag
        """

        if labeled:
            table = LabeledImageTable.read(path, prefix)

        else:
            table = ImageTable.read(path, prefix)

        return cls(table, transformer)

    def __len__(self) -> int:
        """Dataset length
        """

        return len(self.table)

    def __getitem__(self, index: int) -> int:
        """Gets tensor and optionally label from dataset
        """

        if isinstance(self.table, LabeledImageTable):
            image, label = self.table[index]
            return self.transformer(image.tensor), label

        elif isinstance(self.table, ImageTable):
            image = self.table[index]
            return self.transformer(image.tensor)

        else:
            raise NotImplementedError
