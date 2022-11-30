"""PyTorch dataset wrapper on table
"""

from .image import Image
from .table import ImageTable, LabeledImageTable
from .transform import TransformModule

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """PyTorch dataset wrapper on table
    """

    def __init__(
        self, table: ImageTable or LabeledImageTable, 
        transform: TransformModule
    ):
        """Creates PyTorch dataset on table

        Args:
            table: image table (may be labeled or not)
            transformer: preprocess module
        """

        super().__init__()

        self.table = table
        self.transform = transform

    def __len__(self) -> int:
        """Dataset length
        """

        return len(self.table)

    def __getitem__(self, index: int) -> int:
        """Gets tensor and optionally label from dataset
        """

        if isinstance(self.table, LabeledImageTable):
            image, label = self.table[index]
            return self.transform(image.tensor), label

        elif isinstance(self.table, ImageTable):
            image = self.table[index]
            return self.transform(image.tensor)

        else:
            raise NotImplementedError
