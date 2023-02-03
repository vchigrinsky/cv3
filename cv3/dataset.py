"""PyTorch dataset wrapper on table
"""

from torch import Tensor

from .image import Image
from .table import ImageTable
from .transform import TransformModule

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """PyTorch dataset wrapper on table
    """

    def __init__(
        self, table: ImageTable, transform: TransformModule
    ):
        """Creates PyTorch dataset on table

        Args:
            table: image table (may be labeled or not)
            transformer: preprocess module
        """

        super().__init__()

        self.table = table
        self.transform = transform

        self.labeled = self.table.labels is not None

    def __len__(self) -> int:
        """Dataset length
        """

        return len(self.table)

    def __getitem__(self, index: int) -> Tensor or (Tensor, int):
        """Gets tensor and optionally label from dataset

        Args:
            index: table index

        Returns:
            tensor or tensor and label
        """

        image = self.table[index]
        tensor = self.transform(image.tensor)

        if self.labeled:
            label = self.table.labels[index].item()
            return tensor, label
        else:
            return tensor
