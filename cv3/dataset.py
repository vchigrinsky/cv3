"""PyTorch dataset wrapper on table
"""

from .image import Image
from .table import ImageTable
from .transform import TransformModule

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """PyTorch dataset wrapper on table
    """

    def __init__(
        self, table: ImageTable, transform: TransformModule, 
        return_index: bool = False, return_transform: bool = False
    ):
        """Creates PyTorch dataset on table

        Args:
            table: image table (may be labeled or not)
            transformer: preprocess module
            return_index: return index flag
            return_transform: return transforms parameters
        """

        super().__init__()

        self.table = table
        self.transform = transform

        self.labeled = self.table.labels is not None

        self.return_index = return_index
        self.return_transform = return_transform

    def __len__(self) -> int:
        """Dataset length
        """

        return len(self.table)

    def __getitem__(self, index: int) -> int:
        """Gets tensor and optionally label from dataset

        Args:
            index: table index
        """

        image = self.table[index]
        tensor, transform = self.transform(image.tensor)
        
        meta = dict()
        if self.return_index:
            meta['index'] = index
        if self.return_transform:
            meta['transform'] = transform

        if self.labeled:
            label = self.table.labels[index].item()
            return tensor, label, meta
        else:
            return tensor, meta
