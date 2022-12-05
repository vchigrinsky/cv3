"""PyTorch sampler wrappers
"""

import math
import random

from torch.utils.data import Sampler as TorchSampler


class Sampler(TorchSampler):
    """Sequential batch sampler
    """

    def __init__(self, length: int, batch_size: int, drop_last: bool = False):
        """Creates sequential sampler

        Args:
            length: dataset length
            batch_size: batch size
            drop_last: boolean flag of droping last batch
        """

        super().__init__(None)

        self.sequence = list(range(length))
        self.batch_size = batch_size

        self.drop_last = drop_last
    
    def __len__(self) -> int:
        """Sampler length
        """

        if self.drop_last:
            return math.floor(len(self.sequence) / self.batch_size)
        else:
            return math.ceil(len(self.sequence) / self.batch_size)

    def __iter__(self):
        """Iterate over generated sequence
        """

        batch = list()
        for index in self.sequence:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = list()

        if batch and not self.drop_last:
            yield batch


class ShuffledSampler(Sampler):
    """Shuffled batch sampler
    """

    def __init__(self, length: int, batch_size: int, drop_last: bool = False):
        """Creates shuffled sampler

        Args:
            length: dataset length
            batch_size: batch size
            drop_last: boolean flag of droping last batch
        """

        super().__init__(length, batch_size, drop_last)

        random.shuffle(self.sequence)


class PredefinedSampler(Sampler):
    """Sampler on sequence from file
    """

    def __init__(self, length: int, batch_size: int, path: str):
        """Creates predefined sampler on file

        Args:
            length: dataset length
            batch_size: batch size
            path: path to a file with indices sequence to generate sampler on
        """

        with open(path, 'r') as f:
            sequence = [int(index) for index in f.read().split()]

        drop_last = len(sequence) < length

        super().__init__(length, batch_size, drop_last)

        self.sequence = sequence
