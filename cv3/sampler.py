"""PyTorch sampler wrappers
"""

import math

import torch.utils.data


class SequentialSampler(torch.utils.data.Sampler):
    """Sequential batch sampler
    """

    def __init__(self, length: int, batch_size: int, drop_last: bool = False):
        """Creates sequential sampler
        """

        super().__init__(None)

        self.sequence = range(length)
        self.batch_size = batch_size

        self.drop_last = drop_last
    
    def __len__(self) -> int:
        """Sampler length
        """

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
