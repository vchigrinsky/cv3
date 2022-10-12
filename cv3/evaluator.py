"""Evaluate models
"""

from .io import LabeledImageTable
from .dataset import Dataset
from .transformer import Transformer
from .sampler import SequentialSampler
from torch.utils.data import DataLoader

import torch
from torch import nn
from .model import ConvNet, Head

from tqdm import tqdm


def evaluate_classification(
    table: LabeledImageTable, transformer: Transformer, 
    body: nn.Module, head: Head, 
    batch_size: int = 1,
    verbose: bool = True
) -> float:
    """Classification evaluation scenario

    Args:
        table: dataset table with labels to evaluate on
        transformer: preprocess module
        body: CNN to evaluate
        head: classifier module
        batch_size: inference batch size
        verbose: verbosity boolean flag

    Returns:
        accuracy value in [0.0, 1.0], float
    """

    labels = table.labels

    dataset = Dataset(table, transformer)
    sampler = SequentialSampler(len(dataset), batch_size)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler)

    body = body.eval()
    head = head.eval()

    progress_bar = tqdm(total=len(dataset)) if verbose else None

    predictions = list()
    with torch.no_grad():
        for batch, _ in loader:
            descriptors = body(batch)
            scores = head(descriptors)
            predictions.append(scores.argmax(dim=1))
            if progress_bar is not None:
                progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    predictions = torch.cat(predictions)

    accuracy = torch.eq(labels, predictions).to(torch.float32).mean().item()

    return accuracy
