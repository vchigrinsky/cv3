"""Evaluate model
"""

from .table import LabeledImageTable
from .dataset import Dataset
from .transformer import Transformer
from .sampler import Sampler
from torch.utils.data import DataLoader

import torch
from torch import nn

from tqdm import tqdm


def evaluate_classification(
    table: LabeledImageTable, transformer: Transformer, 
    stem: nn.Module, body: nn.Module, neck: nn.Module, head: nn.Module, 
    batch_size: int = 64,
    verbose: bool = True
) -> float:
    """Classification evaluation scenario

    Args:
        table: dataset table with labels to evaluate on
        transformer: preprocess module
        stem: stem module
        body: body module
        neck: neck module
        head: head module
        batch_size: inference batch size
        verbose: verbosity boolean flag

    Returns:
        accuracy value in [0.0, 1.0], float
    """

    labels = table.labels

    dataset = Dataset(table, transformer)
    sampler = Sampler(len(dataset), batch_size)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler)

    stem = stem.eval()
    body = body.eval()
    neck = neck.eval()
    head = head.eval()

    progress_bar = tqdm(total=len(dataset)) if verbose else None

    predictions = list()
    with torch.no_grad():
        for batch, _ in loader:
            feature_map = stem(batch)
            feature_map = body(feature_map)
            descriptors = neck(feature_map)
            logits = head(descriptors)
            predictions.append(logits.argmax(dim=1))
            if progress_bar is not None:
                progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

    predictions = torch.cat(predictions)

    accuracy = torch.eq(labels, predictions).to(torch.float32).mean().item()

    return accuracy
