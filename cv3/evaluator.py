"""Evaluate model
"""

from .table import ImageTable
from .transform import TransformModule
from .dataset import Dataset
from .sampler import Sampler
from torch.utils.data import DataLoader

import torch
from torch import nn

from .stem import Stem
from .convnet import ConvNet
from .neck import Neck
from .head import Head

from tqdm import tqdm


def evaluate_classification(config: dict, verbose: bool = True) -> float:
    """Classification evaluation scenario

    Args:
        config: evaluation configuration file
        verbose: verbosity boolean flag

    Returns:
        accuracy value in [0.0, 1.0], float
    """

    # >>>>> test table    
    test_table_type = config['test_table'].pop('type')
    if test_table_type == 'default':
        test_table = ImageTable.read(**config['test_table'])

    else:
        raise NotImplementedError

    # >>>>> test loader
    test_transform = TransformModule(*config['test_transform'])

    test_dataset = Dataset(test_table, test_transform)

    test_sampler = Sampler(len(test_dataset), config['batch_size'])
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_sampler=test_sampler, num_workers=0
    )

    # >>>>> model
    stem_weights = torch.load(config['stem'].pop('weights'))

    stem_type = config['stem'].pop('type')

    if stem_type == 'default':
        stem = Stem(in_channels=test_table.channels, **config['stem'])

    else:
        raise NotImplementedError

    stem.load_state_dict(stem_weights)

    stem.eval()

    body_weights = torch.load(config['body'].pop('weights'))

    body_type = config['body'].pop('type')

    if body_type == 'convnet':
        body = ConvNet(in_channels=stem.out_channels, **config['body'])

    else:
        raise NotImplementedError

    body.load_state_dict(body_weights)

    body.eval()

    neck_weights = torch.load(config['neck'].pop('weights'))

    neck_type = config['neck'].pop('type')

    if neck_type == 'default':
        neck = Neck(in_channels=body.out_channels, **config['neck'])

    else:
        raise NotImplementedError

    neck.load_state_dict(neck_weights)

    neck.eval()

    head_weights = torch.load(config['head'].pop('weights'))

    head_type = config['head'].pop('type')

    if head_type == 'default':
        head = Head(
            classes=test_table.classes,
            descriptor_size=neck.out_channels,
            **config['head']
        )

    else:
        raise NotImplementedError

    head.load_state_dict(head_weights)

    head.eval()

    # >>>>> evaluation
    progress_bar = tqdm(total=len(test_dataset)) if verbose else None

    predictions = list()
    with torch.no_grad():
        for batch, *_ in test_loader:
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

    accuracy = torch.eq(
        test_table.labels, predictions
    ).to(torch.float32).mean().item()

    return accuracy
