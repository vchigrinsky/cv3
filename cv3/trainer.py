"""Train model
"""

from .io import LabeledImageTable
from .transformer import Transformer
from .dataset import Dataset
from .sampler import Sampler, RandomSampler
from torch.utils.data import DataLoader

import torch
from torch import nn
from .model import ConvNet, Head

from torch.optim import Optimizer, SGD
from .scheduler import Scheduler, StepScheduler

from torch.nn.modules.loss import CrossEntropyLoss


def train(
    train_table: LabeledImageTable, train_loader: DataLoader,
    body: nn.Module, head: nn.Module,
    optimizer: Optimizer, scheduler: Scheduler,
    loss: nn.Module,
    validation_table: LabeledImageTable, validation_loader: DataLoader,
    validation: dict,
    path: str
):
    """Train model

    Args:
        train_table: labeled image table for training
        train_loader: dataloader of train dataset
        body: model body
        head: classifier module
        optimizer: optimizer
        scheduler: learning rate scheduler
        loss: loss module
        validation_table" labeled image table for validation
        validation_loader: dataloader of validation dataset
        validation: validation type and rate
        path: experiment path to dump logs and weights
    """

    pass


def parse_config(config: dict) -> (
    LabeledImageTable, DataLoader,
    nn.Module, nn.Module,
    Optimizer, Scheduler,
    nn.Module,
    LabeledImageTable, DataLoader, dict,
    str
):
    """Prepare train experiment

    Args:
        config: configuration dictionary
    """

    # >>>>> train table
    train_table_type = config['train_table'].pop('type')

    if train_table_type == 'default':
        train_table = LabeledImageTable.read(**config['train_table'])

    else:
        raise NotImplementedError

    # >>>>> train transformer
    train_transformer_type = config['train_transformer']

    if train_transformer_type == 'default':
        train_transformer = Transformer(**config['train_transformer'])

    else:
        raise NotImplementedError

    # >>>>> train loader
    train_dataset = Dataset(train_table, train_transformer)

    train_sampler_type = config['train_sampler'].pop('type')

    train_loader_workers = config['train_sampler'].pop('workers')

    if train_sampler_type == 'default':
        train_sampler = RandomSampler(**config['train_sampler'])

    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=train_loader_workers
    )

    # >>>>> model
    body_type = config['body'].pop('type')

    if body_type == 'default':
        body = ConvNet(**config['body'])

    else:
        raise NotImplementedError

    head_type = config['head'].pop('type')

    if head_type == 'default':
        head = Head(**config['head'])

    else:
        raise NotImplementedError

    # >>>>> optimizer
    optimizer_type = config['optimizer'].pop('type')
    lr = config['optimizer'].pop('lr')

    if optimizer_type == 'default':
        optimizer = SGD(
            [
                {'params': body.parameters()}, 
                {'params': head.parameters()}
            ], lr=lr, **config['optimizer']
        )

    else:
        raise NotImplementedError

    # >>>>> scheduler
    scheduler_type = config['scheduler'].pop('type')
    epochs = config['scheduler'].pop('epochs')
    steps = epochs * len(train_table)

    if scheduler_type == 'default':
        scheduler = StepScheduler(lr, steps, **config['scheduler'])

    else:
        raise NotImplementedError

    # >>>>> loss
    loss_type = config['loss'].pop('type')

    if loss_type == 'default':
        loss = CrossEntropyLoss(**config['loss'])

    else:
        raise NotImplementedError

    # >>>>> validation table
    validation_table_type = config['validation_table'].pop('type')

    if validation_table_type == 'default':
        validation_table = LabeledImageTable.read(**config['validation_table'])

    else:
        raise NotImplementedError

    # >>>>> validation transformer
    validation_transformer_type = config['validation_transformer']

    if validation_transformer_type == 'default':
        validation_transformer = Transformer(**config['validation_transformer'])

    else:
        raise NotImplementedError

    # >>>>> validation loader
    validation_dataset = Dataset(validation_table, validation_transformer)

    validation_sampler_type = config['validation_sampler'].pop('type')

    validation_loader_workers = config['validation_sampler'].pop('workers')

    if validation_sampler_type == 'default':
        validation_sampler = Sampler(**config['validation_sampler'])

    else:
        raise NotImplementedError

    validation_loader = DataLoader(
        dataset=validation_dataset, 
        batch_sampler=validation_sampler, 
        num_workers=validation_loader_workers
    )

    # >>>>> validation
    validation = config['validation']

    return (
        train_table, train_loader,
        body, head,
        optimizer, scheduler,
        loss,
        validation_table, validation_loader, validation,
        config['path']
    )
