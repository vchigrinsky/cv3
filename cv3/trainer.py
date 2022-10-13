"""Train model
"""

from .table import LabeledImageTable
from .transformer import Transformer
from .dataset import Dataset
from .sampler import Sampler, RandomSampler
from torch.utils.data import DataLoader

import torch
from torch import nn
from .model import ConvNet, Head

from torch.optim import Optimizer, SGD
from .scheduler import Scheduler

from torch.nn.modules.loss import CrossEntropyLoss

from tqdm import tqdm

import os.path as osp


def train(
    train_table: LabeledImageTable, train_loader: DataLoader,
    body: nn.Module, head: nn.Module,
    optimizer: Optimizer, scheduler: Scheduler,
    loss_module: nn.Module,
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
        loss_module: loss module
        path: experiment root
    """

    steps = scheduler.steps
    assert steps % len(train_table) == 0, \
        'number of steps should be divisible by train dataset length'
    epochs = steps // len(train_table)

    progress_bar = tqdm(total=steps * 2)

    body = body.train()
    head = head.train()

    for epoch in range(epochs):
        for batch, labels in train_loader:
            optimizer.zero_grad()

            descriptors = body(batch)
            logits = head(descriptors)

            loss = loss_module(logits, labels)

            progress_bar.set_description(f'{loss.item():.4f}')
            progress_bar.update(len(batch))

            loss.backward()

            optimizer.step()

            scheduler.step()
            for parameters_group in optimizer.param_groups:
                parameters_group['lr'] = scheduler.lr

            progress_bar.update(len(batch))

    progress_bar.close()

    body = body.eval()
    head = head.eval()

    torch.save(body.state_dict(), osp.join(path, 'weights.body.pth'))
    torch.save(head.state_dict(), osp.join(path, 'weights.head.pth'))


def parse_config(config: dict) -> (
    LabeledImageTable, DataLoader,
    nn.Module, nn.Module,
    Optimizer, Scheduler,
    nn.Module
):
    """Prepare train experiment

    Args:
        config: configuration dictionary

    Returns:
        train table, train loader, 
        backbone module, classifier module,
        optimizer, scheduler,
        loss module
    """

    # >>>>> train table
    train_table_type = config['train_table'].pop('type')

    if train_table_type == 'default':
        train_table = LabeledImageTable.read(**config['train_table'])

    else:
        raise NotImplementedError

    # >>>>> train transformer
    train_transformer_type = config['train_transformer'].pop('type')

    if train_transformer_type == 'default':
        train_transformer = Transformer(**config['train_transformer'])

    else:
        raise NotImplementedError

    # >>>>> train loader
    train_dataset = Dataset(train_table, train_transformer)

    batch_size = config['train_sampler'].pop('batch_size')

    train_loader_workers = config['train_sampler'].pop('workers')

    train_sampler_type = config['train_sampler'].pop('type')

    if train_sampler_type == 'default':
        train_sampler = RandomSampler(
            length=len(train_dataset), 
            batch_size=batch_size,
            **config['train_sampler']
        )

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
        head = Head(
            descriptor_size=body.descriptor_size,
            n_classes=train_table.n_classes,
            **config['head']
        )

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
        scheduler = Scheduler(lr, steps, **config['scheduler'])

    else:
        raise NotImplementedError

    # >>>>> loss
    loss_type = config['loss'].pop('type')

    if loss_type == 'default':
        loss_module = CrossEntropyLoss(**config['loss'])

    else:
        raise NotImplementedError

    return (
        train_table, train_loader,
        body, head,
        optimizer, scheduler,
        loss_module,
        config['path']
    )
