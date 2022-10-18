"""Train model
"""

from .table import LabeledImageTable
from .transformer import Transformer
from .dataset import Dataset
from .sampler import Sampler, RandomSampler
from torch.utils.data import DataLoader

import torch
from torch import nn
from .convnet import ConvNet
from .head import Neck, Head

from torch.optim import Optimizer, SGD
from .scheduler import Scheduler, SequentialScheduler, \
    LinearScheduler, CosineScheduler

from torch.nn.modules.loss import CrossEntropyLoss

from .manager import Manager

from tqdm import tqdm

import os.path as osp


def train(
    train_table: LabeledImageTable, train_loader: DataLoader,
    body: nn.Module, neck: nn.Module, head: nn.Module,
    optimizer: Optimizer, scheduler: Scheduler,
    loss_module: nn.Module,
    manager: Manager
):
    """Train model

    Args:
        train_table: labeled image table for training
        train_loader: dataloader of train dataset
        body: body module
        neck: neck module
        head: head module
        optimizer: optimizer
        scheduler: learning rate scheduler
        loss_module: loss module
        manager: experiment helper
    """

    assert scheduler.steps % len(train_table) == 0, \
        'number of steps should be divisible by train dataset length'
    epochs = scheduler.steps // len(train_table)

    progress_bar = tqdm(total=scheduler.steps)

    body = body.train()
    neck = neck.train()
    head = head.train()

    step = 0
    for epoch in range(epochs):
        for batch, labels in train_loader:
            lr = scheduler.lr(step)
            
            info = dict()
            info['step'] = step
            info['lr'] = lr

            for parameters_group in optimizer.param_groups:
                parameters_group['lr'] = lr
            optimizer.zero_grad()

            descriptors = body(batch)
            descriptors = neck(descriptors)
            logits = head(descriptors)

            loss = loss_module(logits, labels)
            train_loss = loss.item()
            smoothed_train_loss = manager.smooth_train_loss(
                train_loss=train_loss, images_seen=len(batch)
            )

            info['train_loss'] = train_loss
            info['smoothed_train_loss'] = smoothed_train_loss
            progress_bar.set_description(
                f'LR {lr:.4f}; Train loss {smoothed_train_loss:.4f}'
            )

            loss.backward()

            optimizer.step()
            step += len(batch)

            manager.update_info(info)
            progress_bar.update(len(batch))

    progress_bar.close()

    manager.save_info()

    for attribute in manager.attributes:
        manager.plot_attribute_info(attribute)

    body = body.eval()
    neck = neck.eval()
    head = head.eval()

    body_weights = body.state_dict()
    body_weights.update(neck.state_dict())
    head_weights = head.state_dict()

    manager.save_weights(body_weights, head_weights)

# -----------------------------------------------------------------------------


def parse_config(config: dict) -> (
    LabeledImageTable, DataLoader,
    nn.Module, nn.Module, nn.Module,
    Optimizer, Scheduler,
    nn.Module
):
    """Prepare train experiment

    Args:
        config: configuration dictionary

    Returns:
        train table, train loader, 
        body module, neck module, head module,
        optimizer, scheduler,
        loss module
    """

    # >>>>> manager
    manager = Manager(config)

    exists, overwrite = manager.check_experiment_root()
    if exists:
        if overwrite:
            manager.clear_experiment_root()
            manager.mkdir_experiment_root()
        else:
            manager.quit()

    else:
        manager.mkdir_experiment_root()

    manager.init_experiment()

    # >>>>> train table
    train_table_type = config['train_table'].pop('type')

    if train_table_type == 'default':
        train_table = LabeledImageTable.read(**config['train_table'])

    else:
        raise NotImplementedError

    # >>>>> train transformer
    train_transformer_type = config['train_transformer'].pop('type')

    if train_transformer_type == 'default':
        train_transformer = Transformer(
            mode='train', **config['train_transformer']
        )

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

    # descriptor size in body config is deprecated
    if 'descriptor_size' in config['body']:
        config['head']['descriptor_size'] \
            = config['body'].pop('descriptor_size')

    if body_type == 'default':
        body = ConvNet(channels=train_table.channels, **config['body'])

    else:
        raise NotImplementedError

    hidden_descriptor_size = None
    for layer in body.modules():        
        if isinstance(layer, nn.Conv2d):
            hidden_descriptor_size = layer.out_channels

    head_type = config['head'].pop('type')

    descriptor_size = config['head'].pop('descriptor_size')

    if head_type == 'default':
        neck = Neck(
            channels=hidden_descriptor_size,
            descriptor_size=descriptor_size,
            linear_bias=(
                config['body']['conv_bias'] 
                if 'conv_bias' in config['body'] else True
            ),
            init_weights=(
                config['body']['init_weights'] 
                if 'init_weights' in config['body'] else False
            )
        )
        head = Head(
            descriptor_size=descriptor_size,
            classes=train_table.classes,
            **config['head']
        )

    else:
        raise NotImplementedError

    # >>>>> loss
    loss_type = config['loss'].pop('type')
    if loss_type == 'default':
        loss_type = 'xent'

    if loss_type == 'xent':
        loss_module = CrossEntropyLoss(**config['loss'])

    else:
        raise NotImplementedError

    # >>>>> optimizer
    optimizer_type = config['optimizer'].pop('type')
    if optimizer_type == 'default':
        optimizer_type = 'sgd'

    lr = config['optimizer'].pop('lr')

    if optimizer_type == 'sgd':
        optimizer = SGD(
            [
                {'params': body.parameters()}, 
                {'params': neck.parameters()},
                {'params': head.parameters()}
            ], lr=lr, **config['optimizer']
        )

    else:
        raise NotImplementedError

    # >>>>> scheduler
    scheduler_type = config['scheduler'].pop('type')
    if scheduler_type == 'default':
        scheduler_type = 'constant'

    if scheduler_type == 'sequential':
        schedulers = list()
        for scheduler in config['scheduler']['schedulers']:
            scheduler_type = scheduler.pop('type')
            if scheduler_type == 'default':
                scheduler_type = 'constant'

            epochs = scheduler.pop('epochs')
            steps = epochs * len(train_table)

            if scheduler_type == 'constant':
                scheduler = Scheduler(steps, lr, **scheduler)

            elif scheduler_type == 'linear':
                scheduler = LinearScheduler(steps, lr, **scheduler)

            elif scheduler_type == 'cosine':
                scheduler = CosineScheduler(steps, lr, **scheduler)

            else:
                raise NotImplementedError

            schedulers.append(scheduler)

        scheduler = SequentialScheduler(*schedulers, lr=lr)

    else:
        epochs = config['scheduler'].pop('epochs')
        steps = epochs * len(train_table)

        if scheduler_type == 'constant':
            scheduler = Scheduler(steps, lr, **config['scheduler'])

        elif scheduler_type == 'linear':
            scheduler = LinearScheduler(steps, lr, **config['scheduler'])

        elif scheduler_type == 'cosine':
            scheduler = CosineScheduler(steps, lr, **config['scheduler'])

        else:
            raise NotImplementedError

    manager.plot_lr_schedule(scheduler)

    return (
        train_table, train_loader,
        body, neck, head,
        optimizer, scheduler,
        loss_module,
        manager
    )
