"""Train model
"""

import math

from .table import LabeledImageTable
from .transformer import Transformer
from .dataset import Dataset
from .sampler import Sampler, ShuffledSampler
from torch.utils.data import DataLoader

import torch
from torch import nn

from .stem import Stem
from .convnet import ConvNet
from .neck import Neck
from .head import Head

from torch.optim import Optimizer, SGD
from .scheduler import Scheduler, SequentialScheduler, \
    LinearScheduler, CosineScheduler

from torch.nn.modules.loss import CrossEntropyLoss

from .manager import Manager

from tqdm import tqdm


def train(
    train_table: LabeledImageTable, train_loader: DataLoader,
    stem: nn.Module, body: nn.Module, neck: nn.Module, head: nn.Module,
    optimizer: Optimizer, scheduler: Scheduler,
    loss_module: nn.Module,
    manager: Manager
):
    """Train model

    Args:
        train_table: labeled image table for training
        train_loader: dataloader of train dataset
        stem: stem module
        body: body module
        neck: neck module
        head: head module
        optimizer: optimizer
        scheduler: learning rate scheduler
        loss_module: loss module
        manager: experiment helper
    """

    epochs = len(scheduler) // len(train_table)

    progress_bar = tqdm(total=len(scheduler))

    stem = stem.train()
    body = body.train()
    neck = neck.train()
    head = head.train()

    schedule = iter(scheduler)

    step = 0
    for epoch in range(epochs):
        for batch, labels in train_loader:
            lr = next(schedule)
            
            info = dict()
            info['step'] = step
            info['lr'] = lr

            for parameters_group in optimizer.param_groups:
                parameters_group['lr'] = lr
            optimizer.zero_grad()

            feature_map = stem(batch)
            feature_map = body(feature_map)
            descriptors = neck(feature_map)
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

    stem = stem.eval()
    body = body.eval()
    neck = neck.eval()
    head = head.eval()

    manager.save_weights(
        stem.state_dict(), 
        body.state_dict(), 
        neck.state_dict(), 
        head.state_dict()
    )

# -----------------------------------------------------------------------------


def parse_config(config: dict) -> (
    LabeledImageTable, DataLoader,
    nn.Module, nn.Module, nn.Module, nn.Module,
    Optimizer, Scheduler,
    nn.Module
):
    """Prepare train experiment

    Args:
        config: configuration dictionary

    Returns:
        train table, train loader, 
        stem, body, neck, head modules,
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
        if 'transforms' in config['train_transformer']:
            transforms = config['train_transformer'].pop('transforms')
        else:
            transforms = list()

        train_transformer = Transformer(**config['train_transformer'])

        for transform in transforms:
            transform_type = transform.pop('type')

            if transform_type == 'random_crop':
                train_transformer.add_random_crop(**transform)

            else:
                raise NotImplementedError

    else:
        raise NotImplementedError

    # >>>>> train loader
    train_dataset = Dataset(train_table, train_transformer)

    train_loader_workers = config['train_sampler'].pop('workers')

    train_sampler_type = config['train_sampler'].pop('type')

    if train_sampler_type == 'default':
        train_sampler = ShuffledSampler(
            length=len(train_dataset), 
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
    stem_type = config['stem'].pop('type')

    if stem_type == 'default':
        stem = Stem(in_channels=train_table.channels, **config['stem'])

    else:
        raise NotImplementedError

    body_type = config['body'].pop('type')

    if body_type == 'convnet':
        body = ConvNet(in_channels=stem.conv.out_channels, **config['body'])

    else:
        raise NotImplementedError

    neck_type = config['neck'].pop('type')

    if neck_type == 'default':
        neck = Neck(in_channels=body.descriptor_size, **config['neck'])

    else:
        raise NotImplementedError

    head_type = config['head'].pop('type')

    if head_type == 'default':
        head = Head(
            descriptor_size=neck.descriptor_size,
            classes=train_table.classes,
            **config['head']
        )

    else:
        raise NotImplementedError

    # >>>>> loss
    loss_type = config['loss'].pop('type')
    
    if loss_type == 'xent':
        loss_module = CrossEntropyLoss(**config['loss'])

    else:
        raise NotImplementedError

    # >>>>> optimizer
    optimizer_type = config['optimizer'].pop('type')

    lr = config['optimizer'].pop('lr')

    if optimizer_type == 'sgd':
        optimizer = SGD(
            [
                {'params': stem.parameters()},
                {'params': body.parameters()}, 
                {'params': neck.parameters()},
                {'params': head.parameters()}
            ], lr=lr, **config['optimizer']
        )

    else:
        raise NotImplementedError

    # >>>>> scheduler
    scheduler_type = config['scheduler'].pop('type')

    epochs = config['scheduler'].pop('epochs')

    if scheduler_type == 'sequential':
        schedulers = list()
        for scheduler in config['scheduler']['schedulers']:
            scheduler_type = scheduler.pop('type')
            
            if 'steps' in scheduler:
                steps = scheduler.pop('steps')

            elif 'epochs' in scheduler:
                steps = len(train_table) * scheduler.pop('epochs')

            elif 'fraction' in scheduler:
                steps = math.ceil(
                    len(train_table) * epochs * scheduler.pop('fraction')
                )

            else:
                raise KeyError(
                    'scheduler must have either "steps" or "epochs" argument'
                )

            if 'lr' not in scheduler:
                scheduler['lr'] = lr

            if scheduler_type == 'constant':
                scheduler = Scheduler(steps, **scheduler)

            elif scheduler_type == 'linear':
                scheduler = LinearScheduler(steps, **scheduler)

            elif scheduler_type == 'cosine':
                scheduler = CosineScheduler(steps, **scheduler)

            else:
                raise NotImplementedError

            schedulers.append(scheduler)

        scheduler = SequentialScheduler(*schedulers, lr=lr)

    else:
        if 'steps' in config['scheduler']:
            steps = config['scheduler'].pop('steps')

        else:
            steps = epochs * len(train_table)

        if 'lr' not in config['scheduler']:
            config['scheduler']['lr'] = lr

        if scheduler_type == 'constant':
            scheduler = Scheduler(steps, **config['scheduler'])

        elif scheduler_type == 'linear':
            scheduler = LinearScheduler(steps, **config['scheduler'])

        elif scheduler_type == 'cosine':
            scheduler = CosineScheduler(steps, **config['scheduler'])

        else:
            raise NotImplementedError

    assert len(scheduler) >= epochs * len(train_table), \
        'schedule must be longer than train steps'

    if len(scheduler) > epochs * len(train_table):
        scheduler.steps = epochs * len(train_table)

    manager.plot_lr_schedule(scheduler)

    return (
        train_table, train_loader,
        stem, body, neck, head,
        optimizer, scheduler,
        loss_module,
        manager
    )
