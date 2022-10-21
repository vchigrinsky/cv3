"""Run evaluation
"""

import cv3

import os.path as osp
import json
from collections import OrderedDict

import torch

from argparse import ArgumentParser


if __name__ == '__main__':
    """Run classification evaluation
    """

    argument_parser = ArgumentParser('Classification evaluation')

    argument_parser.add_argument(
        '--table', help='Path to CSV table', 
        type=str, required=True
    )
    argument_parser.add_argument(
        '--prefix', help='Common prefix of all images in table', 
        type=str, required=True
    )

    argument_parser.add_argument(
        '--experiment', help='Path to experiment directory',
        type=str, required=True
    )

    argument_parser.add_argument(
        '--batch_size', help='Batch size', 
        type=int, default=64
    )

    arguments = argument_parser.parse_args()

    # -------------------------------------------------------------------------

    with open(osp.join(arguments.experiment, 'config.json'), 'r') as f:
        config = json.load(f)

    stem_weights = torch.load(
        osp.join(arguments.experiment, 'weights.stem.pth')
    )
    body_weights = torch.load(
        osp.join(arguments.experiment, 'weights.body.pth')
    )
    neck_weights = torch.load(
        osp.join(arguments.experiment, 'weights.neck.pth')
    )
    head_weights = torch.load(
        osp.join(arguments.experiment, 'weights.head.pth')
    )

    # >>>>> test table
    channels = stem_weights['conv.weight'].size(1)

    if channels == 3:
        mode = 'rgb'
    elif channels == 1:
        mode = 'gray'
    else:
        raise NotImplementedError

    table = cv3.table.LabeledImageTable.read(
        arguments.table, arguments.prefix, mode
    )

    # >>>>> test transformer
    transformer_type = config['test_transformer'].pop('type')

    if transformer_type == 'default':
        if 'transforms' in config['test_transformer']:
            transforms = config['test_transformer'].pop('transforms')
        else:
            transforms = list()

        transformer = cv3.transformer.Transformer(**config['test_transformer'])

        for transform in transforms:
            transform_type = transform.pop('type')

            raise NotImplementedError

    else:
        raise NotImplementedError

    # >>>>> model
    stem_type = config['stem'].pop('type')

    if stem_type == 'default':
        stem = cv3.stem.Stem(in_channels=table.channels, **config['stem'])

    else:
        raise NotImplementedError

    body_type = config['body'].pop('type')

    if body_type == 'convnet':
        body = cv3.convnet.ConvNet(in_channels=stem.conv.out_channels, **config['body'])

    else:
        raise NotImplementedError

    neck_type = config['neck'].pop('type')

    if neck_type == 'default':
        neck = cv3.neck.Neck(in_channels=body.descriptor_size, **config['neck'])

    else:
        raise NotImplementedError

    head_type = config['head'].pop('type')

    if head_type == 'default':
        head = cv3.head.Head(
            descriptor_size=neck.descriptor_size,
            classes=table.classes,
            **config['head']
        )

    else:
        raise NotImplementedError

    stem.load_state_dict(stem_weights)
    body.load_state_dict(body_weights)
    neck.load_state_dict(neck_weights)
    head.load_state_dict(head_weights)

    # >>>>> evaluate
    accuracy = cv3.evaluator.evaluate_classification(
        table, transformer, stem, body, neck, head, 
        arguments.batch_size, verbose=True
    )

    print(f'Accuracy: {accuracy:.4f}')
