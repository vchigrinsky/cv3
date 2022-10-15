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

    if 'conv_bias' in config['body']:
        bias = config['body']['conv_bias']
    else:
        bias = True

    body_weights = torch.load(
        osp.join(arguments.experiment, 'weights.body.pth')
    )
    head_weights = torch.load(
        osp.join(arguments.experiment, 'weights.head.pth')
    )

    neck_weights = OrderedDict()
    for layer_name, layer in list(body_weights.items()):
        if layer_name.startswith('linear.'):
            neck_weights[layer_name] = body_weights.pop(layer_name)
        elif layer_name.startswith('bn.'):
            neck_weights[layer_name] = body_weights.pop(layer_name)

    for _, layer in body_weights.items():
        break
    channels = layer.size(1)

    for _, layer in neck_weights.items():
        break
    hidden_descriptor_size = layer.size(1)

    for _, layer in head_weights.items():
        break
    descriptor_size = layer.size(1)
    classes = layer.size(0)

    if channels == 3:
        mode = 'rgb'
    elif channels == 1:
        mode = 'gray'
    else:
        raise NotImplementedError

    table = cv3.table.LabeledImageTable.read(
        arguments.table, arguments.prefix, mode
    )

    transformer_config = config['train_transformer']
    height = transformer_config['height']
    width = transformer_config['width']
    mean = transformer_config['mean']
    std = transformer_config['std']
    crop_height = (
        transformer_config['crop_height'] 
        if 'crop_height' in transformer_config else None
    )
    crop_width = (
        transformer_config['crop_width'] 
        if 'crop_width' in transformer_config else None
    )

    transformer = cv3.transformer.Transformer(
        height, width, mean, std, crop_height, crop_width, mode='test'
    )

    body = cv3.convnet.ConvNet(
        channels=channels,
        conv_bias=bias
    )
    neck = cv3.head.Neck(
        channels=hidden_descriptor_size,
        descriptor_size=descriptor_size,
        linear_bias=bias
    )
    head = cv3.head.Head(
        descriptor_size=descriptor_size,
        classes=classes
    )

    body.load_state_dict(body_weights)
    neck.load_state_dict(neck_weights)
    head.load_state_dict(head_weights)

    accuracy = cv3.evaluator.evaluate_classification(
        table, transformer, body, neck, head, 
        arguments.batch_size, verbose=True
    )

    print(f'Accuracy: {accuracy:.4f}')
