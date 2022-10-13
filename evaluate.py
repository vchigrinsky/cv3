"""Run evaluation
"""

import cv3

import os.path as osp

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
        type=str, required=False
    )

    argument_parser.add_argument(
        '--height', help='Input height', 
        type=int, default=28
    )
    argument_parser.add_argument(
        '--width', help='Input width', 
        type=int, default=28
    )

    argument_parser.add_argument(
        '--mean', help='Normalization mean', 
        type=float, default=0.1309
    )
    argument_parser.add_argument(
        '--std', help='Normalization std', 
        type=float, default=0.3084
    )

    argument_parser.add_argument(
        '--descriptor_size', help='Descriptor size',
        type=int, default=128
    )

    argument_parser.add_argument(
        '--batch_size', help='Batch size', 
        type=int, default=1
    )

    arguments = argument_parser.parse_args()

    # -------------------------------------------------------------------------

    table = cv3.table.LabeledImageTable.read(arguments.table, arguments.prefix)

    mean = (arguments.mean, arguments.mean, arguments.mean)
    std = (arguments.std, arguments.std, arguments.std)
    transformer = cv3.transformer.Transformer(
        arguments.height, arguments.width, mean, std
    )

    body = cv3.model.ConvNet(arguments.descriptor_size)
    head = cv3.model.Head(arguments.descriptor_size, table.n_classes)

    if arguments.experiment is not None:
        body.load_state_dict(
            torch.load(osp.join(arguments.experiment, 'weights.body.pth'))
        )
        head.load_state_dict(
            torch.load(osp.join(arguments.experiment, 'weights.head.pth'))
        )

    accuracy = cv3.evaluator.evaluate_classification(
        table, transformer, body, head, 
        arguments.batch_size, verbose=True
    )

    print(f'Accuracy: {accuracy:.4f}')
