"""Run evaluation
"""

import cv3

import os.path as osp
import json

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

    argument_parser.add_argument(
        '--verbose', help='Display progress bar',
        action='store_true', default=True
    )

    argument_parser.add_argument(
        '--silent', help='Hide progress bar', 
        action='store_false', dest='verbose'
    )

    arguments = argument_parser.parse_args()

    # -------------------------------------------------------------------------

    with open(osp.join(arguments.experiment, 'config.json'), 'r') as f:
        config = json.load(f)

    config['test_table']['path'] = arguments.table
    config['test_table']['prefix'] = arguments.prefix

    config['stem']['weights'] = osp.join(
        arguments.experiment, 'weights.stem.pth'
    )
    config['body']['weights'] = osp.join(
        arguments.experiment, 'weights.body.pth'
    )
    config['neck']['weights'] = osp.join(
        arguments.experiment, 'weights.neck.pth'
    )
    config['head']['weights'] = osp.join(
        arguments.experiment, 'weights.head.pth'
    )

    config['batch_size'] = arguments.batch_size

    accuracy = cv3.evaluator.evaluate_classification(config, arguments.verbose)

    print(f'Accuracy: {accuracy:.4f}')
