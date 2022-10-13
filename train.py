"""Run train
"""

import cv3

from argparse import ArgumentParser

import os
import os.path as osp

import json


if __name__ == '__main__':
    """Run classification train
    """

    argument_parser = ArgumentParser('Classification train')

    argument_parser.add_argument(
        '--config', help='Path to JSON config', 
        type=str, required=True
    )

    arguments = argument_parser.parse_args()

    # -------------------------------------------------------------------------

    assert arguments.config.endswith('.json')

    with open(arguments.config, 'r') as f:
        config = json.load(f)

    config['path'] = arguments.config[:-5]
    if not osp.exists(config['path']):
        os.mkdir(config['path'])

    with open(osp.join(config['path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    cv3.trainer.train(*cv3.trainer.parse_config(config))
