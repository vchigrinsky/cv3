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

    if 'experiment_name' not in config:
        config['experiment_name'] = osp.split(arguments.config)[1][:-5]    

    if 'root' not in config:
        config['root'] = osp.join(
            osp.split(arguments.config)[0], config['experiment_name']
        )

    cv3.trainer.train(config)
