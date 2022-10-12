"""Run train
"""

import cv3

from argparse import ArgumentParser
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

    config['path'] = arguments.json[:-5]
    if not osp.exists(config['path']):
        osp.mkdir(config['path'])

    cv3.trainer.train(*cv3.trainer.parse_config(config))
