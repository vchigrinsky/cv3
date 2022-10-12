"""Run evaluation
"""

import cv3

from argparse import ArgumentParser


if __name__ == '__main__':
    """Run classification evaluation
    """

    argument_parser = ArgumentParser('Classification evaluation')

    argument_parser.add_argument(
        '--path', help='Path to CSV table', 
        type=str, required=True
    )
    argument_parser.add_argument(
        '--prefix', help='Common prefix of all images in table', 
        type=str, required=True
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

    table = cv3.io.LabeledImageTable.read(arguments.path, arguments.prefix)

    n_classes = len(table.labels.unique())

    mean = (arguments.mean, arguments.mean, arguments.mean)
    std = (arguments.std, arguments.std, arguments.std)
    transformer = cv3.transformer.Transformer(
        arguments.height, arguments.width, mean, std
    )

    body = cv3.model.ConvNet(arguments.descriptor_size)
    head = cv3.model.Head(arguments.descriptor_size, n_classes)

    accuracy = cv3.evaluator.evaluate_classification(
        table, transformer, body, head, 
        arguments.batch_size, verbose=True
    )

    print(f'Accuracy: {accuracy:.4f}')
