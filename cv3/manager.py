"""Experiment manager
"""

import math

import sys
import os
import os.path as osp
import shutil
import copy
import json

import logging
from plotly import graph_objects as go

from collections import OrderedDict
import numpy as np
import torch

from pandas import DataFrame
from .table import Table
from .scheduler import Scheduler


class Manager:
    """Manager helping to keep necessary information during experiment
    """

    def __init__(
        self, config: dict, watch: tuple = ('lr', 'train_loss'), 
        smoothing_window: int = 1024
    ):
        """Creates manager instance

        Args:
            config: experiment configuration dictionary
            watch: attributes to watch on during experiment
            smoothing window: number of last seen images 
                to smooth train loss over
        """

        self.config = config
        self.root = osp.abspath(self.config.pop('root'))

        self.running_train_losses = list()
        self.running_images_seen = list()

        self.smoothing_window = smoothing_window

        logging.basicConfig(level=logging.INFO)

        assert isinstance(watch, tuple), 'attributes to watch must be unmutable'
        self.attributes = watch

    # -------------------------------------------------------------------------

    def quit(self):
        """Quits experiment
        """

        logging.info('quitting')

        sys.exit(1)

    # -------------------------------------------------------------------------

    def check_experiment_root(self) -> (bool, bool):
        """Check whether experiment directory exists, warn if it does
        
        Returns:
            bool tuple: (exists, overwrite)
        """

        if osp.exists(self.root):
            exists = True

            ret = ''

            while ret not in {'y', 'n'}:
                logging.info(f'{self.root} exists, overwrite? (y/n)')
                
                ret = input().lower()

                if ret == 'y':
                    overwrite = True
                elif ret == 'n':
                    overwrite = False

        else:
            exists = False
            overwrite = False

        return exists, overwrite

    def clear_experiment_root(self):
        """Creates empty experiment directory
        """

        logging.info(f'removing {self.root}')

        shutil.rmtree(self.root)

    def mkdir_experiment_root(self):
        """Creates empty experiment directory
        """

        os.mkdir(self.root)

    # -------------------------------------------------------------------------

    def init_experiment(self):
        """Initialize experiment with config file
        """

        with open(osp.join(self.root, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        with open(osp.join(self.root, 'log.csv'), 'w') as f:
            f.write('step')
            for attribute in self.attributes:
                f.write(f',{attribute}')
            f.write('\n')

    def plot_lr_schedule(
        self, scheduler: Scheduler, 
        points: int = 1024, margin: float = 0.1
    ):
        """Plots lr scheduler and saves plot as html file in experiment root

        Args:
            scheduler: learning rate scheduler
            points: number of points to plot
            margin: top margin of the float (leave default if confused)
        """

        logging.info('plotting lr schedule')

        step = math.ceil(len(scheduler) / points)

        x, y = list(), list()
        for n, lr in enumerate(scheduler):
            if n % step == 0:
                x.append(n)
                y.append(lr)

        xaxis = go.layout.XAxis(
            title='Images seen', 
            range=[min(x), max(x)]
        )
        yaxis = go.layout.YAxis(
            title='Learning rate', 
            range=[0.0, (1.0 + margin) * max(y)]
        )

        layout = go.Layout(
            title=self.root, 
            xaxis=xaxis, yaxis=yaxis
        )

        fig = go.Figure(layout=layout)

        trace = go.Scatter(x=x, y=y)
        fig.add_trace(trace)

        fig.write_html(osp.join(self.root, 'lr_schedule.html'))

    # -------------------------------------------------------------------------

    def smooth_train_loss(
        self, exact_train_loss: float, images_seen: int
    ) -> float:
        """Smooth train loss

        Args:
            exact_train_loss: exact value of train loss on current batch
            images_seen: current batch size

        Returns:
            smoothed train loss
        """

        self.running_train_losses.append(exact_train_loss)
        self.running_images_seen.append(images_seen)

        smoothing_window = sum(self.running_images_seen)
        while smoothing_window > self.smoothing_window:
            _ = self.running_train_losses.pop(0)
            smoothing_window -= self.running_images_seen.pop(0)

        train_loss = sum([
            running_images_seen * running_train_loss 
            for running_images_seen, running_train_loss 
            in zip(self.running_images_seen, self.running_train_losses)
        ]) / smoothing_window

        return train_loss

    def log(self, step: int, log: dict):
        """Logs attributes on current step to a file

        Args:
            log: values of attributes to log
        """

        with open(osp.join(self.root, 'log.csv'), 'a') as f:
            f.write(f'{step}')
            for attribute in self.attributes:
                f.write(f',{log[attribute]}')
            f.write('\n')

    # -------------------------------------------------------------------------

    def read_log_table(self) -> Table:
        """Reads experiment log as table object

        Returns:
            log table
        """

        table = Table.read(osp.join(self.root, 'log.csv'))

        return table

    def plot_attribute_log(
        self, attribute: str, log_table: Table = None,
        points: int = 1024, margin: float = 0.1
    ):
        """Plots log for specific attribute

        Args:
            attribute: attribute name, i.e. "train_loss" or "lr"
            log_table: full log table, if None then it will be read from file
            points: number of points to plot
            margin: top margin of the plot (default value is quite nice)
        """

        if log_table is None:
            log_table = self.read_log_table()

        logging.info(f'plotting {attribute}')

        step = math.ceil(len(log_table) / points)

        x, y = list(), list()
        for n, value in zip(
            log_table.dataframe.step.values, 
            log_table.dataframe[attribute].values
        ):
            if n % step == 0:
                x.append(n)
                y.append(value)

        x = np.asarray(x)
        y = np.asarray(y)

        xaxis = go.layout.XAxis(
            title='Images seen', 
            range=[x.min(), x.max()]
        )

        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        if attribute in ('lr', 'train_loss'):
            bottom = 0.0
        else:
            bottom = y.min() - (y.max() - y.min()) * margin

        top = y.max() + (y.max() - y.min()) * margin

        yaxis = go.layout.YAxis(
            title=attribute.replace('_', ' ').title(), 
            range=[bottom, top]
        )

        layout = go.Layout(
            title=self.root, 
            xaxis=xaxis, yaxis=yaxis
        )

        fig = go.Figure(layout=layout)

        trace = go.Scatter(x=x, y=y)
        fig.add_trace(trace)

        fig.write_html(osp.join(self.root, f'{attribute}.html'))

    # -------------------------------------------------------------------------

    def save_weights(
        self, stem_weights: OrderedDict,
        body_weights: OrderedDict, 
        neck_weights: OrderedDict, 
        head_weights: OrderedDict
    ):
        """Save model state dicts (body and head) to the experiment directory

        Args:
            stem_weights: model stem state dict
            body_weights: model body state dict
            neck_weights: model neck state dict
            head_weights: model head state dict
        """

        torch.save(stem_weights, osp.join(self.root, 'weights.stem.pth'))
        torch.save(body_weights, osp.join(self.root, 'weights.body.pth'))
        torch.save(neck_weights, osp.join(self.root, 'weights.neck.pth'))
        torch.save(head_weights, osp.join(self.root, 'weights.head.pth'))
