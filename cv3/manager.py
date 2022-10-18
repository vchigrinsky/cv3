"""Experiment manager
"""

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

    def __init__(self, config: dict, smoothing_window: int = 1024):
        """Creates manager instance

        Args:
            config: experiment configuration dictionary
            smoothing window: number of last seen images 
                to smooth train loss over
        """

        self.config = config
        self.root = osp.abspath(self.config.pop('root'))

        self.info = list()

        self.running_train_losses = list()
        self.running_images_seen = list()

        self.smoothing_window = smoothing_window

        logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------

    @property
    def attributes(self) -> tuple:
        """Get experiment attributes

        Returns:
            Info table columns except "step" and "lr"
        """

        table = self.get_info()

        attributes = [
            attribute for attribute in table.columns
            if attribute not in {'step', 'lr'}
        ]

        return attributes

    def plot_attribute_info(self, attribute: str):
        """Plots info for specific attribute

        Args:
            attribute: attribute name, i.e. "train_loss" or "lr"
        """

        table = self.get_info()

        x = table.dataframe.step.values
        y = table.dataframe[attribute].values

        xaxis = go.layout.XAxis(
            title='Images seen', 
            range=[x.min(), x.max()]
        )

        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        yaxis = go.layout.YAxis(
            title=attribute.replace('_', ' ').title(), 
            range=[
                y.min() - (y.max() - y.min()) * 0.1, 
                y.max() + (y.max() - y.min()) * 0.1
            ]
        )

        layout = go.Layout(
            title=self.root, 
            xaxis=xaxis, yaxis=yaxis
        )

        fig = go.Figure(layout=layout)

        trace = go.Scatter(x=x, y=y)
        fig.add_trace(trace)

        fig.write_html(osp.join(self.root, f'{attribute}.html'))

    def save_weights(
        self, body_weights: OrderedDict, head_weights: OrderedDict
    ):
        """Save model state dicts (body and head) to the experiment directory

        Args:
            body_weights: model body state dict
            head_weights: model head state dict
        """

        torch.save(body_weights, osp.join(self.root, 'weights.body.pth'))
        torch.save(head_weights, osp.join(self.root, 'weights.head.pth'))

    def save_info(self):
        """Dumps experiment info to "ROOT/info.csv"
        """

        table = self.get_info()
        table.dataframe.to_csv(osp.join(self.root, 'info.csv'), index=False)

    def get_info(self) -> Table:
        """Returns experiment info as table object
        """

        dataframe = DataFrame(self.info)
        table = Table(dataframe)

        return table

    def update_info(self, info: dict):
        """Updates manager info

        Args:
            info: experiment info dictionary, i.e.
                {
                    'step': step,
                    'lr': lr,
                    'train_loss': train_loss,
                    ...
                }, must have 'step' and 'lr' keys
        """

        assert 'step' in info, 'info should has "step" key'
        assert 'lr' in info, 'info should has "lr" key'

        self.info.append(copy.deepcopy(info))

    # -------------------------------------------------------------------------

    def smooth_train_loss(self, train_loss: float, images_seen: int) -> float:
        """Smooth train loss

        Args:
            train_loss: train loss on current batch
            images_seen: current batch size

        Returns:
            smoothed train loss
        """

        self.running_train_losses.append(train_loss)
        self.running_images_seen.append(images_seen)

        smoothing_window = sum(self.running_images_seen)
        while smoothing_window > self.smoothing_window:
            _ = self.running_train_losses.pop(0)
            smoothing_window -= self.running_images_seen.pop(0)

        smoothed_train_loss = sum([
            running_images_seen * running_train_loss 
            for running_images_seen, running_train_loss 
            in zip(self.running_images_seen, self.running_train_losses)
        ]) / smoothing_window

        return smoothed_train_loss

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

    def mkdir_experiment_root(self):
        """Creates empty experiment directory
        """

        os.mkdir(self.root)

    def clear_experiment_root(self):
        """Creates empty experiment directory
        """

        logging.info(f'removing {self.root}')

        shutil.rmtree(self.root)

    def init_experiment(self):
        """Initialize experiment with config file
        """

        with open(osp.join(self.root, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    def plot_lr_schedule(self, scheduler: Scheduler):
        """Plots lr scheduler and saves plot as html file in experiment root

        Args:
            scheduler: learning rate scheduler
        """

        x = list(range(scheduler.steps))
        y = scheduler.schedule

        xaxis = go.layout.XAxis(
            title='Images seen', 
            range=[min(x), max(x)]
        )
        yaxis = go.layout.YAxis(
            title='Learning rate', 
            range=[0.0, 1.1 * scheduler.lr_base]
        )

        layout = go.Layout(
            title=self.root, 
            xaxis=xaxis, yaxis=yaxis
        )

        fig = go.Figure(layout=layout)

        trace = go.Scatter(x=x, y=y)
        fig.add_trace(trace)

        fig.write_html(osp.join(self.root, 'lr_schedule.html'))

    def quit(self):
        """Quits experiment
        """

        logging.info('quitting')

        sys.exit(1)
