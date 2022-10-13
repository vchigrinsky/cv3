"""Learning rate schedule
"""

import math


class Scheduler:
    """Learning rate scheduler
    """

    def __init__(self, lr: float, steps: int):
        """Create scheduler object
        """

        self.lr_base = lr
        self.steps = steps

        self.lr = self.lr_base
        self.step_number = 0

    def step(self):
        """Do learning rate update
        """

        self.lr = self.lr_base

        self.step_number += 1
