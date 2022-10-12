"""Learning rate schedule
"""

import math


class Scheduler
    """Learning rate scheduler
    """

    def __init__(self, lr: float, iterations: int):
        """Create scheduler object
        """

        self.lr_base = lr
        self.iterations = iterations

        self.lr = self.lr_base
        self.iteration = 0

    def do_iteration(self):
        """Do learning rate update
        """

        self.lr = self.lr_base

        self.iteration += 1


class StepScheduler(Scheduler):
    """Learning rate step scheduler
    """

    def __init__(self, lr: float, iterations: int, steps: int, factor: float):
        """Creates step scheduler object
        """

        super().__init__(lr, iterations)

        self.steps = steps
        self.factor = factor

        self.step = 0

    def do_iteration(self):
        """Do learning rate update
        """

        step = math.floor(self.steps * self.iteration / self.iterations)
        
        if step - self.step == 1:
            self.step = step
            self.lr *= self.factor

        elif step - self.step == 0:
            pass

        else:
            raise ValueError, 'step difference cannot be greater than one'

        self.iteration += 1
