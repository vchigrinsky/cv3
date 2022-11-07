"""Learning rate schedule
"""

import math

from tqdm import tqdm


class Scheduler:
    """Learning rate scheduler
    """

    def __init__(self, steps: int, lr: float):
        """Constant learning rate scheduler

        Args:
            lr: learning rate
            steps: number of steps in schedule
        """

        assert steps > 0, 'steps must be greater than zero'

        self.lr_base = lr
        self.steps = steps

    def __len__(self) -> int:
        """Schedule length
        """

        return self.steps

    def __iter__(self):
        """Generates lr step by step
        """

        for _ in range(len(self)):
            yield self.lr_base


class LinearScheduler(Scheduler):
    """Linear learning rate scheduler
    """

    def __init__(
        self, steps: int, lr: float, 
        start: float = 1.0, stop: float = 0.0
    ):
        """Creates linear scheduler as following: schedule (x: step, y: lr)
            segment y = x @ [0, 1] is scaled with lr over y, 
            clipped with [start, stop] over x 
            and scaled with steps / (stop - start) over x,
            schedule direction (ascending or descending) is determined by 
            which of the values (start or stop) is greater

        Args:
            lr: y-scale of y = x @ [0, 1] learning rate schedule
            steps: x-scale of y = lr * x @ [start, stop] learning rate schedule
            start: relative left clip at [0, 1]
            stop: relative right clip at [0, 1]
        """

        super().__init__(steps, lr)

        assert not math.isclose(start, stop), 'start and stop must not be equal'

        self.start = start
        self.stop = stop

    def __iter__(self):
        """Generates lr step by step according to the linear rule
        """

        for _ in range(len(self)):
            x = self.start + (step / len(self)) * (self.stop - self.start)
            if self.start > self.stop:
                yield self.lr_base * x
            else:
                yield self.lr_base * (1.0 - x)


class CosineScheduler(Scheduler):
    """Cosine learning rate scheduler
    """

    def __init__(
        self, steps: int, lr: float, 
        start: float = 1.0, stop: float = 0.0
    ):
        """Creates linear scheduler as following: schedule (x: step, y: lr)
            segment y = cos(x) @ [0, pi/2] is scaled with lr over y, 
            clipped with [(pi/2) * start, (pi/2) * stop] over x 
            and scaled with steps * (pi/2) / (stop - start) over x, 
            schedule direction (ascending or descending) is determined by 
            which of the values (start or stop) is greater

        Args:
            lr: y-scale of y = cos(x) @ [0, pi/2] learning rate schedule
            steps: x-scale of y = lr * cos(x) @ [(pi/2) * start,  (pi/2) * stop]
                learning rate schedule
            start: relative left clip at [0, pi/2]
            stop: relative right clip at [0, pi/2]
        """

        super().__init__(steps, lr)

        assert not math.isclose(start, stop), 'start and stop must not be equal'

        self.start = start
        self.stop = stop

    def __iter__(self):
        """Generates lr step by step according to the cosine rule
        """

        for step in range(len(self)):
            x = (math.pi / 2) * (
                self.start + (step / len(self)) * (self.stop - self.start)
            )
            yield self.lr_base * math.sin(x)


class SequentialScheduler(Scheduler):
    """Sequence of schedulers
    """

    def __init__(self, *schedulers, lr: float):
        """Combine several schedulers into a single one
        """

        steps = sum([len(scheduler) for scheduler in schedulers])

        super().__init__(steps, lr)

        self.schedulers = schedulers

    def __iter__(self):
        """Iterate over concatenation of schedulers
        """

        step = 0
        for scheduler in self.schedulers:
            for lr in scheduler:
                yield lr
                step += 1
                if step >= len(self):
                    break
