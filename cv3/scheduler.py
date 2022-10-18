"""Learning rate schedule
"""

import math


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

        self.schedule = self.generate_schedule()

    def lr(self, step: int) -> float:
        """Get lerning rate by step 
        """

        return self.schedule[step]

    def generate_schedule(self) -> list:
        """Generate learning rate schedule
        """

        return [self.lr_base for _ in range(self.steps)]


class SequentialScheduler(Scheduler):
    """Sequence of schedulers
    """

    def __init__(self, *schedulers, lr: float):
        """Combine several schedulers into a single one
        """

        steps = sum([scheduler.steps for scheduler in schedulers])

        self.schedulers = schedulers

        super().__init__(steps, lr)

    def generate_schedule(self) -> list:
        """Stack learning rate schedules
        """

        schedule = list()
        for scheduler in self.schedulers:
            schedule.extend(scheduler.schedule)

        return schedule


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

        assert not math.isclose(start, stop), 'start and stop must not be equal'

        self.start = start
        self.stop = stop

        super().__init__(steps, lr)

    def generate_schedule(self) -> list:
        """Generate linear learning rate schedule (see __init__ docstring)
        """

        schedule = list()
        for step in range(self.steps):
            x = self.start + (step / self.steps) * (self.stop - self.start)
            if self.start > self.stop:
                schedule.append(self.lr_base * x)
            else:
                schedule.append(self.lr_base * (1.0 - x))

        return schedule


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

        assert not math.isclose(start, stop), 'start and stop must not be equal'

        self.start = start
        self.stop = stop

        super().__init__(steps, lr)

    def generate_schedule(self) -> list:
        """Generate linear learning rate schedule (see __init__ docstring)
        """

        schedule = list()
        for step in range(self.steps):
            x = (math.pi / 2) * (
                self.start + (step / self.steps) * (self.stop - self.start)
            )
            schedule.append(self.lr_base * math.sin(x))

        return schedule
