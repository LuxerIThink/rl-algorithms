import math


class ExplorationStrategy:
    def __init__(self, start: float = None, end: float = None, decay: float = None):
        self.start = start or 0.9
        self.end = end or 0.05
        self.decay = decay or 1000
        self.steps_done = 0

    def get_epsilon(self):
        epsilon = self.end + (self.start - self.end) * math.exp(
            -1.0 * self.steps_done / self.decay
        )
        self.steps_done += 1
        return epsilon
