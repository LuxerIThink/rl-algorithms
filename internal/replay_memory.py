import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity, transition):
        self.transition = transition
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
