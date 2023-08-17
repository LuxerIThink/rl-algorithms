from torch import nn as nn
from nnets.abstract import NNet


class PPO(nn.Module, NNet):
    def __init__(self, n_observations, n_actions):
        super(PPO, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.policy(x)
