from torch import nn as nn
from torch.nn import functional as f
from nnets.abstract import NNet


class A2CActor(nn.Module, NNet):
    def __init__(self, n_observations, n_actions):
        super(A2CActor, self).__init__()
        self.states = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.states(x)


class A2CCritic(nn.Module, NNet):
    def __init__(self, n_observations, n_actions):
        super(A2CCritic, self).__init__()
        self.states = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = f.relu(self.states(x))
        return x
