import torch.nn as nn
from nnets.abstract import NNet


class DDPGActor(nn.Module, NNet):
    def __init__(
        self, n_observations, n_actions, hidden_size=(128, 128), activation=nn.ReLU()
    ):
        super(DDPGActor, self).__init__()
        self.states = nn.Sequential(
            nn.Linear(n_observations, hidden_size[0]),
            activation,
            nn.Linear(hidden_size[0], hidden_size[1]),
            activation,
            nn.Linear(hidden_size[1], n_actions),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.states(x)


class DDPGCritic(nn.Module, NNet):
    def __init__(
        self, n_observations, n_actions, hidden_size=(128, 128), activation=nn.ReLU()
    ):
        super(DDPGCritic, self).__init__()
        self.states = nn.Sequential(
            nn.Linear(n_observations, hidden_size[0]),
            activation,
            nn.Linear(hidden_size[0], hidden_size[1]),
            activation,
            nn.Linear(hidden_size[1], n_actions),
        )

    def forward(self, x):
        return self.states(x)
