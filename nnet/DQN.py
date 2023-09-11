import torch
from torch import nn as nn
from torch.distributions import Categorical
import gymnasium as gym


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class DiscretePolicy(QNetwork):
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        action_probs = torch.softmax(self(state_tensor), dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_probs[action]


def choose_policy(
    env, state_size, action_space, hidden_size_1=None, hidden_size_2=None
):
    if isinstance(env.action_space, gym.spaces.Discrete):
        policy = DiscretePolicy(state_size, action_space, hidden_size_1, hidden_size_2)
    elif isinstance(env.action_space, gym.spaces.Box):
        raise NotImplementedError("Continuous action space not implemented yet")
    else:
        raise NotImplementedError("Unsupported action space")
    return policy
