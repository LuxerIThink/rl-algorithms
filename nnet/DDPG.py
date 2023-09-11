import torch
from torch import nn as nn
import gymnasium as gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self(state).detach().numpy()
        return action, None


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
        )

    def forward(self, x, a):
        return self.layers(torch.cat([x, a], 1))


def choose_policy(
    env, state_size, action_space, hidden_size_1=None, hidden_size_2=None
):
    if isinstance(env.action_space, gym.spaces.Discrete):
        raise NotImplementedError("Discrete action space not implemented yet")
    elif isinstance(env.action_space, gym.spaces.Box):
        policy = Actor(state_size, action_space, hidden_size_1, hidden_size_2)
    else:
        raise NotImplementedError("Unsupported action space")
    return policy
