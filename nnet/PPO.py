import torch
from torch import nn as nn
from torch.distributions import Categorical, MultivariateNormal
import gymnasium as gym


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, _):
        super(Policy, self).__init__()
        self.action_size = action_size
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, action_size),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.layers(x)


class DiscretePolicy(Policy):
    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float()
        action_probs = torch.softmax(self(state_tensor), dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_probs[action]


class ContinuousPolicy(Policy):
    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float()
        action_mean = self(state_tensor)
        cov_matrix = torch.diag(torch.ones(self.action_size))
        action_dist = MultivariateNormal(action_mean, cov_matrix)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action.numpy(), action_log_prob


def choose_policy(
    env, state_size, action_space, hidden_size_1=None, hidden_size_2=None
):
    if isinstance(env.action_space, gym.spaces.Discrete):
        policy = DiscretePolicy(state_size, action_space, hidden_size_1, hidden_size_2)
    elif isinstance(env.action_space, gym.spaces.Box):
        policy = ContinuousPolicy(
            state_size, action_space, hidden_size_1, hidden_size_2
        )
    else:
        raise Exception("Incorrect environment")
    return policy
