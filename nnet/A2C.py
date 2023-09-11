import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
        )


class ContinousActorCritic(ActorCritic):
    def forward(self, x):
        mean = torch.tanh(self.actor(x))
        log_std = torch.zeros_like(mean)
        return mean, log_std

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mean, log_std = self(state_tensor)
        normal = Normal(mean, log_std.exp())
        action = normal.sample()
        return action.detach().numpy(), normal.log_prob(action).sum()


class DiscreteActorCritic(ActorCritic):
    def forward(self, x):
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_probs[action.item()]


def choose_policy(
    env, state_size, action_space, hidden_size_1=None, hidden_size_2=None
):
    if isinstance(env.action_space, gym.spaces.Discrete):
        policy = DiscreteActorCritic(
            state_size, action_space, hidden_size_1, hidden_size_2
        )
    elif isinstance(env.action_space, gym.spaces.Box):
        policy = ContinousActorCritic(
            state_size, action_space, hidden_size_1, hidden_size_2
        )
    else:
        raise NotImplementedError("Unsupported action space")
    return policy
