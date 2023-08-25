import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, is_continuous=False):
        super(ActorCritic, self).__init__()
        self.is_continuous = is_continuous

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() if is_continuous else nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        action_output = self.actor(x)
        value = self.critic(x)
        return action_output, value


def select_action(action_output, is_continuous):
    if is_continuous:
        action = torch.tanh(action_output)
    else:
        action_probs = torch.softmax(action_output, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
    return action


def train_one_episode(env, model, optimizer, is_continuous):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_output, value = model(state_tensor)

        action = select_action(action_output, is_continuous)
        log_prob = None

        if is_continuous:
            action_dist = torch.distributions.Normal(
                action_output, torch.ones_like(action_output)
            )
            log_prob = action_dist.log_prob(action).sum(dim=-1)
        else:
            action_probs = torch.softmax(action_output, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(action)

        next_state, reward, done, _, _ = env.step(action.detach().numpy())

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = model(next_state_tensor)

        delta = reward + gamma * next_value - value
        actor_loss = -log_prob * delta
        critic_loss = delta.pow(2)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        episode_reward += reward

    return episode_reward


def train_environment(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box)
        else env.action_space.n
    )
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    model = ActorCritic(state_dim, action_dim, hidden_size, is_continuous)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        episode_reward = train_one_episode(env, model, optimizer, is_continuous)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    plt.plot(episode_rewards)
    plt.title(f"{env_name} Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


# Hyperparameters
lr = 0.001
gamma = 0.99
hidden_size = 128
num_episodes = 1000

# Train environments
# train_environment("CartPole-v1")
train_environment("Hopper-v4")
