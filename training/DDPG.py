import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from nnet.DDPG import Actor, Critic


class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        batch_size,
        hidden_size_1,
        hidden_size_2,
    ):
        self.replay_buffer = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2)
        self.actor_target = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2)
        self.critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        target_actions = self.actor_target(next_state_batch)
        target_values = self.critic_target(next_state_batch, target_actions).squeeze()
        target = reward_batch + self.gamma * target_values * (1 - done_batch)

        critic_loss = nn.MSELoss()(
            self.critic(state_batch, action_batch).squeeze(), target.detach()
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


def train(env, parameters):
    (
        state_dim,
        action_dim,
        steps,
        episodes,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        batch_size,
        hidden_size_1,
        hidden_size_2,
    ) = parameters

    agent = DDPGAgent(
        state_dim,
        action_dim,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        batch_size,
        hidden_size_1,
        hidden_size_2,
    )

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(steps):
            action, _ = agent.actor.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            episode_reward += reward

            state = next_state

            agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

    return episode_rewards, agent.actor.state_dict()
