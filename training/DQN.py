import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from nnet.DQN import choose_policy


class DQNAgent:
    def __init__(
        self,
        env,
        state_size,
        action_space,
        hidden_size,
        gamma,
        epsilon,
        learning_rate,
        batch_size,
        epsilon_min,
        epsilon_decay,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = choose_policy(
            env, state_size, action_space, hidden_size, hidden_size
        )
        self.target_network = choose_policy(
            env, state_size, action_space, hidden_size, hidden_size
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            return q_values.argmax().item()

    def train(self, replay_memory):
        if len(replay_memory) < self.batch_size:
            return

        batch = random.sample(replay_memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.target_network.load_state_dict(self.q_network.state_dict())


def train(env, parameters):
    (
        state_size,
        action_space,
        steps,
        episodes,
        learning_rate,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size,
        hidden_size,
        target_reward,
        target_episodes,
    ) = parameters

    # Initialize DQN agent
    agent = DQNAgent(
        env,
        state_size,
        action_space,
        hidden_size,
        gamma,
        epsilon,
        learning_rate,
        batch_size,
        epsilon_min,
        epsilon_decay,
    )

    # Training loop
    replay_memory = []
    episode_rewards = []
    consecutive_success = 0
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(steps):
            action = agent.select_action(state)
            next_state, reward, terminated, _, _ = env.step(action)
            replay_memory.append((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            if terminated:
                break

        agent.train(replay_memory)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        if total_reward >= target_reward:
            consecutive_success += 1
            if consecutive_success >= target_episodes:
                print(
                    f"Target reward achieved for {target_episodes} consecutive episodes. Stopping training."
                )
                break
        else:
            consecutive_success = 0

    return episode_rewards, agent.q_network.state_dict()
