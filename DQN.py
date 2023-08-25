import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.epsilon = 0.2

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            return q_values.argmax().item()

    def train(self, replay_memory):
        if len(replay_memory) < batch_size:
            return

        batch = random.sample(replay_memory, batch_size)
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
        target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

        self.target_network.load_state_dict(self.q_network.state_dict())


# Hyperparameters
gamma = 0.99
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.05
num_episodes = 1500
batch_size = 64
num_steps = 500
target_reward = 500
target_episodes = 10

# Create CartPole environment
consecutive_success = 0
env = gym.make("Acrobot-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# Initialize DQN agent
agent = DQNAgent()

# Training loop
replay_memory = []
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    terminated = False

    for step in range(num_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
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

plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

checkpoint_path = "CartPole-v1_DQN.pth"
torch.save(agent.q_network.state_dict(), checkpoint_path)
print(f"Model checkpoint saved: {checkpoint_path}")

env.close()
