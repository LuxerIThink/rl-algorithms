import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# Define the Actor and Critic neural networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.relu(self.fc1(torch.cat([x, a], 1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DDPG agent class
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = []
        self.batch_size = 256
        self.gamma = 0.98
        self.tau = 1e-3

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).detach().numpy()
        return action

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


# Initialize the environment and agent
env = gym.make("Hopper-v4")
episode_rewards = []
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = DDPGAgent(state_dim, action_dim)

# Training loop
max_episodes = 1000
max_steps = 1000

for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        episode_reward += reward

        state = next_state

        agent.update()

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")


plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

checkpoint_path = "Hopper-v4_A2C.pth"
torch.save(agent.actor.state_dict(), checkpoint_path)
print(f"Model checkpoint saved: {checkpoint_path}")

env.close()
