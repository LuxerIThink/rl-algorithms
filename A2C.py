import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.functional import smooth_l1_loss
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        action_mean = self.actor(x)
        action_std = torch.exp(
            torch.zeros_like(action_mean)
        )  # Initialize standard deviation
        action_dist = Normal(action_mean, action_std)

        value = self.critic(x)
        return action_dist, value


lr = 0.001
gamma = 0.995
hidden_size = 256
num_episodes = 1000

env = gym.make("Hopper-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

model = ActorCritic(state_dim, action_dim, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
episode_rewards = []

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_dist, value = model(state_tensor)

        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        next_state, reward, done, _, _ = env.step(action.detach().numpy())

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = model(next_state_tensor)

        delta = reward + gamma * next_value - value
        actor_loss = -log_prob * delta
        critic_loss = smooth_l1_loss(value, reward + gamma * next_value)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        episode_reward += reward

    episode_rewards.append(episode_reward)

    print(f"Episode {episode+1}, Reward: {episode_reward}")

plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()


checkpoint_path = "Hopper-v4_A2C.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model checkpoint saved: {checkpoint_path}")


env.close()
