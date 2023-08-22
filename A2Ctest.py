import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal


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
        action_std = torch.exp(torch.zeros_like(action_mean))
        action_dist = Normal(action_mean, action_std)

        value = self.critic(x)
        return action_dist, value


checkpoint_path = "Hopper-v4_A2C.pth"
env = gym.make("Hopper-v4", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_size = 256
model = ActorCritic(state_dim, action_dim, hidden_size)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()


num_test_episodes = 10
test_rewards = []

for episode in range(num_test_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_dist, _ = model(state_tensor)
        action = action_dist.mean

        next_state, reward, done, _, _ = env.step(action.detach().numpy())
        state = next_state
        episode_reward += reward

    test_rewards.append(episode_reward)
    print(f"Test Episode {episode+1}, Reward: {episode_reward}")


avg_test_reward = sum(test_rewards) / num_test_episodes
print(f"Average Test Reward: {avg_test_reward}")

env.close()
