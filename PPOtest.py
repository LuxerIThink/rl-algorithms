import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


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


def get_action(state):
    state_tensor = torch.tensor(state, dtype=torch.float)
    action_probs = torch.softmax(policy(state_tensor), dim=-1)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    return action.item()


env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


policy = QNetwork()
checkpoint_path = "Hopper-v4_A2C.pth"
policy.load_state_dict(torch.load(checkpoint_path))
policy.eval()


num_test_episodes = 10
total_rewards = []

for _ in range(num_test_episodes):
    state, _ = env.reset()
    episode_reward = 0
    terminated = False
    truncated = False

    while True:
        action = get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(episode_reward)


env.close()

average_reward = sum(total_rewards) / num_test_episodes
print(f"Average test reward: {average_reward}")
