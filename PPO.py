import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.02
gamma = 0.98
epsilon_clip = 0.2
num_epochs = 20
num_steps = 500
num_episodes = 1000
target_reward = 500
target_episodes = 10
batch_size = 64

# Create CartPole environment
env = gym.make("Acrobot-v1")
state_size = env.observation_space.shape[0]
match type(env.action_space):
    case gym.spaces.Discrete:
        action_size = env.action_space.n
    case gym.spaces.Box:
        action_size = env.action_space.shape[0]
consecutive_success = 0


# Define policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, action_size),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.fc(x)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def get_action(state):
    match type(env.action_space):
        case gym.spaces.Discrete:
            state_tensor = torch.from_numpy(state).float()
            action_probs = torch.softmax(policy(state_tensor), dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item(), action_probs[action]
        case gym.spaces.Box:
            state_tensor = torch.from_numpy(state).float()  # Convert to float tensor
            action_mean = policy(state_tensor)
            cov_matrix = torch.diag(
                torch.ones(action_size)
            )  # Diagonal covariance matrix for simplicity
            action_dist = torch.distributions.MultivariateNormal(
                action_mean, cov_matrix
            )
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            return action.numpy(), action_log_prob


def compute_gae(next_value, rewards, masks, values, gamma=0.99, lamda=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lamda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def PPO_update(
    policy, optimizer, old_probs, states, actions, returns, advantages, epsilon_clip
):
    for _ in range(num_epochs):
        for state, old_prob, action, return_, advantage in zip(
            states, old_probs, actions, returns, advantages
        ):
            state_tensor = torch.tensor(state, dtype=torch.float)
            old_prob_tensor = old_prob.detach().clone().requires_grad_(True)
            return_tensor = torch.tensor(return_, dtype=torch.float)
            advantage_tensor = torch.tensor(advantage, dtype=torch.float)

            action_probs = torch.softmax(policy(state_tensor), dim=-1)

            match type(env.action_space):
                case gym.spaces.Discrete:
                    new_prob = action_probs[action]  # Use action as index directly
                    ratio = new_prob / old_prob_tensor
                case gym.spaces.Box:
                    action_mean = policy(state_tensor)
                    cov_matrix = torch.diag(torch.ones(action_size))
                    action_dist = MultivariateNormal(action_mean, cov_matrix)
                    new_prob = action_dist.log_prob(
                        torch.tensor(action, dtype=torch.float)
                    )
                    ratio = torch.exp(new_prob - old_prob_tensor)

            surrogate1 = ratio * advantage_tensor
            surrogate2 = (
                torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
                * advantage_tensor
            )
            surrogate_loss = -torch.min(surrogate1, surrogate2)

            clipped_value = old_prob_tensor * (1 + epsilon_clip)
            unclipped_value = old_prob_tensor * ratio
            value_loss = torch.max(surrogate1, surrogate2) - torch.min(
                clipped_value, unclipped_value
            )

            loss = surrogate_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


episode_rewards = []
# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    states = []
    actions = []
    rewards = []
    masks = []
    values = []
    old_probs = []

    for step in range(num_steps):
        action, action_prob = get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(
            action
        )  # Update: Unpack the terminated flag

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        masks.append(1 - terminated)  # Update: Use terminated to set masks
        values.append(policy(torch.tensor(state, dtype=torch.float)).detach().numpy())
        old_probs.append(action_prob)

        episode_reward += reward
        state = next_state

        if terminated:
            break

    next_value = policy(torch.tensor(next_state, dtype=torch.float)).detach().numpy()
    returns = compute_gae(next_value, rewards, masks, values)

    advantages = np.array(returns) - np.array(values)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    PPO_update(
        policy, optimizer, old_probs, states, actions, returns, advantages, epsilon_clip
    )

    episode_rewards.append(episode_reward)

    print(f"Episode {episode}, Reward: {episode_reward}")

    if episode_reward >= target_reward:
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


checkpoint_path = "Acrobot-v1_PPO.pth"
torch.save(policy.state_dict(), checkpoint_path)
print(f"Model checkpoint saved: {checkpoint_path}")

env.close()
