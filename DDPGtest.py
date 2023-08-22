import gymnasium as gym
import torch
import torch.nn as nn


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


class TestAgent:
    def __init__(self, actor):
        self.actor = actor

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).detach().numpy()
        return action


def test_ddpg(agent, env, num_episodes=10, max_steps=1000):
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            state = next_state

            if done:
                break

        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")


env = gym.make("Hopper-v4", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actor = Actor(state_dim, action_dim)
actor.load_state_dict(torch.load("Hopper-v4_DDPG.pth"))
actor.eval()


test_agent = TestAgent(actor)
test_ddpg(test_agent, env, num_episodes=5)
env.close()
