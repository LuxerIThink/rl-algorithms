import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def check_model(model, num_episodes=10):
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        while True:
            action = model(state).max(1)[1].view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            if not done:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                break

            state = next_state
        print(f"Episode reward: {total_reward}")


model_path = "model_dqn_cartpol.pth"
env = gym.make('CartPole-v1',render_mode='human')
state = env.reset()
env.render()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)
loaded_model = DQN(n_observations, n_actions).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
check_model(loaded_model)
