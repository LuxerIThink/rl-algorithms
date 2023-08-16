import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from abc import ABC, abstractmethod


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ExplorationStrategy:
    def __init__(self, start: float = None, end: float = None, decay: float = None):
        self.start = start or 0.9
        self.end = end or 0.05
        self.decay = decay or 1000
        self.steps_done = 0

    def get_epsilon(self):
        epsilon = self.end + (self.start - self.end) * math.exp(
            -1.0 * self.steps_done / self.decay
        )
        self.steps_done += 1
        return epsilon


class Algorithm(ABC):
    @abstractmethod
    def forward(self, x):
        pass


class PPO(nn.Module, Algorithm):
    def __init__(self, n_observations, n_actions):
        super(PPO, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.policy(x)


class DQN(nn.Module, Algorithm):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class DDPGActor(nn.Module, Algorithm):
    def __init__(self, n_observations, n_actions):
        super(DDPGActor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.output_activation(self.layer3(x))


class DDPGCritic(nn.Module, Algorithm):
    def __init__(self, n_observations, n_actions):
        super(DDPGCritic, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class A2CActor(nn.Module, Algorithm):
    def __init__(self, n_observations, n_actions):
        super(A2CActor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.output_activation(self.layer3(x))


class A2CCritic(nn.Module, Algorithm):
    def __init__(self, n_observations):
        super(A2CCritic, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    def __init__(
        self,
        policy_net,
        target_net,
        batch_size: int = None,
        gamma: int = None,
        tau: int = None,
    ):
        # Hyperparameters
        self.batch_size = batch_size or 128
        self.gamma = gamma or 0.99
        self.tau = tau or 0.005
        self.lr = 1e-4
        self.policy_net = policy_net
        self.target_net = target_net
        # Others
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()
        self.exploration_strategy = ExplorationStrategy()

    def select_action(self, state):
        sample = random.random()
        if sample > self.exploration_strategy.get_epsilon():
            with torch.no_grad():
                if continuous_actions:
                    return self.policy_net(state).squeeze().cpu().numpy()
                else:
                    return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            if continuous_actions:
                return action_space.sample()
            else:
                return torch.tensor(
                    data=[[env.action_space.sample()]], device=device, dtype=torch.long
                )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        if continuous_actions:
            state_batch = torch.cat([s.to(device) for s in batch.state])
            action_batch = torch.tensor(
                batch.action, device=device, dtype=torch.float32
            )
            reward_batch = torch.cat([r.to(device) for r in batch.reward])
            policy_output = self.policy_net(state_batch)
            state_action_values = torch.sum(
                policy_output * action_batch, dim=1, keepdim=True
            )
        else:
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


class TrainingLoop:
    def __init__(self, policy_net, target_net):
        self.policy_net = policy_net(n_observations, n_actions).to(device)
        self.target_net = policy_net(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.agent = Agent(self.policy_net, self.target_net)
        self.num_episodes = 600 if torch.cuda.is_available() else 50
        self.episode_durations = []

    def run(self):
        for i_episode in range(self.num_episodes):
            self.run_episode()

    def run_episode(self):
        state, info = env.reset()
        state = self.tensorize_state(state)

        for t in count():
            action = self.agent.select_action(state)
            if continuous_actions:
                observation, reward, terminated, truncated, _ = env.step(action)
            else:
                observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_tensor = self.create_reward_tensor(reward)

            if terminated:
                next_state = None
            else:
                next_state = self.tensorize_state(observation)

            self.agent.memory.push(state, action, next_state, reward_tensor)
            state = next_state

            self.agent.optimize_model()
            self.update_target_net_weights()

            if terminated or truncated:
                self.record_episode_duration(t)
                break

    def tensorize_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def create_reward_tensor(self, reward):
        return torch.tensor([reward], device=device)

    def update_target_net_weights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.agent.tau + target_net_state_dict[key] * (1 - self.agent.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def record_episode_duration(self, duration):
        self.episode_durations.append(duration + 1)


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    plt.show()


# Environment settings
env = gym.make("CartPole-v1", render_mode="human")  # Change environment name as needed

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Determine action and observation space
action_space = env.action_space
action_space_type = type(action_space)

if isinstance(action_space, gym.spaces.Discrete):
    n_actions = action_space.n
    continuous_actions = False
elif isinstance(action_space, gym.spaces.Box):
    n_actions = action_space.shape[0]
    continuous_actions = True
else:
    exit(0)

n_observations = env.observation_space.shape[0]

train = TrainingLoop(A2CActor, A2CCritic)
train.run()
print("Complete")
model_path = "model_ppo_swimmer.pth"
torch.save(train.agent.policy_net.state_dict(), model_path)
plot_durations(train.episode_durations, True)
