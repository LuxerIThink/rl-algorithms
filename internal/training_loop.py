from itertools import count
import gymnasium as gym
import random
from collections import namedtuple
import numpy as np
import torch
from torch import optim as optim, nn as nn
from internal.replay_memory import ReplayMemory
from internal.strategy import ExplorationStrategy


class TrainingLoop:
    def __init__(self, policy_net, target_net, env, device):
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )
        self.device = device
        self.env = env
        self.n_actions, self.continuous_actions = self._determine_action_space()
        self.n_observations = self.env.observation_space.shape[0]
        self.policy_net = policy_net(self.n_observations, self.n_actions).to(device)
        self.target_net = target_net(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.num_episodes = 600 if torch.cuda.is_available() else 50
        self.episode_durations = []
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 1e-4
        # Others
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()
        self.exploration_strategy = ExplorationStrategy()
        self.memory = ReplayMemory(10000, self.transition)

    def _determine_action_space(self):
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            n_actions = self.env.action_space.n
            continuous_actions = False
        elif isinstance(action_space, gym.spaces.Box):
            n_actions = self.env.action_space.shape[0]
            continuous_actions = True
        else:
            raise ValueError("Unsupported action space type")
        return n_actions, continuous_actions

    def run(self):
        for i_episode in range(self.num_episodes):
            self.run_episode()

    def run_episode(self):
        state, info = self.env.reset()
        state = self.tensorize_state(state)

        for t in count():
            action = self.select_action(state)
            if self.continuous_actions:
                observation, reward, terminated, truncated, _ = self.env.step(action)
            else:
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
            reward_tensor = self.create_reward_tensor(reward)

            next_state = None if terminated else self.tensorize_state(observation)

            self.memory.push(state, action, next_state, reward_tensor)
            state = next_state

            self.optimize_model()
            self.update_target_net_weights()

            if terminated or truncated:
                self.record_episode_duration(t)
                break

    def tensorize_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def create_reward_tensor(self, reward):
        return torch.tensor([reward], device=self.device)

    def update_target_net_weights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def record_episode_duration(self, duration):
        self.episode_durations.append(duration + 1)

    def select_action(self, state):
        sample = random.random()
        if sample > self.exploration_strategy.get_epsilon():
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        with torch.no_grad():
            if self.continuous_actions:
                return self.policy_net(state).squeeze().cpu().numpy()
            else:
                return self.policy_net(state).max(1)[1].view(1, 1)

    def random_action(self):
        if self.continuous_actions:
            return self.env.action_space.sample()
        else:
            return torch.tensor(
                data=[[self.env.action_space.sample()]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))
        non_final_mask, non_final_next_states, state_batch = self.get_non_final_states(
            batch
        )

        action_batch, reward_batch, state_action_values = self.get_action_values(
            batch, state_batch
        )

        next_state_values = self.calculate_next_state_values(
            non_final_mask, non_final_next_states
        )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.calculate_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def get_non_final_states(self, batch):
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat([s for s in batch.state if s is not None])
        return non_final_mask, non_final_next_states, state_batch

    def get_action_values(self, batch, state_batch):
        if self.continuous_actions:
            action_batch = torch.tensor(
                np.array(batch.action), device=self.device, dtype=torch.float32
            )

            policy_output = self.policy_net(state_batch)
            state_action_values = torch.sum(
                policy_output * action_batch, dim=1, keepdim=True
            )
        else:
            action_batch = torch.cat(batch.action)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        reward_batch = torch.cat(batch.reward)
        return action_batch, reward_batch, state_action_values

    def calculate_next_state_values(self, non_final_mask, non_final_next_states):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        return next_state_values

    def calculate_loss(self, state_action_values, expected_state_action_values):
        return self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
