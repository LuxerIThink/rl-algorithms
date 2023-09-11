import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal
from nnet.A2C import choose_policy


class A2CAgent:
    def __init__(
        self, env, state_size, action_size, hidden_size_1, hidden_size_2, lr, gamma
    ):
        self.env = env
        self.model = choose_policy(
            self.env,
            state_size,
            action_size,
            hidden_size_1,
            hidden_size_2,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def train_one_episode(self, steps):
        state, _ = self.env.reset()
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        values = []  # Store the values

        for _ in range(steps):
            action, log_prob = self.model.get_action(state)

            next_state, reward, done, _, _ = self.env.step(action)

            states.append(state)
            actions.append((action, log_prob))
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        returns = []
        advantage = 0

        for reward in reversed(rewards):
            advantage = reward + self.gamma * advantage
            returns.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32)
        actor_loss = 0
        critic_loss = 0

        for t in range(len(states)):
            if isinstance(self.env.action_space, gym.spaces.Box):
                mean, log_std = self.model(torch.tensor(states[t], dtype=torch.float32))
                action_dist = Normal(mean, log_std.exp())
                log_prob = action_dist.log_prob(
                    torch.tensor(actions[t][0], dtype=torch.float32)
                ).sum()
                value = self.model.critic(torch.tensor(states[t], dtype=torch.float32))
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                action_probs, value = self.model(
                    torch.tensor(states[t], dtype=torch.float32)
                )
                action_dist = Categorical(action_probs)
                log_prob = action_dist.log_prob(
                    torch.tensor(actions[t][0], dtype=torch.int64)
                )
            else:
                raise NotImplemented

            delta = returns[t] - value
            actor_loss += -(log_prob * delta)  # Accumulate actor loss
            critic_loss += delta.pow(2)  # Accumulate critic loss

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return episode_reward


def train(env, parameters):
    (
        state_dim,
        action_dim,
        steps,
        episodes,
        lr,
        gamma,
        hidden_size_1,
        hidden_size_2,
        target_reward,
        target_episodes,
    ) = parameters

    agent = A2CAgent(
        env, state_dim, action_dim, hidden_size_1, hidden_size_2, lr, gamma
    )

    episode_rewards = []
    consecutive_success = 0

    for episode in range(episodes):
        episode_reward = agent.train_one_episode(steps)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

        if episode_reward >= target_reward:
            consecutive_success += 1
            if consecutive_success >= 10:  # Adjust this value as needed
                print(
                    f"Target reward achieved for 10 consecutive episodes. Stopping training."
                )
                break
        else:
            consecutive_success = 0

    return episode_rewards, agent.model.state_dict()
