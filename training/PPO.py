import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from nnet import PPO


def compute_gae(next_value, rewards, masks, values, gamma, lamda):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lamda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def PPO_update(
    policy,
    optimizer,
    old_probs,
    states,
    actions,
    returns,
    advantages,
    epsilon_clip,
    epochs,
    action_space,
    action_size,
):
    for _ in range(epochs):
        for state, old_prob, action, return_, advantage in zip(
            states, old_probs, actions, returns, advantages
        ):
            state_tensor = torch.tensor(state, dtype=torch.float)
            old_prob_tensor = old_prob.detach().clone().requires_grad_(True)
            advantage_tensor = torch.tensor(advantage, dtype=torch.float)

            action_probs = torch.softmax(policy(state_tensor), dim=-1)

            match type(action_space):
                case gym.spaces.Discrete:
                    new_prob = action_probs[action]
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


def train(env, parameters):
    (
        state_dim,
        action_dim,
        steps,
        episodes,
        learning_rate,
        gamma,
        lambda_value,
        epsilon_clip,
        epochs,
        hidden_size,
        target_reward,
        target_episodes,
    ) = parameters

    policy = PPO.choose_policy(
        env,
        state_dim,
        action_dim,
        hidden_size,
        hidden_size,
    )

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_rewards = []
    consecutive_success = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        masks = []
        values = []
        old_probs = []

        for step in range(steps):
            action, action_prob = policy.get_action(state)
            next_state, reward, terminated, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(1 - terminated)
            values.append(
                policy(torch.tensor(state, dtype=torch.float)).detach().numpy()
            )
            old_probs.append(action_prob)

            episode_reward += reward
            state = next_state

            if terminated:
                break

        next_value = (
            policy(torch.tensor(next_state, dtype=torch.float)).detach().numpy()
        )
        returns = compute_gae(next_value, rewards, masks, values, gamma, lambda_value)

        advantages = np.array(returns) - np.array(values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        PPO_update(
            policy,
            optimizer,
            old_probs,
            states,
            actions,
            returns,
            advantages,
            epsilon_clip,
            epochs,
            env.action_space,
            action_dim,
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

    return episode_rewards, policy.state_dict()
