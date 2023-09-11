import gymnasium as gym
import torch

from nnet import PPO, A2C, DQN, DDPG


class EnvironmentRunner:
    def __init__(self, human_mode, drl, env_name, checkpoint, episodes, steps=None):
        if human_mode:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)
        self.policy = self.set_policy(drl, checkpoint)
        self.episodes = episodes
        self.steps = steps or 999999999

    def set_policy(self, drl, checkpoint_path):
        env = self.env
        checkpoint = torch.load(checkpoint_path)
        state_size = env.observation_space.shape[0]
        hidden_size_1 = list(checkpoint.values())[0].shape[0]
        hidden_size_2 = list(checkpoint.values())[2].shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_space = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            action_space = env.action_space.shape[0]
        else:
            raise Exception("Incorrect environment")

        policy = drl.choose_policy(
            env, state_size, action_space, hidden_size_1, hidden_size_2
        )
        policy.load_state_dict(torch.load(checkpoint_path))
        policy.eval()
        return policy

    def run_episodes(self):
        rewards = []
        for _ in range(self.episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for _ in range(self.steps):
                action = self.policy.get_action(state)[0]
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)
        self.env.close()
        return rewards


def print_mean(num_test_episodes, total_rewards):
    average_reward = sum(total_rewards) / num_test_episodes
    print(f"Average test reward: {average_reward}")


def main():
    tests = [
        # [DQN, "CartPole-v1", "models/CartPole-v1_DQN.pth", 10, 500]
        # [DQN, "Acrobot-v1", "models/Acrobot-v1_DQN.pth", 10, 500]
        # [PPO, "CartPole-v1", "models/CartPole-v1_PPO.pth", 10, 500]
        # [PPO, "Acrobot-v1", "models/Acrobot-v1_PPO.pth", 10, 500]
        # [PPO, "Hopper-v4", "models/Hopper-v4_PPO.pth", 10, 500]
        # [PPO, "Swimmer-v4", "models/Hopper-v4_PPO.pth", 10, 500]
        # [A2C, "CartPole-v1", "models/CartPole-v1_A2C.pth", 10, 500]
        # [A2C, "Acrobot-v1", "models/Acrobot-v1_A2C.pth", 10, 500]
        # [A2C, "Hopper-v4", "models/Hopper-v4_A2C.pth", 10, 500]
        # [A2C, "Swimmer-v4", "models/Swimmer-v4_A2C.pth", 10, 500]
        # [DDPG, "Hopper-v4", "models/Hopper-v4_DDPG.pth", 10, 500]
        # [DDPG, "Swimmer-v4", "models/Swimmer-v4_DDPG.pth", 10, 500]
    ]
    human_mode = False
    for test in tests:
        runner = EnvironmentRunner(human_mode, *test)
        total_rewards = runner.run_episodes()
        print_mean(test[3], total_rewards)


if __name__ == "__main__":
    main()
