from training import A2C, DDPG, DQN, PPO
import gymnasium as gym
import torch
import matplotlib.pyplot as plt


def train_environment(environment):
    policy, env_name, parameters = environment

    env = gym.make(env_name)

    all_parameters = prepare_params(env, parameters)

    episode_rewards, state_dict = policy.train(env, all_parameters)

    filename = create_filename(env_name, parameters, policy)

    create_chart(episode_rewards, filename)

    save(filename, state_dict)

    env.close()


def save(filename, state_dict):
    file_path = f"models/{filename}"
    torch.save(state_dict, file_path)
    print(f"Model checkpoint saved: {file_path}")


def create_chart(episode_rewards, filename):
    plt.plot(episode_rewards)
    plt.title(f"{filename}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def create_filename(env_name, parameters, policy):
    parameters_string = "_".join(str(value) for value in parameters)
    filename = f"{env_name}_{policy.__name__.split('.')[-1]}_{parameters_string}.pth"
    return filename


def prepare_params(env, parameters):
    all_parameters = parameters.copy()
    state_dim = env.observation_space.shape[0]

    action_dim = select_action_space(env)

    all_parameters.insert(0, action_dim)
    all_parameters.insert(0, state_dim)

    return all_parameters


def select_action_space(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space")
    return action_dim


environments = [
    # [DQN, "CartPole-v1", [500, 2000, 0.0025, 0.99, 0.2, 0.01, 0.995, 64, 64, 500, 20]],
    # [DQN, "Acrobot-v1", [500, 2000, 0.00063, 0.99, 0.2, 0.01, 0.995, 32, 64, 80, 20]],
    # [PPO, "CartPole-v1", [500, 1000, 0.02, 0.995, 0.95, 0.2, 20, 64, 500, 10]],
    # [PPO, "Acrobot-v1", [500, 1000, 0.0006, 0.99, 0.9, 0.01, 2, 128, 80, 10]],
    # [PPO, "Hopper-v4", [1000, 1000, 0.0003, 0.99, 0.95, 0.2, 2, 128, 9999, 10]],
    # [PPO, "Swimmer-v4", [1000, 2000, 0.0001, 0.999, 0.9, 0.2, 2, 256, 9999, 10]],
    # [A2C, "CartPole-v1", [500, 1000, 0.001, 0.99, 128, 128, 500, 10]],
    # [A2C, "Acrobot-v1", [500, 1500, 0.01, 0.999, 128, 128, 80, 10]],
    # [A2C, "Hopper-v4", [2000, 1000, 2e-4, 0.999, 256, 256, 1800, 10]],
    # [A2C, "Swimmer-v4", [1000, 2000, 1e-3, 0.999, 256, 256, 1800, 10]],
    # [DDPG, "Hopper-v4", [2000, 2000, 1e-5, 1e-3, 0.999, 2e-4, 256, 256, 128]],
    # [DDPG, "Swimmer-v4", [1000, 2000, 1e-5, 1e-3, 0.999, 1e-3, 128, 256, 128]],
]


def train_environments():
    for environment in environments:
        train_environment(environment)


train_environments()
