import gymnasium as gym
import torch
from nnets.dqn import DQN
from nnets.ppo import PPO
from nnets.ddpg import DDPGActor, DDPGCritic
from nnets.a2c import A2CActor, A2CCritic


def check_model(model, num_episodes=10):
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        while True:
            action = model(state).max(1)[1].view(1, 1)
            if continuous_actions:
                observation, reward, terminated, truncated, _ = env.step(action)
            else:
                observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
            else:
                break

            state = next_state

        print(f"Episode reward: {total_reward}")


@staticmethod
def gen_filename(env, nnet):
    return f"{env}_{nnet}.pth"


@staticmethod
def choose_nnet(env_name):
    match env_name:
        case "DQN":
            actor = DQN
        case "PPO":
            actor = PPO
        case "A2C":
            actor = A2CActor
        case "DDPG":
            actor = DDPGActor
        case _:
            raise ModuleNotFoundError("QNetwork doesn't exist")
    return actor


model_path = "CartPole-v1_DQN.pth"
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()
env.render()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

continuous_actions = []
action_space = env.action_space
if isinstance(action_space, gym.spaces.Discrete):
    n_actions = env.action_space.n
    continuous_actions = False
elif isinstance(action_space, gym.spaces.Box):
    n_actions = env.action_space.shape[0]
    continuous_actions = True
else:
    raise ValueError("Unsupported action space type")

state, info = env.reset()
n_observations = len(state)


loaded_model = DQN(n_observations, n_actions).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

check_model(loaded_model)
