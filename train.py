import gymnasium as gym
import torch

from internal.training_loop import TrainingLoop
from nnets.dqn import DQN
from nnets.ppo import PPO
from nnets.ddpg import DDPGActor, DDPGCritic
from nnets.a2c import A2CActor, A2CCritic
from utils.plot import plot_durations


class Environment:
    def __init__(self, env_name, nnet_name, render_mode="human", device="auto"):
        self.filename = self.gen_filename(env_name, nnet_name)
        self.env = gym.make(env_name, render_mode=render_mode)
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor, self.critic = self.choose_nnet(nnet_name)
        self.train = TrainingLoop(self.actor, self.critic, self.env, self.device)

    @staticmethod
    def gen_filename(env, nnet):
        return f"{env}_{nnet}.pth"

    @staticmethod
    def choose_nnet(env_name):
        match env_name:
            case "DQN":
                actor = critic = DQN
            case "PPO":
                actor = critic = PPO
            case "A2C":
                actor = A2CActor
                critic = A2CCritic
            case "DDPG":
                actor = DDPGActor
                critic = DDPGCritic
            case _:
                raise ModuleNotFoundError("QNetwork doesn't exist")
        return actor, critic

    def run_training(self):
        self.train.run()
        print("Training complete")
        torch.save(self.train.policy_net.state_dict(), self.filename)
        plot_durations(self.train.episode_durations, True)


# Usage
if __name__ == "__main__":
    trainer = Environment("CartPole-v1", "PPO")
    trainer.run_training()
