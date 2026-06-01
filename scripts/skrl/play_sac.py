import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Kinova Gen3 grasp with SKRL SAC.")
parser.add_argument("--task", type=str, default="Gen3-Grasp-SAC-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent_<step>.pt checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler

import gen3.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from gen3.tasks.manager_based.gen3_grasp.agents.sac_cfg import Gen3GraspSACCfg

cfg = Gen3GraspSACCfg()


class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
        )

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layers = cfg.net_arch

        self.net = nn.Sequential(
            nn.Linear(obs_dim, layers[0]),
            nn.ELU(),
            nn.Linear(layers[0], layers[1]),
            nn.ELU(),
            nn.Linear(layers[1], layers[2]),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(layers[-1], act_dim)
        self.log_std_layer = nn.Linear(layers[-1], act_dim)

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return self.mean_layer(x), self.log_std_layer(x), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layers = cfg.net_arch

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, layers[0]),
            nn.ELU(),
            nn.Linear(layers[0], layers[1]),
            nn.ELU(),
            nn.Linear(layers[1], layers[2]),
            nn.ELU(),
            nn.Linear(layers[-1], 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.observations.policy.enable_corruption = False

    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = wrap_env(isaac_env, wrapper="isaaclab")
    device = env.device

    models = {
        "policy": Actor(env.observation_space, env.action_space, device),
        "critic_1": Critic(env.observation_space, env.action_space, device),
        "critic_2": Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }

    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }

    agent = SAC(
        models=models,
        memory=None,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, _, terminated, truncated, _ = env.step(actions)
        if (terminated | truncated).any():
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
