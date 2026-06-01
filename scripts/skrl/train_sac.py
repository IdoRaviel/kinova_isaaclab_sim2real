import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Kinova Gen3 grasp with SKRL SAC.")
parser.add_argument("--task", type=str, default="Gen3-Grasp-SAC-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import gen3.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from gen3.tasks.manager_based.gen3_grasp.agents.sac_cfg import Gen3GraspSACCfg

cfg = Gen3GraspSACCfg()
if args_cli.seed is not None:
    cfg.seed = args_cli.seed

set_seed(cfg.seed)


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
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path("logs/skrl") / cfg.log_name / run_name

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.seed = cfg.seed

    # IsaacLab curriculum counts env.step() calls, not total transitions.
    # With multi-env: total env.step() calls = total_timesteps // num_envs.
    _curr_steps = int((cfg.total_timesteps // args_cli.num_envs) * 0.7)
    env_cfg.curriculum.action_rate.params["num_steps"] = _curr_steps
    env_cfg.curriculum.joint_vel.params["num_steps"] = _curr_steps

    # SAC stores observations in a replay buffer; noisy obs destabilise Q-learning.
    env_cfg.observations.policy.enable_corruption = False

    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = wrap_env(isaac_env, wrapper="isaaclab")

    device = env.device

    memory = RandomMemory(
        memory_size=cfg.buffer_size // args_cli.num_envs,
        num_envs=args_cli.num_envs,
        device=device,
    )

    models = {
        "policy": Actor(env.observation_space, env.action_space, device),
        "critic_1": Critic(env.observation_space, env.action_space, device),
        "critic_2": Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }

    def init_model(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)
        # small output-head init so initial actions/Q-values stay near zero
        if hasattr(model, "mean_layer"):
            nn.init.normal_(model.mean_layer.weight, mean=0.0, std=0.01)
            nn.init.normal_(model.log_std_layer.weight, mean=0.0, std=0.01)
        else:
            last = model.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=0.01)

    for model in models.values():
        init_model(model)

    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    agent_cfg.update(
        {
            "gradient_steps": cfg.gradient_steps,
            "batch_size": cfg.batch_size,
            "discount_factor": cfg.gamma,
            "polyak": cfg.tau,
            "actor_learning_rate": cfg.learning_rate,
            "critic_learning_rate": cfg.learning_rate,
            "entropy_learning_rate": cfg.learning_rate,
            "learn_entropy": True,
            "target_entropy": cfg.target_entropy,
            "initial_entropy_value": 0.2,
            "random_timesteps": cfg.learning_starts // args_cli.num_envs,
            "learning_starts": cfg.learning_starts // args_cli.num_envs,
            "grad_norm_clip": 1.0,
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {
                "size": env.observation_space,
                "device": device,
            },
            "experiment": {
                "directory": str(log_root),
                "experiment_name": "",
                "write_interval": 8000 // args_cli.num_envs,
                "checkpoint_interval": cfg.checkpoint_freq // args_cli.num_envs,
                "store_separately": False,
            },
        }
    )

    agent = SAC(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.resume:
        agent.load(args_cli.resume)

    trainer = SequentialTrainer(
        cfg={
            "timesteps": cfg.total_timesteps // args_cli.num_envs,
            "headless": args_cli.headless,
        },
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
