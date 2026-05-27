import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play Kinova Gen3 SAC+HER policy.")
parser.add_argument("--task", type=str, default="Gen3-LiftAndPlace-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--reward_type", type=str, choices=["sparse", "dense"], default="sparse")
parser.add_argument("--success_threshold", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import gen3.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import combine_frame_transforms

from stable_baselines3 import SAC


class IsaacLabGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, isaac_env, reward_type="sparse", success_threshold=0.05):
        super().__init__()
        self.isaac_env = isaac_env
        self.unwrapped_env = isaac_env.unwrapped
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.device = self.unwrapped_env.device

        obs_raw = self._reset_isaac()
        obs_vec = self._policy_obs(obs_raw)

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=obs_vec.shape, dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def _reset_isaac(self):
        out = self.isaac_env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def _policy_obs(self, obs_raw):
        obs = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        return np.asarray(obs[0], dtype=np.float32)

    def _object_pos_w(self):
        obj = self.unwrapped_env.scene["object"]
        return obj.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)

    def _desired_goal_w(self):
        robot = self.unwrapped_env.scene["robot"]
        command = self.unwrapped_env.command_manager.get_command("object_pose")
        desired_pos_b = command[:, :3]
        desired_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            desired_pos_b,
        )
        return desired_pos_w[0].detach().cpu().numpy().astype(np.float32)

    def _make_goal_obs(self, obs_raw):
        return {
            "observation": self._policy_obs(obs_raw),
            "achieved_goal": self._object_pos_w(),
            "desired_goal": self._desired_goal_w(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.reward_type == "dense":
            return -d.astype(np.float32)
        return -(d > self.success_threshold).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_raw = self._reset_isaac()
        return self._make_goal_obs(obs_raw), {"is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        obs_raw, _, terminated, truncated, extras = self.isaac_env.step(action_t)
        obs = self._make_goal_obs(obs_raw)

        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

        info = {
            "is_success": dist < self.success_threshold,
            "goal_distance": dist,
            "TimeLimit.truncated": bool(truncated[0].item()),
        }

        return obs, float(reward), bool(terminated[0].item()), bool(truncated[0].item()), info

    def close(self):
        self.isaac_env.close()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.seed = args_cli.seed

    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabGoalEnv(
        isaac_env,
        reward_type=args_cli.reward_type,
        success_threshold=args_cli.success_threshold,
    )

    model = SAC.load(args_cli.checkpoint, env=env, device=args_cli.device)

    obs, info = env.reset()

    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"reward={reward:.3f} "
            f"distance={info['goal_distance']:.3f} "
            f"success={info['is_success']}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
