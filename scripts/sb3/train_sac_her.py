import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Kinova Gen3 lift/place with SAC + HER.")
parser.add_argument("--task", type=str, default="Gen3-LiftAndPlace-v0")
parser.add_argument("--total_timesteps", type=int, default=2_000_000)
parser.add_argument("--reward_type", type=str, choices=["sparse", "dense"], default="sparse")
parser.add_argument("--success_threshold", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_name", type=str, default="sac_her_2f140")
parser.add_argument("--save_freq", type=int, default=100_000)
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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class IsaacLabGoalEnv(gym.Env):
    """Single-environment GoalEnv wrapper for SB3 SAC+HER.

    This wrapper converts the IsaacLab manipulation task into the standard
    goal-conditioned API expected by HER:

        observation:    proprioception + task observations
        achieved_goal:  current object xyz position in world frame
        desired_goal:   commanded target xyz position in world frame

    Reward:
        sparse: 0 if ||achieved - desired|| < threshold else -1
        dense:  -||achieved - desired||
    """

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
        obs_dim = obs_vec.shape[0]

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # IsaacLab action manager receives normalized actions.
        # Your task has 8 actions: 7 arm joints + 1 2F140 finger_joint.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def _reset_isaac(self):
        out = self.isaac_env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def _policy_obs(self, obs_raw):
        if isinstance(obs_raw, dict):
            obs = obs_raw["policy"]
        else:
            obs = obs_raw
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
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)

        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == "dense":
            return -d.astype(np.float32)

        return -(d > self.success_threshold).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_raw = self._reset_isaac()
        obs = self._make_goal_obs(obs_raw)
        info = {"is_success": False}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        obs_raw, _, terminated, truncated, extras = self.isaac_env.step(action_t)
        obs = self._make_goal_obs(obs_raw)

        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

        # Standard Fetch-style behavior: success is reported in info.
        # The episode still usually ends by time-limit/truncation.
        is_success = dist < self.success_threshold

        terminated_bool = bool(terminated[0].item())
        truncated_bool = bool(truncated[0].item())

        info = {
            "is_success": is_success,
            "goal_distance": dist,
            "TimeLimit.truncated": truncated_bool,
        }

        return obs, float(reward), terminated_bool, truncated_bool, info

    def close(self):
        self.isaac_env.close()


def main():
    log_root = Path("logs/sb3") / args_cli.log_name
    log_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.seed = args_cli.seed

    # Important: SAC+HER here is single-env first. Get it correct, then optimize.
    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabGoalEnv(
        isaac_env,
        reward_type=args_cli.reward_type,
        success_threshold=args_cli.success_threshold,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args_cli.save_freq,
        save_path=str(log_root / "checkpoints"),
        name_prefix="sac_her_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_root / "tb"),
        seed=args_cli.seed,
        device=args_cli.device,
    )

    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name=args_cli.reward_type,
    )

    final_path = log_root / f"final_sac_her_{args_cli.reward_type}"
    model.save(str(final_path))
    model.save_replay_buffer(str(log_root / f"final_replay_buffer_{args_cli.reward_type}.pkl"))

    print(f"[DONE] Saved model to: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
