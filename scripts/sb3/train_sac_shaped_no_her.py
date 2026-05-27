import argparse
import os
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Kinova Gen3 2F140 with SAC shaped reward, no HER.")
parser.add_argument("--task", type=str, default="Gen3-LiftAndPlace-v0")
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_name", type=str, default="sac_shaped_no_her_2f140")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--max_dist", type=float, default=0.70)
parser.add_argument("--success_threshold", type=float, default=0.05)
parser.add_argument("--checkpoint_freq", type=int, default=200_000)
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


class RewardComponentCallback(BaseCallback):
    """Log shaped reward components like the Fetch reward-shaping repo."""

    def __init__(self, window: int = 50):
        super().__init__()
        self.window = window
        self.dist_grip = []
        self.dist_goal = []
        self.successes = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                if "reward_dist_grip" in info:
                    self.dist_grip.append(info["reward_dist_grip"])
                if "reward_dist_goal" in info:
                    self.dist_goal.append(info["reward_dist_goal"])
                if "is_success" in info:
                    self.successes.append(float(info["is_success"]))

        w = self.window
        if self.dist_grip:
            self.logger.record("reward/dist_grip", float(np.mean(self.dist_grip[-w:])))
        if self.dist_goal:
            self.logger.record("reward/dist_goal", float(np.mean(self.dist_goal[-w:])))
        if self.successes:
            self.logger.record("rollout/success_rate_custom", float(np.mean(self.successes[-w:])))

        return True


class IsaacLabShapedFetchWrapper(gym.Env):
    """Goal-conditioned Dict env, but no HER.

    Observation dict follows Fetch-style structure:
        observation:   policy observation vector from IsaacLab
        achieved_goal: object xyz world position
        desired_goal:  commanded target xyz world position

    Reward follows reward-shaping-no-her:
        reward = -(distance_gripper_to_object + distance_object_to_goal) / MAX_DIST
    """

    metadata = {"render_modes": []}

    def __init__(self, isaac_env, max_dist=0.70, success_threshold=0.05):
        super().__init__()
        self.isaac_env = isaac_env
        self.unwrapped_env = isaac_env.unwrapped
        self.device = self.unwrapped_env.device
        self.max_dist = float(max_dist)
        self.success_threshold = float(success_threshold)

        self._ep_dist_grip = 0.0
        self._ep_dist_goal = 0.0
        self._ep_steps = 0

        obs_raw = self._reset_isaac()
        obs_vec = self._policy_obs(obs_raw)

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=obs_vec.shape, dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # Your action manager has 8 actions:
        # 7 arm joints + 1 2F140 finger_joint.
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

    def _ee_pos_w(self):
        ee_frame = self.unwrapped_env.scene["ee_frame"]
        return ee_frame.data.target_pos_w[0, 0].detach().cpu().numpy().astype(np.float32)

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

    def _make_obs(self, obs_raw):
        return {
            "observation": self._policy_obs(obs_raw),
            "achieved_goal": self._object_pos_w(),
            "desired_goal": self._desired_goal_w(),
        }

    def _reward_components(self, obs):
        ee_pos = self._ee_pos_w()
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        dist_grip = float(np.linalg.norm(ee_pos - achieved))
        dist_goal = float(np.linalg.norm(achieved - desired))

        return dist_grip, dist_goal

    def _compute_shaped_reward(self, obs):
        dist_grip, dist_goal = self._reward_components(obs)
        reward = -((dist_grip + dist_goal) / self.max_dist)
        return float(reward), dist_grip, dist_goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._ep_dist_grip = 0.0
        self._ep_dist_goal = 0.0
        self._ep_steps = 0

        obs_raw = self._reset_isaac()
        obs = self._make_obs(obs_raw)

        return obs, {"is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        obs_raw, _, terminated, truncated, extras = self.isaac_env.step(action_t)
        obs = self._make_obs(obs_raw)

        reward, dist_grip, dist_goal = self._compute_shaped_reward(obs)

        self._ep_dist_grip += dist_grip
        self._ep_dist_goal += dist_goal
        self._ep_steps += 1

        n = max(self._ep_steps, 1)
        is_success = dist_goal < self.success_threshold

        terminated_bool = bool(terminated[0].item()) if hasattr(terminated, "__len__") else bool(terminated)
        truncated_bool = bool(truncated[0].item()) if hasattr(truncated, "__len__") else bool(truncated)

        info = {
            "is_success": is_success,
            "goal_distance": dist_goal,
            "gripper_object_distance": dist_grip,
            "reward_dist_grip": -self._ep_dist_grip / (n * self.max_dist),
            "reward_dist_goal": -self._ep_dist_goal / (n * self.max_dist),
            "TimeLimit.truncated": truncated_bool,
        }

        return obs, reward, terminated_bool, truncated_bool, info

    def close(self):
        self.isaac_env.close()


def main():
    log_root = Path("logs/sb3") / args_cli.log_name
    checkpoint_dir = log_root / "checkpoints"
    tb_dir = log_root / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.seed = args_cli.seed

    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabShapedFetchWrapper(
        isaac_env,
        max_dist=args_cli.max_dist,
        success_threshold=args_cli.success_threshold,
    )

    env = Monitor(
        env,
        filename=str(log_root / "monitor.csv"),
        info_keywords=(
            "is_success",
            "goal_distance",
            "gripper_object_distance",
            "reward_dist_grip",
            "reward_dist_goal",
        ),
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=args_cli.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="sac_shaped_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            ),
            RewardComponentCallback(window=50),
        ]
    )

    if args_cli.resume:
        model = SAC.load(
            args_cli.resume,
            env=env,
            tensorboard_log=str(tb_dir),
            device=args_cli.device,
        )
        model.action_noise = action_noise

        replay_buffer_path = args_cli.resume.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print(f"Loaded replay buffer: {replay_buffer_path}")
        else:
            print("No replay buffer found. Continuing with empty replay buffer.")

        reset_num_timesteps = False

    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            buffer_size=1_000_000,
            batch_size=1024,
            gamma=0.97,
            tau=0.05,
            learning_rate=5e-4,
            learning_starts=30_000,
            action_noise=action_noise,
            policy_kwargs={
                "net_arch": [512, 512, 512],
                "n_critics": 2,
            },
            verbose=1,
            tensorboard_log=str(tb_dir),
            seed=args_cli.seed,
            device=args_cli.device,
        )
        reset_num_timesteps = True

    try:
        model.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
            tb_log_name="SAC_shaped_no_HER",
        )

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model.")

    final_path = log_root / "final_sac_shaped_no_her"
    model.save(str(final_path))
    model.save_replay_buffer(str(log_root / "final_sac_shaped_no_her_replay_buffer.pkl"))

    print(f"Saved final model to: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
