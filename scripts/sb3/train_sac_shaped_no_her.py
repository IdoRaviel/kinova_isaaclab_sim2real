import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Kinova Gen3 2F140 with SAC shaped reward, no HER.")
parser.add_argument("--task", type=str, default="Gen3-Grasp-v0")
parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
parser.add_argument("--resume", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import gen3.tasks  # noqa: F401
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from gen3.tasks.manager_based.gen3_grasp.agents.sac_cfg import Gen3GraspSACCfg
from gen3.tasks.manager_based.gen3_grasp.mdp import (
    ee_to_cube_distance,
    cube_to_target_distance_l2,
    cube_to_target_distance_tanh,
)

cfg = Gen3GraspSACCfg()
if args_cli.seed is not None:
    cfg.seed = args_cli.seed


class RewardComponentCallback(BaseCallback):
    """Log shaped reward components like the Fetch reward-shaping repo."""

    def __init__(self, window: int = 50):
        super().__init__()
        self.window = window
        self.dist_grip = []
        self.dist_goal = []
        self.fine_bonus = []
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
                if "reward_fine_bonus" in info:
                    self.fine_bonus.append(info["reward_fine_bonus"])
                if "is_success" in info:
                    self.successes.append(float(info["is_success"]))

        w = self.window
        if self.dist_grip:
            self.logger.record("reward/dist_grip", float(np.mean(self.dist_grip[-w:])))
        if self.dist_goal:
            self.logger.record("reward/dist_goal", float(np.mean(self.dist_goal[-w:])))
        if self.fine_bonus:
            self.logger.record("reward/fine_bonus", float(np.mean(self.fine_bonus[-w:])))
        if self.successes:
            self.logger.record("rollout/success_rate_custom", float(np.mean(self.successes[-w:])))

        return True


class IsaacLabShapedFetchWrapper(gym.Env):
    """Goal-conditioned Dict env, no HER.

    Observation dict follows Fetch-style structure:
        observation:   policy observation vector from IsaacLab
        achieved_goal: object xyz world position
        desired_goal:  commanded target xyz world position

    Reward:  -(dist_grip + dist_goal) / max_dist
    Both distances are computed via mdp reward functions in rewards.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, isaac_env, sac_cfg: Gen3GraspSACCfg):
        super().__init__()
        self.isaac_env = isaac_env
        self.unwrapped_env = isaac_env.unwrapped
        self.device = self.unwrapped_env.device
        self.max_dist = sac_cfg.max_dist
        self.success_threshold = sac_cfg.success_threshold

        self._ep_dist_grip = 0.0
        self._ep_dist_goal = 0.0
        self._ep_fine_bonus = 0.0
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
        # 7 arm joints + 1 finger_joint
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
        desired_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command[:, :3],
        )
        return desired_pos_w[0].detach().cpu().numpy().astype(np.float32)

    def _make_obs(self, obs_raw):
        return {
            "observation": self._policy_obs(obs_raw),
            "achieved_goal": self._object_pos_w(),
            "desired_goal": self._desired_goal_w(),
        }

    def _compute_shaped_reward(self):
        grip_reward = float(ee_to_cube_distance(self.unwrapped_env, std=0.1)[0].item())
        dist_goal = float(cube_to_target_distance_l2(self.unwrapped_env)[0].item())
        fine_bonus = float(cube_to_target_distance_tanh(self.unwrapped_env, std=0.1)[0].item())
        reward = -(grip_reward + dist_goal / self.max_dist) + 0.25 * fine_bonus
        return reward, grip_reward, dist_goal, fine_bonus

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ep_dist_grip = 0.0
        self._ep_dist_goal = 0.0
        self._ep_fine_bonus = 0.0
        self._ep_steps = 0
        obs_raw = self._reset_isaac()
        return self._make_obs(obs_raw), {"is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        obs_raw, _, terminated, truncated, extras = self.isaac_env.step(action_t)
        obs = self._make_obs(obs_raw)
        reward, grip_reward, dist_goal, fine_bonus = self._compute_shaped_reward()

        self._ep_dist_grip += grip_reward
        self._ep_dist_goal += dist_goal
        self._ep_fine_bonus += fine_bonus
        self._ep_steps += 1

        n = max(self._ep_steps, 1)
        is_success = dist_goal < self.success_threshold

        terminated_bool = bool(terminated[0].item()) if hasattr(terminated, "__len__") else bool(terminated)
        truncated_bool = bool(truncated[0].item()) if hasattr(truncated, "__len__") else bool(truncated)

        info = {
            "is_success": is_success,
            "goal_distance": dist_goal,
            "gripper_object_distance": grip_reward,
            "reward_dist_grip": -self._ep_dist_grip / n,
            "reward_dist_goal": -self._ep_dist_goal / (n * self.max_dist),
            "reward_fine_bonus": 0.25 * self._ep_fine_bonus / n,
            "TimeLimit.truncated": truncated_bool,
        }

        return obs, reward, terminated_bool, truncated_bool, info

    def close(self):
        self.isaac_env.close()


def main():
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path("logs/sb3") / cfg.log_name / run_name

    log_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = cfg.seed

    isaac_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabShapedFetchWrapper(isaac_env, cfg)
    env = Monitor(
        env,
        filename=str(log_root / "monitor.csv"),
        info_keywords=(
            "is_success",
            "goal_distance",
            "gripper_object_distance",
            "reward_dist_grip",
            "reward_dist_goal",
            "reward_fine_bonus",
        ),
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=cfg.action_noise_sigma * np.ones(n_actions),
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=cfg.checkpoint_freq,
                save_path=str(log_root),
                name_prefix="model",
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
            tensorboard_log=str(log_root),
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
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            tau=cfg.tau,
            learning_rate=cfg.learning_rate,
            learning_starts=cfg.learning_starts,
            target_entropy=cfg.target_entropy,
            action_noise=action_noise,
            policy_kwargs={
                "net_arch": cfg.net_arch,
                "n_critics": cfg.n_critics,
            },
            verbose=1,
            tensorboard_log=str(log_root),
            seed=cfg.seed,
            device=args_cli.device,
        )
        reset_num_timesteps = True

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
            tb_log_name="SAC_shaped_no_HER",
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model.")

    final_path = log_root / "model_final"
    model.save(str(final_path))
    model.save_replay_buffer(str(log_root / "model_final_replay_buffer.pkl"))
    print(f"Saved final model to: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
