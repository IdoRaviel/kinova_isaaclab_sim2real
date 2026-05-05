"""Gripper toggle test: opens and closes the 2F-85 gripper every 2 seconds.

Run from kinova_isaaclab_sim2real/:
    conda activate env_isaaclab
    python scripts/test_gripper.py --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gripper open/close toggle test")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is launched."""

import gymnasium as gym
import torch

import gen3.tasks       # noqa: F401
import isaaclab_tasks   # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# (short_name, full_name, close_cmd_from_current_config, urdf_lower, urdf_upper)
GRIPPER_JOINTS = [
    ("left_knuckle",        "gen3_robotiq_85_left_knuckle_joint",        +0.8, +0.0, +0.8),
    ("right_knuckle",       "gen3_robotiq_85_right_knuckle_joint",       +0.8, +0.0, +0.8),   # axis flipped
    ("left_inner_knuckle",  "gen3_robotiq_85_left_inner_knuckle_joint",  -0.8, -0.8, +0.0),   # axis flipped
    ("right_inner_knuckle", "gen3_robotiq_85_right_inner_knuckle_joint", -0.8, -0.8, +0.0),
    ("left_finger_tip",     "gen3_robotiq_85_left_finger_tip_joint",     -0.8, -0.8, +0.0),
    ("right_finger_tip",    "gen3_robotiq_85_right_finger_tip_joint",    +0.8, +0.0, +0.8),
]


def print_gripper_state(robot, gripper_ids, step_dt, step, is_closing):
    state = "CLOSE" if is_closing else "OPEN "
    pos = robot.data.joint_pos[0, gripper_ids]
    print(f"\n[{state}] t={step * step_dt:.1f}s")
    print(f"  {'joint':<24} {'actual':>8}  {'cmd':>8}  {'range':>14}  status")
    print(f"  {'-'*72}")
    for (short, _, cmd, lo, hi), p in zip(GRIPPER_JOINTS, pos.tolist()):
        expected = cmd if is_closing else 0.0
        out_of_range = is_closing and not (lo <= cmd <= hi)
        if out_of_range:
            status = "!! CMD OUT OF RANGE"
        elif abs(p - expected) < 0.05:
            status = "OK"
        else:
            status = "NOT REACHED"
        print(f"  {short:<24} {p:>+8.4f}  {expected:>+8.4f}  [{lo:+.1f}, {hi:+.1f}]  {status}")


def main():
    env_cfg = parse_env_cfg("Gen3-Grasp-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make("Gen3-Grasp-v0", cfg=env_cfg)
    base_env = env.unwrapped

    step_dt = base_env.step_dt
    steps_per_2sec = max(1, int(2.0 / step_dt))
    print(f"[INFO] step_dt={step_dt:.4f}s  |  steps per 2s = {steps_per_2sec}")

    robot = base_env.scene["robot"]
    full_names = [j[1] for j in GRIPPER_JOINTS]
    gripper_ids, _ = robot.find_joints(full_names)

    num_actions = env.action_space.shape[-1]
    print(f"[INFO] Action space dim = {num_actions}  (arm=7, gripper=1)")
    print("[INFO] Starting gripper toggle — watch the gripper in the viewport.\n")

    obs, _ = env.reset()
    step = 0

    while simulation_app.is_running():
        phase = (step // steps_per_2sec) % 2
        is_closing = (phase == 1)

        # arm action = 0 (hold default pose), gripper = +1 close / -1 open
        action = torch.zeros(args_cli.num_envs, num_actions, device=base_env.device)
        action[:, -1] = 1.0 if is_closing else -1.0

        obs, _, terminated, truncated, _ = env.step(action)

        if (terminated | truncated).any():
            obs, _ = env.reset()

        if step % steps_per_2sec == 0:
            print_gripper_state(robot, gripper_ids, step_dt, step, is_closing)

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
