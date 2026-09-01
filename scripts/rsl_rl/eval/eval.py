# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy headlessly and report quantitative metrics.

Unlike ``play.py`` (which renders for visual inspection), this runs a policy for a
fixed number of steps across many parallel envs with no rendering, then prints
mean/std position and orientation error plus mean reward -- for numeric comparison
between checkpoints (e.g. before/after a retrain).

Loads either a TorchScript JIT-exported policy or a raw rsl-rl training checkpoint,
whichever ``resume_path`` turns out to be. Metrics are pulled from the env's
``command_manager`` terms (``position_error``/``orientation_error``), so they're only
reported for tasks whose command term exposes them (e.g. reach's ``ee_pose`` command);
other tasks just get the mean-reward line.
"""

from importlib.metadata import version as get_version
import sys
import os

# add parent directory to path so cli_args can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of steps to evaluate.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import gen3.tasks  # noqa: F401


def main():
    """Evaluate RSL-RL agent and report metrics."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, get_version("rsl-rl-lib"))

    # resolve checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load policy — handle both JIT models and rsl-rl checkpoints
    device = env.unwrapped.device
    try:
        # An exported policy (from play.py's --video/export flow, if any) is a
        # TorchScript module; a raw training checkpoint is not, and jit.load raises.
        jit_model = torch.jit.load(resume_path, map_location=device)
        policy = jit_model
        print("[INFO]: Loaded as TorchScript JIT model.")
    except Exception:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=device)
        print("[INFO]: Loaded as rsl-rl checkpoint.")

    # metric storage
    all_pos_errors = []
    all_ori_errors = []
    all_rewards = []

    # run evaluation
    obs = env.get_observations()
    print(f"\n[INFO]: Running evaluation for {args_cli.num_steps} steps with {args_cli.num_envs} envs...")

    is_jit = isinstance(policy, torch.jit.ScriptModule)

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if is_jit:
                # JIT-traced models were exported to take a flat tensor input, not the
                # TensorDict/dict-of-tensors the env normally hands the rsl-rl policy.
                obs_tensor = torch.cat([v for v in obs.values()], dim=-1) if hasattr(obs, "values") else obs
                actions = policy(obs_tensor)
            else:
                actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            # Pull position/orientation error straight from the command term's own
            # metrics dict rather than recomputing them here, so this matches exactly
            # what the reward/logging already consider "the error".
            unwrapped = env.unwrapped
            for term in unwrapped.command_manager._terms.values():
                if hasattr(term, "metrics"):
                    if "position_error" in term.metrics:
                        all_pos_errors.append(term.metrics["position_error"].mean().item())
                    if "orientation_error" in term.metrics:
                        all_ori_errors.append(term.metrics["orientation_error"].mean().item())

            all_rewards.append(rewards.mean().item())

    # compute and print results
    import sys
    print("\n" + "=" * 60, flush=True)
    print(f"  EVALUATION RESULTS — {os.path.basename(resume_path)}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Checkpoint : {resume_path}", flush=True)
    print(f"  Steps      : {args_cli.num_steps}", flush=True)
    print(f"  Envs       : {args_cli.num_envs}", flush=True)
    print(f"  ---", flush=True)
    print(f"  Mean reward          : {np.mean(all_rewards):.4f}", flush=True)
    print(f"  Mean position error  : {np.mean(all_pos_errors):.4f} m", flush=True)
    print(f"  Mean orientation err : {np.mean(all_ori_errors):.4f} rad", flush=True)
    print(f"  ---", flush=True)
    print(f"  Std position error   : {np.std(all_pos_errors):.4f} m", flush=True)
    print(f"  Std orientation err  : {np.std(all_ori_errors):.4f} rad", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
