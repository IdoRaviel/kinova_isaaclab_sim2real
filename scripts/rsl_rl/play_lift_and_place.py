# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chain the frozen reach + grasp policies into a lift-and-place-then-carry rollout.

A per-env state machine runs five phases in sequence:

    REACH  : reach policy drives the fingertip to `reach_target_z` above the cube
             (gripper open), yaw aligned to the cube.
    PAUSE1 : hold still for `pause_after_reach` s (reach policy holds the pose).
    GRASP  : grasp policy grabs the cube and lifts it to `grasp_lift_z` (a small,
             in-place vertical lift; the place target tracks the cube's xy).
    PAUSE2 : hold still for `pause_after_grasp` s (grasp policy keeps holding the cube).
    CARRY  : reach policy flies the arm to a random top-down pose in the air; the
             gripper replays the command grasp last held (it stays closed), so the
             cube is carried along.

The env exposes two observation groups: ``policy`` (grasp layout) and ``reach`` (reach
layout); each policy is fed the group it was trained on.

All tunable values live in ``lift_and_place_cfg.py`` (the ``CFG`` object). Only Isaac's
launcher options (``--headless``, ``--device``, ...) are taken from the command line.
"""

import argparse

from isaaclab.app import AppLauncher

from lift_and_place_cfg import CFG

# Only launcher args are taken from the CLI; everything else comes from CFG.
parser = argparse.ArgumentParser(description="Chain reach + grasp policies (lift-and-place).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import time

import gymnasium as gym
import torch
import yaml

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import gen3.tasks  # noqa: F401

# Phase ids for the state machine.
REACH, PAUSE1, GRASP, PAUSE2, CARRY = 0, 1, 2, 3, 4
PHASE_NAME = {REACH: "reach", PAUSE1: "pause1", GRASP: "grasp", PAUSE2: "pause2", CARRY: "carry"}

# Binary gripper action convention (see BinaryJointAction.process_actions):
# action >= 0 -> open, action < 0 -> close.
GRIPPER_OPEN = 1.0
HALF_PI = math.pi / 2.0
# Cube within this (m) of grasp_lift_z counts as "lifted" -> advance past the grasp phase.
LIFT_TOL = 0.005


class _ActionDimProxy:
    """Wrap the env but report a different action dim.

    The reach actor must be built with ``output_dim = 7`` (arm only), while the env's real
    action space is 8 (arm + gripper). The rsl-rl runner reads ``num_actions`` from the env
    at construction, so we hand it a proxy that overrides only that value. The proxy is never
    stepped; the real env is.
    """

    def __init__(self, env, num_actions):
        self._env = env
        self.num_actions = num_actions

    def get_observations(self):
        return self._env.get_observations()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _load_agent_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    device = args_cli.device

    # --- build the combined env (exposes both 'policy' (grasp) and 'reach' obs groups) ---
    env_cfg = parse_env_cfg(CFG.task, device=device, num_envs=CFG.num_envs)
    env = RslRlVecEnvWrapper(gym.make(CFG.task, cfg=env_cfg), clip_actions=None)
    num_envs = env.num_envs

    # --- grasp policy: reads group 'policy', outputs 8 actions (arm + gripper) ---
    grasp_policy = None
    grasp_clip = None
    if not CFG.reach_only:
        grasp_cfg = _load_agent_cfg(CFG.grasp_agent_cfg)
        grasp_clip = grasp_cfg.get("clip_actions")  # read before the runner mutates the cfg
        grasp_runner = OnPolicyRunner(env, grasp_cfg, log_dir=None, device=device)
        grasp_runner.load(retrieve_file_path(CFG.grasp_checkpoint))
        grasp_policy = grasp_runner.get_inference_policy(device)

    # --- reach policy: reads group 'reach', outputs 7 actions (arm only) ---
    reach_cfg = _load_agent_cfg(CFG.reach_agent_cfg)
    reach_cfg["obs_groups"] = {"actor": ["reach"], "critic": ["reach"]}  # both read the 'reach' group
    reach_clip = reach_cfg.get("clip_actions")
    reach_runner = OnPolicyRunner(_ActionDimProxy(env, num_actions=7), reach_cfg, log_dir=None, device=device)
    reach_runner.load(retrieve_file_path(CFG.reach_checkpoint))
    reach_policy = reach_runner.get_inference_policy(device)
    print(f"[info] action clipping: reach={reach_clip}  grasp={grasp_clip}")

    # --- scene / command handles ---
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]          # fingertip frame (+0.21 m off the flange)
    ee_cmd = env.unwrapped.command_manager.get_term("ee_pose")      # reach target (phases REACH/CARRY)
    obj_cmd = env.unwrapped.command_manager.get_term("object_pose")  # grasp place target (phases GRASP/PAUSE2)

    # --- per-env state ---
    phase = torch.full((num_envs,), REACH, dtype=torch.long, device=device)
    reach_steps = torch.zeros(num_envs, device=device)                    # steps spent in REACH (for timeout)
    pause_left = torch.zeros(num_envs, dtype=torch.long, device=device)   # countdown for the current pause
    held_grip = torch.full((num_envs, 1), GRIPPER_OPEN, device=device)    # gripper cmd replayed during CARRY

    # pause durations (s) -> control steps
    step_dt = getattr(env.unwrapped, "step_dt", None) or (env.unwrapped.physics_dt * env.unwrapped.cfg.decimation)
    pause1_steps = max(1, round(CFG.pause_after_reach / step_dt))
    pause2_steps = max(1, round(CFG.pause_after_grasp / step_dt))

    # --- target-setting helpers (write the command terms the policies read) ---
    def set_above_cube_target(ids=None):
        """Reach target: directly above the cube at `reach_target_z`, gripper yaw aligned to
        the cube's yaw (wrapped to the nearest 90 deg face -> a valid grasp on a square cube)."""
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        cube_p_b, cube_q_b = subtract_frame_transforms(rp, rq, obj.data.root_pos_w, obj.data.root_quat_w)
        _, _, cube_yaw = euler_xyz_from_quat(cube_q_b)
        yaw = cube_yaw - HALF_PI * torch.round(cube_yaw / HALF_PI)   # nearest face, in [-pi/4, pi/4]
        quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.full_like(yaw, math.pi), yaw)  # top-down
        if ee_cmd.cfg.make_quat_unique:
            quat = quat_unique(quat)
        s = slice(None) if ids is None else ids
        ee_cmd.pose_command_b[s, 0] = cube_p_b[s, 0]
        ee_cmd.pose_command_b[s, 1] = cube_p_b[s, 1]
        ee_cmd.pose_command_b[s, 2] = CFG.reach_target_z
        ee_cmd.pose_command_b[s, 3:7] = quat[s]

    def set_lift_target(ids):
        """Grasp place target: directly above the cube (xy tracks the cube) at `grasp_lift_z`,
        so the grasp policy just lifts the cube straight up, in place."""
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        cube_p_b, _ = subtract_frame_transforms(rp, rq, obj.data.root_pos_w, obj.data.root_quat_w)
        obj_cmd.pose_command_b[ids, 0] = cube_p_b[ids, 0]
        obj_cmd.pose_command_b[ids, 1] = cube_p_b[ids, 1]
        obj_cmd.pose_command_b[ids, 2] = CFG.grasp_lift_z

    def set_air_target(ids):
        """Reach target for CARRY: a random top-down pose in the air (inside the CFG box)."""
        n = ids.numel()
        ax = CFG.air_x[0] + (CFG.air_x[1] - CFG.air_x[0]) * torch.rand(n, device=device)
        ay = CFG.air_y[0] + (CFG.air_y[1] - CFG.air_y[0]) * torch.rand(n, device=device)
        az = CFG.air_z[0] + (CFG.air_z[1] - CFG.air_z[0]) * torch.rand(n, device=device)
        quat = quat_from_euler_xyz(torch.zeros(n, device=device), torch.full((n,), math.pi, device=device),
                                   torch.zeros(n, device=device))  # top-down: keep the cube under the gripper
        if ee_cmd.cfg.make_quat_unique:
            quat = quat_unique(quat)
        ee_cmd.pose_command_b[ids, 0] = ax
        ee_cmd.pose_command_b[ids, 1] = ay
        ee_cmd.pose_command_b[ids, 2] = az
        ee_cmd.pose_command_b[ids, 3:7] = quat

    # initialise the reach target above the cube, then refresh the observation
    set_above_cube_target()
    obs = env.get_observations()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()

        # --- choose actions per phase, then step the env ---
        with torch.inference_mode():
            a_reach = reach_policy(obs)  # (N, 7) arm-only
            if reach_clip is not None:
                a_reach = a_reach.clamp(-reach_clip, reach_clip)
            # gripper for the reach-driven phases: open while approaching, but during CARRY
            # replay the (closed) command grasp last held so the cube is not dropped.
            grip = torch.where((phase == CARRY).unsqueeze(-1), held_grip,
                               torch.full((num_envs, 1), GRIPPER_OPEN, device=device))
            a_reach_full = torch.cat([a_reach, grip], dim=-1)  # (N, 8)

            if CFG.reach_only:
                action = a_reach_full
            else:
                a_grasp = grasp_policy(obs)  # (N, 8)
                if grasp_clip is not None:
                    a_grasp = a_grasp.clamp(-grasp_clip, grasp_clip)
                # grasp policy drives GRASP + PAUSE2 (it holds the cube); reach drives the rest.
                use_grasp = ((phase == GRASP) | (phase == PAUSE2)).unsqueeze(-1)
                action = torch.where(use_grasp, a_grasp, a_reach_full)
            obs, _, dones, _ = env.step(action)

        # --- signals used by the state machine ---
        target_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                ee_cmd.pose_command_b[:, :3])
        dist = torch.norm(ee_frame.data.target_pos_w[:, 0, :] - target_w, dim=-1)  # fingertip -> reach target
        cube_z = obj.data.root_pos_w[:, 2]                                          # cube height

        # --- state machine (transitions evaluated latest-phase-first to avoid skipping a phase) ---
        if not CFG.reach_only:
            # PAUSE2 -> CARRY: hold done -> freeze grasp's gripper cmd and pick a random air target
            in_pause2 = phase == PAUSE2
            pause_left = torch.where(in_pause2, pause_left - 1, pause_left)
            done_pause2 = in_pause2 & (pause_left <= 0)
            if done_pause2.any():
                ids = done_pause2.nonzero(as_tuple=False).flatten()
                held_grip[ids] = a_grasp[ids, 7:8]  # freeze grasp's current (closed) gripper command
                set_air_target(ids)
            phase = torch.where(done_pause2, torch.full_like(phase, CARRY), phase)

            # GRASP -> PAUSE2: the cube has been lifted to the target height
            lifted = (phase == GRASP) & (cube_z >= CFG.grasp_lift_z - LIFT_TOL)
            phase = torch.where(lifted, torch.full_like(phase, PAUSE2), phase)
            pause_left = torch.where(lifted, torch.full_like(pause_left, pause2_steps), pause_left)

            # keep the place target a pure vertical lift above the cube while grasping/holding
            holding = (phase == GRASP) | (phase == PAUSE2)
            if holding.any():
                set_lift_target(holding.nonzero(as_tuple=False).flatten())

            # PAUSE1 -> GRASP: hold done
            in_pause1 = phase == PAUSE1
            pause_left = torch.where(in_pause1, pause_left - 1, pause_left)
            phase = torch.where(in_pause1 & (pause_left <= 0), torch.full_like(phase, GRASP), phase)

            # REACH -> PAUSE1: fingertip is above the cube (or reach timed out)
            reach_steps += (phase == REACH).float()
            done_reach = (phase == REACH) & ((dist < CFG.handoff_dist) | (reach_steps > CFG.reach_timeout))
            phase = torch.where(done_reach, torch.full_like(phase, PAUSE1), phase)
            pause_left = torch.where(done_reach, torch.full_like(pause_left, pause1_steps), pause_left)

        # --- on episode reset: restart at REACH and retarget above the new cube ---
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten()
            phase[done_ids] = REACH
            reach_steps[done_ids] = 0.0
            pause_left[done_ids] = 0
            held_grip[done_ids] = GRIPPER_OPEN
            set_above_cube_target(done_ids)

        if timestep % 50 == 0:
            cube_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                  obj.data.root_pos_w)
            ph = "reach" if CFG.reach_only else PHASE_NAME[int(phase[0])]
            print(f"[t {timestep}] env0 phase={ph} dist={dist[0]:.3f}m cube_z={cube_z[0]:.3f} "
                  f"cube_b=({cube_b[0, 0]:.2f},{cube_b[0, 1]:.2f},{cube_b[0, 2]:.2f})")
        timestep += 1

        # real-time pacing
        if CFG.real_time:
            sleep_time = env.unwrapped.physics_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
