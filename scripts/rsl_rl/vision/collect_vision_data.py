# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect an RGB + cube-pose dataset from lift-and-place chain rollouts.

Runs the same frozen reach + grasp chain as ``play_lift_and_place.py`` (same phase
state machine: REACH -> PAUSE1 -> GRASP -> PAUSE2 -> CARRY), but instead of running
forever for interactive viewing, it saves the camera's RGB frame and the cube's pose
(robot-root frame) at each step, until a target number of frames has been collected.

To avoid over-representing the near-static PAUSE1/PAUSE2 phases (the arm holds still,
so consecutive frames there are nearly identical), only 1 in every ``PAUSE_SAVE_STRIDE``
steps is saved during those phases; REACH/GRASP/CARRY are saved every step.

Output layout (under ``--output_dir``):
    meta.json        camera intrinsics/extrinsics, saved once (constant across frames)
    labels.jsonl      one JSON record per saved frame
    images/NNNNNN.png  the corresponding RGB frame

All chain-timing/policy knobs (checkpoints, pause durations, handoff thresholds, ...)
are shared with the play script via ``lift_and_place_cfg.CFG`` -- this script only adds
CLI options for what's specific to data collection (output location, frame budget).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add parent directory to path so lift_and_place_cfg can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lift_and_place_cfg import CFG

parser = argparse.ArgumentParser(
    description="Collect vision dataset from lift-and-place chain rollouts."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="vision_dataset",
    help="Directory to write images/labels to.",
)
parser.add_argument(
    "--num_frames",
    type=int,
    default=10000,
    help="Stop once this many frames have been saved.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=32,
    help="Parallel envs to run (overrides CFG.num_envs).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = (
    True  # required for the camera sensor to render in headless mode
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os

import gymnasium as gym
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import gen3.tasks  # noqa: F401

# Phase ids for the state machine (same convention as play_lift_and_place.py).
REACH, PAUSE1, GRASP, PAUSE2, CARRY = 0, 1, 2, 3, 4
PHASE_NAME = {
    REACH: "reach",
    PAUSE1: "pause1",
    GRASP: "grasp",
    PAUSE2: "pause2",
    CARRY: "carry",
}

GRIPPER_OPEN = 1.0
HALF_PI = math.pi / 2.0

# Only 1 in this many steps is saved while the arm is holding still (PAUSE1/PAUSE2), so
# near-duplicate static frames don't dominate the dataset. REACH/GRASP/CARRY save every step.
PAUSE_SAVE_STRIDE = 5

# Cube half-size (m): the DexCube USD asset's base half-size (0.03 m) times the spawn
# scale factor used by the grasp/lift_and_place env configs' cube RigidObjectCfg
# (scale=(1.2, 1.2, 1.2)) -- expressed as a product, not a bare literal, so it stays
# self-evidently correct if that scale factor ever changes.
_DEXCUBE_BASE_HALF_SIZE_M = 0.03
_CUBE_SPAWN_SCALE = 1.2
CUBE_HALF_SIZE = _DEXCUBE_BASE_HALF_SIZE_M * _CUBE_SPAWN_SCALE
_CORNER_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)


def project_cube_corners(
    obj_pos_w, obj_quat_w, cam_pos_w, cam_quat_w_ros, intrinsic_matrices
):
    """Project the cube's 8 local corners into 2D pixel coordinates, per env.

    Standard pinhole projection: local corners -> world (via the cube's pose) -> camera
    frame (via the camera's pose, ROS convention: +X right, +Y down, +Z forward) -> pixels
    (via the intrinsic matrix). Some corners will be behind the cube itself (self-occlusion)
    or the gripper -- that's expected (see PnP discussion), not handled here.

    Returns:
        (N, 8, 2) tensor of (u, v) pixel coordinates.
    """
    n = obj_pos_w.shape[0]
    corners_local = _CORNER_SIGNS.to(obj_pos_w.device) * CUBE_HALF_SIZE  # (8, 3)
    corners_local = corners_local.unsqueeze(0).expand(n, -1, -1)  # (N, 8, 3)

    quat_obj = obj_quat_w.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 4)
    corners_w = quat_apply(quat_obj, corners_local) + obj_pos_w.unsqueeze(
        1
    )  # (N, 8, 3)

    quat_cam = cam_quat_w_ros.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 4)
    rel_w = corners_w - cam_pos_w.unsqueeze(1)  # (N, 8, 3)
    corners_cam = quat_apply_inverse(quat_cam, rel_w)  # (N, 8, 3)

    fx = intrinsic_matrices[:, 0, 0].unsqueeze(1)
    fy = intrinsic_matrices[:, 1, 1].unsqueeze(1)
    cx = intrinsic_matrices[:, 0, 2].unsqueeze(1)
    cy = intrinsic_matrices[:, 1, 2].unsqueeze(1)
    z = corners_cam[..., 2].clamp(min=1e-6)
    u = fx * (corners_cam[..., 0] / z) + cx
    v = fy * (corners_cam[..., 1] / z) + cy
    return torch.stack([u, v], dim=-1)  # (N, 8, 2)


class _ActionDimProxy:
    """Wrap the env but report a different action dim (reach actor is arm-only: 7)."""

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

    images_dir = os.path.join(args_cli.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_path = os.path.join(args_cli.output_dir, "labels.jsonl")
    labels_file = open(labels_path, "a")

    # --- build the combined env (exposes both 'policy' (grasp) and 'reach' obs groups) ---
    env_cfg = parse_env_cfg(CFG.task, device=device, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(gym.make(CFG.task, cfg=env_cfg), clip_actions=None)
    num_envs = env.num_envs

    # --- grasp policy: reads group 'policy', outputs 8 actions (arm + gripper) ---
    grasp_cfg = _load_agent_cfg(CFG.grasp_agent_cfg)
    grasp_clip = grasp_cfg.get("clip_actions")
    grasp_runner = OnPolicyRunner(env, grasp_cfg, log_dir=None, device=device)
    grasp_runner.load(retrieve_file_path(CFG.grasp_checkpoint))
    grasp_policy = grasp_runner.get_inference_policy(device)

    # --- reach policy: reads group 'reach', outputs 7 actions (arm only) ---
    reach_cfg = _load_agent_cfg(CFG.reach_agent_cfg)
    reach_cfg["obs_groups"] = {"actor": ["reach"], "critic": ["reach"]}
    reach_clip = reach_cfg.get("clip_actions")
    reach_runner = OnPolicyRunner(
        _ActionDimProxy(env, num_actions=7), reach_cfg, log_dir=None, device=device
    )
    reach_runner.load(retrieve_file_path(CFG.reach_checkpoint))
    reach_policy = reach_runner.get_inference_policy(device)
    print(f"[info] action clipping: reach={reach_clip}  grasp={grasp_clip}")

    # --- scene / command handles ---
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    camera = env.unwrapped.scene["camera"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    ee_cmd = env.unwrapped.command_manager.get_term("ee_pose")
    obj_cmd = env.unwrapped.command_manager.get_term("object_pose")

    # --- write camera intrinsics/extrinsics once (constant across every frame) ---
    cam_offset = env_cfg.scene.camera.offset
    meta = {
        "image_width": camera.image_shape[1],
        "image_height": camera.image_shape[0],
        "intrinsic_matrix": camera.data.intrinsic_matrices[0].cpu().tolist(),
        "extrinsics_frame": "robot_root",  # robot root == world origin in this scene (identity transform)
        "camera_pos": list(cam_offset.pos),
        "camera_rot_wxyz": list(cam_offset.rot),
        "camera_convention": cam_offset.convention,
    }
    with open(os.path.join(args_cli.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- per-env state (mirrors play_lift_and_place.py's state machine) ---
    phase = torch.full((num_envs,), REACH, dtype=torch.long, device=device)
    reach_steps = torch.zeros(num_envs, device=device)
    pause_left = torch.zeros(num_envs, dtype=torch.long, device=device)
    carry_settled_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    held_grip = torch.full((num_envs, 1), GRIPPER_OPEN, device=device)
    prev_ee_w = ee_frame.data.target_pos_w[:, 0, :].clone()
    best_dist = torch.full((num_envs,), float("inf"), device=device)
    plateau_count = torch.zeros(num_envs, device=device)

    step_dt = getattr(env.unwrapped, "step_dt", None) or (
        env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
    )
    pause1_steps = max(1, round(CFG.pause_after_reach / step_dt))
    pause2_steps = max(1, round(CFG.pause_after_grasp / step_dt))

    def set_above_cube_target(ids=None):
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        cube_p_b, cube_q_b = subtract_frame_transforms(
            rp, rq, obj.data.root_pos_w, obj.data.root_quat_w
        )
        _, _, cube_yaw = euler_xyz_from_quat(cube_q_b)
        yaw = cube_yaw - HALF_PI * torch.round(cube_yaw / HALF_PI)
        quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.full_like(yaw, math.pi), yaw
        )
        if ee_cmd.cfg.make_quat_unique:
            quat = quat_unique(quat)
        s = slice(None) if ids is None else ids
        ee_cmd.pose_command_b[s, 0] = cube_p_b[s, 0]
        ee_cmd.pose_command_b[s, 1] = cube_p_b[s, 1]
        ee_cmd.pose_command_b[s, 2] = CFG.reach_target_z
        ee_cmd.pose_command_b[s, 3:7] = quat[s]

    def set_lift_target(ids):
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        cube_p_b, _ = subtract_frame_transforms(
            rp, rq, obj.data.root_pos_w, obj.data.root_quat_w
        )
        obj_cmd.pose_command_b[ids, 0] = cube_p_b[ids, 0]
        obj_cmd.pose_command_b[ids, 1] = cube_p_b[ids, 1]
        obj_cmd.pose_command_b[ids, 2] = CFG.grasp_lift_z

    def set_air_target(ids):
        n = ids.numel()
        ax = CFG.air_x[0] + (CFG.air_x[1] - CFG.air_x[0]) * torch.rand(n, device=device)
        ay = CFG.air_y[0] + (CFG.air_y[1] - CFG.air_y[0]) * torch.rand(n, device=device)
        az = CFG.air_z[0] + (CFG.air_z[1] - CFG.air_z[0]) * torch.rand(n, device=device)
        roll = CFG.air_roll[0] + (CFG.air_roll[1] - CFG.air_roll[0]) * torch.rand(
            n, device=device
        )
        pitch = CFG.air_pitch[0] + (CFG.air_pitch[1] - CFG.air_pitch[0]) * torch.rand(
            n, device=device
        )
        yaw = CFG.air_yaw[0] + (CFG.air_yaw[1] - CFG.air_yaw[0]) * torch.rand(
            n, device=device
        )
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        if ee_cmd.cfg.make_quat_unique:
            quat = quat_unique(quat)
        ee_cmd.pose_command_b[ids, 0] = ax
        ee_cmd.pose_command_b[ids, 1] = ay
        ee_cmd.pose_command_b[ids, 2] = az
        ee_cmd.pose_command_b[ids, 3:7] = quat

    set_above_cube_target()
    obs = env.get_observations()

    timestep = 0
    saved_count = 0
    progress = tqdm(total=args_cli.num_frames, unit="frame", desc="collecting")
    while simulation_app.is_running() and saved_count < args_cli.num_frames:
        with torch.inference_mode():
            a_reach = reach_policy(obs)
            if reach_clip is not None:
                a_reach = a_reach.clamp(-reach_clip, reach_clip)
            grip = torch.where(
                (phase == CARRY).unsqueeze(-1),
                held_grip,
                torch.full((num_envs, 1), GRIPPER_OPEN, device=device),
            )
            a_reach_full = torch.cat([a_reach, grip], dim=-1)

            a_grasp = grasp_policy(obs)
            if grasp_clip is not None:
                a_grasp = a_grasp.clamp(-grasp_clip, grasp_clip)
            use_grasp = ((phase == GRASP) | (phase == PAUSE2)).unsqueeze(-1)
            action = torch.where(use_grasp, a_grasp, a_reach_full)
            obs, _, dones, _ = env.step(action)

        target_w, _ = combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_cmd.pose_command_b[:, :3],
            ee_cmd.pose_command_b[:, 3:7],
        )
        ee_now_w = ee_frame.data.target_pos_w[:, 0, :]
        dist = torch.norm(ee_now_w - target_w, dim=-1)
        ee_speed = torch.norm(ee_now_w - prev_ee_w, dim=-1) / step_dt
        prev_ee_w = ee_now_w.clone()
        cube_z = obj.data.root_pos_w[:, 2]

        in_reach = phase == REACH
        improved = in_reach & (dist < best_dist - CFG.handoff_plateau_eps)
        best_dist = torch.where(improved, dist, best_dist)
        plateau_count = torch.where(
            improved, torch.zeros_like(plateau_count), plateau_count
        )
        plateau_count = torch.where(
            in_reach & ~improved, plateau_count + 1.0, plateau_count
        )

        # --- decide which envs to save this step: every step outside pauses, 1-in-N inside ---
        in_pause = (phase == PAUSE1) | (phase == PAUSE2)
        # once CARRY has arrived and stopped moving, it just sits there until the episode
        # times out -- stop saving entirely for that env (latched, so it doesn't flicker
        # back on if dist/speed jitter near the threshold) rather than trickle in duplicates.
        carry_settled = (
            (phase == CARRY)
            & (dist < CFG.handoff_dist)
            & (ee_speed < CFG.handoff_speed)
        )
        carry_settled_latch = carry_settled_latch | carry_settled
        save_mask = ((~in_pause) | (pause_left % PAUSE_SAVE_STRIDE == 0)) & (
            ~carry_settled_latch
        )

        if save_mask.any():
            rgb = camera.data.output["rgb"]  # (N, H, W, 3) uint8
            # 3-D position + orientation (robot-root frame) -- labels for the direct-regression
            # baseline (Phase 5), and for computing PnP/keypoint error against ground truth.
            cube_pos_b, cube_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                obj.data.root_pos_w,
                obj.data.root_quat_w,
            )
            # 2-D corner keypoints (pixel coords) -- labels for the keypoint+PnP model (Phase 3).
            # Some corners are self-/gripper-occluded; kept as-is (see earlier discussion) rather
            # than filtered.
            corners_2d = project_cube_corners(
                obj.data.root_pos_w,
                obj.data.root_quat_w,
                camera.data.pos_w,
                camera.data.quat_w_ros,
                camera.data.intrinsic_matrices,
            )
            for env_id in save_mask.nonzero(as_tuple=False).flatten().tolist():
                img = Image.fromarray(rgb[env_id].cpu().numpy())
                fname = f"{saved_count:06d}.png"
                img.save(os.path.join(images_dir, fname))
                record = {
                    "file": f"images/{fname}",
                    "env_id": env_id,
                    "step": timestep,
                    "phase": PHASE_NAME[int(phase[env_id])],
                    "cube_pos": cube_pos_b[env_id].cpu().tolist(),
                    "cube_quat_wxyz": cube_quat_b[env_id].cpu().tolist(),
                    "corners_2d": corners_2d[env_id].cpu().tolist(),
                }
                labels_file.write(json.dumps(record) + "\n")
                saved_count += 1
                progress.update(1)
                if saved_count >= args_cli.num_frames:
                    break
            labels_file.flush()

        # --- state machine (same transitions as play_lift_and_place.py) ---
        in_pause2 = phase == PAUSE2
        pause_left = torch.where(in_pause2, pause_left - 1, pause_left)
        done_pause2 = in_pause2 & (pause_left <= 0)
        if done_pause2.any():
            ids = done_pause2.nonzero(as_tuple=False).flatten()
            held_grip[ids] = a_grasp[ids, 7:8]
            set_air_target(ids)
        phase = torch.where(done_pause2, torch.full_like(phase, CARRY), phase)

        lifted = (phase == GRASP) & (
            (cube_z - CFG.grasp_lift_z).abs() <= CFG.grasp_lift_tol
        )
        phase = torch.where(lifted, torch.full_like(phase, PAUSE2), phase)
        pause_left = torch.where(
            lifted, torch.full_like(pause_left, pause2_steps), pause_left
        )

        holding = (phase == GRASP) | (phase == PAUSE2)
        if holding.any():
            set_lift_target(holding.nonzero(as_tuple=False).flatten())

        in_pause1 = phase == PAUSE1
        pause_left = torch.where(in_pause1, pause_left - 1, pause_left)
        phase = torch.where(
            in_pause1 & (pause_left <= 0), torch.full_like(phase, GRASP), phase
        )

        reach_steps += (phase == REACH).float()
        settled = (dist < CFG.handoff_dist) & (ee_speed < CFG.handoff_speed)
        plateaued = (plateau_count >= CFG.handoff_plateau_steps) & (
            dist < CFG.handoff_accept_dist
        )
        timed_out = reach_steps > CFG.reach_timeout
        done_reach = (phase == REACH) & (settled | plateaued | timed_out)
        phase = torch.where(done_reach, torch.full_like(phase, PAUSE1), phase)
        pause_left = torch.where(
            done_reach, torch.full_like(pause_left, pause1_steps), pause_left
        )

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten()
            phase[done_ids] = REACH
            reach_steps[done_ids] = 0.0
            pause_left[done_ids] = 0
            carry_settled_latch[done_ids] = False
            held_grip[done_ids] = GRIPPER_OPEN
            best_dist[done_ids] = float("inf")
            plateau_count[done_ids] = 0.0
            prev_ee_w[done_ids] = ee_frame.data.target_pos_w[done_ids, 0, :]
            set_above_cube_target(done_ids)

        if timestep % CFG.diagnostic_print_interval_steps == 0:
            print(f"[t {timestep}] saved {saved_count}/{args_cli.num_frames} frames")
        timestep += 1

    labels_file.close()
    print(f"[done] saved {saved_count} frames to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
