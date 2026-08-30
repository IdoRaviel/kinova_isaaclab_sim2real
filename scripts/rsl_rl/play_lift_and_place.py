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

With ``--vision``, the chain runs on the ``Gen3-LiftAndPlace-Chain-Vision-v0`` env and
every use of the cube's pose comes from the trained CubePoseNet's prediction on the
workspace camera image instead of privileged simulator state: the grasp policy's
``object_position``/``object_orientation`` observations (swapped inside the env cfg),
the reach target above the cube (position + yaw), the in-place lift target, and the
GRASP -> PAUSE2 "cube is lifted" check (predicted cube height). The true cube pose is
then read only for diagnostics (periodic prediction-error print and a per-episode
success count). Because the prediction refreshes after each sim step, the reach target
is re-set from the latest prediction on every REACH step -- this also covers the
one-step camera lag right after a reset.
"""

import argparse

from isaaclab.app import AppLauncher

from lift_and_place_cfg import CFG

# Only launcher args (plus the vision switch) are taken from the CLI; everything else comes from CFG.
parser = argparse.ArgumentParser(description="Chain reach + grasp policies (lift-and-place).")
parser.add_argument(
    "--vision",
    action="store_true",
    help="Drive the chain from CubePoseNet's camera prediction instead of ground-truth cube pose.",
)
parser.add_argument(
    "--vision_checkpoint",
    type=str,
    default="pretrained_models/cube_pose/best.pt",
    help="CubePoseNet checkpoint (used with --vision).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.vision:
    args_cli.enable_cameras = True  # the vision obs terms need the camera rendered

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

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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

    # --- build the combined env (exposes both 'policy' (grasp) and 'reach' obs groups);
    # with --vision, the env's object obs terms are fed by CubePoseNet instead ---
    task = "Gen3-LiftAndPlace-Chain-Vision-v0" if args_cli.vision else CFG.task
    env_cfg = parse_env_cfg(task, device=device, num_envs=CFG.num_envs)
    if args_cli.vision:
        env_cfg.observations.policy.object_position.params["checkpoint"] = args_cli.vision_checkpoint
        env_cfg.observations.policy.object_orientation.params["checkpoint"] = args_cli.vision_checkpoint
        print(f"[info] vision checkpoint: {args_cli.vision_checkpoint}")
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=None)
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

    def cube_pose():
        """Cube pose in the robot-root frame: CubePoseNet's prediction (--vision) or ground truth."""
        if args_cli.vision:
            cache = env.unwrapped._vision_pose_cache  # latest prediction, cached by the obs terms
            return cache["pos"], cache["quat"]
        return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         obj.data.root_pos_w, obj.data.root_quat_w)

    # --- phase-colored target marker: one sphere at the active phase's target ---
    # Same radius as the (now-hidden) ee_frame debug sphere, for visual consistency.
    def _sphere(color):
        return sim_utils.SphereCfg(
            radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        )
    hidden_sphere = _sphere((0.0, 0.0, 0.0))
    hidden_sphere.visible = False
    target_vis = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/Command/active_target",
        markers={
            "blue": _sphere((0.0, 0.0, 1.0)),    # first reach + pause1 (above the cube)
            "green": _sphere((0.0, 1.0, 0.0)),   # second reach / carry (in the air)
            "hidden": hidden_sphere,              # grasp / pause2: no target sphere shown
        },
    ))

    # --- per-env state ---
    phase = torch.full((num_envs,), REACH, dtype=torch.long, device=device)
    reach_steps = torch.zeros(num_envs, device=device)                    # steps spent in REACH (for timeout)
    pause_left = torch.zeros(num_envs, dtype=torch.long, device=device)   # countdown for the current pause
    held_grip = torch.full((num_envs, 1), GRIPPER_OPEN, device=device)    # gripper cmd replayed during CARRY
    # reach-handoff bookkeeping (hybrid switch): fingertip speed + plateau detection
    prev_ee_w = ee_frame.data.target_pos_w[:, 0, :].clone()               # prev fingertip pos (for speed)
    best_dist = torch.full((num_envs,), float("inf"), device=device)      # running-min reach dist (plateau)
    plateau_count = torch.zeros(num_envs, device=device)                  # steps since dist last improved
    # success bookkeeping (--vision diagnostics): episodes finished / episodes that reached CARRY
    episodes_done = 0
    episodes_carried = 0

    # pause durations (s) -> control steps
    step_dt = getattr(env.unwrapped, "step_dt", None) or (env.unwrapped.physics_dt * env.unwrapped.cfg.decimation)
    pause1_steps = max(1, round(CFG.pause_after_reach / step_dt))
    pause2_steps = max(1, round(CFG.pause_after_grasp / step_dt))

    # --- target-setting helpers (write the command terms the policies read) ---
    def set_above_cube_target(ids=None):
        """Reach target: directly above the cube at `reach_target_z`, gripper yaw aligned to
        the cube's yaw (wrapped to the nearest 90 deg face -> a valid grasp on a square cube)."""
        cube_p_b, cube_q_b = cube_pose()
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
        cube_p_b, _ = cube_pose()
        obj_cmd.pose_command_b[ids, 0] = cube_p_b[ids, 0]
        obj_cmd.pose_command_b[ids, 1] = cube_p_b[ids, 1]
        obj_cmd.pose_command_b[ids, 2] = CFG.grasp_lift_z

    def set_air_target(ids):
        """Reach target for CARRY: a random top-down pose in the air (inside the CFG box)."""
        n = ids.numel()
        ax = CFG.air_x[0] + (CFG.air_x[1] - CFG.air_x[0]) * torch.rand(n, device=device)
        ay = CFG.air_y[0] + (CFG.air_y[1] - CFG.air_y[0]) * torch.rand(n, device=device)
        az = CFG.air_z[0] + (CFG.air_z[1] - CFG.air_z[0]) * torch.rand(n, device=device)
        # sampled orientation around top-down (yaw varies freely; roll/pitch small tilt)
        roll = CFG.air_roll[0] + (CFG.air_roll[1] - CFG.air_roll[0]) * torch.rand(n, device=device)
        pitch = CFG.air_pitch[0] + (CFG.air_pitch[1] - CFG.air_pitch[0]) * torch.rand(n, device=device)
        yaw = CFG.air_yaw[0] + (CFG.air_yaw[1] - CFG.air_yaw[0]) * torch.rand(n, device=device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
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
        target_w, target_quat_w = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                            ee_cmd.pose_command_b[:, :3],
                                                            ee_cmd.pose_command_b[:, 3:7])
        ee_now_w = ee_frame.data.target_pos_w[:, 0, :]
        dist = torch.norm(ee_now_w - target_w, dim=-1)                              # fingertip -> reach target
        ee_speed = torch.norm(ee_now_w - prev_ee_w, dim=-1) / step_dt              # fingertip speed (m/s)
        prev_ee_w = ee_now_w.clone()
        if args_cli.vision:
            cube_z = cube_pose()[0][:, 2]   # predicted height (robot root == world origin in this scene)
        else:
            cube_z = obj.data.root_pos_w[:, 2]                                      # cube height

        # plateau tracking (REACH only): count steps since the reach distance last improved
        in_reach = phase == REACH
        improved = in_reach & (dist < best_dist - CFG.handoff_plateau_eps)
        best_dist = torch.where(improved, dist, best_dist)
        plateau_count = torch.where(improved, torch.zeros_like(plateau_count), plateau_count)
        plateau_count = torch.where(in_reach & ~improved, plateau_count + 1.0, plateau_count)

        # --- draw the active phase's target as a colored sphere ---
        grasp_target_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                      obj_cmd.pose_command_b[:, :3])
        use_grasp_vis = (phase == GRASP) | (phase == PAUSE2)
        active_pos = torch.where(use_grasp_vis.unsqueeze(-1), grasp_target_w, target_w)
        midx = torch.zeros(num_envs, dtype=torch.long, device=device)  # blue (reach/pause1)
        midx[use_grasp_vis] = 2                                         # hidden (grasp/pause2)
        midx[phase == CARRY] = 1                                        # green (carry)
        target_vis.visualize(translations=active_pos, marker_indices=midx)

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

            # GRASP -> PAUSE2: the cube has reached the lift target (within a band, both sides).
            # No timeout: a grasp that never lifts just keeps trying until the episode ends.
            lifted = (phase == GRASP) & ((cube_z - CFG.grasp_lift_z).abs() <= CFG.grasp_lift_tol)
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

            # REACH -> PAUSE1: hybrid handoff (whichever fires first):
            #   (a) close AND settled, (b) converged/plateaued and reasonably close, (c) hard timeout.
            reach_steps += (phase == REACH).float()
            settled = (dist < CFG.handoff_dist) & (ee_speed < CFG.handoff_speed)
            plateaued = (plateau_count >= CFG.handoff_plateau_steps) & (dist < CFG.handoff_accept_dist)
            timed_out = reach_steps > CFG.reach_timeout
            done_reach = (phase == REACH) & (settled | plateaued | timed_out)
            phase = torch.where(done_reach, torch.full_like(phase, PAUSE1), phase)
            pause_left = torch.where(done_reach, torch.full_like(pause_left, pause1_steps), pause_left)

        # --vision: keep the reach target glued to the latest prediction while approaching
        # (also covers the one-step camera lag right after a reset)
        if args_cli.vision:
            still_reaching = phase == REACH
            if still_reaching.any():
                set_above_cube_target(still_reaching.nonzero(as_tuple=False).flatten())

        # --- on episode reset: restart at REACH and retarget above the new cube ---
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if args_cli.vision:  # success diagnostics: count before the phase is cleared
                episodes_done += done_ids.numel()
                episodes_carried += int((phase[done_ids] == CARRY).sum())
            phase[done_ids] = REACH
            reach_steps[done_ids] = 0.0
            pause_left[done_ids] = 0
            held_grip[done_ids] = GRIPPER_OPEN
            best_dist[done_ids] = float("inf")
            plateau_count[done_ids] = 0.0
            prev_ee_w[done_ids] = ee_frame.data.target_pos_w[done_ids, 0, :]  # avoid a spurious speed spike
            set_above_cube_target(done_ids)
            if args_cli.vision:
                pct = 100.0 * episodes_carried / episodes_done
                print(f"[episodes] {episodes_carried}/{episodes_done} reached CARRY ({pct:.0f}%)")

        if timestep % 50 == 0:
            cube_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                  obj.data.root_pos_w)
            ph = "reach" if CFG.reach_only else PHASE_NAME[int(phase[0])]
            if args_cli.vision:
                # diagnostics: prediction error vs the true (privileged) cube position
                err_cm = (cube_pose()[0] - cube_b).norm(dim=-1) * 100.0
                print(f"[t {timestep}] env0 phase={ph} dist={dist[0]:.3f}m cube_z_pred={cube_z[0]:.3f} "
                      f"vision_err={err_cm[0]:.2f}cm (mean {err_cm.mean():.2f}cm)")
            else:
                print(f"[t {timestep}] env0 phase={ph} dist={dist[0]:.3f}m ee_speed={ee_speed[0]:.3f} "
                      f"cube_z={cube_z[0]:.3f} cube_b=({cube_b[0, 0]:.2f},{cube_b[0, 1]:.2f},{cube_b[0, 2]:.2f})")
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
