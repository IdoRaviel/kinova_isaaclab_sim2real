# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK-based start-state reset for the grasp task.

At every reset this places the cube at a random pose on the table and uses inverse
kinematics to put the gripper **top-down, directly above the cube center**, with a
varied joint configuration (random IK seed per env). The gripper yaw is aligned to the
cube's nearest face (square-cube symmetry) plus a small jitter.

This trains a grasp policy that is robust to the range of start configurations the reach
policy hands off in the reach->grasp chain, instead of the single canonical pose the
original grasp policy saw.

IK is a hand-written batched damped-least-squares loop over a ``pytorch_kinematics``
forward-kinematics + Jacobian. The arm URDF (``gen3_2f85``) was FK-verified to match the
2F-140 USD exactly; the gripper is irrelevant for arm IK.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from gen3.assets import FINGERTIP_OFFSET_M, KINOVA_GEN3_2F140_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# --- arm description ---
_URDF_PATH = Path(__file__).resolve().parents[4] / "assets" / "urdf" / "gen3_2f85.urdf"
_BASE_LINK = "gen3_base_link"
_EE_LINK = "gen3_end_effector_link"

# arm joint position limits; continuous joints (1,3,5,7) have no URDF limit -> cap to +-pi.
# Naturalness of the configuration is handled by a null-space pull toward canonical in solve().
_PI = math.pi
_LOWER = [-_PI, -2.41, -_PI, -2.66, -_PI, -2.23, -_PI]
_UPPER = [_PI, 2.41, _PI, 2.66, _PI, 2.23, _PI]
# Derived from the robot's own default pose (KINOVA_GEN3_2F140_CFG) rather than a second
# hardcoded copy, so this can never silently drift from the arm's actual spawn/rest pose.
_CANONICAL = [KINOVA_GEN3_2F140_CFG.init_state.joint_pos[f"joint_{i}"] for i in range(1, 8)]


class _ArmIK:
    """Batched damped-least-squares IK for the Gen3 7-DOF arm (cached one per device)."""

    def __init__(self, device: torch.device):
        import pytorch_kinematics as pk

        self.device = device
        with open(_URDF_PATH, "rb") as f:
            self.chain = pk.build_serial_chain_from_urdf(f.read(), _EE_LINK, _BASE_LINK)
        self.chain = self.chain.to(dtype=torch.float32, device=device)
        self.lower = torch.tensor(_LOWER, dtype=torch.float32, device=device)
        self.upper = torch.tensor(_UPPER, dtype=torch.float32, device=device)
        self.canonical = torch.tensor(_CANONICAL, dtype=torch.float32, device=device)
        # top-down orientation: maps the EE local z-axis (approach) to world -z
        self.R_topdown = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _so3_log(R: torch.Tensor) -> torch.Tensor:
        """Rotation matrices (B,3,3) -> rotation vectors (B,3)."""
        c = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1.0) / 2.0
        th = torch.acos(c.clamp(-1.0, 1.0))
        w = torch.stack(
            [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]],
            dim=1,
        )
        s = torch.sin(th)
        f = torch.where(th < 1e-5, torch.full_like(th, 0.5), th / (2.0 * s.clamp_min(1e-9)))
        return f.unsqueeze(1) * w

    @torch.no_grad()
    def solve(
        self,
        target_pos: torch.Tensor,
        target_R: torch.Tensor,
        num_seeds: int = 8,
        iters: int = 120,
        damp: float = 0.08,
        pos_tol: float = 2e-3,
        rot_tol: float = 1e-2,
        seed_noise: float = 0.5,
    ):
        """Solve IK for flange targets in the robot base frame.

        Args:
            target_pos: (N,3) flange target positions (base frame).
            target_R:   (N,3,3) flange target rotations (base frame).
        Returns:
            q:         (N,7) a random *converged* seed per target (canonical-ish if none).
            converged: (N,) bool — whether at least one seed converged.
        """
        n = target_pos.shape[0]
        tp = target_pos.repeat_interleave(num_seeds, dim=0)
        tR = target_R.repeat_interleave(num_seeds, dim=0)
        # Seed near the canonical pose (+ small noise) instead of fully random, so the IK
        # converges to NATURAL arm configs (elbow-up, like the reach policy delivers) rather
        # than contorted branches -- this is what removes the "unwind" at episode start.
        noise = (torch.rand(n * num_seeds, 7, device=self.device) - 0.5) * 2.0 * seed_noise
        q = (self.canonical.unsqueeze(0) + noise).clamp(self.lower, self.upper)
        eye6 = torch.eye(6, device=self.device)
        eye7 = torch.eye(7, device=self.device)
        canon = self.canonical.unsqueeze(0)
        for it in range(iters):
            m = self.chain.forward_kinematics(q).get_matrix()
            p = m[:, :3, 3]
            R = m[:, :3, :3]
            err = torch.cat([tp - p, self._so3_log(tR @ R.transpose(1, 2))], dim=1)
            J = self.chain.jacobian(q)  # (B,6,7); linear rows 0:3, angular rows 3:6
            JT = J.transpose(1, 2)
            jpinv = JT @ torch.linalg.inv(J @ JT + (damp * damp) * eye6)  # (B,7,6)
            dq_task = (jpinv @ err.unsqueeze(-1)).squeeze(-1)
            # null-space pull toward canonical (keeps redundant joints natural); decays to 0 so
            # the task still converges tightly at the end.
            k_null = 0.6 * (1.0 - it / iters)
            null_proj = eye7 - jpinv @ J  # (B,7,7)
            dq_null = (null_proj @ (canon - q).unsqueeze(-1)).squeeze(-1)
            q = torch.clamp(q + dq_task + k_null * dq_null, self.lower, self.upper)

        m = self.chain.forward_kinematics(q).get_matrix()
        pe = (m[:, :3, 3] - tp).norm(dim=1)
        re = self._so3_log(tR @ m[:, :3, :3].transpose(1, 2)).norm(dim=1)
        conv = ((pe < pos_tol) & (re < rot_tol)).view(n, num_seeds)
        q = q.view(n, num_seeds, 7)
        # Among the converged seeds, pick the one CLOSEST to canonical -> the natural arm
        # branch (no flipped wrist / wound base), which avoids the "unwind" spin at reset.
        dist_canon = (q - self.canonical).norm(dim=2)  # (n, num_seeds)
        score = torch.where(conv, dist_canon, torch.full_like(dist_canon, 1e9))
        pick = score.argmin(dim=1)
        q_best = q[torch.arange(n, device=self.device), pick]
        return q_best, conv.any(dim=1)


_IK_CACHE: dict[str, _ArmIK] = {}


def _get_ik(device) -> _ArmIK:
    """Return the singleton _ArmIK for the given device, creating it on first call."""
    key = str(device)
    if key not in _IK_CACHE:
        _IK_CACHE[key] = _ArmIK(torch.device(device))
    return _IK_CACHE[key]


def _rot_z(yaw: torch.Tensor) -> torch.Tensor:
    """Batched rotation about world z: (N,) -> (N,3,3)."""
    o = torch.zeros_like(yaw)
    i = torch.ones_like(yaw)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([c, -s, o], dim=1),
            torch.stack([s, c, o], dim=1),
            torch.stack([o, o, i], dim=1),
        ],
        dim=1,
    )


# Precomputed (cube_local_pose (M,7), arm_joints (M,7)) start-state tables, one per device.
_TABLE_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@torch.no_grad()
def _build_start_table(
    ik: _ArmIK,
    cube_x,
    cube_y,
    cube_z,
    yaw_range,
    hover,
    tool_offset,
    jitter_deg,
    num_seeds,
    n_samples=10000,
    chunk=2500,
):
    """Run the IK once to precompute a table of valid start states.

    Returns (cube_pose (M,7) local [pos+quat], arm_joints (M,7)) for the converged samples.
    Built in chunks to keep peak memory low (shares the GPU with Isaac Sim).
    """
    device = ik.device
    half_pi = math.pi / 2.0
    cube_poses, arm_qs = [], []
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        cx = cube_x[0] + (cube_x[1] - cube_x[0]) * torch.rand(b, device=device)
        cy = cube_y[0] + (cube_y[1] - cube_y[0]) * torch.rand(b, device=device)
        cyaw = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * torch.rand(b, device=device)
        # gripper yaw: nearest cube face (square symmetry) + jitter
        face = cyaw - half_pi * torch.round(cyaw / half_pi)
        jitter = (torch.rand(b, device=device) - 0.5) * 2.0 * math.radians(jitter_deg)
        target_pos = torch.stack(
            [cx, cy, torch.full_like(cx, cube_z + hover + tool_offset)], dim=1
        )
        target_R = _rot_z(face + jitter) @ ik.R_topdown
        q, conv = ik.solve(target_pos, target_R, num_seeds=num_seeds)
        cx, cy, cyaw, q = cx[conv], cy[conv], cyaw[conv], q[conv]
        quat = torch.zeros(cx.shape[0], 4, device=device)
        quat[:, 0] = torch.cos(cyaw / 2.0)
        quat[:, 3] = torch.sin(cyaw / 2.0)
        cube_poses.append(
            torch.cat([torch.stack([cx, cy, torch.full_like(cx, cube_z)], dim=1), quat], dim=1)
        )
        arm_qs.append(q)
        done += b
    return torch.cat(cube_poses, dim=0), torch.cat(arm_qs, dim=0)


def reset_above_cube_ik(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_x: tuple[float, float] = (0.35, 0.55),
    cube_y: tuple[float, float] = (-0.30, 0.30),
    cube_z: float = 0.055,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    hover: float = 0.12,
    tool_offset: float = FINGERTIP_OFFSET_M,
    jitter_deg: float = 15.0,
    num_seeds: int = 4,
    cube_jitter: float = 0.0,
):
    """Reset event: place cube at a random pose and IK the arm top-down above it.

    Args:
        cube_x:      (min, max) range for the cube x position in the robot base frame (m).
        cube_y:      (min, max) range for the cube y position in the robot base frame (m).
        cube_z:      Fixed cube spawn height (m); default 0.055 matches the table surface.
        yaw_range:   (min, max) range for the cube yaw (rad).
        hover:       Gripper hover height above the cube top (m).
        tool_offset: Distance from the wrist flange to the fingertip grasp point (m).
                     Must match the ee_frame offset in the env config
                     (``gen3.assets.FINGERTIP_OFFSET_M``).
        jitter_deg:  Random yaw jitter added on top of the nearest-face snap (degrees).
        num_seeds:   IK seeds tried per target; more seeds -> higher convergence rate.
        cube_jitter: Max lateral offset applied to the cube after table placement (m),
                     so the cube is not exactly under the gripper at reset start.

    The IK is run once at startup to build a table of valid start states; each
    reset samples a row, so per-reset cost is negligible (one index + two state writes).
    """
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["object"]
    key = str(device)
    if key not in _TABLE_CACHE:
        print("[ik-reset] building grasp start-state table (one-time IK)...")
        _TABLE_CACHE[key] = _build_start_table(
            _get_ik(device), cube_x, cube_y, cube_z, yaw_range, hover, tool_offset,
            jitter_deg, num_seeds,
        )
        print(f"[ik-reset] start-state table ready: {_TABLE_CACHE[key][0].shape[0]} states")
    cube_pose_tab, q_tab = _TABLE_CACHE[key]

    n = len(env_ids)
    idx = torch.randint(0, cube_pose_tab.shape[0], (n,), device=device)
    q_arm = q_tab[idx]
    cube_local = cube_pose_tab[idx]

    # --- write arm joints (joint_1..7) ---
    arm_ids = [robot.joint_names.index(f"joint_{i}") for i in range(1, 8)]
    robot.write_joint_state_to_sim(
        position=q_arm,
        velocity=torch.zeros_like(q_arm),
        joint_ids=arm_ids,
        env_ids=env_ids,
    )

    # --- write cube pose (world frame = env origin + local) ---
    cube_pos = env.scene.env_origins[env_ids] + cube_local[:, :3]
    # nudge the cube laterally so it's NOT exactly under the gripper -> the policy must
    # localize and adjust each episode (robustness for the reach->grasp handoff).
    if cube_jitter > 0.0:
        cube_pos[:, :2] += (torch.rand(n, 2, device=device) - 0.5) * 2.0 * cube_jitter
    obj.write_root_pose_to_sim(
        torch.cat([cube_pos, cube_local[:, 3:7]], dim=1), env_ids=env_ids
    )
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
