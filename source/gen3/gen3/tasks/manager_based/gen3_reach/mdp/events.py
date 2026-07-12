# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reset events for the Gen3 reach task.

Provides two reset events used in Gen3ReachEnvCfg:
  reset_joints_held_uniform — randomize a joint AND hold it with the PD drive target
                               (needed for the gripper so it does not snap back).
  reset_joints_by_offset    — standard offset reset, re-exported from Isaac Lab MDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_held_uniform(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the given joints to a uniform-random position AND set their drive target to match.

    Unlike ``reset_joints_by_offset`` (which only writes the joint *state*), this also writes
    the drive *target*, so a non-actioned joint with a stiff PD drive — e.g. the gripper
    finger — actually HOLDS the sampled position instead of the drive snapping it back to the
    default. Used to randomize the gripper open<->closed so the finger observation covers the
    carry case.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(robot.num_joints))
    lo, hi = position_range
    pos = lo + (hi - lo) * torch.rand(len(env_ids), len(joint_ids), device=robot.device)
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
