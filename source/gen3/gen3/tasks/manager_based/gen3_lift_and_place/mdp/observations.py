"""Custom observation terms for the Gen3 lift-and-place task.

Provides object_orientation_in_robot_root_frame, which exposes the cube's
orientation as a quaternion in the robot base frame. Identical to the grasp
task's observation of the same name; kept separate to preserve each task's
mdp namespace independence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object in the robot's root frame (quaternion wxyz)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w, object_quat_w
    )
    return object_quat_b
