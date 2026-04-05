from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_from_above(
    env: ManagerBasedRLEnv,
    margin: float = 0.04,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the gripper for being above the cube before grasping.

    Returns a tanh-shaped reward that is high when the gripper is directly
    above the object (xy close, z higher by at least `margin`).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # Horizontal distance (xy) between gripper and cube
    xy_distance = torch.norm(ee_pos_w[:, :2] - cube_pos_w[:, :2], dim=1)

    # Reward for being directly above: small xy distance
    xy_reward = 1 - torch.tanh(xy_distance / std)

    # Only reward when gripper is above the cube
    above_cube = ee_pos_w[:, 2] > (cube_pos_w[:, 2] + margin)

    return xy_reward * above_cube.float()
