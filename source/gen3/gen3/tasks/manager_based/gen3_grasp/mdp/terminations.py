from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_lifted_success(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate the episode when the cube is lifted above the success threshold.

    Returns True (terminate) when cube_z > height_threshold (15cm above ground,
    ~10cm above its spawn position). This ends the episode on success so a new
    one starts immediately.
    """
    object: RigidObject = env.scene[object_cfg.name]
    cube_z = object.data.root_pos_w[:, 2]
    return cube_z > height_threshold
