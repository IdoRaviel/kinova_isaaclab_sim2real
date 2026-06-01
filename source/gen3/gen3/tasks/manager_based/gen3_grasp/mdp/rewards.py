from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# PPO rewards — wired into RewardManager via joint_pos_env_cfg.py RewTerms.
# =============================================================================


def ee_to_cube_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for 3D distance between EE fingertip center and cube center.

    Returns tanh(dist / std), range [0, 1].
    Use with a negative weight to form a penalty.
    Near 0 when EE is at the cube, near 1 when far away.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    dist = torch.norm(ee_pos - cube_pos, dim=1)
    return torch.tanh(dist / std)


def cube_to_target_distance_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """L2 distance between cube and command target (both in world frame).

    Command is in robot-root frame; we transform it to world before comparing
    to the cube's world position so multi-env spacing doesn't break the metric.
    Constant gradient at all ranges — the coarse "pull" toward the target.
    Use with a negative weight.  [PPO penalty + SAC d_goal component]
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    des_pos_b = env.command_manager.get_command(command_name)[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    return torch.norm(des_pos_w - object.data.root_pos_w, dim=1)


def cube_to_target_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 - tanh(dist / std): positive bonus that rises sharply near zero.

    Use with a positive weight. With std=0.1 the bonus is nearly zero past
    ~20cm and grows steeply inside ~10cm, providing the fine-grained signal
    needed to drive the cube precisely onto the target.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    des_pos_b = env.command_manager.get_command(command_name)[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


# =============================================================================
# SAC rewards — called directly from the SB3 wrapper (not through RewardManager).
# These return raw (unscaled) distances so the wrapper can form the combined
# shaped reward:  r = -(dist_grip + dist_goal) / max_dist
# =============================================================================


def ee_to_cube_distance_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Raw L2 distance between EE fingertip and cube center.  [SAC d_grip component]

    Unlike ee_to_cube_distance (which applies tanh for PPO), this returns the
    unscaled metric so the SAC wrapper can combine it with d_goal and normalise
    by max_dist in one formula.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(ee_pos - cube_pos, dim=1)
