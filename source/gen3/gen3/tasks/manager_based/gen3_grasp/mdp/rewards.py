from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_cube_alignment(
    env: ManagerBasedRLEnv,
    xy_std: float = 0.08,
    z_std: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for having the cube centered between the gripper fingers.

    Rewards when the EE (fingertip center) is aligned with the cube both
    horizontally (XY) and vertically (Z). The wide std (0.08) makes this
    work as both a long-range pull and a short-range precision reward,
    replacing the separate reaching_object term.

    R = (1 - tanh(xy_dist / xy_std)) * (1 - tanh(z_dist / z_std))
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos = object.data.root_pos_w                   # (N, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]      # (N, 3)

    # XY: horizontal distance between EE center and cube center
    xy_dist = torch.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=1)
    xy_reward = 1 - torch.tanh(xy_dist / xy_std)

    # Z: EE fingertip level should match cube center height
    z_dist = torch.abs(ee_pos[:, 2] - cube_pos[:, 2])
    z_reward = 1 - torch.tanh(z_dist / z_std)

    return xy_reward * z_reward


def object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for EE approach to the cube using 3D distance tanh.

    R = 1 - tanh(dist / std)
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos = object.data.root_pos_w                   # (N, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]      # (N, 3)

    dist = torch.norm(ee_pos - cube_pos, dim=1)
    return 1 - torch.tanh(dist / std)


def finger_closure_when_aligned(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward finger closure only when the gripper is aligned with the cube.

    The gate prevents the agent from learning to always keep fingers closed,
    which would block it from positioning correctly in the first place.

    R = is_aligned * (left_knuckle_pos / 0.8)
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    cube_pos = object.data.root_pos_w                   # (N, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]      # (N, 3)

    # Gate: is the EE close enough to the cube?
    distance = torch.norm(ee_pos - cube_pos, dim=1)
    is_aligned = (distance < distance_threshold).float()

    # Finger closure: left knuckle joint goes from 0 (open) to 0.8 (closed)
    left_knuckle_idx = robot.find_joints("gen3_robotiq_85_left_knuckle_joint")[0][0]
    closure = robot.data.joint_pos[:, left_knuckle_idx] / 0.8  # normalise to [0, 1]
    closure = torch.clamp(closure, 0.0, 1.0)

    return is_aligned * closure


def dense_finger_closure(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward finger closure weighted by distance to the object.

    R = (left_knuckle_pos / 0.8) * (1 - tanh(dist / std))
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    cube_pos = object.data.root_pos_w                   # (N, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]      # (N, 3)

    # Distance pull
    dist = torch.norm(ee_pos - cube_pos, dim=1)
    dist_reward = 1 - torch.tanh(dist / std)

    # Finger closure
    left_knuckle_idx = robot.find_joints("gen3_robotiq_85_left_knuckle_joint")[0][0]
    closure = robot.data.joint_pos[:, left_knuckle_idx] / 0.8
    closure = torch.clamp(closure, 0.0, 1.0)

    return closure * dist_reward


def object_height_reward(
    env: ManagerBasedRLEnv,
    table_z: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward proportional to how high the cube is above the table.

    Even 1mm of lift gives some reward, providing a continuous gradient
    from the table surface all the way to the success threshold.
    The reward is zero when the cube is at or below table level.

    R = max(0, cube_z - table_z)
    """
    object: RigidObject = env.scene[object_cfg.name]
    cube_z = object.data.root_pos_w[:, 2]
    return torch.clamp(cube_z - table_z, min=0.0)


def object_lifted_binary(
    env: ManagerBasedRLEnv,
    threshold: float = 0.04,
    table_z: float = 0.055,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward when the cube clears a specific height threshold.

    R = 1.0 if (cube_z - table_z) > threshold else 0.0
    """
    object: RigidObject = env.scene[object_cfg.name]
    cube_z = object.data.root_pos_w[:, 2]
    return (cube_z - table_z > threshold).float()


def cube_lifted_bonus(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-time bonus when the cube is lifted above the success threshold.

    Returns 1.0 when cube_z > height_threshold (15cm above ground,
    ~10cm above its spawn position). Combined with the success termination
    at the same height, this fires exactly once per successful episode.

    R = 1.0 if cube_z > height_threshold else 0.0
    """
    object: RigidObject = env.scene[object_cfg.name]
    cube_z = object.data.root_pos_w[:, 2]
    return torch.where(cube_z > height_threshold, 1.0, 0.0)


_GRASP_TICKS = {}


def stable_grasp_duration(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    duration: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary reward if a stable grasp is maintained for at least 'duration' seconds.

    A stable grasp is defined as the EE being close to the object and fingers being closed.
    """
    if env not in _GRASP_TICKS:
        _GRASP_TICKS[env] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Reset ticks for environments that were reset
    _GRASP_TICKS[env][env.reset_buf] = 0

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    cube_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    # Distance between EE and object
    dist = torch.norm(ee_pos - cube_pos, dim=1)
    is_close = dist < threshold

    # Fingers are closed
    # gen3_robotiq_85_left_knuckle_joint goes from 0 to 0.8
    left_knuckle_idx = robot.find_joints("gen3_robotiq_85_left_knuckle_joint")[0][0]
    is_closed = robot.data.joint_pos[:, left_knuckle_idx] > 0.4  # half way closed

    is_grasping = is_close & is_closed

    # Update ticks: increment if grasping, reset if not
    _GRASP_TICKS[env] += 1
    _GRASP_TICKS[env][~is_grasping] = 0

    # How many ticks are needed for 'duration' seconds?
    needed_ticks = int(duration / env.step_dt)
    return (_GRASP_TICKS[env] >= needed_ticks).float()
