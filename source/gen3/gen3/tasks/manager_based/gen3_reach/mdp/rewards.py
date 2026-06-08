# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local reward functions for the Gen3 reach task.

These track the *fingertip* grasp point (the space between the 2F-140 fingers),
not the wrist flange. The fingertip is the FrameTransformer "ee_frame" defined in
joint_pos_env_cfg.py as a +0.21 m offset along the end_effector_link z-axis, so its
position differs from the flange while its orientation is identical (pure translation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize L2 distance between the fingertip and the commanded position.

    The command is in the robot-root frame; it is transformed to world before
    comparing with the fingertip world position from the ee_frame sensor.
    Use with a negative weight.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    curr_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def ee_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fine-grained reward for fingertip position using a tanh kernel.

    Returns 1 - tanh(dist / std): a positive bonus that rises sharply as the
    fingertip nears the target. Use with a positive weight.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    curr_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def ee_orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize gripper orientation error vs. the commanded orientation (shortest path).

    The command quaternion is in the robot-root frame; it is rotated into world
    before comparing with the fingertip world orientation (identical to the flange,
    since the ee_frame offset is a pure translation). Use with a negative weight.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def ee_orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fine-grained orientation reward using a tanh kernel.

    Returns 1 - tanh(angle_error / std): a positive bonus that rises sharply as the
    gripper aligns with the commanded orientation. Use with a positive weight.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    angle = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(angle / std)
