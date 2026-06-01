from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _asset(env: ManagerBasedRLEnv, cfg: SceneEntityCfg):
    return env.scene[cfg.name]


def _body_pos_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    asset = _asset(env, cfg)
    if cfg.body_ids is not None:
        body_id = cfg.body_ids[0]
    else:
        body_ids, _ = asset.find_bodies(cfg.body_names)
        body_id = body_ids[0]
    return asset.data.body_pos_w[:, body_id, :3]


def _body_quat_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    asset = _asset(env, cfg)
    if cfg.body_ids is not None:
        body_id = cfg.body_ids[0]
    else:
        body_ids, _ = asset.find_bodies(cfg.body_names)
        body_id = body_ids[0]
    return asset.data.body_quat_w[:, body_id, :]


def _object_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3]


def _object_vel_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w[:, :3]


def _command_goal_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        des_pos_b,
    )
    return des_pos_w


def _finger_features(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
):
    left_pos = _body_pos_w(env, left_finger_cfg)
    right_pos = _body_pos_w(env, right_finger_cfg)
    midpoint = 0.5 * (left_pos + right_pos)
    gap_vec = right_pos - left_pos
    gap = torch.norm(gap_vec, dim=-1)
    return left_pos, right_pos, midpoint, gap_vec, gap


def _finger_closed_amount(
    env: ManagerBasedRLEnv,
    closed_joint_pos: float = 0.70,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    try:
        finger_idx = robot.joint_names.index("finger_joint")
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device)

    q = robot.data.joint_pos[:, finger_idx]
    return torch.clamp(q / closed_joint_pos, 0.0, 1.0)


def _bilateral_grasp_proxy(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    cube_length: float,
    closed_joint_pos: float = 0.70,
) -> torch.Tensor:
    """Proxy for the contact-grasp reward from ReachCubePick.

    The reference repo uses contact sensors on both fingers. To avoid crashing if
    contact sensors are not present in your current scene, this uses bilateral
    finger proximity + gripper closure as a grasp proxy.
    """
    left_pos, right_pos, midpoint, _, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)
    cube_pos = _object_pos_w(env, object_cfg)

    half = cube_length * 0.5
    std = cube_length * 0.35

    d_left = torch.clamp(torch.norm(left_pos - cube_pos, dim=-1) - half, min=0.0)
    d_right = torch.clamp(torch.norm(right_pos - cube_pos, dim=-1) - half, min=0.0)

    left_touch = torch.exp(-d_left / std)
    right_touch = torch.exp(-d_right / std)
    bilateral = torch.minimum(left_touch, right_touch)

    closed = _finger_closed_amount(env, closed_joint_pos=closed_joint_pos)
    return bilateral * closed


# ---------------------------------------------------------------------
# ReachCubePick-style reward terms
# ---------------------------------------------------------------------

def native_finger_midpoint_to_target_reward(
    env: ManagerBasedRLEnv,
    std_dist: float,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Approach reward: midpoint between fingers to cube center."""
    _, _, midpoint, _, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)
    target_pos = _object_pos_w(env, target_asset_cfg)
    dist = torch.norm(midpoint - target_pos, dim=-1)
    return torch.exp(-dist / std_dist)


def finger_symmetry_reward(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    std_grasp: float,
) -> torch.Tensor:
    """Reward both fingers being similarly distant from cube center."""
    target_pos = _object_pos_w(env, target_asset_cfg)
    left_pos, right_pos, _, _, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)

    l_dist = torch.norm(left_pos - target_pos, dim=-1)
    r_dist = torch.norm(right_pos - target_pos, dim=-1)
    diff = torch.abs(l_dist - r_dist)

    return torch.exp(-diff / std_grasp)


def finger_opposition_reward(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    opposition_scale: float = 3.0,
) -> torch.Tensor:
    """Reward fingers lying on opposite sides of the cube."""
    target_pos = _object_pos_w(env, target_asset_cfg)
    left_pos, right_pos, _, _, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)

    v_left = left_pos - target_pos
    v_right = right_pos - target_pos

    v_left_n = v_left / (torch.norm(v_left, dim=-1, keepdim=True) + 1e-6)
    v_right_n = v_right / (torch.norm(v_right, dim=-1, keepdim=True) + 1e-6)

    dot = (v_left_n * v_right_n).sum(dim=-1)

    # ideal: dot ≈ -1
    return torch.exp(-torch.abs(dot + 1.0) * opposition_scale)


def finger_height_alignment_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_height: float = 0.02,
) -> torch.Tensor:
    """Reward side grasp: finger midpoint height near cube center height."""
    target_pos = _object_pos_w(env, target_asset_cfg)
    _, _, midpoint, _, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)

    height_diff = torch.abs(midpoint[:, 2] - target_pos[:, 2])
    return torch.exp(-height_diff / std_height)


def finger_closure_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    target_width: float,
    activation_dist: float,
    std_gap: float,
    gap_closure_factor: float = 0.99,
) -> torch.Tensor:
    """Reward closing the gripper to cube width only near the cube."""
    _, _, midpoint, _, gap = _finger_features(env, left_finger_cfg, right_finger_cfg)
    cube_pos = _object_pos_w(env, target_asset_cfg)

    dist = torch.norm(midpoint - cube_pos, dim=-1)
    target_gap = target_width * gap_closure_factor
    gap_error = torch.abs(gap - target_gap)

    closure = torch.clamp(1.0 - gap_error / std_gap, min=0.0, max=1.0)
    return torch.where(dist <= activation_dist, closure, torch.zeros_like(closure))


def contact_grasp_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    cube_length: float,
    closed_joint_pos: float = 0.70,
) -> torch.Tensor:
    """Bilateral grasp proxy.

    ReachCubePick uses real ContactSensor force on both fingers. This proxy is
    safer for your current scene because your scene may not have those sensors
    wired yet.
    """
    return _bilateral_grasp_proxy(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        object_cfg=target_asset_cfg,
        cube_length=cube_length,
        closed_joint_pos=closed_joint_pos,
    )


def finger_line_horizontal_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    std_angle: float = 0.3,
) -> torch.Tensor:
    """Reward the line between fingertips being approximately horizontal."""
    left_pos, right_pos, _, gap_vec, _ = _finger_features(env, left_finger_cfg, right_finger_cfg)

    horizontal_norm = torch.norm(gap_vec[:, :2], dim=-1)
    vertical_abs = torch.abs(gap_vec[:, 2])

    angle = torch.atan2(vertical_abs, horizontal_norm + 1e-6)
    return torch.exp(-(angle ** 2) / std_angle)


def wrist_outside_normal_to_target_reward(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_angle: float = 0.5,
) -> torch.Tensor:
    """Approximate wrist orientation reward.

    It rewards one wrist axis pointing toward the cube. The exact useful axis
    may differ across USDs; this uses the EE local +Y axis as in many gripper setups.
    """
    ee_pos = _body_pos_w(env, ee_link_cfg)
    ee_quat = _body_quat_w(env, ee_link_cfg)
    target_pos = _object_pos_w(env, target_asset_cfg)

    # Convert local Y axis [0,1,0] by quaternion.
    # Quaternion format in IsaacLab is usually wxyz.
    w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]

    # Rotation matrix column for local Y axis.
    y_axis = torch.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        dim=-1,
    )

    to_target = target_pos - ee_pos
    to_target = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-6)

    dot = (y_axis * to_target).sum(dim=-1).clamp(-1.0, 1.0)

    # Reward either +Y or -Y pointing toward cube, because USD axis convention may flip.
    alignment = torch.abs(dot)
    return torch.exp(-((1.0 - alignment) ** 2) / std_angle)


def position_command_error_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    max_track_dist: float = 1.2,
    dist_sigma: float = 0.12,
    cube_length: float = 0.05,
    closed_joint_pos: float = 0.70,
    grasp_threshold: float = 0.25,
) -> torch.Tensor:
    """Transport reward: cube to command, gated by grasp proxy.

    This mirrors ReachCubePick's cube_command_dist term, where cube-target
    progress is only paid when bilateral contact/grasp exists.
    """
    cube_pos = _object_pos_w(env, asset_cfg)
    goal_pos = _command_goal_w(env, command_name, robot_cfg)

    dist = torch.norm(cube_pos - goal_pos, dim=-1)

    far_reward = torch.clamp(1.0 - dist / max_track_dist, min=0.0, max=1.0)
    near_reward = torch.exp(-dist / dist_sigma)
    dist_reward = far_reward + near_reward

    grasp = _bilateral_grasp_proxy(
        env,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        object_cfg=asset_cfg,
        cube_length=cube_length,
        closed_joint_pos=closed_joint_pos,
    )

    grasp_gate = (grasp > grasp_threshold).float()
    return dist_reward * grasp_gate


def asset_vel_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    cube_length: float,
) -> torch.Tensor:
    """Reward cube velocity pointing toward the command target."""
    cube_pos = _object_pos_w(env, asset_cfg)
    cube_vel = _object_vel_w(env, asset_cfg)
    goal_pos = _command_goal_w(env, command_name, robot_cfg)

    to_goal = goal_pos - cube_pos
    dist = torch.norm(to_goal, dim=-1, keepdim=True)
    direction = to_goal / (dist + 1e-6)

    speed_toward_goal = (cube_vel * direction).sum(dim=-1)
    reward = torch.clamp(speed_toward_goal / 0.5, min=0.0, max=1.0)

    close = (dist.squeeze(-1) < cube_length * 0.5)
    return torch.where(close, torch.ones_like(reward), reward)


def joint_limit_distance_clamped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.15,
) -> torch.Tensor:
    """Penalty near joint limits, adapted from ReachCubePick."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids is not None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = list(range(min(7, asset.data.joint_pos.shape[1])))

    limits = asset.data.soft_joint_pos_limits[:, joint_ids]
    lower = limits[:, :, 0]
    upper = limits[:, :, 1]
    current = asset.data.joint_pos[:, joint_ids]

    dist_to_lower = torch.clamp(margin - (current - lower), min=0.0)
    dist_to_upper = torch.clamp(margin - (upper - current), min=0.0)

    return (dist_to_lower ** 2 + dist_to_upper ** 2).sum(dim=1)


def object_lifted_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    lift_height: float,
) -> torch.Tensor:
    cube_pos = _object_pos_w(env, object_cfg)
    return (cube_pos[:, 2] > lift_height).float()


def object_goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    lift_height: float,
    goal_threshold: float,
) -> torch.Tensor:
    cube_pos = _object_pos_w(env, object_cfg)
    goal_pos = _command_goal_w(env, command_name, robot_cfg)

    dist = torch.norm(cube_pos - goal_pos, dim=-1)
    lifted = cube_pos[:, 2] > lift_height

    return ((dist < goal_threshold) & lifted).float()
