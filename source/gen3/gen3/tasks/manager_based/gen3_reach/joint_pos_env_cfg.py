# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from . import mdp
from .agents.rsl_rl_ppo_cfg import Gen3ReachPPORunnerCfg as _RunnerCfg

##
# Pre-defined configs
##
from gen3.assets import KINOVA_GEN3_2F140_CFG  # isort: skip

##
# Environment configuration
##


@configclass
class Gen3ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to gen3 with 2f-140 gripper
        self.scene.robot = KINOVA_GEN3_2F140_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # override events — use offset reset so joint_7=0 still gets randomized
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.4, 0.4), "velocity_range": (0.0, 0.0)},
        )
        # override rewards — track the fingertip grasp point (ee_frame), not the
        # wrist flange. Custom ee_frame-based functions measure fingertip position;
        # orientation is identical at the fingertip but uses the ee_frame variant too.
        self.rewards.end_effector_position_tracking = RewTerm(
            func=mdp.ee_position_command_error,
            weight=-0.45,
            params={
                "command_name": "ee_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.ee_position_command_error_tanh,
            weight=0.25,
            params={
                "command_name": "ee_pose",
                "std": 0.15,  # tanh width (m): fine-grained position bonus, grows as the fingertip nears the target
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.rewards.end_effector_orientation_tracking = RewTerm(
            func=mdp.ee_orientation_command_error,
            weight=-0.2,  # coarse orientation-error penalty (shortest-path angle)
            params={
                "command_name": "ee_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
            func=mdp.ee_orientation_command_error_tanh,
            weight=0.30,
            params={
                "command_name": "ee_pose",
                "std": 0.5,  # tanh width (rad): fine-grained orientation bonus across the full angular range
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        # override actions — arm joints only, gripper excluded
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        self.commands.ee_pose.body_name = "end_effector_link"
        # Orientation convention (blue/z = gripper approach axis toward the fingertips):
        #   pitch=pi   -> approach points straight DOWN (top-down grasp)
        #   pitch=pi/2 -> approach points FORWARD (horizontal)
        #   yaw        -> swings the approach direction left/right (to the sides)
        # Ranges span forward (pitch=pi/2) to straight-down (pitch=pi) with +/-90deg yaw,
        # covering top-down grasp poses and angled place poses.
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi)
        self.commands.ee_pose.ranges.yaw = (-math.pi / 2, math.pi / 2)
        # target height range (m): floor at 0.05 reaches table / cube-grasp height (cube sits at z≈0.055)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.50)
        # Smoothness penalties (action-rate and joint-velocity) that suppress wavering.
        # They activate at 50% of training so the policy first learns to reach and orient,
        # then to hold still; num_steps scales with the run length.
        _curriculum_steps = int(
            _RunnerCfg().max_iterations * _RunnerCfg().num_steps_per_env * 0.5
        )
        self.curriculum.action_rate.params["weight"] = -0.01
        self.curriculum.action_rate.params["num_steps"] = _curriculum_steps
        self.curriculum.joint_vel.params["weight"] = -0.003
        self.curriculum.joint_vel.params["num_steps"] = _curriculum_steps
        # Target pose: frame arrows (RGB axes) — goal position + orientation.
        goal_marker_cfg = FRAME_MARKER_CFG.copy()
        goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_marker_cfg.prim_path = "/Visuals/Command/goal_pose"
        self.commands.ee_pose.goal_pose_visualizer_cfg = goal_marker_cfg
        # Gripper orientation: frame arrows at end_effector_link (flange). Orientation
        # matches the fingertip; compare these against the target arrows for success.
        current_marker_cfg = FRAME_MARKER_CFG.copy()
        current_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        current_marker_cfg.prim_path = "/Visuals/Command/current_pose"
        self.commands.ee_pose.current_pose_visualizer_cfg = current_marker_cfg

        # EE frame: +0.21 m offset from the wrist flange to the fingertip grasp
        # point, shown as a small sphere. Also drives the position/orientation rewards.
        ee_marker_cfg = SPHERE_MARKER_CFG.copy()
        ee_marker_cfg.markers["sphere"].radius = 0.01
        ee_marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gen3n7_instanceable/end_effector_link",
            debug_vis=True,
            visualizer_cfg=ee_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gen3n7_instanceable/end_effector_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.21]),
                ),
            ],
        )
