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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
        # override events — use offset reset so joint_7=0 still gets randomized.
        # Scoped to the ARM joints so it doesn't also jiggle the finger (gripper is handled below).
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.4, 0.4),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]"]),
            },
        )
        # Gripper state: random open(0) -> closed(0.7) each reset, so the finger-joint
        # observation covers the carry case (gripper closed), not just open-ish.
        self.events.reset_gripper_state = EventTerm(
            func=mdp.reset_joints_held_uniform,
            mode="reset",
            params={
                "position_range": (0.0, 0.7),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            },
        )
        # Payload randomization: ADD a random mass on top of the gripper's own mass each reset,
        # so the policy reaches accurately whether empty or carrying the cube (chain's 2nd reach).
        # operation="add" -> robotiq_base_link's 0.289 kg stays; we add 0 (empty) .. 0.3 kg on top.
        # Cube measured at 0.216 kg, so it sits inside the trained span. (end_effector_link is
        # massless, hence robotiq_base_link.) Policy doesn't observe mass -> load-robust controller.
        self.events.randomize_gripper_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["robotiq_base_link"]),
                "mass_distribution_params": (0.0, 0.3),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        # override rewards — track the fingertip grasp point (ee_frame), not the
        # wrist flange. Custom ee_frame-based functions measure fingertip position;
        # orientation is identical at the fingertip but uses the ee_frame variant too.
        self.rewards.end_effector_position_tracking = RewTerm(
            func=mdp.ee_position_command_error,
            weight=-0.6,
            params={
                "command_name": "ee_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.ee_position_command_error_tanh,
            weight=0.6,
            params={
                "command_name": "ee_pose",
                "std": 0.07,  # tanh width (m): fine-grained position bonus, grows as the fingertip nears the target
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        self.rewards.end_effector_orientation_tracking = RewTerm(
            func=mdp.ee_orientation_command_error,
            weight=-0.3,  # coarse orientation-error penalty (shortest-path angle)
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
        # Orientation ranges (base-frame RPY; verified against quat_from_euler_xyz):
        #   pitch (base Y) -> tilts the approach (blue) forward<->back: pi=straight DOWN, pi/2=FORWARD.
        #                     keep the DOWN end at pi (the chain's first reach needs exact top-down).
        #   yaw   (base Z) -> WRIST TWIST: spins the gripper about the (downward) approach axis.
        #   roll  (base X) -> tilts the approach SIDEWAYS (left/right lean).
        self.commands.ee_pose.ranges.roll = (
            -math.pi / 18,
            math.pi / 18,
        )  # +/-10 deg: a small sideways approach tilt (reduced)
        self.commands.ee_pose.ranges.pitch = (
            math.pi / 2 + math.pi / 6,
            math.pi,
        )  # 120 deg (less forward) .. straight DOWN (pi kept); forward tilt reduced
        self.commands.ee_pose.ranges.yaw = (
            -math.pi / 4,
            math.pi / 4,
        )  # +/-45 deg twist: covers all 4 faces of the square cube
        # target height range (m): floor at 0.05 reaches table / cube-grasp height (cube sits at z≈0.055)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.50)
        # Smoothness penalties: OFF for the first 25% (learn to reach freely), then ramp
        # LINEARLY up to max between 25% and 60% (ease in, no shock), max for the rest.
        self.rewards.action_rate.weight = 0.0
        _total = _RunnerCfg().max_iterations * _RunnerCfg().num_steps_per_env
        _start, _end = int(_total * 0.25), int(_total * 0.6)
        self.curriculum.action_rate = CurrTerm(
            func=mdp.modify_reward_weight_ramp,
            params={
                "term_name": "action_rate",
                "start_weight": 0.0,
                "end_weight": -0.005,
                "start_step": _start,
                "end_step": _end,
            },
        )
        # joint velocity: global penalty, ramped 0 -> max over 25%-60% (matches action_rate).
        self.rewards.joint_vel.weight = 0.0
        self.curriculum.joint_vel = CurrTerm(
            func=mdp.modify_reward_weight_ramp,
            params={
                "term_name": "joint_vel",
                "start_weight": 0.0,
                "end_weight": -0.0015,
                "start_step": _start,
                "end_step": _end,
            },
        )
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
