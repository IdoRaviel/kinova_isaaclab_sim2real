# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from . import mdp

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
        # override rewards — same as base, point to end_effector_link
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_position_tracking.weight = -0.5
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.25
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["end_effector_link"]
        self.rewards.end_effector_orientation_tracking.weight = 0.0
        # override actions — arm joints only, gripper excluded
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        self.commands.ee_pose.body_name = "end_effector_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.ee_pose.ranges.pos_z = (0.30, 0.65)
        # goal as a small green sphere, no current-EE arrows
        self.commands.ee_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
        self.commands.ee_pose.current_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/current_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.001,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    visible=False,
                )
            },
        )
