import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm, SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from gen3.assets import KINOVA_GEN3_2F140_CFG  # isort: skip


@configclass
class Gen3LiftAndPlaceEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Gen3+gripper as robot
        self.scene.robot = KINOVA_GEN3_2F140_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Set actions for Gen3 arm (7 joints, continuous)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.2,  # Reduced to stop the "smashing" from Day 1
            use_default_offset=True,
        )
        # Set actions for Robotiq gripper (binary open/close)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.700},
        )

        # Set the body name for the command generator
        self.commands.object_pose.body_name = "end_effector_link"

        # Set cube as object — random position on table
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Add object orientation to observations
        self.observations.policy.object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_root_frame
        )

        # ReachCubePick-style PPO reward design adapted to Kinova Gen3 + Robotiq 2F140.
        #
        # Based on:
        #   midpoint-to-cube approach
        #   bilateral grasp/contact
        #   finger symmetry/opposition
        #   finger horizontal/height alignment
        #   wrist orientation
        #   cube-to-command transport progress
        #   curriculum increasing grasp and transport rewards

        left_finger_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_inner_finger"],
        )
        right_finger_cfg = SceneEntityCfg(
            "robot",
            body_names=["right_inner_finger"],
        )
        ee_link_cfg = SceneEntityCfg(
            "robot",
            body_names=["end_effector_link"],
        )
        object_cfg = SceneEntityCfg("object")
        robot_cfg = SceneEntityCfg("robot")

        cube_length = 0.05

        self.rewards.gripper_cube_dist = RewTerm(
            func=mdp.native_finger_midpoint_to_target_reward,
            weight=5.0,
            params={
                "std_dist": 0.15,
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "target_asset_cfg": object_cfg,
            },
        )

        self.rewards.finger_symmetry = RewTerm(
            func=mdp.finger_symmetry_reward,
            weight=3.0,
            params={
                "std_grasp": cube_length / 4.0,
                "target_asset_cfg": object_cfg,
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
            },
        )

        self.rewards.finger_opposition = RewTerm(
            func=mdp.finger_opposition_reward,
            weight=3.0,
            params={
                "target_asset_cfg": object_cfg,
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "opposition_scale": 3.0,
            },
        )

        self.rewards.finger_height_alignment = RewTerm(
            func=mdp.finger_height_alignment_reward,
            weight=3.0,
            params={
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "target_asset_cfg": object_cfg,
                "std_height": cube_length / 4.0,
            },
        )

        self.rewards.contact_grasp = RewTerm(
            func=mdp.contact_grasp_reward,
            weight=5.0,
            params={
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "target_asset_cfg": object_cfg,
                "cube_length": cube_length,
                "closed_joint_pos": 0.70,
            },
        )

        self.rewards.wrist_outside_normal_to_target = RewTerm(
            func=mdp.wrist_outside_normal_to_target_reward,
            weight=3.0,
            params={
                "ee_link_cfg": ee_link_cfg,
                "target_asset_cfg": object_cfg,
                "std_angle": 0.5,
            },
        )

        self.rewards.finger_line_horizontal = RewTerm(
            func=mdp.finger_line_horizontal_reward,
            weight=3.0,
            params={
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "std_angle": 0.3,
            },
        )

        self.rewards.finger_closure = RewTerm(
            func=mdp.finger_closure_reward,
            weight=5.0,
            params={
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "target_asset_cfg": object_cfg,
                "target_width": cube_length,
                "activation_dist": cube_length * 0.75,
                "std_gap": cube_length * 1.5,
            },
        )

        # Transport rewards start at 0 and are increased by curriculum.
        self.rewards.cube_command_dist = RewTerm(
            func=mdp.position_command_error_progress,
            weight=0.0,
            params={
                "command_name": "object_pose",
                "asset_cfg": object_cfg,
                "robot_cfg": robot_cfg,
                "left_finger_cfg": left_finger_cfg,
                "right_finger_cfg": right_finger_cfg,
                "max_track_dist": 1.2,
                "dist_sigma": 0.12,
                "cube_length": cube_length,
                "closed_joint_pos": 0.70,
                "grasp_threshold": 0.25,
            },
        )

        self.rewards.cube_move_towards_command = RewTerm(
            func=mdp.asset_vel_to_command,
            weight=0.0,
            params={
                "command_name": "object_pose",
                "asset_cfg": object_cfg,
                "robot_cfg": robot_cfg,
                "cube_length": cube_length,
            },
        )

        self.rewards.object_lifted_bonus = RewTerm(
            func=mdp.object_lifted_bonus,
            weight=5.0,
            params={
                "object_cfg": object_cfg,
                "lift_height": 0.12,
            },
        )

        self.rewards.object_goal_success_bonus = RewTerm(
            func=mdp.object_goal_success_bonus,
            weight=25.0,
            params={
                "command_name": "object_pose",
                "object_cfg": object_cfg,
                "robot_cfg": robot_cfg,
                "lift_height": 0.12,
                "goal_threshold": 0.05,
            },
        )

        self.rewards.joint_limit_avoidance = RewTerm(
            func=mdp.joint_limit_distance_clamped,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]"]),
                "margin": 0.15,
            },
        )

        # Keep penalties from becoming dominant.
        self.rewards.action_rate.weight = -0.0001
        self.rewards.joint_vel.weight = -0.001

        # Disable old/default IsaacLab lift terms if present.
        for _reward_name in [
            "reaching_object",
            "lifting_object",
            "object_goal_tracking",
            "object_goal_tracking_fine_grained",
            "approach_from_above",
            "single_gated_pick_place",
            "diag_reach_object",
            "diag_lift_progress",
            "diag_lifted",
            "diag_goal_tracking_lifted",
            "diag_success",
        ]:
            if hasattr(self.rewards, _reward_name):
                getattr(self.rewards, _reward_name).weight = 0.0

        # ReachCubePick-style curriculum.
        # First learn secure grasp/contact. Then increase transport-to-target reward.
        self.curriculum.contact_grasp_150k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "contact_grasp", "weight": 7.0, "num_steps": 150000},
        )
        self.curriculum.contact_grasp_250k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "contact_grasp", "weight": 11.0, "num_steps": 250000},
        )
        self.curriculum.contact_grasp_350k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "contact_grasp", "weight": 15.0, "num_steps": 350000},
        )
        self.curriculum.contact_grasp_550k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "contact_grasp", "weight": 13.0, "num_steps": 550000},
        )
        self.curriculum.contact_grasp_750k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "contact_grasp", "weight": 9.0, "num_steps": 750000},
        )

        self.curriculum.cube_command_dist_200k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 1.0, "num_steps": 200000},
        )
        self.curriculum.cube_command_dist_500k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 5.0, "num_steps": 500000},
        )
        self.curriculum.cube_command_dist_800k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 9.0, "num_steps": 800000},
        )
        self.curriculum.cube_command_dist_1200k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 13.0, "num_steps": 1200000},
        )
        self.curriculum.cube_command_dist_1600k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 17.0, "num_steps": 1600000},
        )
        self.curriculum.cube_command_dist_2200k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_command_dist", "weight": 23.0, "num_steps": 2200000},
        )

        self.curriculum.cube_move_towards_command_300k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_move_towards_command", "weight": 0.5, "num_steps": 300000},
        )
        self.curriculum.cube_move_towards_command_700k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_move_towards_command", "weight": 1.5, "num_steps": 700000},
        )
        self.curriculum.cube_move_towards_command_1100k = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "cube_move_towards_command", "weight": 2.5, "num_steps": 1100000},
        )

        # Easier initial target distribution than the UR10e repo, because your
        # Kinova task is currently less stable.
        self.commands.object_pose.ranges.pos_x = (0.42, 0.58)
        self.commands.object_pose.ranges.pos_y = (-0.12, 0.12)
        self.commands.object_pose.ranges.pos_z = (0.14, 0.24)

        # Cube reset with random yaw
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.06, 0.06),
            "z": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
# End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.markers["connecting_line"].radius = 0.0001
        marker_cfg.markers["connecting_line"].height = 0.0001
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gen3n7_instanceable/end_effector_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gen3n7_instanceable/end_effector_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.10],
                    ),
                ),
            ],
        )

        # Hide the command's current-pose marker — only show the goal target
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (
            0.0,
            0.0,
            0.0,
        )


@configclass
class Gen3LiftAndPlaceEnvCfg_PLAY(Gen3LiftAndPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
