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

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from gen3.assets import KINOVA_GEN3_2F85_CFG  # isort: skip


@configclass
class Gen3GraspEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Gen3+gripper as robot
        self.scene.robot = KINOVA_GEN3_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for Gen3 arm (7 joints, continuous)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gen3_joint_[1-7]"],
            scale=0.2, # Reduced to stop the "smashing" from Day 1
            use_default_offset=True,
        )
        # Set actions for Robotiq gripper (binary open/close)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "gen3_robotiq_85_left_knuckle_joint",
                "gen3_robotiq_85_right_knuckle_joint",
                "gen3_robotiq_85_left_inner_knuckle_joint",
                "gen3_robotiq_85_right_inner_knuckle_joint",
                "gen3_robotiq_85_left_finger_tip_joint",
                "gen3_robotiq_85_right_finger_tip_joint",
            ],
            open_command_expr={
                "gen3_robotiq_85_left_knuckle_joint": 0.0,
                "gen3_robotiq_85_right_knuckle_joint": 0.0,
                "gen3_robotiq_85_left_inner_knuckle_joint": 0.0,
                "gen3_robotiq_85_right_inner_knuckle_joint": 0.0,
                "gen3_robotiq_85_left_finger_tip_joint": 0.0,
                "gen3_robotiq_85_right_finger_tip_joint": 0.0,
            },
            close_command_expr={
                "gen3_robotiq_85_left_knuckle_joint": 0.8,
                "gen3_robotiq_85_right_knuckle_joint": 0.8,
                "gen3_robotiq_85_left_inner_knuckle_joint": 0.8,
                "gen3_robotiq_85_right_inner_knuckle_joint": 0.8,
                "gen3_robotiq_85_left_finger_tip_joint": 0.8,
                "gen3_robotiq_85_right_finger_tip_joint": 0.8,
            },
        )

        # Set the body name for the command generator
        self.commands.object_pose.body_name = "gen3_bracelet_link"

        # Set cube as object — random position on table
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
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
        self.observations.policy.object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)

        # Increase reaching reward (base LiftEnvCfg has it at 1.0)
        self.rewards.reaching_object.weight = 2.0

        # Add approach-from-above reward
        self.rewards.approach_from_above = RewTerm(
            func=mdp.approach_from_above,
            params={"margin": 0.04, "std": 0.05},
            weight=2.0,
        )

        # Very gradual curriculum ramp (Mastery Timeline)
        # Reaches -0.01 over 50 million steps (~1000 iterations)
        self.curriculum.action_rate.params["weight"] = -1e-2
        self.curriculum.action_rate.params["num_steps"] = 50000000
        self.curriculum.joint_vel.params["weight"] = -1e-2
        self.curriculum.joint_vel.params["num_steps"] = 50000000

        # Cube reset with random yaw
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.25, 0.25),
            "z": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

        # End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gen3_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gen3_bracelet_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.18],
                    ),
                ),
            ],
        )


@configclass
class Gen3GraspEnvCfg_PLAY(Gen3GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
