import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import SPHERE_MARKER_CFG  # isort: skip
from gen3.assets import KINOVA_GEN3_2F140_CFG  # isort: skip


@configclass
class Gen3LiftAndPlaceEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot: use USD default joint positions ---
        self.scene.robot = KINOVA_GEN3_2F140_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # --- Actions: arm (7 joints) + gripper (binary open/close) ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.7},
        )

        # --- Command: random 3D target for the cube (pick-and-place) ---
        self.commands.object_pose.body_name = "end_effector_link"
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.45),
            pos_y=(-0.2, 0.2),
            pos_z=(0.30, 0.45),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
        target_marker_cfg = SPHERE_MARKER_CFG.copy()
        target_marker_cfg.markers["sphere"].radius = 0.02
        target_marker_cfg.prim_path = "/Visuals/Command/goal_pose"
        self.commands.object_pose.goal_pose_visualizer_cfg = target_marker_cfg
        current_marker_cfg = SPHERE_MARKER_CFG.copy()
        current_marker_cfg.markers["sphere"].visible = False
        current_marker_cfg.prim_path = "/Visuals/Command/current_pose"
        self.commands.object_pose.current_pose_visualizer_cfg = current_marker_cfg

        # --- Object: cube spawned on table, random position ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
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

        # --- Observations ---
        self.observations.policy.object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_root_frame
        )

        # --- Object reset: random on table surface ---
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.10, 0.14),
            "y": (-0.2, 0.2),
            "z": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

        # --- EE frame: offset from wrist flange to approximate finger center ---
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.markers["sphere"].radius = 0.01
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
                        pos=[0.0, 0.0, 0.21],
                    ),
                ),
            ],
        )


@configclass
class Gen3LiftAndPlaceEnvCfg_PLAY(Gen3LiftAndPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
