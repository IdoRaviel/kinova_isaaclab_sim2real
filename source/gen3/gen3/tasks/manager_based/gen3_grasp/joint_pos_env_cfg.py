from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

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

        # --- Robot ---
        self.scene.robot = KINOVA_GEN3_2F85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "gen3_joint_1": 0.0,
                    "gen3_joint_2": 0.3,
                    "gen3_joint_3": 0.0,
                    "gen3_joint_4": 1.4,
                    "gen3_joint_5": 0.0,
                    "gen3_joint_6": 0.9,
                    "gen3_joint_7": -1.57,
                    "gen3_robotiq_85_left_knuckle_joint": 0.0,
                    "gen3_robotiq_85_right_knuckle_joint": 0.0,
                    "gen3_robotiq_85_left_inner_knuckle_joint": 0.0,
                    "gen3_robotiq_85_right_inner_knuckle_joint": 0.0,
                    "gen3_robotiq_85_left_finger_tip_joint": 0.0,
                    "gen3_robotiq_85_right_finger_tip_joint": 0.0,
                }
            ),
        )

        # --- Actions: arm (7 joints) + gripper (binary open/close) ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gen3_joint_[1-7]"],
            scale=0.2,
            use_default_offset=True,
        )
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

        # --- Command: not used for grasp-only, disable its visualization ---
        self.commands.object_pose.body_name = "gen3_bracelet_link"
        self.commands.object_pose.debug_vis = False

        # --- Object: cube spawned directly below the EE default pose ---
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

        # --- Observations ---
        # Remove target_object_position (uses the command we don't need)
        self.observations.policy.target_object_position = None
        # Add object orientation
        self.observations.policy.object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_root_frame
        )

        # --- Rewards ---
        # Remove inherited rewards that don't apply to grasp-only
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None

        # Reaching: wider tanh for better global gradient
        self.rewards.reaching_object_tanh = RewTerm(
            func=mdp.object_distance_tanh,
            params={"std": 0.15},
            weight=2.0,
        )

        # Dense Grasp: subtle guide to keep fingers shut
        self.rewards.dense_finger_closure = RewTerm(
            func=mdp.dense_finger_closure,
            params={"std": 0.05},
            weight=1.0,
        )

        # Stability: 5.0 bonus — a solid anchor for the grasp
        self.rewards.stable_grasp = RewTerm(
            func=mdp.stable_grasp_duration,
            params={"threshold": 0.05, "duration": 0.5},
            weight=4.0,
        )

        # Dense lifting: massive upward pull to overcome table-safety
        self.rewards.object_height_reward = RewTerm(
            func=mdp.object_height_reward,
            params={"table_z": 0.055},
            weight=1000.0,
        )

        # Lifting: binary bonus for clearing 4cm threshold
        self.rewards.object_lifted_binary = RewTerm(
            func=mdp.object_lifted_binary,
            params={"threshold": 0.04, "table_z": 0.055},
            weight=50.0,
        )

        # Success bonus: one-time spike at 15cm (the final goal)
        self.rewards.cube_lifted_bonus = RewTerm(
            func=mdp.cube_lifted_bonus,
            params={"height_threshold": 0.15},
            weight=100.0,
        )

        # Action rate penalty: start low, ramp up via curriculum
        self.rewards.action_rate.weight = 0.0
        # Joint velocity penalty: start low, ramp up via curriculum
        self.rewards.joint_vel.weight = 0.0

        # --- Termination: end episode on success ---
        self.terminations.cube_lifted_success = DoneTerm(
            func=mdp.cube_lifted_success,
            params={"height_threshold": 0.15},
        )

        # --- Curriculum: gradually increase penalties to enforce smoothness ---
        # Ramp from 0.0 to -5e-4 over ~2000 iterations (98M steps)
        self.curriculum.action_rate.params["weight"] = -5e-4
        self.curriculum.action_rate.params["num_steps"] = 98_000_000
        self.curriculum.joint_vel.params["weight"] = -5e-4
        self.curriculum.joint_vel.params["num_steps"] = 98_000_000

        # --- Object reset: small random range near EE ---
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
        }

        # --- End-effector frame transformer (same fix as lift-and-place) ---
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.markers["connecting_line"].radius = 0.0001
        marker_cfg.markers["connecting_line"].height = 0.0001
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gen3_robotiq_85_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gen3_robotiq_85_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.10],
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
