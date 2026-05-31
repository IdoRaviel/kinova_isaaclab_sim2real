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

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp
from .agents.rsl_rl_ppo_cfg import Gen3GraspPPORunnerCfg as _RunnerCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import SPHERE_MARKER_CFG  # isort: skip
from gen3.assets import KINOVA_GEN3_2F140_CFG  # isort: skip


@configclass
class Gen3GraspEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = KINOVA_GEN3_2F140_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint_1": 0.0,
                    "joint_2": 0.3,
                    "joint_3": 0.0,
                    "joint_4": 1.8,
                    "joint_5": 0.0,
                    "joint_6": 0.7,
                    "joint_7": 0.0,
                    "finger_joint": 0.0,
                }
            ),
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
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.30, 0.65),
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

        # --- Object: cube spawned directly below the EE default pose ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.43, 0, 0.055], rot=[1, 0, 0, 0]
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

        # --- Rewards ---
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None

        # Penalty: keep EE near cube (don't drift away)
        self.rewards.ee_to_cube = RewTerm(
            func=mdp.ee_to_cube_distance,
            params={"std": 0.1},
            weight=-0.2,
        )

        # Coarse pull: L2 distance from cube to random 3D target (constant gradient)
        self.rewards.cube_to_target = RewTerm(
            func=mdp.cube_to_target_distance_l2,
            params={"command_name": "object_pose"},
            weight=-1.0,
        )
        # Fine-grained bonus: sharp positive reward when cube is near target
        self.rewards.cube_to_target_fine = RewTerm(
            func=mdp.cube_to_target_distance_tanh,
            params={"std": 0.1, "command_name": "object_pose"},
            weight=+0.25,
        )

        # Action/joint penalties: start at zero, ramp very slowly
        self.rewards.action_rate.weight = 0.0
        self.rewards.joint_vel.weight = 0.0

        # --- Curriculum: ramp penalties over first 30% of training ---
        # num_steps scales automatically when max_iterations or num_steps_per_env changes
        _runner = _RunnerCfg()
        _total_steps = _runner.max_iterations * _runner.num_steps_per_env
        _curriculum_steps = int(_total_steps * 0.7)
        self.curriculum.action_rate.params["weight"] = -5e-4
        self.curriculum.action_rate.params["num_steps"] = _curriculum_steps
        self.curriculum.joint_vel.params["weight"] = -5e-4
        self.curriculum.joint_vel.params["num_steps"] = _curriculum_steps

        # --- Object reset: small random range near default spawn ---
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
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
class Gen3GraspEnvCfg_PLAY(Gen3GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
