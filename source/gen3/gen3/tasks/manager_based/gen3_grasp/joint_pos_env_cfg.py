"""Environment configuration for the Gen3 grasp task (Gen3-Grasp-v0).

The arm picks a cube from a random IK-initialized start pose (gripper top-down
above the cube, varied joint configuration) and places it at a random 3-D target.

Key design choices:
  - IK-randomized resets (reset_above_cube_ik) train robustness to the range of
    start configurations the reach policy delivers at handoff.
  - Reward structure: coarse L2 pull (constant gradient) + fine tanh bonus for
    both EE-to-cube and cube-to-target, plus a hard lifting bonus to break the
    "hover without grasping" local optimum.
  - Sim timing (dt=1/60, decimation=2) matches the reach task so both frozen
    policies run at the same control frequency in the chain.
"""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp
from .agents.rsl_rl_ppo_cfg import Gen3GraspPPORunnerCfg as _RunnerCfg

from isaaclab.markers.config import SPHERE_MARKER_CFG  # isort: skip
from gen3.assets import FINGERTIP_OFFSET_M, KINOVA_GEN3_2F140_CFG  # isort: skip


@configclass
class Gen3GraspEnvCfg(LiftEnvCfg):
    """Gen3 grasp environment: pick cube from IK-randomized start, place at 3-D target.

    Inherits scene, observations, rewards, and terminations from LiftEnvCfg and
    overrides them for the Gen3 + 2F-140 hardware and the robust-grasp training setup.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Sim ----
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 60.0

        # --- Robot ---
        # (init_state is inherited from KINOVA_GEN3_2F140_CFG unchanged -- this task
        # does not need a different default pose; the IK-randomized reset below
        # overwrites joint state every episode anyway.)
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
            pos_x=(0.40, 0.55),
            pos_y=(-0.15, 0.15),
            pos_z=(0.08, 0.20),
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
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None

        # Lifting hard reward: +1/step whenever the cube is off the table.
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_is_lifted,
            params={
                "minimal_height": 0.05
            },  # cube rests at 0.055 -> triggers once lifted ~2.5cm
            weight=1.0,
        )

        # Coarse pull: L2 distance from EE to cube (constant, non-saturating gradient).
        # Dense "get to / stay on the cube" signal that counters the drift toward canonical.
        self.rewards.ee_to_cube_l2 = RewTerm(
            func=mdp.ee_to_cube_distance_l2,
            weight=-1.0,
        )
        # Fine-grained: tanh, sharp signal right at the cube (fine-tune)
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
            weight=+1.0,
        )

        # Action/joint penalties: start at zero, ramp very slowly
        self.rewards.action_rate.weight = 0.0
        self.rewards.joint_vel.weight = 0.0

        # --- Curriculum: ramp penalties over first 30% of training ---
        # num_steps scales automatically when max_iterations or num_steps_per_env changes
        _runner = _RunnerCfg()
        _total_steps = _runner.max_iterations * _runner.num_steps_per_env
        _curriculum_steps = int(_total_steps * 0.7)
        self.curriculum.action_rate.params["weight"] = -1e-3
        self.curriculum.action_rate.params["num_steps"] = _curriculum_steps
        self.curriculum.joint_vel.params["weight"] = -5e-4
        self.curriculum.joint_vel.params["num_steps"] = _curriculum_steps

        # --- Reset: random cube pose + IK so the gripper starts top-down above the cube
        # with a varied joint configuration (robustness for the reach->grasp handoff).
        # Replaces the default random object reset (this event sets cube + arm together).
        self.events.reset_object_position = None
        self.events.reset_above_cube_ik = EventTerm(
            func=mdp.reset_above_cube_ik,
            mode="reset",
            params={
                "cube_x": (0.35, 0.55),
                "cube_y": (-0.30, 0.30),
                "cube_z": 0.055,
                "yaw_range": (-math.pi, math.pi),
                "hover": 0.12,
                "jitter_deg": 15.0,
                "num_seeds": 8,
                "cube_jitter": 0.04,  # ±4 cm lateral offset from directly-below
            },
        )

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
                        pos=[0.0, 0.0, FINGERTIP_OFFSET_M],
                    ),
                ),
            ],
        )


@configclass
class Gen3GraspEnvCfg_PLAY(Gen3GraspEnvCfg):
    """Play variant: fewer envs, wider spacing, observation noise disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
