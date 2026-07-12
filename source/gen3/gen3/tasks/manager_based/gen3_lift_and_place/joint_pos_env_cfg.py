"""Environment configurations for the Gen3 lift-and-place tasks.

Defines three environments:

  Gen3LiftAndPlaceEnvCfg        — base training env: single end-to-end policy
                                   lifts and places a cube at a random 3-D target.
  Gen3LiftAndPlaceEnvCfg_PLAY   — play variant with fewer envs and no obs noise.
  Gen3LiftAndPlaceChainEnvCfg   — inference-only env that exposes both the grasp
                                   ('policy') and reach ('reach') observation groups
                                   so the play script can feed each frozen policy
                                   the observations it was trained on.
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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp

from isaaclab.markers.config import SPHERE_MARKER_CFG  # isort: skip
from gen3.assets import KINOVA_GEN3_2F140_CFG  # isort: skip


@configclass
class Gen3LiftAndPlaceEnvCfg(LiftEnvCfg):
    """Gen3 lift-and-place base environment.

    Inherits from LiftEnvCfg and configures the Gen3 + 2F-140 robot, the DexCube
    object, and the ee_frame sensor. Rewards are inherited from LiftEnvCfg unchanged.
    Sim timing (dt=1/60, decimation=2) matches the reach and grasp tasks.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Sim ----
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 60.0

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
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
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
    """Play variant: fewer envs, wider spacing, observation noise disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class _ReachPolicyObsCfg(ObsGroup):
    """Observation layout the frozen reach_with_orientation policy was trained on.

    Term order must match the reach policy's training env exactly:
    joint_pos, joint_vel, ee_pose command, last (arm) action.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    pose_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "ee_pose"}
    )
    # reach was trained arm-only (7 dims) -> slice last_action to arm_action
    actions = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Gen3LiftAndPlaceChainEnvCfg(Gen3LiftAndPlaceEnvCfg):
    """Inference env that chains the two frozen pretrained policies.

    Flow (driven by the play script): reach -> pause -> grasp.
      * The arm starts like the reach task (USD default pose + a small random joint
        offset), so the reach policy has real work to do (it is NOT pre-placed above
        the cube).
      * The cube spawns randomly on the table, in the intersection of grasp's cube
        range and reach's trained workspace (x in [0.35, 0.55], y in [-0.2, 0.2]).
      * ``ee_pose`` command -> phase-1 target for the reach policy; never auto-resamples
        within an episode (the play script overwrites ``pose_command_b`` each reset to
        sit above the cube, aligned to its yaw).
      * ``reach`` obs group -> matches the reach policy's training layout (the inherited
        ``policy`` group already matches grasp).
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Episode length: the chain runs reach -> pause -> grasp -> pause -> carry in
        # sequence, so it needs more than the 5 s single-task default (else it ends
        # mid-chain). Frozen policies don't observe time, so a longer episode is safe.
        self.episode_length_s = 10.0

        # --- Arm start: like the reach task — USD default pose + small random joint
        # offset, so the reach policy has real work to do. Set position_range to
        # (0.0, 0.0) for a fixed default start.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.4, 0.4), "velocity_range": (0.0, 0.0)},
        )

        # --- Cube: random on the table. Range is grasp's cube range clipped to reach's
        # trained workspace (reach was trained on y in [-0.2, 0.2], grasp on [-0.3, 0.3]).
        self.scene.object.init_state.pos = [0.45, 0.0, 0.055]
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.10, 0.10),   # -> x in [0.35, 0.55]
            "y": (-0.20, 0.20),   # -> y in [-0.20, 0.20]
            "z": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

        # --- Place target (object_pose): the range the grasp policy was trained on.
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.45),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
        # Never auto-resample within an episode: the play script overwrites the place
        # target (a 2 cm in-place lift) once grasp takes over.
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

        # --- Phase-1 reach target (ee_pose). Huge resampling interval -> stays fixed
        # within an episode; the play script writes pose_command_b above the cube after
        # every reset.
        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="end_effector_link",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.43, 0.43),
                pos_y=(0.0, 0.0),
                pos_z=(0.30, 0.30),
                roll=(0.0, 0.0),
                pitch=(math.pi, math.pi),
                yaw=(0.0, 0.0),
            ),
        )

        # Target markers are drawn by the play script (phase-colored), so disable the
        # commands' built-in debug markers to avoid duplicate spheres.
        self.commands.object_pose.debug_vis = False
        self.commands.ee_pose.debug_vis = False

        # Second obs group for the reach policy; corruption off for inference.
        self.observations.reach = _ReachPolicyObsCfg()
        self.observations.policy.enable_corruption = False
