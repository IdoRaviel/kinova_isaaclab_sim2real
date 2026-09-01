"""Kinova Gen3 7-DoF arm + Robotiq 2F-140 gripper asset configuration.

Loads the robot from the vendored USD at gen3/assets/gen3_2f140/ (alongside
this file) and exposes KINOVA_GEN3_2F140_CFG, the ArticulationCfg imported by
all training environments (reach, grasp, lift-and-place).

Actuator gains (stiffness/damping) were tuned for stable joint-position
control at dt=1/60 s, decimation=2, matching the sim timing used across
all three tasks.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


USD_PATH = Path(__file__).resolve().parent / "gen3_2f140" / "kinova_gen3_robotiq_2f_140.usd"

# Distance (m) from the wrist flange (end_effector_link) to the fingertip grasp point,
# along the flange's local z-axis. Used everywhere an `ee_frame` FrameTransformer or the
# grasp task's IK reset needs the fingertip pose rather than the flange pose -- keeping a
# single source of truth here means the reach/grasp/lift_and_place env configs and the
# grasp IK reset can never drift out of sync with each other.
FINGERTIP_OFFSET_M = 0.21


KINOVA_GEN3_2F140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        activate_contact_sensors=True,
    ),
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
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            stiffness={
                "joint_[1-4]": 60.0,
                "joint_[5-7]": 25.0,
            },
            damping={
                "joint_[1-4]": 4.0,
                "joint_[5-7]": 2.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=37.52,
            damping=0.00125,
            effort_limit_sim=1000.0,
        ),
    },
)
