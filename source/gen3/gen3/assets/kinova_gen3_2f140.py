from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_REPO_ROOT = Path(__file__).resolve().parents[4]
USD_PATH = _REPO_ROOT / "test_sim" / "kinova_gen3_robotiq_2f_140.usd"


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
