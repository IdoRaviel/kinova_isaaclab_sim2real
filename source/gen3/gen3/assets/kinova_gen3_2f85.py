import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

KINOVA_GEN3_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ido-raviel/RL_spot/kinova_arm/kinova_isaaclab_sim2real/source/gen3/gen3_2f85/gen3_2f85_fixed/gen3_2f85_fixed.usd",
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
            "gen3_joint_1": 0.0,
            "gen3_joint_2": 0.65,
            "gen3_joint_3": 0.0,
            "gen3_joint_4": 1.89,
            "gen3_joint_5": 0.0,
            "gen3_joint_6": 0.6,
            "gen3_joint_7": -1.57,
            "gen3_robotiq_85_left_knuckle_joint": 0.0,   # open
            "gen3_robotiq_85_right_knuckle_joint": 0.0,  # open
            "gen3_robotiq_85_left_inner_knuckle_joint": 0.0,
            "gen3_robotiq_85_right_inner_knuckle_joint": 0.0,
            "gen3_robotiq_85_left_finger_tip_joint": 0.0,
            "gen3_robotiq_85_right_finger_tip_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["gen3_joint_[1-7]"],
            effort_limit_sim={
                "gen3_joint_[1-4]": 87.0,
                "gen3_joint_[5-7]": 20.0,
            },
            stiffness={
                "gen3_joint_[1-4]": 60.0,
                "gen3_joint_[5-7]": 25.0,
            },
            damping={
                "gen3_joint_[1-4]": 4.0,
                "gen3_joint_[5-7]": 2.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gen3_robotiq_85_.*_knuckle_joint",
                "gen3_robotiq_85_.*_finger_tip_joint",
            ],
            effort_limit_sim=50.0,
            stiffness=40.0,
            damping=1.0,
        ),
    },
)
