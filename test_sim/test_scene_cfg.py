import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os

USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kinova_gen3_robotiq_2f_140.usd"
)

GRIPPER_OPEN = {"finger_joint": 0.0}
GRIPPER_CLOSE = {"finger_joint": 0.700}

# Arm hover pose — gripper open, hovering above table.
# Tune these values until the EE printed position matches z ~ 0.055 + desired_height.
ARM_HOVER_POS = {
    "joint_1": 0.0,
    "joint_2": 0.3,
    "joint_3": 0.0,
    "joint_4": 1.8,
    "joint_5": 0.0,
    "joint_6": 0.7,
    "joint_7": 0.0,
}

# Cube spawn position — adjust to align with gripper fingers after seeing EE print.
CUBE_POS = [0.4283, -0.025, 0.0]

KINOVA_GEN3_2F140_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            stiffness={"joint_[1-4]": 60.0, "joint_[5-7]": 25.0},
            damping={"joint_[1-4]": 4.0, "joint_[5-7]": 2.0},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=37.52,
            damping=0.00125,
            effort_limit=1000.0,
        ),
    },
)


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = KINOVA_GEN3_2F140_CFG

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.3, 1.3, 1.3),
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

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
