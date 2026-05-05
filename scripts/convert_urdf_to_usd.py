"""Re-convert the Gen3 + 2F-85 URDF into USD with mimic joints flattened.

Run from kinova_isaaclab_sim2real/:
    conda activate env_isaaclab
    python scripts/convert_urdf_to_usd.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert Gen3+2F85 URDF to USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is launched."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

REPO_ROOT = "/home/ido-raviel/RL_spot/kinova_arm/kinova_isaaclab_sim2real"
URDF_PATH = f"{REPO_ROOT}/source/gen3/gen3_2f85/gen3_2f85_fixed.urdf"
USD_DIR   = f"{REPO_ROOT}/source/gen3/gen3_2f85/gen3_2f85_fixed"
USD_FILE  = "gen3_2f85_fixed.usd"


def main():
    assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"

    cfg = UrdfConverterCfg(
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        usd_file_name=USD_FILE,
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,   # <-- the fix
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
        ),
    )

    print("=" * 80)
    print(f"Input URDF : {URDF_PATH}")
    print(f"Output USD : {os.path.join(USD_DIR, USD_FILE)}")
    print("convert_mimic_joints_to_normal_joints = True")
    print("=" * 80)

    converter = UrdfConverter(cfg)
    print(f"\n[OK] Generated USD: {converter.usd_path}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
