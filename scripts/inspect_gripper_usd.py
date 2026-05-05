"""Inspect the gripper joints in the existing USD.

Prints, for each gripper joint:
  - joint type
  - axis
  - lower / upper limits
  - drive stiffness / damping (if a drive is present)
  - mimic info: master joint, gear ratio, offset (if a mimic is present)

Run from kinova_isaaclab_sim2real/:
    conda activate env_isaaclab
    python scripts/inspect_gripper_usd.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect gripper joints in USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is launched."""

from pxr import Usd, UsdPhysics, PhysxSchema

USD_PATH = (
    "/home/ido-raviel/RL_spot/kinova_arm/kinova_isaaclab_sim2real/"
    "source/gen3/gen3_2f85/gen3_2f85_fixed/gen3_2f85_fixed.usd"
)

GRIPPER_KEYWORDS = ("knuckle_joint", "finger_tip_joint", "robotiq_85")


def has_api(prim, api_cls):
    return prim.HasAPI(api_cls)


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        print(f"[ERROR] Cannot open: {USD_PATH}")
        return

    print(f"[INFO] Opened: {USD_PATH}\n")

    joints_found = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            if any(k in name for k in GRIPPER_KEYWORDS):
                joints_found.append(prim)

    if not joints_found:
        print("[WARN] No gripper joints found by name. Showing every revolute joint instead.")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joints_found.append(prim)

    for prim in joints_found:
        print("=" * 80)
        print(f"Joint: {prim.GetName()}")
        print(f"Path : {prim.GetPath()}")
        print(f"Type : {prim.GetTypeName()}")

        # axis
        axis_attr = prim.GetAttribute("physics:axis")
        if axis_attr and axis_attr.HasValue():
            print(f"Axis : {axis_attr.Get()}")

        # limits
        lo = prim.GetAttribute("physics:lowerLimit")
        hi = prim.GetAttribute("physics:upperLimit")
        if lo and lo.HasValue():
            print(f"Lower limit: {lo.Get()}")
        if hi and hi.HasValue():
            print(f"Upper limit: {hi.Get()}")

        # drive
        if UsdPhysics.DriveAPI.CanApply(prim, "angular"):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                stiff = drive.GetStiffnessAttr()
                damp = drive.GetDampingAttr()
                target = drive.GetTargetPositionAttr()
                if stiff and stiff.HasValue():
                    print(f"Drive stiffness: {stiff.Get()}")
                if damp and damp.HasValue():
                    print(f"Drive damping  : {damp.Get()}")
                if target and target.HasValue():
                    print(f"Drive target   : {target.Get()}")

        # mimic (PhysxMimicJointAPI)
        try:
            for inst in ("rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
                if PhysxSchema.PhysxMimicJointAPI.CanApply(prim, inst):
                    mimic = PhysxSchema.PhysxMimicJointAPI.Get(prim, inst)
                    if mimic:
                        ref = mimic.GetReferenceJointRel()
                        gear = mimic.GetGearingAttr()
                        offset = mimic.GetOffsetAttr()
                        ref_axis = mimic.GetReferenceJointAxisAttr()
                        targets = ref.GetTargets() if ref else []
                        if targets or (gear and gear.HasValue()):
                            print(f"Mimic [{inst}]:")
                            print(f"  reference joint     : {targets}")
                            if ref_axis and ref_axis.HasValue():
                                print(f"  reference axis      : {ref_axis.Get()}")
                            if gear and gear.HasValue():
                                print(f"  gearing (multiplier): {gear.Get()}")
                            if offset and offset.HasValue():
                                print(f"  offset              : {offset.Get()}")
        except Exception as e:
            print(f"[mimic check error]: {e}")

        print()


if __name__ == "__main__":
    main()
    simulation_app.close()
