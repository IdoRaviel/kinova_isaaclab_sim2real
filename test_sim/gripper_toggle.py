"""Hold arm in hover pose above table with cube between open gripper fingers."""

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Static hover pose with cube")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg

from test_scene_cfg import ARM_HOVER_POS, GRIPPER_OPEN, TestSceneCfg


def main():
    sim_cfg = SimulationCfg(dt=1 / 60)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.2), target=(0.5, 0.0, 0.5))

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    arm_names    = list(ARM_HOVER_POS.keys())
    gripper_names = list(GRIPPER_OPEN.keys())
    arm_ids,     _ = robot.find_joints(arm_names)
    gripper_ids, _ = robot.find_joints(gripper_names)

    arm_targets    = torch.tensor([ARM_HOVER_POS[n] for n in arm_names],    device=sim.device).unsqueeze(0)
    gripper_targets = torch.tensor([GRIPPER_OPEN[n]  for n in gripper_names], device=sim.device).unsqueeze(0)

    dt = sim_cfg.dt
    step = 0

    while simulation_app.is_running():
        robot.set_joint_position_target(arm_targets,    joint_ids=arm_ids)
        robot.set_joint_position_target(gripper_targets, joint_ids=gripper_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        # Print EE position once after settling (at t=2s)
        if step == int(2.0 / dt):
            ee_idx = robot.body_names.index("end_effector_link")
            ee_pos = robot.data.body_pos_w[0, ee_idx]
            print(f"\n[INFO] EE position after settling:")
            print(f"       x={ee_pos[0]:.4f}  y={ee_pos[1]:.4f}  z={ee_pos[2]:.4f}")
            print(f"       table surface z=0.055  →  gripper is {(ee_pos[2]-0.055)*100:.1f} cm above table\n")

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
