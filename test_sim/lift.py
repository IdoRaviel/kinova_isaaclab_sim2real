"""Grasp cube, lift to random position, hold 2 s, then reset and repeat."""

import argparse
import random
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Lift cube to random positions with reset")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg

from test_scene_cfg import ARM_HOVER_POS, GRIPPER_CLOSE, GRIPPER_OPEN, TestSceneCfg

JOINT_RANDOM_BOUNDS = {
    "joint_1": (-0.8, 0.8),
    "joint_2": (0.1, 0.5),
    "joint_3": (-0.3, 0.3),
    "joint_4": (1.1, 1.8),
    "joint_5": (-0.3, 0.3),
    "joint_6": (0.5, 1.2),
    "joint_7": (-0.5, 0.5),
}

SETTLE_TIME = 1  # s — arm and cube settle
CLOSE_TIME = 0.5  # s — gripper closes
HOLD_TIME = 2.0  # s — hold random position before reset


def sample_arm_target(arm_names, device):
    vals = [random.uniform(*JOINT_RANDOM_BOUNDS[n]) for n in arm_names]
    return torch.tensor(vals, device=device).unsqueeze(0)


def main():
    sim_cfg = SimulationCfg(dt=1 / 60)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.2), target=(0.5, 0.0, 0.5))

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]

    arm_names = list(ARM_HOVER_POS.keys())
    gripper_names = list(GRIPPER_OPEN.keys())
    arm_ids, _ = robot.find_joints(arm_names)
    gripper_ids, _ = robot.find_joints(gripper_names)

    arm_hover = torch.tensor(
        [ARM_HOVER_POS[n] for n in arm_names], device=sim.device
    ).unsqueeze(0)
    g_open = torch.tensor(
        [GRIPPER_OPEN[n] for n in gripper_names], device=sim.device
    ).unsqueeze(0)
    g_close = torch.tensor(
        [GRIPPER_CLOSE[n] for n in gripper_names], device=sim.device
    ).unsqueeze(0)

    env_ids = torch.tensor([0], device=sim.device, dtype=torch.long)

    # One warm-up step to populate world-frame data buffers
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_cfg.dt)
    cube_init_state = cube.data.root_state_w.clone()

    def reset_scene():
        robot.write_joint_state_to_sim(
            arm_hover, torch.zeros_like(arm_hover), joint_ids=arm_ids, env_ids=env_ids
        )
        robot.write_joint_state_to_sim(
            g_open, torch.zeros_like(g_open), joint_ids=gripper_ids, env_ids=env_ids
        )
        robot.reset(env_ids)
        cube.write_root_state_to_sim(cube_init_state, env_ids=env_ids)
        cube.reset(env_ids)

    dt = sim_cfg.dt
    t_close = SETTLE_TIME
    t_lift = SETTLE_TIME + CLOSE_TIME
    t_reset = SETTLE_TIME + CLOSE_TIME + HOLD_TIME

    arm_target, gripper_target = arm_hover, g_open
    step = 0
    episode = 1
    print(f"\n[Episode {episode}] Starting...")

    while simulation_app.is_running():
        t = step * dt

        if t >= t_reset:
            episode += 1
            print(f"\n[Episode {episode}] Resetting scene...")
            reset_scene()
            step = 0
            arm_target, gripper_target = arm_hover, g_open

        elif t >= t_lift:
            if step == int(t_lift / dt):
                arm_target = sample_arm_target(arm_names, sim.device)
                named = {
                    n: f"{v:.2f}" for n, v in zip(arm_names, arm_target[0].tolist())
                }
                print(f"[t={t:.1f}s] Random target: {named}")
            gripper_target = g_close

        elif t >= t_close:
            if step == int(t_close / dt):
                print(f"[t={t:.1f}s] Closing gripper...")
            arm_target, gripper_target = arm_hover, g_close

        else:
            arm_target, gripper_target = arm_hover, g_open

        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
