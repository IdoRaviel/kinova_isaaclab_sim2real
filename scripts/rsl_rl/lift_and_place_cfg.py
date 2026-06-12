# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tunable settings for ``play_lift_and_place.py`` (the reach + grasp chaining rollout).

All knobs live here so the play script stays focused on logic. Edit the values below to
tune the chain. (Launcher options such as ``--headless`` / ``--device`` are still passed
on the command line, since they are consumed by Isaac's AppLauncher.)
"""

from dataclasses import dataclass


@dataclass
class ChainCfg:
    # ---- environment ----
    task: str = "Gen3-LiftAndPlace-Chain-v0"
    num_envs: int = 1

    # ---- frozen policies: each is a checkpoint + the agent.yaml saved with that run ----
    reach_checkpoint: str = "pretrained_models/reach_with_orientation/policy.pt"
    reach_agent_cfg: str = "pretrained_models/reach_with_orientation/agent.yaml"
    # retrained robust grasp; swap to "pretrained_models/grasp/policy.pt" + ".../agent.yaml"
    # to use the old single-pose grasp instead.
    grasp_checkpoint: str = "pretrained_models/robust_grasp/policy.pt"
    grasp_agent_cfg: str = "pretrained_models/robust_grasp/agent.yaml"

    # ---- phase REACH: drive the fingertip to a point above the cube ----
    reach_target_z: float = 0.115   # fingertip target height (m, base frame): 6 cm above the cube (rest 0.055)
    handoff_dist: float = 0.04      # reach is "done" when the fingertip is within this (m) of the target
    reach_timeout: int = 300        # fallback: end the reach phase after this many control steps

    # ---- pauses: hold still between phases (seconds) ----
    pause_after_reach: float = 1.0  # held by the reach policy, before grasp takes over
    pause_after_grasp: float = 1.0  # held by the grasp policy (cube stays lifted), before carry

    # ---- phase GRASP: lift the cube straight up, in place ----
    grasp_lift_z: float = 0.075     # cube-center target height (m): ~2 cm above the table (rest 0.055)

    # ---- phase CARRY: fly the held cube to a random top-down pose in the air ----
    # sampled uniformly inside this box (m, base frame), kept within the reach policy's workspace
    air_x: tuple = (0.40, 0.55)
    air_y: tuple = (-0.15, 0.15)
    air_z: tuple = (0.25, 0.40)

    # ---- misc ----
    real_time: bool = False
    reach_only: bool = False        # debug: run only the reach phase (skip grasp + carry)


CFG = ChainCfg()
