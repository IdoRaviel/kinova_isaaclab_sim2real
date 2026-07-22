# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tunable settings for ``play_lift_and_place.py`` (the reach + grasp chaining rollout).

All knobs live here so the play script stays focused on logic. Edit the values below to
tune the chain. (Launcher options such as ``--headless`` / ``--device`` are still passed
on the command line, since they are consumed by Isaac's AppLauncher.)
"""

import math
from dataclasses import dataclass


@dataclass
class ChainCfg:
    """All tunable knobs for play_lift_and_place.py.

    Edit these values to tune the reach->grasp->carry chain without touching
    the play script itself.  See inline comments for the meaning of each field.
    """

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
    # Handoff is a hybrid of three tests (whichever fires first):
    #   (a) close AND settled       -> dist < handoff_dist AND ee_speed < handoff_speed
    #   (b) converged/plateaued      -> no progress for handoff_plateau_steps AND dist < handoff_accept_dist
    #   (c) hard timeout fallback    -> reach_steps > reach_timeout
    reach_target_z: float = (
        0.115  # fingertip target height (m, base frame): 6 cm above the cube (rest 0.055)
    )
    handoff_dist: float = 0.05  # (a) "close": fingertip within this (m) of the target
    handoff_speed: float = (
        0.03  # (a) "settled": fingertip speed below this (m/s) -> clean static handoff
    )
    handoff_accept_dist: float = (
        0.07  # (b) only accept a plateaued handoff if at least this close (m)
    )
    handoff_plateau_steps: int = (
        30  # (b) steps with no improvement -> reach has converged (~1 s at 30 Hz)
    )
    handoff_plateau_eps: float = (
        0.003  # (b) min dist improvement (m) that counts as "progress"
    )
    reach_timeout: int = (
        120  # (c) fallback: end the reach phase after this many control steps (~4 s)
    )

    # ---- pauses: hold still between phases (seconds) ----
    pause_after_reach: float = 0.3  # held by the reach policy, before grasp takes over
    pause_after_grasp: float = (
        0.1  # held by the grasp policy (cube stays lifted), before carry
    )

    # ---- phase GRASP: lift the cube straight up, in place ----
    grasp_lift_z: float = (
        0.155  # cube-center target height (m): 10 cm above the resting cube (rest 0.055)
    )
    grasp_lift_tol: float = (
        0.02  # advance when |cube_z - grasp_lift_z| <= this (m): cube is AT the target
    )
    # NOTE: no grasp timeout -- if the grasp never lifts the cube, it keeps trying until the
    # episode ends (intentional: failures run through rather than aborting early).

    # ---- phase CARRY: fly the held cube to a random top-down pose in the air ----
    # sampled uniformly inside this box (m, base frame), kept within the reach policy's workspace
    air_x: tuple = (0.40, 0.60)
    air_y: tuple = (-0.25, 0.25)
    air_z: tuple = (0.25, 0.40)
    # second-reach (carry) target orientation ranges (rad), sampled uniformly. Kept inside
    # the reach policy's TRAINED range (gen3_reach: pitch in [pi/2, pi], yaw in [-pi/2, pi/2],
    # roll fixed 0) but shrunk -- asking for poses outside that range makes reach flail.
    #   pitch=pi -> blue/approach points straight DOWN; pitch<pi tilts it FORWARD (away from arm).
    #   Never let pitch exceed pi (that tilts toward the arm/back, which reach never trained on).
    air_roll: tuple = (0.0, 0.0)  # reach trained roll=0 -> keep fixed (else OOD)
    air_pitch: tuple = (
        math.pi / 2 + math.pi / 6,
        math.pi,
    )  # 120 deg .. straight-down: matches the reach training pitch range (no OOD)
    air_yaw: tuple = (-0.3, 0.3)  # swing approach left/right a little

    # ---- misc ----
    real_time: bool = False
    reach_only: bool = False  # debug: run only the reach phase (skip grasp + carry)


CFG = ChainCfg()
