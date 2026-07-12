# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom curriculum terms for the Gen3 reach task.

Provides modify_reward_weight_ramp, which linearly ramps a reward term's
weight over a training window. Used to ease in the action-rate and
joint-velocity smoothness penalties after the policy has learned to reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight_ramp(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
):
    """Linearly ramp a reward term's weight from ``start_weight`` to ``end_weight`` as the
    global step counter goes from ``start_step`` to ``end_step`` (clamped outside the window).

    Unlike ``modify_reward_weight`` (a step change), this eases the weight in gradually.
    """
    step = env.common_step_counter
    if step <= start_step:
        w = start_weight
    elif step >= end_step:
        w = end_weight
    else:
        t = (step - start_step) / (end_step - start_step)
        w = start_weight + t * (end_weight - start_weight)
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = w
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return w
