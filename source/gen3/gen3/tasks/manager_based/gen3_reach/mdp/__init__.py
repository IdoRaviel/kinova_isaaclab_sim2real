# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for the Gen3 reach task.

Re-exports Isaac Lab's built-in MDP terms (observations, actions, events, rewards)
and adds task-specific fingertip-tracking rewards, gripper reset events, and the
linear reward-weight curriculum.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
