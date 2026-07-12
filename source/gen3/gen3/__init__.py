# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""gen3 extension — Kinova Gen3 RL tasks for Isaac Lab.

Registers all Gen3 gymnasium environments (reach, grasp, lift-and-place,
lift-and-place chain) by importing the tasks sub-package.
"""

from .tasks import *
