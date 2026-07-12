# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registry for the gen3 extension.

Discovers and imports all task sub-packages (gen3_reach, gen3_grasp,
gen3_lift_and_place) so their gym.register() calls run at import time.
"""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
